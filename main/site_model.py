import numpy as np
from collections import deque
from config import *


class GreenInferenceSite:
    """
    对应论文第3章：受限功率下的LLM推理站点模型
    扩展Valve在线/离线混部仿真：online时延敏感，offline吞吐优先
    """
    def __init__(self, site_id, total_gpus):
        self.site_id = site_id
        self.total_gpus = total_gpus
        self.active_gpus = total_gpus
        self.current_freq = GPU_MAX_FREQ

        # 请求队列：存储字典，区分online / offline
        self.request_queue = []

        # KV拆分统计
        self.kv_online_usage = 0.0
        self.kv_offline_usage = 0.0
        self.kv_cache_usage = 0.0

        self.tbt_history = deque(maxlen=100)
        self.current_tbt = 20.0

        # 功率相关
        self.current_power = 0.0
        self.power_budget = 0.0

        # 全局统计
        self.total_requests = 0
        self.dropped_requests = 0
        self.queue_length_history = []
        self.tbt_history_log = []

        # ===== 新增离线统计指标（用于计算最大吞吐） =====
        self.total_online_finished = 0
        self.total_offline_finished = 0
        self.total_offline_generated_tokens = 0
        self.offline_evict_count = 0  # 模拟Valve KV回收事件计数


    def set_power_budget(self, budget_kw):
        """设置站点功率预算（来自风电预测）"""
        self.power_budget = budget_kw * 1000


    def set_config(self, active_gpus, freq):
        """设置当前激活GPU数与运行频率"""
        self.active_gpus = max(1, min(active_gpus, self.total_gpus))
        self.current_freq = np.clip(freq, GPU_MIN_FREQ, GPU_MAX_FREQ)


    def add_requests(self, num_online: int, num_offline: int = 0):
        """
        向站点加入请求
        :param num_online: 在线请求数量
        :param num_offline: 离线请求数量
        """
        for _ in range(int(num_online)):
            self.request_queue.append({
                "req_type": "online",
                "generated_token": 0
            })
        for _ in range(int(num_offline)):
            self.request_queue.append({
                "req_type": "offline",
                "generated_token": 0
            })
        self.total_requests += num_online + num_offline


    def _compute_power(self):
        """计算当前实际功耗：激活卡动态功耗 + 闲置卡静态功耗"""
        active_power = self.active_gpus * (GPU_IDLE_POWER +
                        (GPU_PEAK_POWER - GPU_IDLE_POWER) * (self.current_freq / GPU_MAX_FREQ)**1.8)
        idle_power = (self.total_gpus - self.active_gpus) * GPU_IDLE_POWER
        self.current_power = active_power + idle_power
        return self.current_power


    def _update_kv_and_tbt(self):
        """
        Valve混部逻辑：算力优先online；KV压力大时离线吞吐打折（模拟子层KV回收重算开销）
        返回：(总队列, online队列长度, offline队列长度)
        """
        throughput_per_gpu = self.current_freq / GPU_MAX_FREQ * 60
        total_throughput = self.active_gpus * throughput_per_gpu

        # 拆分队列
        online_queue = [r for r in self.request_queue if r["req_type"] == "online"]
        offline_queue = [r for r in self.request_queue if r["req_type"] == "offline"]

        # KV占用统计：每条未结束请求占用等效KV
        kv_online_sum = len(online_queue) * AVG_OUTPUT_TOKENS
        kv_offline_sum = len(offline_queue) * AVG_OUTPUT_TOKENS
        max_kv_capacity = self.active_gpus * 1000

        self.kv_online_usage = min(1.0, kv_online_sum / max_kv_capacity)
        self.kv_offline_usage = min(1.0, kv_offline_sum / max_kv_capacity)
        self.kv_cache_usage = min(1.0, (kv_online_sum + kv_offline_sum) / max_kv_capacity)

        remain_throughput = total_throughput

        # ---------- 第一阶段：算力优先全部给online（Valve：online优先级最高） ----------
        idx = 0
        while remain_throughput > 0 and idx < len(online_queue):
            req = online_queue[idx]
            rem_tok = AVG_OUTPUT_TOKENS - req["generated_token"]
            add_tok = min(remain_throughput, rem_tok)
            req["generated_token"] += add_tok
            remain_throughput -= add_tok
            if req["generated_token"] >= AVG_OUTPUT_TOKENS:
                online_queue.pop(idx)
                self.total_online_finished += 1
            else:
                idx += 1

        # ---------- 第二阶段：剩余算力给offline；KV压力大，离线吞吐打折，模拟Valve局部KV回收 ----------
        offline_scale = OFFLINE_THROUGHPUT_SCALE_BASE
        if self.kv_cache_usage > KV_THRESHOLD:
            pressure_ratio = (self.kv_cache_usage - KV_THRESHOLD) / (1.0 - KV_THRESHOLD)
            offline_scale = max(0.05, 1.0 - pressure_ratio)
            self.offline_evict_count += 1

        offline_throughput = remain_throughput * offline_scale
        idx = 0
        while offline_throughput > 0 and idx < len(offline_queue):
            req = offline_queue[idx]
            rem_tok = AVG_OUTPUT_TOKENS - req["generated_token"]
            add_tok = min(offline_throughput, rem_tok)
            req["generated_token"] += add_tok
            offline_throughput -= add_tok
            if req["generated_token"] >= AVG_OUTPUT_TOKENS:
                offline_queue.pop(idx)
                self.total_offline_finished += 1
                self.total_offline_generated_tokens += AVG_OUTPUT_TOKENS
            else:
                idx += 1

        # 合并回总队列
        self.request_queue = online_queue + offline_queue

        # ---------- TBT时延：仅统计online负载，离线不计入时延SLA ----------
        if self.kv_online_usage < KV_THRESHOLD:
            base_tbt = 20.0 * (GPU_MAX_FREQ / self.current_freq)
        else:
            overflow = (self.kv_online_usage - KV_THRESHOLD) / (1 - KV_THRESHOLD)
            base_tbt = 20.0 * (GPU_MAX_FREQ / self.current_freq) * (1 + overflow * 8)

        self.current_tbt = base_tbt
        self.tbt_history.append(base_tbt)

        return len(self.request_queue), len(online_queue), len(offline_queue)


    def step(self):
        """
        单步仿真推进，不再接收incoming_requests；请求使用add_requests加入
        返回站点全部遥测指标，上层控制器完全不变，对cw‑slc透明
        """
        queue_len, q_online, q_offline = self._update_kv_and_tbt()
        power = self._compute_power()

        self.queue_length_history.append(queue_len)
        self.tbt_history_log.append(self.current_tbt)

        return {
            "active_gpus": self.active_gpus,
            "freq": self.current_freq,
            "power_w": power,
            "kv_total_usage": self.kv_cache_usage,
            "kv_online_usage": self.kv_online_usage,
            "kv_offline_usage": self.kv_offline_usage,
            "queue_depth": queue_len,
            "queue_online": q_online,
            "queue_offline": q_offline,
            "tbt_ms": self.current_tbt,
            "offline_evict_cnt": self.offline_evict_count,
            "offline_total_tokens": self.total_offline_generated_tokens
        }


    def get_percentile_tbt(self, p=99):
        """获取P99 TBT时延（仅online负载）"""
        if not self.tbt_history_log:
            return 0
        return np.percentile(self.tbt_history_log, p)


    def get_percentile_queue(self, p=99):
        """获取P99总队列长度"""
        if not self.queue_length_history:
            return 0
        return np.percentile(self.queue_length_history, p)
