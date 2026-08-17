import numpy as np
from collections import deque
from config import *

class GreenInferenceSite:
    """
    对应论文第3章：受限功率下的LLM推理站点模型
    核心输出：实时功耗、KV缓存占用、队列长度、TBT时延
    """
    def __init__(self, site_id, total_gpus):
        self.site_id = site_id
        self.total_gpus = total_gpus
        self.active_gpus = total_gpus
        self.current_freq = GPU_MAX_FREQ #存疑？
        
        # 运行状态
        self.request_queue = []   # 使用普通list
        self.kv_cache_usage = 0.0
        self.tbt_history = deque(maxlen=100)
        self.current_tbt = 20.0
        
        # 功率相关
        self.current_power = 0.0
        self.power_budget = 0.0
        
        # 统计指标
        self.total_requests = 0
        self.dropped_requests = 0
        self.queue_length_history = []
        self.tbt_history_log = []

    def set_power_budget(self, budget_kw):
        """设置站点功率预算（来自风电预测）"""
        self.power_budget = budget_kw * 1000  # 转W

    def set_config(self, active_gpus, freq):
        """设置当前激活GPU数与运行频率"""
        self.active_gpus = max(1, min(active_gpus, self.total_gpus))
        self.current_freq = np.clip(freq, GPU_MIN_FREQ, GPU_MAX_FREQ)

    def _compute_power(self):
        """计算当前实际功耗：激活卡动态功耗 + 闲置卡静态功耗"""
        active_power = self.active_gpus * (GPU_IDLE_POWER + 
                           (GPU_PEAK_POWER - GPU_IDLE_POWER) * (self.current_freq / GPU_MAX_FREQ)**1.8)
        idle_power = (self.total_gpus - self.active_gpus) * GPU_IDLE_POWER
        self.current_power = active_power + idle_power
        return self.current_power

    def _update_kv_and_tbt(self, incoming_requests):
        """
        模拟KV缓存占用与TBT时延，对应论文Fig.5的非线性关系
        频率越低、并发越高，KV占用越高，超过阈值后TBT指数恶化
        """
        # 把原来的 8 改成 60，提升单卡基准吞吐，匹配真实推理速度
        throughput_per_gpu = self.current_freq / GPU_MAX_FREQ * 60  # 单卡满频 60 token/s
        total_throughput = self.active_gpus * throughput_per_gpu
        
        # 原来：self.request_queue.append(AVG_OUTPUT_TOKENS)
        # 修改后：新请求刚入队，0个token生成，KV占用为0
        for _ in range(int(incoming_requests)):
            self.request_queue.append(0)

        # 处理逻辑同步修改
        processed = 0
        if total_throughput > 0 and self.request_queue:
            gen_need = total_throughput  # 本轮可生成token总数
            idx = 0
            while gen_need > 0 and idx < len(self.request_queue):
                curr_generated = self.request_queue[idx]
                remaining_token = AVG_OUTPUT_TOKENS - curr_generated
                add = min(gen_need, remaining_token)
                self.request_queue[idx] += add
                gen_need -= add
                # 一条请求全部生成完毕，弹出队列
                if self.request_queue[idx] >= AVG_OUTPUT_TOKENS:
                    self.request_queue.pop(idx)
                    processed += 1
                else:
                    idx += 1

        # 原来错误：
        # total_generated_kv = sum(self.request_queue)
        # max_kv_capacity = self.active_gpus * 1000
        # self.kv_cache_usage = min(1.0, total_generated_kv / max_kv_capacity)

        # ✅修正：队列中每一条未结束请求都占用KV缓存，和已经生成token无关
        num_pending_req = len(self.request_queue)   # 队列中未完成请求总条数
        per_req_kv_cost = AVG_OUTPUT_TOKENS         # 单条请求占用KV等效开销
        total_kv_demand = num_pending_req * per_req_kv_cost

        max_kv_capacity = self.active_gpus * 1000
        self.kv_cache_usage = min(1.0, total_kv_demand / max_kv_capacity)
        
        # TBT计算：KV低于阈值时线性，超过后指数恶化
        if self.kv_cache_usage < KV_THRESHOLD:
            base_tbt = 20.0 * (GPU_MAX_FREQ / self.current_freq)
        else:
            overflow = (self.kv_cache_usage - KV_THRESHOLD) / (1 - KV_THRESHOLD)
            base_tbt = 20.0 * (GPU_MAX_FREQ / self.current_freq) * (1 + overflow * 8)
        
        self.current_tbt = base_tbt
        self.tbt_history.append(base_tbt)
        return len(self.request_queue)

    def step(self, incoming_requests):
        """单步仿真推进，返回当前状态指标"""
        queue_len = self._update_kv_and_tbt(incoming_requests)
        power = self._compute_power()
        
        # 记录历史
        self.queue_length_history.append(queue_len)
        self.tbt_history_log.append(self.current_tbt)
        self.total_requests += incoming_requests
        
        return {
            "active_gpus": self.active_gpus,
            "freq": self.current_freq,
            "power_w": power,
            "kv_usage": self.kv_cache_usage,
            "queue_depth": queue_len,
            "tbt_ms": self.current_tbt
        }

    def get_percentile_tbt(self, p=99):
        """获取P99 TBT时延"""
        if not self.tbt_history_log:
            return 0
        return np.percentile(self.tbt_history_log, p)

    def get_percentile_queue(self, p=99):
        """获取P99队列长度"""
        if not self.queue_length_history:
            return 0
        return np.percentile(self.queue_length_history, p)