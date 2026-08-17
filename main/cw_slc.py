import numpy as np
from config import *
from site_model import GreenInferenceSite

class CWSlcController:
    """
    论文算法1：Reactive Site-Level Controller (CW-Slc)
    核心逻辑：功率预算约束下，联合优化「活跃GPU数 + 运行频率」
    基于 KV缓存、队列深度、TBT时延 三个实时信号做反馈控制
    """
    def __init__(self, site: GreenInferenceSite):
        self.site = site
        self.N_curr = site.total_gpus  # 当前激活GPU数状态
        self.F_floor = GPU_MIN_FREQ    # 频率下限状态
        self.congestion_flag = False   # 拥塞标记

    def _generate_candidates(self, power_budget_w):
        """枚举所有满足功率预算的 (N, f) 候选配置"""
        candidates = []
        for n in range(1, self.site.total_gpus + 1):
            # 计算n张卡最大可运行频率
            remaining_power = power_budget_w - (self.site.total_gpus - n) * GPU_IDLE_POWER
            if remaining_power <= n * GPU_IDLE_POWER:
                continue
            # 反推频率（幂律关系近似）
            power_per_gpu = remaining_power / n
            dynamic_power = power_per_gpu - GPU_IDLE_POWER
            if dynamic_power <= 0:
                continue
            freq_ratio = (dynamic_power / (GPU_PEAK_POWER - GPU_IDLE_POWER)) ** (1/1.8)
            max_freq = freq_ratio * GPU_MAX_FREQ
            # 离散化到步长
            #max_freq = int(max_freq // FREQ_STEP) * FREQ_STEP
            #max_freq = np.clip(max_freq, GPU_MIN_FREQ, GPU_MAX_FREQ)
            candidates.append((n, max_freq))
        return candidates

    def step(self, power_budget_w, telemetry):
        """
        单步决策：输入功率预算 + 实时遥测数据，输出最优配置
        对应论文Algorithm 1完整逻辑
        """
        kv = telemetry["kv_usage"]
        q = telemetry["queue_depth"]
        tbt = telemetry["tbt_ms"]
        
        # Step1: 拥塞判断
        self.congestion_flag = q > MAX_QUEUE_DEPTH

        # Step2: 更新频率下限 F_floor（非拥塞状态下）
        if not self.congestion_flag:
            #print(f"kv: {kv}, tbt: {tbt}, q: {q}")
            if kv > KV_THRESHOLD:
                # KV超限，激进抬升频率（2倍步长，论文§4.2）
                self.F_floor += 2 * FREQ_STEP
            elif tbt > TBT_THRESHOLD:
                # TBT超限，正常抬升频率
                self.F_floor += FREQ_STEP
            elif kv < KV_THRESHOLD and tbt < TBT_THRESHOLD and q < MAX_QUEUE_DEPTH/2:
                # 三项全健康，缓慢降频节能
                self.F_floor -= FREQ_STEP
        
        # 钳位频率下限
        self.F_floor = np.clip(self.F_floor, GPU_MIN_FREQ, GPU_MAX_FREQ)

        # Step3: 生成所有可行候选
        all_candidates = self._generate_candidates(power_budget_w)
        if not all_candidates:
            # 极端情况：功率不足以支撑1张卡
            self.N_curr = 1
            self.F_floor = GPU_MIN_FREQ
            self.site.set_config(1, GPU_MIN_FREQ)
            return

        # Step4: 按约束筛选候选
        if self.congestion_flag:
            # 拥塞模式：容量优先，只保留GPU数 ≥ 当前值的候选（论文§4.2）
            valid = [(n, f) for n, f in all_candidates if n >= self.N_curr]
            if not valid:
                valid = all_candidates  # 兜底
        else:
            # 正常模式：频率不低于下限
            valid = [(n, f) for n, f in all_candidates if f >= self.F_floor]
            if not valid:
                valid = all_candidates  # 兜底

        # Step5: 选择 N×f 最大的配置（最大化算力容量，论文§4.2）
        best = max(valid, key=lambda x: x[0] * x[1])
        self.N_curr, best_freq = best

        # 应用配置到站点
        self.site.set_config(self.N_curr, best_freq)