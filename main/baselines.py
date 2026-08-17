import numpy as np
from config import *
from site_model import GreenInferenceSite

class DownclockOnly:
    """基线1：仅全局降频，GPU数量固定（论文§5.2）"""
    def __init__(self, site: GreenInferenceSite):
        self.site = site

    def step(self, power_budget_w, telemetry):
        n = self.site.total_gpus
        idle_total = n * GPU_IDLE_POWER
        if power_budget_w <= idle_total:
            # 功率预算连全部GPU空闲功耗都不够，直接压到硬件最低频率
            freq = GPU_MIN_FREQ
        else:
            remaining_power = power_budget_w
            power_per_gpu = remaining_power / n
            dynamic_power = power_per_gpu - GPU_IDLE_POWER
            # 这里dynamic_power一定>0，不再需要max(10,...)
            freq_ratio = (dynamic_power / (GPU_PEAK_POWER - GPU_IDLE_POWER)) ** (1/1.8)
            freq = freq_ratio * GPU_MAX_FREQ

        freq = np.clip(freq, GPU_MIN_FREQ, GPU_MAX_FREQ)
        self.site.set_config(n, freq)



class IdleOnly:
    """基线2：仅闲置整卡，频率固定最高（论文§5.2）"""
    def __init__(self, site: GreenInferenceSite):
        self.site = site

    def step(self, power_budget_w, telemetry):
        # 计算最多能开多少张满频卡
        """
        原循环
        max_n = 0
        for n in range(1, self.site.total_gpus + 1): #依旧n尽可能大
            power = n * GPU_PEAK_POWER + (self.site.total_gpus - n) * GPU_IDLE_POWER
            if power <= power_budget_w:
                max_n = n
        max_n = max(1, max_n)
        """

        # 满频下最大活跃GPU，等价原循环
        dyn_total_budget = power_budget_w - self.site.total_gpus * GPU_IDLE_POWER
        max_n = dyn_total_budget // (GPU_PEAK_POWER - GPU_IDLE_POWER)
        # 上下界约束：最少1卡，最多整机全部GPU
        max_n = max(1, min(max_n, self.site.total_gpus))

        self.site.set_config(max_n, GPU_MAX_FREQ)





class PowerCapping:
    """基线3：硬件功率封顶，匹配nvidia-smi -pl硬件功耗墙约束
    逻辑：永远全开所有GPU，单卡功耗不能超过硬件峰值，多余风电功率闲置浪费
    """
    def __init__(self, site):
        self.site = site

    def step(self, power_budget_w, telemetry):
        total_gpus = self.site.total_gpus
        P_idle = GPU_IDLE_POWER
        P_peak = GPU_PEAK_POWER
        max_dyn_per_gpu = P_peak - P_idle
        total_static = total_gpus * P_idle
        total_dyn_budget = power_budget_w - total_static

        best_n = total_gpus
        if total_dyn_budget <= 0:
            # 仅够整机静态待机，无动态算力
            freq = GPU_MIN_FREQ
        else:
            dyn_per = total_dyn_budget / best_n
            if dyn_per > max_dyn_per_gpu:
                # 触及单卡硬件功耗上限，跑满最高频率
                freq = GPU_MAX_FREQ
            else:
                # 动态功率不足，按DVFS幂次公式降频
                freq_ratio = (dyn_per / max_dyn_per_gpu) ** (1 / 1.8)
                freq = GPU_MAX_FREQ * freq_ratio

        freq = np.clip(freq, GPU_MIN_FREQ, GPU_MAX_FREQ)
        self.site.set_config(best_n, freq)



class MaxFLOPS:
    """基线4：纯最大化N×f，无遥测反馈（论文§5.2，最强对比基线）"""
    def __init__(self, site: GreenInferenceSite):
        self.site = site

    def step(self, power_budget_w, telemetry):
        # 枚举所有(N,f)，选N*f最大的，完全不考虑KV和队列


        best = (1, GPU_MIN_FREQ)
        """
        best_score = 0
        for n in range(1, self.site.total_gpus + 1):
            remaining = power_budget_w - (self.site.total_gpus - n) * GPU_IDLE_POWER
            if remaining <= n * GPU_IDLE_POWER:
                continue
            dyn = remaining / n - GPU_IDLE_POWER
            freq = GPU_MAX_FREQ * (dyn / (GPU_PEAK_POWER - GPU_IDLE_POWER)) ** (1/1.8)
            freq = np.clip(freq, GPU_MIN_FREQ, GPU_MAX_FREQ)
            score = n * freq
            if score > best_score:
                best_score = score
                best = (n, freq)
        self.site.set_config(best[0], best[1])
        """
        # 优化说明：算力目标 N*f 随活跃GPU数量单调递增，
        # 因此从最大n向下搜索，第一个满足功耗约束的n即为最优解，无需完整遍历打分。
        for n in range(self.site.total_gpus, 0, -1):
            remaining = power_budget_w - (self.site.total_gpus - n) * GPU_IDLE_POWER
            if remaining > n * GPU_IDLE_POWER:
                dyn = remaining / n - GPU_IDLE_POWER
                freq = GPU_MAX_FREQ * (dyn / (GPU_PEAK_POWER - GPU_IDLE_POWER)) ** (1/1.8)
                freq = np.clip(freq, GPU_MIN_FREQ, GPU_MAX_FREQ)  
                best = (n, freq)
                break

        self.site.set_config(best[0], best[1])