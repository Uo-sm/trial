import numpy as np
from config import *

class CWindRouter:
    """
    论文算法2：CWind Adaptive Weight Update
    双环路设计：
    1. 主动路径：站点容量/频率变化时，立即按 N×f 重置权重
    2. 反应路径：每15s用EMA‑TBT做时延反馈修正，仅惩罚慢站点
    Note：本实现做工程简化：proactive直接归一化为概率；论文原版proactive输出中间权重，reactive再做缩放收拢。
    """
    def __init__(self, site_controllers: list):
        self.sites = site_controllers
        self.n_sites = len(site_controllers)

        # 路由权重：概率分布，sum(weights)=1
        self.weights = np.ones(self.n_sites) / self.n_sites

        # 时延EMA状态
        self.tbt_ema = np.zeros(self.n_sites)

        # 上一周期容量记录，用于检测硬件状态变化
        self.prev_capacity = np.zeros(self.n_sites)
        self.prev_freq = np.zeros(self.n_sites)

        self.time_since_last_reactive = 0

    def _get_live_capacity(self):
        """获取各站点实时算力容量 N × f
        返回数组顺序与self.sites严格一一对应
        """
        caps = []
        freqs = []
        for c in self.sites:
            active_gpu = c.site.active_gpus
            freq = c.site.current_freq
            caps.append(active_gpu * freq)
            freqs.append(freq)
        return np.array(caps), np.array(freqs)

    def _proactive_update(self, caps, freqs):
        """主动路径：容量变化时立即更新权重（工程简化：直接归一为概率）"""
        total = caps.sum()
        if total <= 1e-9:
            self.weights = np.ones(self.n_sites) / self.n_sites
        else:
            self.weights = caps / total
        self.prev_capacity = caps.copy()
        self.prev_freq = freqs.copy()

    def _reactive_update(self, current_tbts):
        """反应路径：时延反馈修正，仅惩罚高于均值的站点，抑制震荡迁移"""
        # EMA平滑TBT
        self.tbt_ema = (1 - EMA_ALPHA) * self.tbt_ema + EMA_ALPHA * np.array(current_tbts)
        mean_tbt = np.mean(self.tbt_ema)
        if mean_tbt < 1e-9:
            return

        rho = self.tbt_ema / mean_tbt
        rho = np.clip(rho, 1 - SENSITIVITY_DELTA, 1 + SENSITIVITY_DELTA)

        # 非对称惩罚：只惩罚慢站点(rho>1)，快站点权重不变
        for i in range(self.n_sites):
            if rho[i] > 1.0:
                self.weights[i] /= rho[i]

        sum_w = self.weights.sum()
        if sum_w < 1e-9:
            # 极端保护：所有权重被惩罚至接近0，恢复均等分配
            self.weights = np.ones(self.n_sites) / self.n_sites
        else:
            self.weights = self.weights / sum_w

    def step(self, current_tbts, dt=1.0):
        """
        每秒钟调用一次，对应论文每秒更新权重
        :param current_tbts: list[float]，各站点当前TBT时延，长度等于n_sites
        :param dt: 仿真步长，单位秒
        """
        caps, freqs = self._get_live_capacity()
        # 检测硬件是否发生变化
        if not np.allclose(caps, self.prev_capacity) or not np.allclose(freqs, self.prev_freq):
            self._proactive_update(caps, freqs)
            self.time_since_last_reactive = 0
        else:
            self.time_since_last_reactive += dt
            if self.time_since_last_reactive >= ROUTER_UPDATE_CYCLE:
                self._reactive_update(current_tbts)
                self.time_since_last_reactive = 0

    def distribute_requests(self, total_requests):
        """
        按权重分发请求；floor之后余数优先分配给权重高的站点，避免分配偏差
        """
        counts = np.floor(self.weights * total_requests).astype(int)
        remainder = int(total_requests - counts.sum())

        if remainder > 0:
            # 余数优先分给权重从大到小的站点
            idx_sorted = np.argsort(-self.weights)
            for k in range(remainder):
                counts[idx_sorted[k]] += 1
        return counts
