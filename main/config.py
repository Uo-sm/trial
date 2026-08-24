"""
对应论文第3.2节、第5.2节的硬件参数与阈值标定
默认基于 NVIDIA A100 40GB 配置
"""

# ==================== GPU 硬件参数（A100 40GB） ====================
GPU_PEAK_POWER = 400.0        # 单卡峰值功耗 W
GPU_IDLE_POWER = 120.0        # 单卡闲置静态功耗 W（约30%峰值，对应论文描述）
GPU_MAX_FREQ = 1410           # 最高频率 MHz
GPU_MIN_FREQ = 500            # 最低频率 MHz
FREQ_STEP = 60                # 频率调节步长 Δf = 60MHz（对应论文§4.2）

# KV缓存阈值（A100: 20%，H100: 35%，论文§3.2）
KV_THRESHOLD = 0.20
# TBT时延阈值 ms（交互式业务SLO）
TBT_THRESHOLD = 100.0
# 单实例最大排队请求数（触发容量优先模式）
MAX_QUEUE_DEPTH = 5

# ==================== 控制器时序参数 ====================
SLC_DECISION_CYCLE = 180      # CW-Slc 决策周期 3分钟 = 180s（论文§4.2）
ROUTER_UPDATE_CYCLE = 15      # 路由权重更新周期 15s
EMA_ALPHA = 0.1               # TBT 指数滑动平均系数
SENSITIVITY_DELTA = 0.2       # 时延惩罚灵敏度边界 δ（论文§4.3）

# ==================== 三站点仿真配置（对应论文Fig.8） ====================
# 站点GPU总数配比 2:1:1
SITE_GPU_TOTAL = [32, 16, 16]
SITE_NAMES = ["Site-0 (大)", "Site-1 (中)", "Site-2 (小)"]



REQUEST_ARRIVAL_RATE = 90.0  # 每秒总请求数，满功率下接近饱和，功率下降后出现差异
PREFILL_DECODE_RATIO = 0.16
AVG_OUTPUT_TOKENS = 20

# 风电功率曲线（单位：kW），设置为低于GPU满负载功耗，强制触发功率约束
WIND_POWER_PROFILE = {
    0:  [10.0, 5.0, 5.0],    # 0分钟：Site0 10kW < 12.8kW满载，Site1/2 5kW < 6.4kW满载
    20: [10.5, 5.2, 5.1],
    40: [9.8,  4.8, 4.9],
    60: [5.0,  5.0, 4.5],    # 60分钟：Site0 功率腰斩到 5kW，触发剧烈调度
    80: [5.5,  5.3, 4.8],
    100:[9.5,  4.9, 5.0],    # 100分钟：Site0 功率恢复
    120:[10.0, 5.0, 5.0]
}

# ==================== 在线/离线混部仿真参数（参考Valve论文） ====================
# 负载模式: "online_only" / "offline_only" / "mixed"
WORKLOAD_MODE = "mixed"

# mixed模式下，每秒请求中离线占比 0~1
OFFLINE_RATIO = 0.3

# 离线KV回收压力系数：当总KV超过阈值，离线吞吐的打折系数（模拟Valve子层重算开销）
# kv压力越大，离线能拿到的算力比例越低，0代表完全无法处理，1无损失
OFFLINE_THROUGHPUT_SCALE_BASE = 1.0
