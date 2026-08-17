import numpy as np
import matplotlib.pyplot as plt
from config import *
from site_model import GreenInferenceSite
from cw_slc import CWSlcController
from baselines import DownclockOnly, IdleOnly, PowerCapping, MaxFLOPS
from cwind_router import CWindRouter


# 覆盖原有字体设置，换成支持全符号字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def run_simulation(controller_class, name):
    """运行单组仿真，返回指标结果"""
    # 初始化站点
    sites = []
    controllers = []
    for i, gpu_num in enumerate(SITE_GPU_TOTAL):
        site = GreenInferenceSite(i, gpu_num)
        ctrl = controller_class(site)
        sites.append(site)
        controllers.append(ctrl)
    
    # 初始化路由器
    router = CWindRouter(controllers)
    
    # 时间序列（120分钟，步长1秒）
    total_seconds = 120 * 60
    power_time_points = sorted(WIND_POWER_PROFILE.keys())
    
    # 记录日志：新增kv
    log = {
        "time": [],
        "power": [[] for _ in range(3)],
        "tbt": [[] for _ in range(3)],
        "queue": [[] for _ in range(3)],
        "kv": [[] for _ in range(3)],
        "weights": [[] for _ in range(3)]
    }
    
    slc_counter = 0
    
    for t in range(total_seconds):
        # 1. 插值获取当前风电功率
        minute = t / 60
        # 线性插值
        for idx in range(len(power_time_points)-1):
            t0 = power_time_points[idx]
            t1 = power_time_points[idx+1]
            if t0 <= minute <= t1:
                alpha = (minute - t0) / (t1 - t0)
                powers = []
                for s in range(3):
                    p0 = WIND_POWER_PROFILE[t0][s]
                    p1 = WIND_POWER_PROFILE[t1][s]
                    powers.append(p0 + alpha*(p1-p0))
                break
        
        # 2. 每SLC决策周期更新一次本地控制器（3分钟）
        if t % SLC_DECISION_CYCLE == 0:
            for i in range(3):
                telem = {
                    "kv_usage": sites[i].kv_cache_usage,
                    "queue_depth": len(sites[i].request_queue),
                    "tbt_ms": sites[i].current_tbt
                }
                controllers[i].step(powers[i] * 1000, telem)  # kW转W
        
        # 3. 每秒更新路由权重
        current_tbts = [s.current_tbt for s in sites]
        router.step(current_tbts)
        
        # 4. 分发请求
        total_req = REQUEST_ARRIVAL_RATE * 1.0  # 每秒请求数
        req_counts = router.distribute_requests(total_req)
        
        # 5. 各站点执行推理
        for i in range(3):
            sites[i].step(req_counts[i])
        
        # 记录日志
        log["time"].append(t/60)
        for i in range(3):
            log["power"][i].append(sites[i].current_power / 1000)
            log["tbt"][i].append(sites[i].current_tbt)
            log["queue"][i].append(len(sites[i].request_queue))
            log["kv"][i].append(sites[i].kv_cache_usage)
            log["weights"][i].append(router.weights[i])

        # 放在 for t in range(total_seconds): 循环的最后
        #if t % 600 == 0:  # 每10分钟打印一次
         #   print(f"时间 {t//60}min: Site0 活跃GPU={sites[0].active_gpus}, 频率={sites[0].current_freq:.0f} MHz")
        # 临时调试，观察60min附近真实kv，运行完删掉
        #if 58*60 < t < 65*60:
        #    print(f"t={t/60:.1f}min site0 kv={sites[0].kv_cache_usage:.3f}, q={len(sites[0].request_queue)}")
    
    # 计算全局指标
    all_tbt = []
    all_queue = []
    all_kv = []
    for i in range(3):
        all_tbt.extend(log["tbt"][i])
        all_queue.extend(log["queue"][i])
        all_kv.extend(log["kv"][i])
    
    p99_tbt = np.percentile(all_tbt, 99)
    p99_queue = np.percentile(all_queue, 99)
    p50_queue = np.percentile(all_queue, 50)

    p99_kv = np.percentile(all_kv, 99)
    p50_kv = np.percentile(all_kv, 50)
    max_kv = np.max(all_kv)
    
    # 计算平均风电利用率
    wind_budget_list = []
    t_points = sorted(WIND_POWER_PROFILE.keys())

    for t_min in np.linspace(0, 120, 120*60):
        # 线性插值计算当前时刻风电预算
        for idx in range(len(t_points)-1):
            t0, t1 = t_points[idx], t_points[idx+1]
            if t0 <= t_min <= t1:
                alpha = (t_min - t0) / (t1 - t0)
                site_powers = []
                for s in range(3):
                    p0 = WIND_POWER_PROFILE[t0][s]
                    p1 = WIND_POWER_PROFILE[t1][s]
                    site_powers.append(p0 + alpha*(p1-p0))
                wind_budget_list.append(site_powers)
                break

    wind_budget_arr = np.array(wind_budget_list)
    power_arr = np.array([log["power"][i] for i in range(3)]).T

    # 逐元素限制最大1.0，消除浮点超100%
    ratio = power_arr / wind_budget_arr
    ratio = np.clip(ratio, a_min=0, a_max=1.0)
    avg_util = np.mean(ratio) * 100

    print(f"\n=== {name} 仿真结果 ===")
    print(f"全局P99 TBT时延: {p99_tbt:.2f} ms")
    print(f"全局P99队列长度: {p99_queue:.2f}")
    print(f"全局P50队列长度: {p50_queue:.2f}")
    print(f"全局P99 KV缓存占用: {p99_kv:.4f}")
    print(f"全局P50 KV缓存占用: {p50_kv:.4f}")
    print(f"全局最大KV缓存占用: {max_kv:.4f}")
    print(f"平均风电利用率: {avg_util:.2f}%")
    
    return log, p99_tbt, p99_queue

def plot_results(results_dict):
    """绘制对比图，对应论文Fig.9样式"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 子图1：P99 TBT 对比
    names = list(results_dict.keys())
    p99_vals = [results_dict[n][1] for n in names]
    

    
    axes[0].bar(names, p99_vals, color=['#2ecc71', '#3498db', '#e67e22', '#9b59b6', '#e74c3c'])
    axes[0].set_ylabel("P99 TBT 时延 (ms)")
    axes[0].set_title("各算法 P99 Token间隔时延对比", fontsize=13)
    axes[0].grid(axis='y', alpha=0.3)
    
    # 子图2：CW-Slc方案下三站点功率曲线
    log_cw = results_dict["CWind (本文方案)"][0]
    for i in range(3):
        axes[1].plot(log_cw["time"], log_cw["power"][i], label=f"{SITE_NAMES[i]} 实际功耗")
    # 叠加风电预算曲线
    wind_times = sorted(WIND_POWER_PROFILE.keys())
    for i in range(3):
        wind_vals = [WIND_POWER_PROFILE[t][i] for t in wind_times]
        axes[1].plot(wind_times, wind_vals, '--', label=f"{SITE_NAMES[i]} 风电预算", alpha=0.7)
    axes[1].set_ylabel("功率 (kW)")
    axes[1].set_title("CWind方案：各站点功耗跟随风电波动情况", fontsize=13)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # 子图3：路由权重变化
    for i in range(3):
        axes[2].plot(log_cw["time"], log_cw["weights"][i], label=f"{SITE_NAMES[i]} 流量权重")
    axes[2].set_xlabel("时间 (分钟)")
    axes[2].set_ylabel("流量权重占比")
    axes[2].set_title("CWind跨站路由权重动态变化", fontsize=13)
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = {}
    
    # 运行所有方案
    results["仅降频 (Downclock)"] = run_simulation(DownclockOnly, "仅降频 (Downclock)")
    results["仅闲置GPU (Idle)"] = run_simulation(IdleOnly, "仅闲置GPU (Idle)")
    results["功率封顶 (PowerCap)"] = run_simulation(PowerCapping, "功率封顶 (PowerCap)")
    results["Max-FLOPS (最强基线)"] = run_simulation(MaxFLOPS, "Max-FLOPS (最强基线)")
    results["CWind (本文方案)"] = run_simulation(CWSlcController, "CWind (本文方案)")
    
    # 绘图
    plot_results(results)
