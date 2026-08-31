import numpy as np
import matplotlib.pyplot as plt
from config import *
from site_model import GreenInferenceSite
from cw_slc import CWSlcController
from baselines import DownclockOnly, IdleOnly, PowerCapping, MaxFLOPS
from cwind_router import CWindRouter
from energy_storage import SharedEnergyStorage


# 覆盖原有字体设置，换成支持全符号字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def run_simulation(controller_cls, exp_name, enable_storage:bool=True):
    """运行单组仿真，返回指标结果，支持online/offline/mixed模式"""
    def get_current_wind_power(t):
        """输入仿真秒t，返回 [site0,site1,site2] 当前风电kW，仿真内完美预知风电（等价风电预测）"""
        minute = t / 60
        for idx in range(len(power_time_points)-1):
            t0 = power_time_points[idx]
            t1 = power_time_points[idx+1]
            if t0 <= minute <= t1:
                alpha = (minute - t0) / (t1 - t0)
                p0_list = WIND_POWER_PROFILE[t0]
                p1_list = WIND_POWER_PROFILE[t1]
                out = [p0_list[s] + alpha*(p1_list[s]-p0_list[s]) for s in range(3)]
                return out
        return WIND_POWER_PROFILE[power_time_points[-1]]

    sites = []
    controllers = []
    for i, gpu_num in enumerate(SITE_GPU_TOTAL):
        site = GreenInferenceSite(i, gpu_num)
        ctrl = controller_cls(site)
        sites.append(site)
        controllers.append(ctrl)
    router = CWindRouter(controllers)

    # =========新增：初始化全局共享储能中心 =========
    shared_storage = SharedEnergyStorage()
    # 储能时序日志
    storage_log = {"soc":[], "charge":[], "discharge":[]}
    # ==============================================


    # =========新增：AI4Science全局离线Job缓冲池 =========
    # buffer_item = (per_job_total_tokens, job_count)
    offline_global_buffer = []
    # 记录每一批原始提交信息，用于统计批完成时延
    batch_submit_records = []
    # 格式：(submit_t_sec, total_job, per_tok, finish_t_sec=None)
    # =============================================

    total_seconds = 120 * 60
    power_time_points = sorted(WIND_POWER_PROFILE.keys())


    # 记录日志
    log = {
        "time": [],
        "power": [[] for _ in range(3)],
        "tbt": [[] for _ in range(3)],
        "queue": [[] for _ in range(3)],
        "kv": [[] for _ in range(3)],
        "kv_online": [[] for _ in range(3)],
        "kv_offline": [[] for _ in range(3)],
        "weights": [[] for _ in range(3)],
        "offline_token_rate": [[] for _ in range(3)]
    }

    for t in range(total_seconds):
        # 1. 插值获取当前风电功率
        # 1. 插值获取【原始未经过储能】各站点风电功率
        minute = t / 60
        raw_wind = []
        for idx in range(len(power_time_points)-1):
            t0 = power_time_points[idx]
            t1 = power_time_points[idx+1]
            if t0 <= minute <= t1:
                alpha = (minute - t0) / (t1 - t0)
                for s in range(3):
                    p0 = WIND_POWER_PROFILE[t0][s]
                    p1 = WIND_POWER_PROFILE[t1][s]
                    raw_wind.append(p0 + alpha*(p1-p0))
                break

                # ============ 共享储能充放电计算 ============
        # ============ 共享储能充放电计算【方案A：仅CWind启用储能】 ============
        if controller_cls.__name__ == "CWSlcController" and enable_storage:
            total_wind_all_site = sum(raw_wind)
            site_demand = [s.current_power / 1000.0 for s in sites]
            total_demand = sum(site_demand)
            supply_total_from_storage = shared_storage.step(total_wind_all_site, total_demand, dt_h=SIM_STEP_H)
            site_budget = []
            for i in range(3):
                if supply_total_from_storage >= 0:
                    # 储能放电：给各个站点补充功率，按均分（或者按原始风电占比）
                    add = supply_total_from_storage / 3.0
                    bud = raw_wind[i] + add
                else:
                    # supply_total_from_storage <0：充电，不削减GPU功率，站点直接使用原始风电
                    bud = raw_wind[i]
                bud = max(0.0, bud)
                site_budget.append(bud)

        else:
            # Downclock / Idle / PowerCapping / MaxFLOPS：基线，不启用储能，直接原始风电
            site_budget = raw_wind.copy()
        # ============================================



        # 2. 每SLC决策周期更新一次本地控制器（3分钟）
        if t % SLC_DECISION_CYCLE == 0:
            for i in range(3):
                telem = {
                    "kv_usage": sites[i].kv_cache_usage,
                    "queue_depth": len(sites[i].request_queue),
                    "tbt_ms": sites[i].current_tbt
                }
                # ✅关键：传入经过共享储能修正后的功率 site_budget[i] *1000 (W)
                controllers[i].step(site_budget[i] * 1000, telem)


        # 3. 每秒更新路由权重
        current_tbts = [s.current_tbt for s in sites]
        router.step(current_tbts)

        # 4. 根据负载模式生成online/offline请求，分发
        total_req = REQUEST_ARRIVAL_RATE
        if WORKLOAD_MODE == "online_only":
            on_req, off_req = total_req, 0
        elif WORKLOAD_MODE == "offline_only":
            on_req, off_req = 0, total_req
        elif WORKLOAD_MODE == "mixed":
            on_req = total_req * (1.0 - OFFLINE_RATIO)
            off_req = total_req * OFFLINE_RATIO
            req_counts = router.distribute_requests(total_req)
            # 按比例拆分online/offline分发到各个站点
            for i in range(3):
                frac = req_counts[i] / total_req if total_req>1e-6 else 0
                site_on = int(on_req * frac)
                site_off = int(off_req * frac)
                sites[i].add_requests(site_on, site_off)

        elif WORKLOAD_MODE == "ai4science_mixed":
            # ========= AI‑for‑Science模式：在线每秒随机；离线使用全局缓冲池 =========
            # 在线流量：依旧立刻分发，在线优先级最高，不受缓冲池影响
            on_req = total_req
            off_req = 0
            req_counts = router.distribute_requests(total_req)
            for i in range(3):
                frac = req_counts[i] / total_req if total_req>1e-6 else 0
                site_on = int(on_req * frac)
                sites[i].add_requests(site_on, 0)

            # 离线批量job到达时刻
            for submit_sec, job_num, per_job_tok in AI4SCIENCE_OFFLINE_BATCHES:
                if t == submit_sec:
                    print(f"[t={t/60:.1f}min] 提交AI4Science离线批量job，总job数={job_num}, per_job_tok={per_job_tok}")
                    batch_submit_records.append({"submit_t": t, "total_job": job_num, "per_tok": per_job_tok, "finish_t": None})
                    if ENABLE_OFFLINE_PREDICT_SCHED:
                        # 开启优化：存入全局缓冲池，暂不下发到站点
                        offline_global_buffer.append( {"per_tok":per_job_tok, "remain_job":job_num} )
                    else:
                        # 关闭优化：旧逻辑，到达立刻全部分发到站点
                        req_counts_batch = router.distribute_requests(job_num)
                        for i in range(3):
                            assign_job = int(req_counts_batch[i])
                            if assign_job > 0:
                                sites[i].add_offline_batch_job(assign_job, per_job_tok)

            # ==========【核心新增】每一秒：根据风电预测，从缓冲池取job下发到站点 ==========
            if ENABLE_OFFLINE_PREDICT_SCHED and len(offline_global_buffer)>0:
                wind_now = get_current_wind_power(t)
                avg_wind_kW = sum(wind_now)/3.0
                # 启发式规则：风电功率越高，允许每秒下发更多离线job；风电低则少下发
                # 风电参考基准：本仿真满载单站点约12‑13kW；>8kW判定为风电富余
                if avg_wind_kW > 8.0:
                    can_dispatch = MAX_DISPATCH_OFFLINE_PER_SEC
                elif avg_wind_kW >5.0:
                    can_dispatch = int(MAX_DISPATCH_OFFLINE_PER_SEC * 0.4)
                else:
                    # 风电低谷：不下发新离线Job，全部留在缓冲池等待
                    can_dispatch = 0

                remain_to_dispatch = can_dispatch
                buf_idx = 0
                loop_cnt = 0
                while remain_to_dispatch>0 and buf_idx < len(offline_global_buffer):
                    loop_cnt +=1
                    if loop_cnt>2000:
                        print(f"warning! heavy dispatch loop t={t}")
                        break
                    #原有逻辑

                    buf_item = offline_global_buffer[buf_idx]
                    if buf_item["remain_job"] <=0:
                        buf_idx +=1
                        continue
                    take = min(remain_to_dispatch, buf_item["remain_job"])
                    # 将take个job分发到各个站点
                    req_counts_batch = router.distribute_requests(take)
                    for i in range(3):
                        assign_job = int(req_counts_batch[i])
                        if assign_job>0:
                            sites[i].add_offline_batch_job(assign_job, buf_item["per_tok"])
                    buf_item["remain_job"] -= take
                    remain_to_dispatch -= take
                    buf_idx +=1
                # 清理缓冲池已经空的条目
                offline_global_buffer = [x for x in offline_global_buffer if x["remain_job"]>0]

        else:
            raise ValueError("WORKLOAD_MODE可选：online_only / offline_only / mixed / ai4science_mixed")

        # 5. 各站点step推进
        for i in range(3):
            _ = sites[i].step()

        # 记录日志
        log["time"].append(t/60)
        for i in range(3):
            log["power"][i].append(sites[i].current_power / 1000)
            log["tbt"][i].append(sites[i].current_tbt)
            log["queue"][i].append(len(sites[i].request_queue))
            log["kv"][i].append(sites[i].kv_cache_usage)
            log["kv_online"][i].append(sites[i].kv_online_usage)
            log["kv_offline"][i].append(sites[i].kv_offline_usage)
            log["weights"][i].append(router.weights[i])

        # 记录共享储能时序：仅CWind保存真实值，基线填充nan
        import math
        if controller_cls.__name__ == "CWSlcController" and enable_storage:
            storage_log["soc"].append(shared_storage.soc)
            storage_log["charge"].append(shared_storage.charge_p_history[-1])
            storage_log["discharge"].append(shared_storage.discharge_p_history[-1])
        else:
            storage_log["soc"].append(math.nan)
            storage_log["charge"].append(math.nan)
            storage_log["discharge"].append(math.nan)


        # 放在 for t in range(total_seconds): 循环的最后
        #if t % 600 == 0:  # 每10分钟打印一次

        # 放在 for t in range(total_seconds): 循环的最后
        #if t % 600 == 0:  # 每10分钟打印一次
        #   print(f"时间 {t//60}min: Site0 活跃GPU={sites[0].active_gpus}, 频率={sites[0].current_freq:.0f} MHz")
        # 临时调试，观察60min附近真实kv，运行完删掉
        #if 58*60 < t < 65*60:
        #    print(f"t={t/60:.1f}min site0 kv={sites[0].kv_cache_usage:.3f}, q={len(sites[0].request_queue)}")

    # =========仿真结束后统计离线job剩余情况 =========
    buf_remain_job = sum(item["remain_job"] for item in offline_global_buffer)
    site_remain_off_job = 0
    for s in sites:
        off_q = [r for r in s.request_queue if r["req_type"]=="offline"]
        site_remain_off_job += len(off_q)
    total_undone_job = buf_remain_job + site_remain_off_job

    print(f"\n--------AI4Science Job统计--------")
    print(f"全局缓冲池剩余job: {buf_remain_job}")
    print(f"各站点队列内剩余离线job: {site_remain_off_job}")
    print(f"仿真结束总未完成离线job: {total_undone_job}")

    # ========== 全局指标统计 ==========
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

    # ===== 计算全系统离线吞吐：总生成token / 总仿真秒数（核心指标，回答你的问题：离线最大吞吐量） =====
    total_off_token = sum(s.total_offline_generated_tokens for s in sites)
    avg_off_token_per_sec = total_off_token / total_seconds

    # 计算平均风电利用率
    wind_budget_list = []
    t_points = sorted(WIND_POWER_PROFILE.keys())
    for t_min in np.linspace(0, 120, 120*60):
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
    ratio = power_arr / wind_budget_arr
    ratio = np.clip(ratio, a_min=0, a_max=1.0)
    avg_util = np.mean(ratio) * 100

    print(f"\n=== {exp_name} | 负载模式:{WORKLOAD_MODE} ===")
    print(f"全局P99 TBT时延: {p99_tbt:.2f} ms")
    print(f"全局P99队列长度: {p99_queue:.2f}")
    print(f"全局P50队列长度: {p50_queue:.2f}")
    print(f"全局P99 KV缓存占用: {p99_kv:.4f}")
    print(f"全局P50 KV缓存占用: {p50_kv:.4f}")
    print(f"全局最大KV缓存占用: {max_kv:.4f}")
    print(f"平均风电利用率: {avg_util:.2f}%")
    print(f"【离线吞吐】系统平均总token/s: {avg_off_token_per_sec:.2f} token/s")

    # return log, p99_tbt, p99_queue, avg_off_token_per_sec
    # return log, p99_tbt, p99_queue, avg_off_token_per_sec, total_undone_job
    return log, p99_tbt, p99_queue, avg_off_token_per_sec, total_undone_job, storage_log


def plot_results(results_dict):
    """
    新版绘图：
    图1：2行3列子图：总指标柱状图3张 + CWind时序3张（标记批量Job提交时刻竖线）
    图2：独立新画布：储能SOC时序、充放电功率时序（仅CWind方案）
    """
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # ========== 第一张图：原有6张子图 ==========
    fig1, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 11))
    ax_bar_tbt, ax_bar_off_tp, ax_bar_undone = axes[0]
    ax_kv_time, ax_queue_time, ax_power_route = axes[1]
    names = list(results_dict.keys())
    bar_colors = ['#2ecc71', '#3498db', '#e67e22', '#9b59b6', '#e74c3c']

    p99_vals = []
    offline_throughput_vals = []
    undone_job_vals = []
    for n in names:
        log, p99_tbt, p99_queue, avg_off_token_per_sec, total_undone_job, storage_log = results_dict[n]
        p99_vals.append(p99_tbt)
        offline_throughput_vals.append(avg_off_token_per_sec)
        undone_job_vals.append(total_undone_job)

    # (0,0) P99 TBT时延
    ax_bar_tbt.bar(names, p99_vals, color=bar_colors)
    ax_bar_tbt.set_ylabel("P99 TBT 时延 (ms)")
    ax_bar_tbt.set_title("各算法 P99 Token间隔时延对比", fontsize=12)
    ax_bar_tbt.grid(axis='y', alpha=0.3)

    # (0,1) 平均离线吞吐
    ax_bar_off_tp.bar(names, offline_throughput_vals, color=bar_colors)
    ax_bar_off_tp.set_ylabel("平均离线吞吐 (token/s)")
    ax_bar_off_tp.set_title("各算法系统平均离线吞吐对比", fontsize=12)
    ax_bar_off_tp.grid(axis='y', alpha=0.3)

    # (0,2) 未完成离线Job
    ax_bar_undone.bar(names, undone_job_vals, color=bar_colors)
    ax_bar_undone.set_ylabel("未完成离线Job数量")
    ax_bar_undone.set_title("仿真结束剩余未完成AI4Science Job", fontsize=12)
    ax_bar_undone.grid(axis='y', alpha=0.3)

    # 取CWind时序数据
    log_cw, _, _, _, _, storage_log_cw = results_dict["CWind+Storage(本文完整方案)"]
    t_arr = log_cw["time"]
    kv_avg = np.mean([log_cw["kv"][i] for i in range(3)], axis=0)
    q_avg = np.mean([log_cw["queue"][i] for i in range(3)], axis=0)
    batch_min = [10, 40, 70, 100]

    # (1,0) KV缓存占用
    ax_kv_time.plot(t_arr, kv_avg, color="#c0392b", label="全局平均KV占用")
    for tm in batch_min:
        ax_kv_time.axvline(x=tm, color="red", linestyle="--", alpha=0.6, label="批量Job提交" if tm == batch_min[0] else "")
    ax_kv_time.set_ylabel("KV缓存占用")
    ax_kv_time.set_xlabel("时间 (分钟)")
    ax_kv_time.set_title("CWind：全局KV缓存时序（标记批量Job提交时刻）", fontsize=11)
    ax_kv_time.legend()
    ax_kv_time.grid(alpha=0.3)

    # (1,1) 全局队列长度
    ax_queue_time.plot(t_arr, q_avg, color="#2980b9", label="全局平均队列长度")
    for tm in batch_min:
        ax_queue_time.axvline(x=tm, color="red", linestyle="--", alpha=0.6)
    ax_queue_time.set_ylabel("队列长度")
    ax_queue_time.set_xlabel("时间 (分钟)")
    ax_queue_time.set_title("CWind：全局请求队列时序（标记批量Job提交时刻）", fontsize=11)
    ax_queue_time.legend()
    ax_queue_time.grid(alpha=0.3)

    # (1,2) 功耗+风电预算 + 路由权重
    for i in range(3):
        ax_power_route.plot(log_cw["time"], log_cw["power"][i], label=f"{SITE_NAMES[i]} 实际功耗")
    wind_times = sorted(WIND_POWER_PROFILE.keys())
    for i in range(3):
        wind_vals = [WIND_POWER_PROFILE[t][i] for t in wind_times]
        ax_power_route.plot(wind_times, wind_vals, '--', alpha=0.7, label=f"{SITE_NAMES[i]} 风电预算")
    ax_route_twin = ax_power_route.twinx()
    for i in range(3):
        ax_route_twin.plot(log_cw["time"], log_cw["weights"][i], linestyle=":", alpha=0.7, label=f"权重{SITE_NAMES[i]}")
    ax_power_route.set_ylabel("功率 (kW)")
    ax_route_twin.set_ylabel("路由权重")
    ax_power_route.set_xlabel("时间 (分钟)")
    ax_power_route.set_title("CWind：站点功耗跟随风电 & 跨站路由权重", fontsize=11)
    ax_power_route.legend(loc="upper left")
    ax_route_twin.legend(loc="upper right")
    ax_power_route.grid(alpha=0.3)

    fig1.tight_layout()

    # ========== 第二张独立画布：储能时序 2行1列 ==========
    fig2, axs_stor = plt.subplots(nrows=2, ncols=1, figsize=(14,9))
    ax_soc, ax_pow = axs_stor

    soc_arr = np.array(storage_log_cw["soc"])
    chg_arr = np.array(storage_log_cw["charge"])
    dis_arr = np.array(storage_log_cw["discharge"])

    # SOC时序
    ax_soc.plot(t_arr, soc_arr, color="#27ae60", label="储能SOC")
    ax_soc.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label="SOC_MAX=0.95")
    ax_soc.axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label="SOC_MIN=0.10")
    for tm in batch_min:
        ax_soc.axvline(x=tm, color="red", linestyle="--", alpha=0.4)
    ax_soc.set_ylabel("SOC（荷电状态）")
    ax_soc.set_title("CWind：共享储能SOC时序曲线", fontsize=12)
    ax_soc.set_xlabel("时间 (分钟)")
    ax_soc.set_ylim(0,1.05)
    ax_soc.legend()
    ax_soc.grid(alpha=0.3)

    # 充放电功率：charge>0充电；discharge>0放电
    ax_pow.plot(t_arr, chg_arr, color="#2980b9", label="充电功率(kW)")
    ax_pow.plot(t_arr, dis_arr, color="#e74c3c", label="放电功率(kW)")
    for tm in batch_min:
        ax_pow.axvline(x=tm, color="red", linestyle="--", alpha=0.4)
    ax_pow.set_ylabel("功率 (kW)")
    ax_pow.set_xlabel("时间 (分钟)")
    ax_pow.set_title("CWind：储能充放电功率时序", fontsize=12)
    ax_pow.legend()
    ax_pow.grid(alpha=0.3)

    fig2.tight_layout()

    plt.show()




if __name__ == "__main__":
    results = {}

    # 基线
    results["仅降频 (Downclock)"] = run_simulation(DownclockOnly, "仅降频 (Downclock)")
    results["仅闲置GPU (Idle)"] = run_simulation(IdleOnly, "仅闲置GPU (Idle)")
    results["功率封顶 (PowerCap)"] = run_simulation(PowerCapping, "功率封顶 (PowerCap)")
    results["Max‑FLOPS (最强基线)"] = run_simulation(MaxFLOPS, "Max‑FLOPS (最强基线)")

    # 消融对照：同一CWind控制器，关闭储能；仍然开启ENABLE_OFFLINE_PREDICT_SCHED
    results["CWind‑noStorage(消融：无储能)"] = run_simulation(CWSlcController, "CWind‑noStorage(消融：无储能)", enable_storage=False)
    # 完整本文方案：CWind + 共享储能
    results["CWind+Storage(本文完整方案)"] = run_simulation(CWSlcController, "CWind+Storage(本文完整方案)", enable_storage=True)

    # 绘图
    plot_results(results)

