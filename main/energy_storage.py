from config import *

class SharedEnergyStorage:
    """
    三数据中心 全局共享集中式储能中心
    全部站点风电汇入本储能池；实现集群级充放电调配
    """
    def __init__(self):
        self.cap_kwh = SHARED_STORAGE_CAPACITY_KWH
        self.soc = STORAGE_INIT_SOC
        self.soc_min = STORAGE_SOC_MIN
        self.soc_max = STORAGE_SOC_MAX
        self.max_charge_kw = SHARED_MAX_CHARGE_KW
        self.max_discharge_kw = SHARED_MAX_DISCHARGE_KW
        self.cha_eff = STORAGE_CHARGE_EFF
        self.dis_eff = STORAGE_DISCHARGE_EFF

        # 记录时序，用于绘图
        self.soc_history = []
        self.charge_p_history = []
        self.discharge_p_history = []

    def step(self, total_wind_kw, total_demand_kw, dt_h):
        """
        每仿真秒执行一次储能更新
        :param total_wind_kw: 三站点合计风光发电总功率 kW
        :param total_demand_kw: 集群整体功耗需求（全部站点希望消耗的总功率）kW
        :param dt_h: 步长，单位小时
        :return: actual_supply_kw:储能对外实际输出总供电功率(>0放电；<0充电)
        """
        delta_power = total_wind_kw - total_demand_kw
        actual_charge_kw = 0.0
        actual_discharge_kw = 0.0

        if delta_power > 0:
            # -------- 风电富余，给储能充电 --------
            avail_charge = min(delta_power, self.max_charge_kw)
            max_energy_can_accept = (self.soc_max - self.soc) * self.cap_kwh
            max_charge_p_by_soc = max_energy_can_accept / dt_h / self.cha_eff
            actual_charge_kw = min(avail_charge, max_charge_p_by_soc)
            # 充电，计入储能（乘以充电效率）
            energy_in = actual_charge_kw * dt_h * self.cha_eff
            self.soc += energy_in / self.cap_kwh

        else:
            # -------- 风电不足，储能放电弥补缺口 --------
            need_discharge = -delta_power
            avail_discharge = min(need_discharge, self.max_discharge_kw)
            max_energy_can_release = (self.soc - self.soc_min) * self.cap_kwh
            max_discharge_p_by_soc = (max_energy_can_release / dt_h) / self.dis_eff
            actual_discharge_kw = min(avail_discharge, max_discharge_p_by_soc)
            # 放电，储能能量减少（除以放电效率）
            energy_out = actual_discharge_kw * dt_h / self.dis_eff
            self.soc -= energy_out / self.cap_kwh

        # 钳位SOC
        self.soc = max(self.soc_min, min(self.soc_max, self.soc))
        supply_from_storage = actual_discharge_kw - actual_charge_kw

        # 记录日志
        self.soc_history.append(self.soc)
        self.charge_p_history.append(actual_charge_kw)
        self.discharge_p_history.append(actual_discharge_kw)
        return supply_from_storage
