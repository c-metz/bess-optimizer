import os
import numpy as np
import math
from ortools.linear_solver import pywraplp


class optimizer:
    def __init__(self):
        pass

    def step1_optimize_daa(self, n_cycles: int, energy_cap: int, power_cap: int, daa_price_vector: list):
        """
        Calculates optimal charge/discharge schedule on the day-ahead auction (daa) for a given 96-d daa_price_vector.

        Parameters:
        - n_cycles: Maximum number of allowed cycles
        - energy_cap: Energy capacity
        - power_cap: Power capacity
        - daa_price_vector: 96-dimensional daa price vector

        Returns:
        - step1_soc_daa: Resulting state of charge schedule
        - step1_cha_daa: Resulting charge schedule / Positions on DA Auction
        - step1_dis_daa: Resulting discharge schedule / Positions on DA Auction
        - step1_profit_daa: Profit from Day-ahead auction trades
        """

        # Create the solver
        solver = pywraplp.Solver.CreateSolver('CBC')

        if not solver:
            return None

        # Number of quarters
        Q = 96

        # Daily discharged energy limit
        volume_limit = energy_cap * n_cycles

        # Variables
        soc = [solver.NumVar(0, energy_cap, f'soc_{i}') for i in range(Q+1)]
        cha_daa = [solver.NumVar(0, 1, f'cha_daa_{i}') for i in range(Q)]
        dis_daa = [solver.NumVar(0, 1, f'dis_daa_{i}') for i in range(Q)]

        # Constraints

        # SOC initial and final
        solver.Add(soc[0] == 0)
        solver.Add(soc[Q] == 0)

        # SOC dynamics
        for q in range(Q):
            solver.Add(soc[q+1] == soc[q] + (power_cap / 4) * cha_daa[q] - (power_cap / 4) * dis_daa[q])

        # Cycle limits
        solver.Add(solver.Sum([(power_cap / 4) * cha_daa[q] for q in range(Q)]) <= volume_limit)
        solver.Add(solver.Sum([(power_cap / 4) * dis_daa[q] for q in range(Q)]) <= volume_limit)

        # Hourly parity for DA Auction (positions same in each hour's 4 quarters)
        for h in range(24):
            for i in range(1, 4):
                solver.Add(cha_daa[4*h] == cha_daa[4*h + i])
                solver.Add(dis_daa[4*h] == dis_daa[4*h + i])

        # Objective: maximize revenue
        objective = solver.Objective()
        for q in range(Q):
            objective.SetCoefficient(cha_daa[q], - (power_cap/4) * daa_price_vector[q])
            objective.SetCoefficient(dis_daa[q], (power_cap/4) * daa_price_vector[q])
        objective.SetMaximization()

        # Solve
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            step1_soc_daa = [soc[q].solution_value() for q in range(Q)]
            step1_cha_daa = [cha_daa[q].solution_value() for q in range(Q)]
            step1_dis_daa = [dis_daa[q].solution_value() for q in range(Q)]
            step1_profit_daa = sum((power_cap/4) * daa_price_vector[q] * (step1_dis_daa[q] - step1_cha_daa[q]) for q in range(Q))
        else:
            # Handle infeasible or other cases
            step1_soc_daa = [0] * Q
            step1_cha_daa = [0] * Q
            step1_dis_daa = [0] * Q
            step1_profit_daa = 0

        return(step1_soc_daa, step1_cha_daa, step1_dis_daa, step1_profit_daa)

    def step2_optimize_ida(self, n_cycles: int, energy_cap: int, power_cap: int, ida_price_vector: list, step1_cha_daa: list, step1_dis_daa: list):
        """
        Calculates optimal charge/discharge schedule on the intraday auction (ida) for a given 96-d ida_price_vector.

        Parameters:
        - n_cycles: Maximum number of allowed cycles
        - energy_cap: Energy capacity
        - power_cap: Power capacity
        - ida_price_vector: 96-dimensional ida price vector
        - step1_cha_daa: Previous Buys on the Day-Ahead auction
        - step1_dis_daa: Previous Sells on the Day-Ahead auction

        Returns:
        - step2_soc_ida: Resulting state of charge schedule
        - step2_cha_ida: Resulting charges on ID Auction
        - step2_dis_ida: Resulting discharges on ID Auction
        - step2_cha_ida_close: Resulting charges on ID Auction to close previous DA Auction positions
        - step2_dis_ida_close: Resulting discharge on ID Auction to close previous DA Auction positions
        - step2_profit_ida: Profit from Day-ahead auction trades
        - step2_cha_daaida: Combined charges from DA Auction and ID Auction
        - step2_dis_daaida: Combined discharges from DA Auction and ID Auction
        """

        # Create the solver
        solver = pywraplp.Solver.CreateSolver('CBC')

        if not solver:
            return None

        # Number of quarters
        Q = len(ida_price_vector)

        # Daily discharged energy limit
        volume_limit = energy_cap * n_cycles

        # Variables
        soc = [solver.NumVar(0, energy_cap, f'soc_{i}') for i in range(Q+1)]
        cha_ida = [solver.NumVar(0, 1, f'cha_ida_{i}') for i in range(Q)]
        dis_ida = [solver.NumVar(0, 1, f'dis_ida_{i}') for i in range(Q)]
        cha_ida_close = [solver.NumVar(0, 1, f'cha_ida_close_{i}') for i in range(Q)]
        dis_ida_close = [solver.NumVar(0, 1, f'dis_ida_close_{i}') for i in range(Q)]

        # Constraints

        # SOC initial and final
        solver.Add(soc[0] == 0)
        solver.Add(soc[Q] == 0)

        # SOC dynamics
        for q in range(Q):
            solver.Add(soc[q+1] == soc[q] + (power_cap / 4) * (cha_ida[q] - dis_ida[q] + cha_ida_close[q] - dis_ida_close[q] + step1_cha_daa[q] - step1_dis_daa[q]))

        # Cycle limits
        charge_total = sum(step1_cha_daa) + sum(cha_ida) - sum(dis_ida_close)
        discharge_total = sum(step1_dis_daa) + sum(dis_ida) - sum(cha_ida_close)
        solver.Add(charge_total * (power_cap / 4) <= volume_limit)
        solver.Add(discharge_total * (power_cap / 4) <= volume_limit)

        # Close logic
        for q in range(Q):
            solver.Add(cha_ida_close[q] <= step1_dis_daa[q])
            solver.Add(dis_ida_close[q] <= step1_cha_daa[q])

        # Rate limits
        for q in range(Q):
            solver.Add(cha_ida[q] + step1_cha_daa[q] <= 1)
            solver.Add(dis_ida[q] + step1_dis_daa[q] <= 1)

        # Objective: maximize revenue
        objective = solver.Objective()
        for q in range(Q):
            coeff = (power_cap / 4) * ida_price_vector[q]
            objective.SetCoefficient(cha_ida[q], -coeff)
            objective.SetCoefficient(dis_ida[q], coeff)
            objective.SetCoefficient(cha_ida_close[q], -coeff)
            objective.SetCoefficient(dis_ida_close[q], coeff)
        objective.SetMaximization()

        # Solve
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            step2_soc_ida = [soc[q].solution_value() for q in range(Q)]
            step2_cha_ida = [cha_ida[q].solution_value() for q in range(Q)]
            step2_dis_ida = [dis_ida[q].solution_value() for q in range(Q)]
            step2_cha_ida_close = [cha_ida_close[q].solution_value() for q in range(Q)]
            step2_dis_ida_close = [dis_ida_close[q].solution_value() for q in range(Q)]
            step2_profit_ida = sum(ida_price_vector[q] * (power_cap/4) * (step2_dis_ida[q] + step2_dis_ida_close[q] - step2_cha_ida[q] - step2_cha_ida_close[q]) for q in range(Q))
            step2_cha_daaida = np.array(step1_cha_daa) - np.array(step2_dis_ida_close) + np.array(step2_cha_ida)
            step2_dis_daaida = np.array(step1_dis_daa) - np.array(step2_cha_ida_close) + np.array(step2_dis_ida)
        else:
            step2_soc_ida = [0] * Q
            step2_cha_ida = [0] * Q
            step2_dis_ida = [0] * Q
            step2_cha_ida_close = [0] * Q
            step2_dis_ida_close = [0] * Q
            step2_profit_ida = 0
            step2_cha_daaida = np.array(step1_cha_daa)
            step2_dis_daaida = np.array(step1_dis_daa)

        return(step2_soc_ida, step2_cha_ida, step2_dis_ida, step2_cha_ida_close, step2_dis_ida_close, step2_profit_ida, step2_cha_daaida, step2_dis_daaida)

    def step3_optimize_idc(self, n_cycles: int, energy_cap: int, power_cap: int, idc_price_vector: list, step2_cha_daaida: list, step2_dis_daaida: list):
        """
        Calculates optimal charge/discharge schedule on the intraday continuous (idc) for a given 96-d idc_price_vector.

        Parameters:
        - n_cycles: Maximum number of allowed cycles
        - energy_cap: Energy capacity
        - power_cap: Power capacity
        - ida_price_vector: 96-dimensional ida price vector
        - step2_cha_daaida: Previous combined Buys on the DA Auction and ID Auction
        - step2_dis_daaida: Previous combined Sells on the DA Auction and ID Auction

        Returns:
        - step3_soc_idc: Resulting state of charge schedule
        - step3_cha_idc: Resulting charges on ID Continuous
        - step3_dis_idc: Resulting discharges on ID Continuous
        - step3_cha_idc_close: Resulting charges on ID Continuous to close previous DA or ID Auction positions
        - step3_dis_idc_close: Resulting discharge on ID Continuous to close previous DA or ID Auction positions
        - step3_profit_idc: Profit from Day-ahead auction trades
        - step3_cha_daaidaidc: Combined charges from DA Auction, ID Auction and ID Continuous
        - step3_dis_daaidaidc: Combined discharges from DA Auction, ID Auction and ID Continuous
        """

        # Create the solver
        solver = pywraplp.Solver.CreateSolver('CBC')

        if not solver:
            return None

        # Number of quarters
        Q = len(idc_price_vector)

        # Daily discharged energy limit
        volume_limit = energy_cap * n_cycles

        # Variables
        soc = [solver.NumVar(0, energy_cap, f'soc_{i}') for i in range(Q+1)]
        cha_idc = [solver.NumVar(0, 1, f'cha_idc_{i}') for i in range(Q)]
        dis_idc = [solver.NumVar(0, 1, f'dis_idc_{i}') for i in range(Q)]
        cha_idc_close = [solver.NumVar(0, 1, f'cha_idc_close_{i}') for i in range(Q)]
        dis_idc_close = [solver.NumVar(0, 1, f'dis_idc_close_{i}') for i in range(Q)]

        # Constraints

        # SOC initial and final
        solver.Add(soc[0] == 0)
        solver.Add(soc[Q] == 0)

        # SOC dynamics
        for q in range(Q):
            solver.Add(soc[q+1] == soc[q] + (power_cap / 4) * (cha_idc[q] - dis_idc[q] + cha_idc_close[q] - dis_idc_close[q] + step2_cha_daaida[q] - step2_dis_daaida[q]))

        # Cycle limits
        charge_total = sum(step2_dis_daaida) + sum(dis_idc) - sum(cha_idc_close)
        discharge_total = sum(step2_cha_daaida) + sum(cha_idc) - sum(dis_idc_close)
        solver.Add(charge_total * (power_cap / 4) <= volume_limit)
        solver.Add(discharge_total * (power_cap / 4) <= volume_limit)

        # Close logic
        for q in range(Q):
            solver.Add(cha_idc_close[q] <= step2_dis_daaida[q])
            solver.Add(dis_idc_close[q] <= step2_cha_daaida[q])

        # Rate limits
        for q in range(Q):
            solver.Add(cha_idc[q] + step2_cha_daaida[q] <= 1)
            solver.Add(dis_idc[q] + step2_dis_daaida[q] <= 1)

        # Objective: maximize revenue
        objective = solver.Objective()
        for q in range(Q):
            coeff = (power_cap / 4) * idc_price_vector[q]
            objective.SetCoefficient(cha_idc[q], -coeff)
            objective.SetCoefficient(dis_idc[q], coeff)
            objective.SetCoefficient(cha_idc_close[q], -coeff)
            objective.SetCoefficient(dis_idc_close[q], coeff)
        objective.SetMaximization()

        # Solve
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            step3_soc_idc = [soc[q].solution_value() for q in range(Q)]
            step3_cha_idc = [cha_idc[q].solution_value() for q in range(Q)]
            step3_dis_idc = [dis_idc[q].solution_value() for q in range(Q)]
            step3_cha_idc_close = [cha_idc_close[q].solution_value() for q in range(Q)]
            step3_dis_idc_close = [dis_idc_close[q].solution_value() for q in range(Q)]
            step3_profit_idc = sum(idc_price_vector[q] * (power_cap/4) * (step3_dis_idc[q] + step3_dis_idc_close[q] - step3_cha_idc[q] - step3_cha_idc_close[q]) for q in range(Q))
            step3_cha_daaidaidc = np.array(step2_cha_daaida) - np.array(step3_dis_idc_close) + np.array(step3_cha_idc)
            step3_dis_daaidaidc = np.array(step2_dis_daaida) - np.array(step3_cha_idc_close) + np.array(step3_dis_idc)
        else:
            step3_soc_idc = [0] * Q
            step3_cha_idc = [0] * Q
            step3_dis_idc = [0] * Q
            step3_cha_idc_close = [0] * Q
            step3_dis_idc_close = [0] * Q
            step3_profit_idc = 0
            step3_cha_daaidaidc = np.array(step2_cha_daaida)
            step3_dis_daaidaidc = np.array(step2_dis_daaida)

        return(step3_soc_idc, step3_cha_idc, step3_dis_idc, step3_cha_idc_close, step3_dis_idc_close, step3_profit_idc, step3_cha_daaidaidc, step3_dis_daaidaidc)
