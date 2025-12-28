
from ortools.linear_solver import pywraplp
import numpy as np


class optimizer:
    """
    Sequential storage optimizer using Google OR-Tools.
    Replacement of the original Pyomo-based optimizer.
    """

    # =====================================================
    # STEP 1 – DAY-AHEAD AUCTION
    # =====================================================
    def step1_optimize_daa(self, n_cycles, energy_cap, power_cap, price):

        Q = len(price)
        H = Q // 4
        volume_limit = energy_cap * n_cycles

        solver = pywraplp.Solver.CreateSolver("GLOP")

        soc = [solver.NumVar(0, energy_cap, f"soc_{q}") for q in range(Q + 1)]
        cha = [solver.NumVar(0, 1, f"cha_{q}") for q in range(Q)]
        dis = [solver.NumVar(0, 1, f"dis_{q}") for q in range(Q)]

        solver.Add(soc[0] == 0)
        solver.Add(soc[Q] == 0)

        for q in range(Q):
            solver.Add(soc[q+1] == soc[q] + power_cap/4 * (cha[q] - dis[q]))

        solver.Add(sum(cha[q] for q in range(Q)) * power_cap/4 <= volume_limit)
        solver.Add(sum(dis[q] for q in range(Q)) * power_cap/4 <= volume_limit)

        for h in range(H):
            q0 = 4*h
            for i in range(3):
                solver.Add(cha[q0+i] == cha[q0+i+1])
                solver.Add(dis[q0+i] == dis[q0+i+1])

        solver.Maximize(
            sum(power_cap/4 * price[q] * (dis[q] - cha[q]) for q in range(Q))
        )

        solver.Solve()

        soc_out = [soc[q].solution_value() for q in range(Q)]
        cha_out = [cha[q].solution_value() for q in range(Q)]
        dis_out = [dis[q].solution_value() for q in range(Q)]

        profit = sum(power_cap/4 * price[q] * (dis_out[q] - cha_out[q]) for q in range(Q))

        return soc_out, cha_out, dis_out, profit

    # =====================================================
    # STEP 2 – INTRADAY AUCTION
    # =====================================================
    def step2_optimize_ida(self, n_cycles, energy_cap, power_cap, price, cha_daa, dis_daa):

        Q = len(price)
        volume_limit = energy_cap * n_cycles

        solver = pywraplp.Solver.CreateSolver("GLOP")

        soc = [solver.NumVar(0, energy_cap, f"soc_{q}") for q in range(Q + 1)]
        cha = [solver.NumVar(0, 1, f"cha_{q}") for q in range(Q)]
        dis = [solver.NumVar(0, 1, f"dis_{q}") for q in range(Q)]
        cha_close = [solver.NumVar(0, 1, f"cha_close_{q}") for q in range(Q)]
        dis_close = [solver.NumVar(0, 1, f"dis_close_{q}") for q in range(Q)]

        solver.Add(soc[0] == 0)
        solver.Add(soc[Q] == 0)

        for q in range(Q):
            solver.Add(
                soc[q+1] == soc[q] + power_cap/4 * (
                    cha[q] - dis[q]
                    + cha_close[q] - dis_close[q]
                    + cha_daa[q] - dis_daa[q]
                )
            )

            solver.Add(cha[q] + cha_daa[q] <= 1)
            solver.Add(dis[q] + dis_daa[q] <= 1)
            solver.Add(cha_close[q] <= dis_daa[q])
            solver.Add(dis_close[q] <= cha_daa[q])

        solver.Add(
            (sum(cha[q] for q in range(Q)) + sum(cha_daa) - sum(dis_close[q] for q in range(Q)))
            * power_cap/4 <= volume_limit
        )

        solver.Add(
            (sum(dis[q] for q in range(Q)) + sum(dis_daa) - sum(cha_close[q] for q in range(Q)))
            * power_cap/4 <= volume_limit
        )

        solver.Maximize(
            sum(power_cap/4 * price[q] * (
                dis[q] + dis_close[q] - cha[q] - cha_close[q]
            ) for q in range(Q))
        )

        solver.Solve()

        soc_out = [soc[q].solution_value() for q in range(Q)]
        cha_out = [cha[q].solution_value() for q in range(Q)]
        dis_out = [dis[q].solution_value() for q in range(Q)]
        cha_c = [cha_close[q].solution_value() for q in range(Q)]
        dis_c = [dis_close[q].solution_value() for q in range(Q)]

        profit = sum(
            power_cap/4 * price[q] * (
                dis_out[q] + dis_c[q] - cha_out[q] - cha_c[q]
            ) for q in range(Q)
        )

        cha_tot = np.array(cha_daa) - np.array(dis_c) + np.array(cha_out)
        dis_tot = np.array(dis_daa) - np.array(cha_c) + np.array(dis_out)

        return soc_out, cha_out, dis_out, cha_c, dis_c, profit, cha_tot, dis_tot

    # =====================================================
    # STEP 3 – INTRADAY CONTINUOUS
    # =====================================================
    def step3_optimize_idc(self, n_cycles, energy_cap, power_cap, price, cha_prev, dis_prev):

        Q = len(price)
        volume_limit = energy_cap * n_cycles

        solver = pywraplp.Solver.CreateSolver("GLOP")

        soc = [solver.NumVar(0, energy_cap, f"soc_{q}") for q in range(Q + 1)]
        cha = [solver.NumVar(0, 1, f"cha_{q}") for q in range(Q)]
        dis = [solver.NumVar(0, 1, f"dis_{q}") for q in range(Q)]
        cha_close = [solver.NumVar(0, 1, f"cha_close_{q}") for q in range(Q)]
        dis_close = [solver.NumVar(0, 1, f"dis_close_{q}") for q in range(Q)]

        solver.Add(soc[0] == 0)
        solver.Add(soc[Q] == 0)

        for q in range(Q):
            solver.Add(
                soc[q+1] == soc[q] + power_cap/4 * (
                    cha[q] - dis[q]
                    + cha_close[q] - dis_close[q]
                    + cha_prev[q] - dis_prev[q]
                )
            )

            solver.Add(cha[q] + cha_prev[q] <= 1)
            solver.Add(dis[q] + dis_prev[q] <= 1)
            solver.Add(cha_close[q] <= dis_prev[q])
            solver.Add(dis_close[q] <= cha_prev[q])

        solver.Add(
            (sum(cha[q] for q in range(Q)) + sum(cha_prev) - sum(dis_close[q] for q in range(Q)))
            * power_cap/4 <= volume_limit
        )

        solver.Add(
            (sum(dis[q] for q in range(Q)) + sum(dis_prev) - sum(cha_close[q] for q in range(Q)))
            * power_cap/4 <= volume_limit
        )

        solver.Maximize(
            sum(power_cap/4 * price[q] * (
                dis[q] + dis_close[q] - cha[q] - cha_close[q]
            ) for q in range(Q))
        )

        solver.Solve()

        soc_out = [soc[q].solution_value() for q in range(Q)]
        cha_out = [cha[q].solution_value() for q in range(Q)]
        dis_out = [dis[q].solution_value() for q in range(Q)]
        cha_c = [cha_close[q].solution_value() for q in range(Q)]
        dis_c = [dis_close[q].solution_value() for q in range(Q)]

        profit = sum(
            power_cap/4 * price[q] * (
                dis_out[q] + dis_c[q] - cha_out[q] - cha_c[q]
            ) for q in range(Q)
        )

        cha_tot = np.array(cha_prev) - np.array(dis_c) + np.array(cha_out)
        dis_tot = np.array(dis_prev) - np.array(cha_c) + np.array(dis_out)

        return soc_out, cha_out, dis_out, cha_c, dis_c, profit, cha_tot, dis_tot
