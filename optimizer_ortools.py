from ortools.linear_solver import pywraplp
import numpy as np


class optimizer:
    """
    Co-located PV + BESS optimizer using OR-Tools allowing for both sequential (DAA -> IDA -> IDC) and simultaneous (single shot) optimization.

    Time resolution:
      - 96 quarters (15 minutes), dt = 0.25 h

    Inputs:
      - pv[q] in MW (length 96)
      - prices per quarter (same length)
      - energy_cap in MWh
      - power_cap in MW (BESS charge/discharge limit)

    Grid constraint (hard):
      - max 1 MW import and 1 MW export at every quarter:
          -GRID_LIMIT <= grid[q] <= GRID_LIMIT
        where:
          grid[q] = (pv[q] - curtail[q]) + (dis_phys[q] - cha_phys[q])

    Notes:
      - Physical BESS charge/discharge after netting:
          cha_phys = prev_cha - close_dis + new_cha
          dis_phys = prev_dis - close_cha + new_dis

      - Enforce "no simultaneous long & short" per market via binary direction variables (requires MILP).
    """

    GRID_LIMIT = 1.0  # MW
    DT = 0.25         # hours per step (15 minutes)

    def __init__(self):
        pass

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _assert_lengths(*arrays):
        L = len(arrays[0])
        for a in arrays[1:]:
            if len(a) != L:
                raise ValueError("All time series must have the same length.")
        return L

    @staticmethod
    def _solve_or_raise(solver):
        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError("Optimization failed (no OPTIMAL solution).")

    @staticmethod
    def _create_milp_solver():
        # Needs MILP because we add binaries to prevent simultaneous buy & sell.
        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
        if solver is None:
            # Some builds accept "CBC" as name.
            solver = pywraplp.Solver.CreateSolver("CBC")
        if solver is None:
            raise RuntimeError("OR-Tools CBC MILP solver is not available in this environment.")
        return solver

    @staticmethod
    def _enforce_single_direction(solver, charge_expr, discharge_expr, limit, name):
        """
        Adds a binary switch such that either charge_expr or discharge_expr can be positive,
        preventing simultaneous charge/discharge artifacts.
        """
        u = solver.BoolVar(name)
        solver.Add(charge_expr <= limit * u)
        solver.Add(discharge_expr <= limit * (1 - u))
        return u

    # =====================================================
    # STEP 1 – DAY-AHEAD AUCTION
    # =====================================================
    def step1_optimize_daa(self, n_cycles: int, energy_cap: float, power_cap: float,
                           daa_price_vector: list, pv_vector: list):
        """
        Returns:
          soc, cha_daa, dis_daa, curtail, pv_to_bess, profit
        where cha_daa/dis_daa are DA positions (MW) applied physically (hourly blocks enforced).
        pv_to_bess is the PV power directly charging the battery.
        """
        Q = self._assert_lengths(daa_price_vector, pv_vector)
        H = Q // 4
        volume_limit = energy_cap * n_cycles

        solver = self._create_milp_solver()

        # Variables (MW for power vars, MWh for SOC)
        soc = [solver.NumVar(0.0, energy_cap, f"soc_{q}") for q in range(Q + 1)]
        cha = [solver.NumVar(0.0, power_cap, f"cha_daa_{q}") for q in range(Q)]
        dis = [solver.NumVar(0.0, power_cap, f"dis_daa_{q}") for q in range(Q)]
        cur = [solver.NumVar(0.0, float(pv_vector[q]), f"curtail_{q}") for q in range(Q)]
        pv_to_bess = [solver.NumVar(0.0, float(pv_vector[q]), f"pv_to_bess_{q}") for q in range(Q)]

        # one binary per hour since DA is hourly-blocked in the original notebook (pre-Oct2025 when quarter-hours were introduced)
        # u_h = 1 => "charging/buying" allowed, dis=0
        # u_h = 0 => "discharging/selling" allowed, cha=0
        u_hour = [solver.BoolVar(f"u_daa_hour_{h}") for h in range(H)]

        # SOC boundary
        solver.Add(soc[0] == 0.0)
        solver.Add(soc[Q] == 0.0)

        # Dynamics + grid constraint
        for q in range(Q):
            cha_total = cha[q] + pv_to_bess[q]
            solver.Add(soc[q + 1] == soc[q] + self.DT * (cha_total - dis[q]))

            # PV allocation: to grid, to battery, or curtailed
            solver.Add(pv_to_bess[q] + cur[q] <= pv_vector[q])
            
            grid = (pv_vector[q] - pv_to_bess[q] - cur[q]) + (dis[q] - cha[q])
            solver.Add(grid <= self.GRID_LIMIT)
            solver.Add(grid >= -self.GRID_LIMIT)

        # Cycle / throughput limits (energy in MWh)
        solver.Add(sum(cha[q] * self.DT for q in range(Q)) <= volume_limit)
        solver.Add(sum(dis[q] * self.DT for q in range(Q)) <= volume_limit)

        # Hourly parity constraints (DA assumed to trade in hourly blocks)
        for h in range(H):
            q0 = 4 * h
            solver.Add(cha[q0] == cha[q0 + 1])
            solver.Add(cha[q0 + 1] == cha[q0 + 2])
            solver.Add(cha[q0 + 2] == cha[q0 + 3])

            solver.Add(dis[q0] == dis[q0 + 1])
            solver.Add(dis[q0 + 1] == dis[q0 + 2])
            solver.Add(dis[q0 + 2] == dis[q0 + 3])

            # no simultaneous long & short in DA (per hour block)
            # Apply to the representative quarter q0 - parity makes it equivalent for all 4
            solver.Add(cha[q0] <= power_cap * u_hour[h])
            solver.Add(dis[q0] <= power_cap * (1 - u_hour[h]))

        # Objective: revenue of net grid export (incl PV minus curtail and PV to BESS) at DA price
        solver.Maximize(
            sum(daa_price_vector[q] * ((pv_vector[q] - pv_to_bess[q] - cur[q]) + (dis[q] - cha[q])) * self.DT
                for q in range(Q))
        )

        self._solve_or_raise(solver)

        soc_out = [soc[q].solution_value() for q in range(Q)]
        cha_out = [cha[q].solution_value() for q in range(Q)]
        dis_out = [dis[q].solution_value() for q in range(Q)]
        cur_out = [cur[q].solution_value() for q in range(Q)]
        pv_to_bess_out = [pv_to_bess[q].solution_value() for q in range(Q)]

        profit = sum(
            daa_price_vector[q] * ((pv_vector[q] - pv_to_bess_out[q] - cur_out[q]) + (dis_out[q] - cha_out[q])) * self.DT
            for q in range(Q)
        )

        # Track PV allocation (to grid + to bess + curtailed)
        pv_allocation_out = [(pv_vector[q] - cur_out[q]) for q in range(Q)]  # PV actually used (not curtailed)
        
        return soc_out, cha_out, dis_out, cur_out, pv_to_bess_out, profit, pv_allocation_out

    # =====================================================
    # STEP 2 – INTRADAY AUCTION (adjust / close DA positions + add new IDA)
    # =====================================================
    def step2_optimize_ida(self, n_cycles: int, energy_cap: float, power_cap: float,
                           ida_price_vector: list, pv_vector: list,
                           step1_cha_daa: list, step1_dis_daa: list,
                           step1_pv_allocation: list = None):
        """
        Returns:
          soc, cha_ida, dis_ida, cha_close, dis_close, curtail, pv_to_bess, profit,
          cha_phys_total, dis_phys_total, pv_allocation

        Close variables:
          - cha_close[q] closes previous DIS positions: cha_close[q] <= step1_dis_daa[q]
          - dis_close[q] closes previous CHA positions: dis_close[q] <= step1_cha_daa[q]

          - For IDA itself, do not allow both (total buys) and (total sells) in same quarter:
              buy_ida[q]  = cha_ida[q] + cha_close[q]
              sell_ida[q] = dis_ida[q] + dis_close[q]
            Enforce: not both > 0 via binary u_ida[q].
        
        pv_to_bess is the PV power directly charging the battery.
        step1_pv_allocation tracks what PV was already used in step1, so we use the remainder.
        """
        # Use remaining PV if step1 allocation is provided
        if step1_pv_allocation is not None:
            pv_remaining = [max(0, pv_vector[q] - step1_pv_allocation[q]) for q in range(len(pv_vector))]
        else:
            pv_remaining = pv_vector  # Fallback for backward compatibility
        
        Q = self._assert_lengths(ida_price_vector, pv_remaining, step1_cha_daa, step1_dis_daa)
        volume_limit = energy_cap * n_cycles

        solver = self._create_milp_solver()

        soc = [solver.NumVar(0.0, energy_cap, f"soc_{q}") for q in range(Q + 1)]

        cha_ida = [solver.NumVar(0.0, power_cap, f"cha_ida_{q}") for q in range(Q)]
        dis_ida = [solver.NumVar(0.0, power_cap, f"dis_ida_{q}") for q in range(Q)]
        cha_close = [solver.NumVar(0.0, power_cap, f"cha_close_{q}") for q in range(Q)]
        dis_close = [solver.NumVar(0.0, power_cap, f"dis_close_{q}") for q in range(Q)]

        cur = [solver.NumVar(0.0, float(pv_remaining[q]), f"curtail_{q}") for q in range(Q)]
        pv_to_bess = [solver.NumVar(0.0, float(pv_remaining[q]), f"pv_to_bess_{q}") for q in range(Q)]

        # binary direction per quarter for the IDA market
        u_ida = [solver.BoolVar(f"u_ida_{q}") for q in range(Q)]

        solver.Add(soc[0] == 0.0)
        solver.Add(soc[Q] == 0.0)

        for q in range(Q):
            cha_phys = step1_cha_daa[q] - dis_close[q] + cha_ida[q] + pv_to_bess[q]
            dis_phys = step1_dis_daa[q] - cha_close[q] + dis_ida[q]

            # SOC dynamics use physical charge/discharge
            solver.Add(soc[q + 1] == soc[q] + self.DT * (cha_phys - dis_phys))

            # Closing logic (only close if position exists)
            solver.Add(cha_close[q] <= step1_dis_daa[q])
            solver.Add(dis_close[q] <= step1_cha_daa[q])

            # prevent simultaneous long & short within IDA (including closes)
            buy_ida = cha_ida[q] + cha_close[q]
            sell_ida = dis_ida[q] + dis_close[q]
            solver.Add(buy_ida <= power_cap * u_ida[q])
            solver.Add(sell_ida <= power_cap * (1 - u_ida[q]))

            # PV allocation: to grid, to battery, or curtailed
            solver.Add(pv_to_bess[q] + cur[q] <= pv_remaining[q])
            
            # Grid constraint with PV
            grid = (pv_remaining[q] - pv_to_bess[q] - cur[q]) + (dis_phys - cha_phys)
            solver.Add(grid <= self.GRID_LIMIT)
            solver.Add(grid >= -self.GRID_LIMIT)

            # Physical BESS can only go in one direction per quarter
            self._enforce_single_direction(solver, cha_phys, dis_phys, power_cap, f"u_phys_ida_{q}")

        # Cycle / throughput limits on physical flows
        solver.Add(
            sum((step1_cha_daa[q] - dis_close[q] + cha_ida[q]) * self.DT for q in range(Q))
            <= volume_limit
        )
        solver.Add(
            sum((step1_dis_daa[q] - cha_close[q] + dis_ida[q]) * self.DT for q in range(Q))
            <= volume_limit
        )

        solver.Maximize(
            sum(ida_price_vector[q] * ((pv_remaining[q] - pv_to_bess[q] - cur[q]) + (dis_ida[q] + dis_close[q]) - (cha_ida[q] + cha_close[q])) * self.DT
                for q in range(Q))
        )

        self._solve_or_raise(solver)

        soc_out = [soc[q].solution_value() for q in range(Q)]
        cha_ida_out = [cha_ida[q].solution_value() for q in range(Q)]
        dis_ida_out = [dis_ida[q].solution_value() for q in range(Q)]
        cha_close_out = [cha_close[q].solution_value() for q in range(Q)]
        dis_close_out = [dis_close[q].solution_value() for q in range(Q)]
        cur_out = [cur[q].solution_value() for q in range(Q)]
        pv_to_bess_out = [pv_to_bess[q].solution_value() for q in range(Q)]

        profit = sum(
            ida_price_vector[q]
            * ((pv_remaining[q] - pv_to_bess_out[q] - cur_out[q]) + (dis_ida_out[q] + dis_close_out[q]) - (cha_ida_out[q] + cha_close_out[q]))
            * self.DT
            for q in range(Q)
        )

        cha_phys_total = np.array(step1_cha_daa) - np.array(dis_close_out) + np.array(cha_ida_out) + np.array(pv_to_bess_out)
        dis_phys_total = np.array(step1_dis_daa) - np.array(cha_close_out) + np.array(dis_ida_out)

        # Track PV allocation in this step
        pv_allocation_out = [(pv_remaining[q] - cur_out[q]) for q in range(Q)]

        return (soc_out, cha_ida_out, dis_ida_out,
                cha_close_out, dis_close_out,
                cur_out, pv_to_bess_out, profit,
                cha_phys_total, dis_phys_total, pv_allocation_out)

    # =====================================================
    # STEP 3 – INTRADAY CONTINUOUS (adjust / close DA+IDA physical schedule + add new IDC)
    # =====================================================
    def step3_optimize_idc(self, n_cycles: int, energy_cap: float, power_cap: float,
                           idc_price_vector: list, pv_vector: list,
                           step2_cha_phys: list, step2_dis_phys: list,
                           step1_pv_allocation: list = None,
                           step2_pv_allocation: list = None):
        """
        Returns:
          soc, cha_idc, dis_idc, cha_close, dis_close, curtail, pv_to_bess, profit,
          cha_phys_total, dis_phys_total
          - For IDC itself, do not allow both (total buys) and (total sells) in same quarter:
              buy_idc[q]  = cha_idc[q] + cha_close[q]
              sell_idc[q] = dis_idc[q] + dis_close[q]
        
        pv_to_bess is the PV power directly charging the battery.
        step1/step2_pv_allocation track what PV was already used, so we use the remainder.
        """
        # Calculate remaining PV after step1 and step2
        if step1_pv_allocation is not None and step2_pv_allocation is not None:
            pv_remaining = [max(0, pv_vector[q] - step1_pv_allocation[q] - step2_pv_allocation[q]) 
                           for q in range(len(pv_vector))]
        else:
            pv_remaining = pv_vector  # Fallback for backward compatibility
        
        Q = self._assert_lengths(idc_price_vector, pv_remaining, step2_cha_phys, step2_dis_phys)
        volume_limit = energy_cap * n_cycles

        solver = self._create_milp_solver()

        soc = [solver.NumVar(0.0, energy_cap, f"soc_{q}") for q in range(Q + 1)]

        cha_idc = [solver.NumVar(0.0, power_cap, f"cha_idc_{q}") for q in range(Q)]
        dis_idc = [solver.NumVar(0.0, power_cap, f"dis_idc_{q}") for q in range(Q)]
        cha_close = [solver.NumVar(0.0, power_cap, f"cha_close_{q}") for q in range(Q)]
        dis_close = [solver.NumVar(0.0, power_cap, f"dis_close_{q}") for q in range(Q)]

        cur = [solver.NumVar(0.0, float(pv_remaining[q]), f"curtail_{q}") for q in range(Q)]
        pv_to_bess = [solver.NumVar(0.0, float(pv_remaining[q]), f"pv_to_bess_{q}") for q in range(Q)]

        # binary direction per quarter for the IDC market
        u_idc = [solver.BoolVar(f"u_idc_{q}") for q in range(Q)]

        solver.Add(soc[0] == 0.0)
        solver.Add(soc[Q] == 0.0)

        for q in range(Q):
            cha_phys = step2_cha_phys[q] - dis_close[q] + cha_idc[q] + pv_to_bess[q]
            dis_phys = step2_dis_phys[q] - cha_close[q] + dis_idc[q]

            solver.Add(soc[q + 1] == soc[q] + self.DT * (cha_phys - dis_phys))

            # Closing only where prior physical exists
            solver.Add(cha_close[q] <= step2_dis_phys[q])
            solver.Add(dis_close[q] <= step2_cha_phys[q])

            # prevent simultaneous long & short within IDC (including closes)
            buy_idc = cha_idc[q] + cha_close[q]
            sell_idc = dis_idc[q] + dis_close[q]
            solver.Add(buy_idc <= power_cap * u_idc[q])
            solver.Add(sell_idc <= power_cap * (1 - u_idc[q]))

            # PV allocation: to grid, to battery, or curtailed
            solver.Add(pv_to_bess[q] + cur[q] <= pv_remaining[q])
            
            grid = (pv_remaining[q] - pv_to_bess[q] - cur[q]) + (dis_phys - cha_phys)
            solver.Add(grid <= self.GRID_LIMIT)
            solver.Add(grid >= -self.GRID_LIMIT)

            # Physical BESS can only go in one direction per quarter
            self._enforce_single_direction(solver, cha_phys, dis_phys, power_cap, f"u_phys_idc_{q}")

        solver.Add(
            sum((step2_cha_phys[q] - dis_close[q] + cha_idc[q]) * self.DT for q in range(Q))
            <= volume_limit
        )
        solver.Add(
            sum((step2_dis_phys[q] - cha_close[q] + dis_idc[q]) * self.DT for q in range(Q))
            <= volume_limit
        )

        solver.Maximize(
            sum(idc_price_vector[q] * ((pv_remaining[q] - pv_to_bess[q] - cur[q]) + (dis_idc[q] + dis_close[q]) - (cha_idc[q] + cha_close[q])) * self.DT
                for q in range(Q))
        )

        self._solve_or_raise(solver)

        soc_out = [soc[q].solution_value() for q in range(Q)]
        cha_idc_out = [cha_idc[q].solution_value() for q in range(Q)]
        dis_idc_out = [dis_idc[q].solution_value() for q in range(Q)]
        cha_close_out = [cha_close[q].solution_value() for q in range(Q)]
        dis_close_out = [dis_close[q].solution_value() for q in range(Q)]
        cur_out = [cur[q].solution_value() for q in range(Q)]
        pv_to_bess_out = [pv_to_bess[q].solution_value() for q in range(Q)]

        profit = sum(
            idc_price_vector[q]
            * ((pv_remaining[q] - pv_to_bess_out[q] - cur_out[q]) + (dis_idc_out[q] + dis_close_out[q]) - (cha_idc_out[q] + cha_close_out[q]))
            * self.DT
            for q in range(Q)
        )

        cha_phys_total = np.array(step2_cha_phys) - np.array(dis_close_out) + np.array(cha_idc_out) + np.array(pv_to_bess_out)
        dis_phys_total = np.array(step2_dis_phys) - np.array(cha_close_out) + np.array(dis_idc_out)

        return (soc_out, cha_idc_out, dis_idc_out,
                cha_close_out, dis_close_out, cur_out, pv_to_bess_out, profit,
                cha_phys_total, dis_phys_total)

    # =====================================================
    # SIMULTANEOUS OPTIMIZATION – all markets at once (perfect foresight = theoretical upper bound benchmark)
    # =====================================================
    def optimize_simultaneous(self, n_cycles: int, energy_cap: float, power_cap: float,
                             daa_price_vector: list, ida_price_vector: list, 
                             idc_price_vector: list, pv_vector: list):
        """
        Optimizes all three markets simultaneously with perfect price foresight.
        This represents the theoretical upper bound on profit.
        
        Returns:
          dict with keys: 'soc', 'cha_daa', 'dis_daa', 'cha_ida', 'dis_ida', 
          'cha_idc', 'dis_idc', 'curtail', 'profit', 'profit_by_market'
        """
        Q = self._assert_lengths(daa_price_vector, ida_price_vector, idc_price_vector, pv_vector)
        H = Q // 4
        volume_limit = energy_cap * n_cycles

        solver = self._create_milp_solver()

        # State variables
        soc = [solver.NumVar(0.0, energy_cap, f"soc_{q}") for q in range(Q + 1)]
        cur = [solver.NumVar(0.0, float(pv_vector[q]), f"curtail_{q}") for q in range(Q)]

        # Market variables (MW)
        cha_daa = [solver.NumVar(0.0, power_cap, f"cha_daa_{q}") for q in range(Q)]
        dis_daa = [solver.NumVar(0.0, power_cap, f"dis_daa_{q}") for q in range(Q)]
        cha_ida = [solver.NumVar(0.0, power_cap, f"cha_ida_{q}") for q in range(Q)]
        dis_ida = [solver.NumVar(0.0, power_cap, f"dis_ida_{q}") for q in range(Q)]
        cha_idc = [solver.NumVar(0.0, power_cap, f"cha_idc_{q}") for q in range(Q)]
        dis_idc = [solver.NumVar(0.0, power_cap, f"dis_idc_{q}") for q in range(Q)]

        # Binary direction variables per market per quarter/hour
        u_daa = [solver.BoolVar(f"u_daa_h{h}") for h in range(H)]
        u_ida = [solver.BoolVar(f"u_ida_{q}") for q in range(Q)]
        u_idc = [solver.BoolVar(f"u_idc_{q}") for q in range(Q)]

        # SOC boundary conditions
        solver.Add(soc[0] == 0.0)
        solver.Add(soc[Q] == 0.0)

        # Quarter-level constraints
        for q in range(Q):
            # Total physical charge/discharge
            cha_total = cha_daa[q] + cha_ida[q] + cha_idc[q] + pv_to_bess[q]
            dis_total = dis_daa[q] + dis_ida[q] + dis_idc[q]

            # SOC dynamics
            solver.Add(soc[q + 1] == soc[q] + self.DT * (cha_total - dis_total))

            # Grid constraint (only PV going to markets, not to battery)
            grid = (pv_to_daa[q] + pv_to_ida[q] + pv_to_idc[q]) + (dis_total - (cha_daa[q] + cha_ida[q] + cha_idc[q]))
            solver.Add(grid <= self.GRID_LIMIT)
            solver.Add(grid >= -self.GRID_LIMIT)

            # Physical device cannot charge and discharge simultaneously
            self._enforce_single_direction(solver, cha_total, dis_total, power_cap, f"u_phys_simul_{q}")

            # Per-market direction constraints
            solver.Add(cha_ida[q] <= power_cap * u_ida[q])
            solver.Add(dis_ida[q] <= power_cap * (1 - u_ida[q]))
            solver.Add(cha_idc[q] <= power_cap * u_idc[q])
            solver.Add(dis_idc[q] <= power_cap * (1 - u_idc[q]))

        # DAA hourly parity and direction constraints
        for h in range(H):
            q0 = 4 * h
            for offset in range(3):
                solver.Add(cha_daa[q0] == cha_daa[q0 + offset + 1])
                solver.Add(dis_daa[q0] == dis_daa[q0 + offset + 1])
            solver.Add(cha_daa[q0] <= power_cap * u_daa[h])
            solver.Add(dis_daa[q0] <= power_cap * (1 - u_daa[h]))

        # Cycle limits
        solver.Add(sum((cha_daa[q] + cha_ida[q] + cha_idc[q]) * self.DT for q in range(Q)) <= volume_limit)
        solver.Add(sum((dis_daa[q] + dis_ida[q] + dis_idc[q]) * self.DT for q in range(Q)) <= volume_limit)

        # PV market allocation: which market does PV go to?
        # Binary variables: PV can only be sold to ONE market per quarter
        pv_to_daa_bin = [solver.BoolVar(f"pv_to_daa_bin_{q}") for q in range(Q)]
        pv_to_ida_bin = [solver.BoolVar(f"pv_to_ida_bin_{q}") for q in range(Q)]
        pv_to_idc_bin = [solver.BoolVar(f"pv_to_idc_bin_{q}") for q in range(Q)]
        
        pv_to_daa = [solver.NumVar(0.0, float(pv_vector[q]), f"pv_to_daa_{q}") for q in range(Q)]
        pv_to_ida = [solver.NumVar(0.0, float(pv_vector[q]), f"pv_to_ida_{q}") for q in range(Q)]
        pv_to_idc = [solver.NumVar(0.0, float(pv_vector[q]), f"pv_to_idc_{q}") for q in range(Q)]
        pv_to_bess = [solver.NumVar(0.0, float(pv_vector[q]), f"pv_to_bess_{q}") for q in range(Q)]
        
        # PV must be allocated to exactly one market
        for q in range(Q):
            # Only one market can be selected
            solver.Add(pv_to_daa_bin[q] + pv_to_ida_bin[q] + pv_to_idc_bin[q] == 1)
            
            # If market is selected, PV flows to that market; otherwise zero
            solver.Add(pv_to_daa[q] <= float(pv_vector[q]) * pv_to_daa_bin[q])
            solver.Add(pv_to_ida[q] <= float(pv_vector[q]) * pv_to_ida_bin[q])
            solver.Add(pv_to_idc[q] <= float(pv_vector[q]) * pv_to_idc_bin[q])
            
            # All PV is allocated: to markets, to battery, or curtailed
            solver.Add(pv_to_daa[q] + pv_to_ida[q] + pv_to_idc[q] + pv_to_bess[q] + cur[q] == pv_vector[q])
        
        # Objective: maximize total profit across all markets
        # Battery trading profit at each market's price
        profit_daa = sum(daa_price_vector[q] * (dis_daa[q] - cha_daa[q] + pv_to_daa[q]) * self.DT for q in range(Q))
        profit_ida = sum(ida_price_vector[q] * (dis_ida[q] - cha_ida[q] + pv_to_ida[q]) * self.DT for q in range(Q))
        profit_idc = sum(idc_price_vector[q] * (dis_idc[q] - cha_idc[q] + pv_to_idc[q]) * self.DT for q in range(Q))

        solver.Maximize(profit_daa + profit_ida + profit_idc)

        self._solve_or_raise(solver)

        # Extract solutions
        result = {
            'soc': [soc[q].solution_value() for q in range(Q)],
            'cha_daa': [cha_daa[q].solution_value() for q in range(Q)],
            'dis_daa': [dis_daa[q].solution_value() for q in range(Q)],
            'cha_ida': [cha_ida[q].solution_value() for q in range(Q)],
            'dis_ida': [dis_ida[q].solution_value() for q in range(Q)],
            'cha_idc': [cha_idc[q].solution_value() for q in range(Q)],
            'dis_idc': [dis_idc[q].solution_value() for q in range(Q)],
            'curtail': [cur[q].solution_value() for q in range(Q)],
            'pv_to_daa': [pv_to_daa[q].solution_value() for q in range(Q)],
            'pv_to_ida': [pv_to_ida[q].solution_value() for q in range(Q)],
            'pv_to_idc': [pv_to_idc[q].solution_value() for q in range(Q)],
            'pv_to_bess': [pv_to_bess[q].solution_value() for q in range(Q)],
        }

        # Calculate actual profits
        result['profit_by_market'] = {
            'daa': sum(daa_price_vector[q] * (result['dis_daa'][q] - result['cha_daa'][q] + result['pv_to_daa'][q]) * self.DT for q in range(Q)),
            'ida': sum(ida_price_vector[q] * (result['dis_ida'][q] - result['cha_ida'][q] + result['pv_to_ida'][q]) * self.DT for q in range(Q)),
            'idc': sum(idc_price_vector[q] * (result['dis_idc'][q] - result['cha_idc'][q] + result['pv_to_idc'][q]) * self.DT for q in range(Q)),
            'pv': sum((daa_price_vector[q] * result['pv_to_daa'][q] + 
                      ida_price_vector[q] * result['pv_to_ida'][q] + 
                      idc_price_vector[q] * result['pv_to_idc'][q]) * self.DT for q in range(Q))
        }
        result['profit'] = sum(result['profit_by_market'].values())

        return result

    # =====================================================
    # ONE-SHOT HIERARCHICAL OPTIMIZATION – All markets in single MILP
    # =====================================================
    def optimize_full_stack_single_milp(self, n_cycles: int, energy_cap: float, power_cap: float,
                                        daa_price_vector: list, ida_price_vector: list,
                                        idc_price_vector: list, pv_vector: list):
        """
        Solves DAA, IDA, and IDC jointly in one MILP with cross-market position closing.
        Positions opened on one market can be closed on another (e.g., buy on DAA, sell on IDA).
        All positions represent physical flows that affect battery SOC and grid constraints.
        This captures inter-market arbitrage opportunities with perfect foresight.
        Returns a dict with market trades, PV allocation, SOC, and profit breakdown.
        """
        Q = self._assert_lengths(daa_price_vector, ida_price_vector, idc_price_vector, pv_vector)
        H = Q // 4
        volume_limit = energy_cap * n_cycles

        solver = self._create_milp_solver()

        soc = [solver.NumVar(0.0, energy_cap, f"soc_{q}") for q in range(Q + 1)]
        cur = [solver.NumVar(0.0, float(pv_vector[q]), f"curtail_{q}") for q in range(Q)]

        # DAA hourly-blocked trades
        cha_daa = [solver.NumVar(0.0, power_cap, f"cha_daa_{q}") for q in range(Q)]
        dis_daa = [solver.NumVar(0.0, power_cap, f"dis_daa_{q}") for q in range(Q)]
        u_daa = [solver.BoolVar(f"u_daa_h{h}") for h in range(H)]

        # IDA trades and DAA closings (all represent physical flows)
        cha_ida = [solver.NumVar(0.0, power_cap, f"cha_ida_{q}") for q in range(Q)]
        dis_ida = [solver.NumVar(0.0, power_cap, f"dis_ida_{q}") for q in range(Q)]
        cha_close_daa = [solver.NumVar(0.0, power_cap, f"cha_close_daa_{q}") for q in range(Q)]  # Close DAA long by selling on IDA
        dis_close_daa = [solver.NumVar(0.0, power_cap, f"dis_close_daa_{q}") for q in range(Q)]  # Close DAA short by buying on IDA
        u_ida = [solver.BoolVar(f"u_ida_{q}") for q in range(Q)]

        # IDC trades and closings of remaining DAA + IDA (all physical)
        cha_idc = [solver.NumVar(0.0, power_cap, f"cha_idc_{q}") for q in range(Q)]
        dis_idc = [solver.NumVar(0.0, power_cap, f"dis_idc_{q}") for q in range(Q)]
        cha_close_daa_idc = [solver.NumVar(0.0, power_cap, f"cha_close_daa_idc_{q}") for q in range(Q)]
        dis_close_daa_idc = [solver.NumVar(0.0, power_cap, f"dis_close_daa_idc_{q}") for q in range(Q)]
        cha_close_ida_idc = [solver.NumVar(0.0, power_cap, f"cha_close_ida_idc_{q}") for q in range(Q)]
        dis_close_ida_idc = [solver.NumVar(0.0, power_cap, f"dis_close_ida_idc_{q}") for q in range(Q)]
        u_idc = [solver.BoolVar(f"u_idc_{q}") for q in range(Q)]

        # PV allocation to markets (exclusive per quarter)
        pv_to_daa_bin = [solver.BoolVar(f"pv_to_daa_bin_{q}") for q in range(Q)]
        pv_to_ida_bin = [solver.BoolVar(f"pv_to_ida_bin_{q}") for q in range(Q)]
        pv_to_idc_bin = [solver.BoolVar(f"pv_to_idc_bin_{q}") for q in range(Q)]
        pv_to_daa = [solver.NumVar(0.0, float(pv_vector[q]), f"pv_to_daa_{q}") for q in range(Q)]
        pv_to_ida = [solver.NumVar(0.0, float(pv_vector[q]), f"pv_to_ida_{q}") for q in range(Q)]
        pv_to_idc = [solver.NumVar(0.0, float(pv_vector[q]), f"pv_to_idc_{q}") for q in range(Q)]
        pv_to_bess = [solver.NumVar(0.0, float(pv_vector[q]), f"pv_to_bess_{q}") for q in range(Q)]

        solver.Add(soc[0] == 0.0)
        solver.Add(soc[Q] == 0.0)

        cha_phys = []
        dis_phys = []

        for q in range(Q):
            # Net physical positions after closings (all represent actual physical execution)
            # DAA positions that haven't been closed yet
            net_cha_daa = cha_daa[q] - dis_close_daa[q] - dis_close_daa_idc[q]
            net_dis_daa = dis_daa[q] - cha_close_daa[q] - cha_close_daa_idc[q]

            # IDA positions (new positions + closings of DAA) minus IDC closings
            net_cha_ida = cha_ida[q] + cha_close_daa[q] - dis_close_ida_idc[q]
            net_dis_ida = dis_ida[q] + dis_close_daa[q] - cha_close_ida_idc[q]

            # IDC positions (new positions + closings of DAA and IDA)
            net_cha_idc = cha_idc[q] + cha_close_daa_idc[q] + cha_close_ida_idc[q]
            net_dis_idc = dis_idc[q] + dis_close_daa_idc[q] + dis_close_ida_idc[q]

            # Total physical charge/discharge
            cha_phys_q = net_cha_daa + net_cha_ida + net_cha_idc + pv_to_bess[q]
            dis_phys_q = net_dis_daa + net_dis_ida + net_dis_idc
            cha_phys.append(cha_phys_q)
            dis_phys.append(dis_phys_q)

            # SOC dynamics
            solver.Add(soc[q + 1] == soc[q] + self.DT * (cha_phys_q - dis_phys_q))

            # Grid constraint (PV to grid + net BESS flow)
            # cha_phys_q includes pv_to_bess, so we need to use the net BESS charge from grid only
            grid = (pv_to_daa[q] + pv_to_ida[q] + pv_to_idc[q]) + (dis_phys_q - cha_phys_q + pv_to_bess[q])
            solver.Add(grid <= self.GRID_LIMIT)
            solver.Add(grid >= -self.GRID_LIMIT)

            # Physical unidirectionality per quarter
            self._enforce_single_direction(solver, cha_phys_q, dis_phys_q, power_cap, f"u_phys_full_{q}")

            # Closings limited by available open positions
            solver.Add(cha_close_daa[q] <= dis_daa[q])  # Can only close DAA short by buying
            solver.Add(dis_close_daa[q] <= cha_daa[q])  # Can only close DAA long by selling
            solver.Add(cha_close_daa_idc[q] <= dis_daa[q] - cha_close_daa[q])  # Remaining DAA short
            solver.Add(dis_close_daa_idc[q] <= cha_daa[q] - dis_close_daa[q])  # Remaining DAA long
            solver.Add(cha_close_ida_idc[q] <= dis_ida[q])  # Can only close IDA short
            solver.Add(dis_close_ida_idc[q] <= cha_ida[q])  # Can only close IDA long

            # Market direction binaries (prevent simultaneous buy/sell in new positions)
            solver.Add(cha_ida[q] <= power_cap * u_ida[q])
            solver.Add(dis_ida[q] <= power_cap * (1 - u_ida[q]))
            solver.Add(cha_idc[q] <= power_cap * u_idc[q])
            solver.Add(dis_idc[q] <= power_cap * (1 - u_idc[q]))

            # PV exclusivity
            solver.Add(pv_to_daa_bin[q] + pv_to_ida_bin[q] + pv_to_idc_bin[q] == 1)
            solver.Add(pv_to_daa[q] <= float(pv_vector[q]) * pv_to_daa_bin[q])
            solver.Add(pv_to_ida[q] <= float(pv_vector[q]) * pv_to_ida_bin[q])
            solver.Add(pv_to_idc[q] <= float(pv_vector[q]) * pv_to_idc_bin[q])
            solver.Add(pv_to_daa[q] + pv_to_ida[q] + pv_to_idc[q] + pv_to_bess[q] + cur[q] == pv_vector[q])

        # DAA hourly parity and direction constraints
        for h in range(H):
            q0 = 4 * h
            for offset in range(3):
                solver.Add(cha_daa[q0] == cha_daa[q0 + offset + 1])
                solver.Add(dis_daa[q0] == dis_daa[q0 + offset + 1])
            solver.Add(cha_daa[q0] <= power_cap * u_daa[h])
            solver.Add(dis_daa[q0] <= power_cap * (1 - u_daa[h]))

        # Cycle limits on physical flows
        solver.Add(sum(cha_phys[q] * self.DT for q in range(Q)) <= volume_limit)
        solver.Add(sum(dis_phys[q] * self.DT for q in range(Q)) <= volume_limit)

        # Profit: Opening positions on one market, closing on another captures price spread
        profit_daa = sum(
            daa_price_vector[q] * ((dis_daa[q] - cha_daa[q]) + pv_to_daa[q]) * self.DT
            for q in range(Q)
        )
        profit_ida = sum(
            ida_price_vector[q] * (
                (dis_ida[q] + dis_close_daa[q]) - (cha_ida[q] + cha_close_daa[q]) + pv_to_ida[q]
            ) * self.DT
            for q in range(Q)
        )
        profit_idc = sum(
            idc_price_vector[q] * (
                (dis_idc[q] + dis_close_daa_idc[q] + dis_close_ida_idc[q])
                - (cha_idc[q] + cha_close_daa_idc[q] + cha_close_ida_idc[q])
                + pv_to_idc[q]
            ) * self.DT
            for q in range(Q)
        )

        solver.Maximize(profit_daa + profit_ida + profit_idc)
        self._solve_or_raise(solver)

        result = {
            'soc': [soc[q].solution_value() for q in range(Q)],
            'cha_daa': [cha_daa[q].solution_value() for q in range(Q)],
            'dis_daa': [dis_daa[q].solution_value() for q in range(Q)],
            'cha_ida': [cha_ida[q].solution_value() for q in range(Q)],
            'dis_ida': [dis_ida[q].solution_value() for q in range(Q)],
            'cha_idc': [cha_idc[q].solution_value() for q in range(Q)],
            'dis_idc': [dis_idc[q].solution_value() for q in range(Q)],
            'cha_close_daa': [cha_close_daa[q].solution_value() for q in range(Q)],
            'dis_close_daa': [dis_close_daa[q].solution_value() for q in range(Q)],
            'cha_close_daa_idc': [cha_close_daa_idc[q].solution_value() for q in range(Q)],
            'dis_close_daa_idc': [dis_close_daa_idc[q].solution_value() for q in range(Q)],
            'cha_close_ida_idc': [cha_close_ida_idc[q].solution_value() for q in range(Q)],
            'dis_close_ida_idc': [dis_close_ida_idc[q].solution_value() for q in range(Q)],
            'curtail': [cur[q].solution_value() for q in range(Q)],
            'pv_to_daa': [pv_to_daa[q].solution_value() for q in range(Q)],
            'pv_to_ida': [pv_to_ida[q].solution_value() for q in range(Q)],
            'pv_to_idc': [pv_to_idc[q].solution_value() for q in range(Q)],
            'pv_to_bess': [pv_to_bess[q].solution_value() for q in range(Q)],
            'cha_phys': [cha_phys[q].solution_value() for q in range(Q)],
            'dis_phys': [dis_phys[q].solution_value() for q in range(Q)],
        }

        result['profit_by_market'] = {
            'daa': sum(daa_price_vector[q] * ((result['dis_daa'][q] - result['cha_daa'][q]) + result['pv_to_daa'][q]) * self.DT for q in range(Q)),
            'ida': sum(ida_price_vector[q] * ((result['dis_ida'][q] + result['dis_close_daa'][q]) - (result['cha_ida'][q] + result['cha_close_daa'][q]) + result['pv_to_ida'][q]) * self.DT for q in range(Q)),
            'idc': sum(idc_price_vector[q] * ((result['dis_idc'][q] + result['dis_close_daa_idc'][q] + result['dis_close_ida_idc'][q]) - (result['cha_idc'][q] + result['cha_close_daa_idc'][q] + result['cha_close_ida_idc'][q]) + result['pv_to_idc'][q]) * self.DT for q in range(Q)),
        }
        result['profit'] = sum(result['profit_by_market'].values())

        return result

    # =====================================================
    # ROLLING HORIZON OPTIMIZATION – Sequential with price discovery
    # =====================================================
    def optimize_rolling_horizon(self, n_cycles: int, energy_cap: float, power_cap: float,
                                daa_price_vector: list, ida_price_vector: list,
                                idc_price_vector: list, pv_vector: list,
                                idc_price_scenarios: list = None,
                                pv_forecast_scenarios: list = None,
                                lookahead_quarters: int = None,
                                enforce_physical_unidirectional: bool = True,
                                time_limit_ms: int = 5000):
        """
        Realistic optimization simulating sequential market clearing and price discovery:
        1. Optimize DAA (24h ahead)
        2. Optimize IDA (hours ahead) with DAA commitments fixed
        3. Rolling optimization for IDC every quarter with updated prices and PV forecasts
        
        Args:
            idc_price_scenarios: list of 96 price vectors representing price uncertainty/evolution
            pv_forecast_scenarios: list of 96 PV forecast vectors representing forecast updates
            lookahead_quarters: optional finite horizon for IDC re-optimization (e.g., 8 quarters = 2h)
            enforce_physical_unidirectional: if True, forbid simultaneous charge/discharge physically
            time_limit_ms: per-horizon CBC time limit to keep rolling solve fast
            
        Returns:
            dict with optimization history, final positions, and profit breakdown
        """
        Q = self._assert_lengths(daa_price_vector, ida_price_vector, idc_price_vector, pv_vector)
        
        # Initialize tracking
        history = {
            'quarter': [],
            'idc_prices_seen': [],
            'pv_forecast_seen': [],
            'decision_cha': [],
            'decision_dis': [],
            'soc': [],
            'grid_flow': [],
            'profit_incremental': []
        }
        
        # Step 1: DAA optimization (done once, 24h before delivery)
        print(f"  Rolling Horizon: Optimizing DAA...")
        soc_daa, cha_daa, dis_daa, cur_daa, profit_daa = self.step1_optimize_daa(
            n_cycles, energy_cap, power_cap, daa_price_vector, pv_vector
        )
        
        # Step 2: IDA optimization (done once, hours before delivery)
        print(f"  Rolling Horizon: Optimizing IDA...")
        soc_ida, cha_ida, dis_ida, cha_ida_c, dis_ida_c, cur_ida, profit_ida, cha_phys_ida, dis_phys_ida = \
            self.step2_optimize_ida(n_cycles, energy_cap, power_cap, ida_price_vector, pv_vector, cha_daa, dis_daa)
        
        # Step 3: Rolling IDC optimization (quarter by quarter)
        print(f"  Rolling Horizon: Starting quarter-by-quarter IDC optimization...")
        
        # Initialize state
        current_soc = 0.0
        cha_phys_current = list(cha_phys_ida)
        dis_phys_current = list(dis_phys_ida)
        
        # Final results
        cha_idc_final = [0.0] * Q
        dis_idc_final = [0.0] * Q
        cha_idc_close_final = [0.0] * Q
        dis_idc_close_final = [0.0] * Q
        curtail_final = list(cur_ida)
        soc_final = [current_soc]
        total_profit_idc = 0.0
        
        # Rolling optimization
        for q_now in range(Q):
            # Simulate price discovery - use scenario if provided, else use actual prices
            if idc_price_scenarios and q_now < len(idc_price_scenarios):
                # Price becomes known at this quarter
                idc_price_known = idc_price_scenarios[q_now][:q_now+1]
                # Forecast remaining prices (could add uncertainty here)
                idc_price_forecast = idc_price_vector[q_now+1:]
                idc_price_current = idc_price_known + idc_price_forecast
            else:
                idc_price_current = idc_price_vector
                
            if pv_forecast_scenarios and q_now < len(pv_forecast_scenarios):
                pv_current = pv_forecast_scenarios[q_now]
            else:
                pv_current = pv_vector
                
            # Optimize remaining horizon from q_now onwards
            horizon_length = Q - q_now
            if lookahead_quarters:
                horizon_length = min(horizon_length, lookahead_quarters)
            q_end = q_now + horizon_length
            
            # Create sub-problem for remaining quarters
            solver = self._create_milp_solver()
            solver.SetTimeLimit(time_limit_ms)  # keep solves bounded for rolling use case
            
            # Variables for remaining horizon
            soc = [solver.NumVar(0.0, energy_cap, f"soc_{i}") for i in range(horizon_length + 1)]
            cha_idc = [solver.NumVar(0.0, power_cap, f"cha_idc_{i}") for i in range(horizon_length)]
            dis_idc = [solver.NumVar(0.0, power_cap, f"dis_idc_{i}") for i in range(horizon_length)]
            cha_close = [solver.NumVar(0.0, power_cap, f"cha_close_{i}") for i in range(horizon_length)]
            dis_close = [solver.NumVar(0.0, power_cap, f"dis_close_{i}") for i in range(horizon_length)]
            cur = [solver.NumVar(0.0, float(pv_current[q_now + i]), f"cur_{i}") for i in range(horizon_length)]
            
            u_idc = [solver.BoolVar(f"u_idc_{i}") for i in range(horizon_length)]
            u_phys = []
            
            # Initial and final SOC
            solver.Add(soc[0] == current_soc)
            solver.Add(soc[horizon_length] == 0.0)
            
            # Constraints for remaining quarters
            for i in range(horizon_length):
                q_global = q_now + i
                
                cha_phys = cha_phys_current[q_global] - dis_close[i] + cha_idc[i]
                dis_phys = dis_phys_current[q_global] - cha_close[i] + dis_idc[i]
                
                solver.Add(soc[i + 1] == soc[i] + self.DT * (cha_phys - dis_phys))
                solver.Add(cha_close[i] <= dis_phys_current[q_global])
                solver.Add(dis_close[i] <= cha_phys_current[q_global])
                
                buy_idc = cha_idc[i] + cha_close[i]
                sell_idc = dis_idc[i] + dis_close[i]
                solver.Add(buy_idc <= power_cap * u_idc[i])
                solver.Add(sell_idc <= power_cap * (1 - u_idc[i]))
                if enforce_physical_unidirectional:
                    u_phys.append(self._enforce_single_direction(solver, cha_phys, dis_phys, power_cap, f"u_phys_roll_{q_global}"))
                
                grid = (pv_current[q_global] - cur[i]) + (dis_phys - cha_phys)
                solver.Add(grid <= self.GRID_LIMIT)
                solver.Add(grid >= -self.GRID_LIMIT)
            
            # Throughput limits (for entire day, including past decisions)
            past_cha = sum(cha_phys_current[i] * self.DT for i in range(q_now))
            past_dis = sum(dis_phys_current[i] * self.DT for i in range(q_now))
            
            solver.Add(
                past_cha + sum((cha_phys_current[q_now+i] - dis_close[i] + cha_idc[i]) * self.DT 
                              for i in range(horizon_length)) <= energy_cap * n_cycles
            )
            solver.Add(
                past_dis + sum((dis_phys_current[q_now+i] - cha_close[i] + dis_idc[i]) * self.DT 
                              for i in range(horizon_length)) <= energy_cap * n_cycles
            )
            
            # Objective: maximize profit over remaining horizon
            solver.Maximize(
                sum(idc_price_current[q_now + i] * ((dis_idc[i] + dis_close[i]) - (cha_idc[i] + cha_close[i])) * self.DT
                    for i in range(horizon_length))
            )
            
            # Solve
            status = solver.Solve()
            if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
                print(f"    Warning: Quarter {q_now} optimization did not find optimal solution")
                # Use zero action as fallback
                cha_decision = 0.0
                dis_decision = 0.0
                cha_close_decision = 0.0
                dis_close_decision = 0.0
                cur_decision = 0.0
            else:
                # Extract ONLY the current quarter decision (first element)
                cha_decision = cha_idc[0].solution_value()
                dis_decision = dis_idc[0].solution_value()
                cha_close_decision = cha_close[0].solution_value()
                dis_close_decision = dis_close[0].solution_value()
                cur_decision = cur[0].solution_value()
            
            # Apply decision and update state
            cha_idc_final[q_now] = cha_decision
            dis_idc_final[q_now] = dis_decision
            cha_idc_close_final[q_now] = cha_close_decision
            dis_idc_close_final[q_now] = dis_close_decision
            curtail_final[q_now] = cur_decision
            
            # Update physical positions
            cha_phys_current[q_now] = cha_phys_current[q_now] - dis_close_decision + cha_decision
            dis_phys_current[q_now] = dis_phys_current[q_now] - cha_close_decision + dis_decision
            
            # Update SOC
            current_soc = current_soc + self.DT * (cha_phys_current[q_now] - dis_phys_current[q_now])
            soc_final.append(current_soc)
            
            # Calculate profit for this quarter (using ACTUAL price, not forecast)
            quarter_profit = idc_price_vector[q_now] * (
                (dis_decision + dis_close_decision) - (cha_decision + cha_close_decision)
            ) * self.DT
            total_profit_idc += quarter_profit
            
            # Track history
            history['quarter'].append(q_now)
            history['idc_prices_seen'].append(idc_price_vector[q_now])
            history['pv_forecast_seen'].append(pv_current[q_now])
            history['decision_cha'].append(cha_decision + cha_close_decision)
            history['decision_dis'].append(dis_decision + dis_close_decision)
            history['soc'].append(current_soc)
            history['grid_flow'].append((pv_current[q_now] - cur_decision) + (dis_phys_current[q_now] - cha_phys_current[q_now]))
            history['profit_incremental'].append(quarter_profit)
        
        print(f"  Rolling Horizon: Completed all {Q} quarters")
        
        return {
            'profit_daa': profit_daa,
            'profit_ida': profit_ida,
            'profit_idc': total_profit_idc,
            'profit_total': profit_daa + profit_ida + total_profit_idc,
            'soc': soc_final[:-1],  # Remove last element to match length
            'cha_daa': cha_daa,
            'dis_daa': dis_daa,
            'cha_ida': cha_ida,
            'dis_ida': dis_ida,
            'cha_idc': cha_idc_final,
            'dis_idc': dis_idc_final,
            'cha_idc_close': cha_idc_close_final,
            'dis_idc_close': dis_idc_close_final,
            'cha_phys_final': cha_phys_current,
            'dis_phys_final': dis_phys_current,
            'curtail': curtail_final,
            'history': history
        }
