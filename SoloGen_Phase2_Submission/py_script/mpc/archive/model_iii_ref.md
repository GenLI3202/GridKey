1. 整体框架设计：三层嵌套结构根据你的 p2_bi_model_ggdp.tex 文件 2，你的整个解决方案实际上包含三个嵌套的循环：外层（Meta-Optimization）- 元优化：目标： 找到最优的“权衡参数” $\alpha$ 3。方法： 循环遍历一系列 $\alpha$ 值（例如从 0.1 到 3.0）。评估： 对每个 $\alpha$ 值，运行一次中层 (MPC) 模拟，得到全年的总利润和总退化成本。然后，根据项目要求计算 10 年期 ROI，并选出 ROI 最高的 $\alpha^*$ 4。中层（MPC Simulation）- 滚动时域模拟：目标： 在给定一个 $\alpha$ 值的情况下，模拟一整年的运行。方法： 这就是你请求的“滚动时域” 5。它不是一次性求解全年，而是（例如）循环 365 次，每次求解一个 24-48 小时的“内层 MILP”。输出： 返回该 $\alpha$ 值下的全年总利润和全年总退化成本。内层（MILP Solve）- 单次求解：目标： 在给定的短期（例如 48 小时）数据和初始 SOC 条件下，求解最优竞价策略。方法： 这就是你的 BESSOptimizerModelIII（你目前 optimizer.py 中 ModelII 的扩展）的核心 solve() 功能。2. 建议的 Python 代码框架（伪代码）你需要创建两个新类来实现这个结构：MPCSimulator 和 MetaOptimizer。步骤 1：完成你的 BESSOptimizerModelIII首先，你需要按照你的 LaTeX 文档 6 完成 Model (iii)。这应该是在 optimizer.py 中：

```python
# -----------------------------------------------
# 1. 扩展你现有的 optimizer.py
# -----------------------------------------------
# (你已经有了 BESSOptimizerModelI 和 BESSOptimizerModelII)

class BESSOptimizerModelIII(BESSOptimizerModelII):
    """
    Phase II Model (iii): Model (ii) + Calendar Aging Cost
    
    继承自 Model II (循环老化成本)，并增加了基于 SOS2 的
    日历老化成本 。
    """
    def __init__(self, alpha: float = 1.0, calendar_config_path: str = None):
        super().__init__(alpha=alpha)
        
        # 加载日历老化成本参数 [cite: 109-110]
        # (来自 "aging_config.json" 或 tex 文件中的表格)
        self.calendar_params = self._load_calendar_config(calendar_config_path)
        logger.info("Initialized BESSOptimizerModelIII with Calendar Aging.")

    def _load_calendar_config(self, config_path):
        # ... 加载你的日历老化断点 (breakpoints) 的逻辑 ...
        # [cite: 109-110]
        # 示例:
        # self.soc_breakpoints_kwh = [0, 1118, 2236, 3354, 4472]
        # self.cost_breakpoints_eur_hr = [1.79, 2.15, 3.58, 6.44, 10.73]
        pass

    def build_optimization_model(self, country_data: pd.DataFrame, 
                               c_rate: float, 
                               initial_segment_soc: Dict[int, float]) -> pyo.ConcreteModel:
        
        # 1. 调用父类的 build_optimization_model
        #    注意：父类会处理 Model(i) 和 Model(ii) 的所有逻辑
        #    你需要修改父类的 build_optimization_model，使其接受
        #    `initial_segment_soc` 而不是 `initial_soc`
        model = super().build_optimization_model(country_data, c_rate, None) # 传入 daily_cycle_limit=None

        # 2. **添加日历老化 (Calendar Aging) 逻辑 **
        
        # 新增 SOS2 相关的 Set 和 Vars
        model.I = pyo.Set(initialize=range(1, len(self.soc_breakpoints_kwh) + 1))
        model.lambda_cal = pyo.Var(model.T, model.I, domain=pyo.NonNegativeReals, bounds=(0, 1))
        model.c_cal_cost_t = pyo.Var(model.T, domain=pyo.NonNegativeReals) # 每小时的日历成本

        # SOS2 约束 [cite: 96-98]
        def calendar_soc_rule(m, t):
            # e_soc(t) 必须由 SOS2 变量加权得出
            return m.e_soc[t] == sum(m.lambda_cal[t, i] * self.soc_breakpoints_kwh[i-1] for i in m.I)
        model.calendar_soc_con = pyo.Constraint(model.T, rule=calendar_soc_rule)

        def calendar_cost_rule(m, t):
            # c_cal_cost_t(t) 必须由 *相同* 的 SOS2 变量加权得出
            return m.c_cal_cost_t[t] == sum(m.lambda_cal[t, i] * self.cost_breakpoints_eur_hr[i-1] for i in m.I)
        model.calendar_cost_con = pyo.Constraint(model.T, rule=calendar_cost_rule)

        def calendar_sos2_sum_rule(m, t):
            # 权重和必须为 1
            return sum(m.lambda_cal[t, i] for i in m.I) == 1
        model.calendar_sos2_sum_con = pyo.Constraint(model.T, rule=calendar_sos2_sum_rule)

        # 定义 SOS2 变量集
        model.calendar_sos2_set = pyo.SOSConstraint(model.T, var=model.lambda_cal, sos=2)
        
        # 3. **从目标函数中减去日历老化成本 **
        
        # 获取已有的目标函数表达式
        existing_objective = model.objective.expr
        
        # 定义新的日历成本项
        cost_calendar = sum(model.c_cal_cost_t[t] * model.dt for t in model.T)
        
        # 设置新的、完整的 Model (iii) 目标函数
        model.objective.set_value(
            existing_objective - model.alpha * cost_calendar
        )
        
        logger.info("Extended Model (ii) to Model (iii) with calendar aging.")
        return model
```

步骤 2：设计 MPCSimulator 类（中层）
这个类将包裹你的 BESSOptimizerModelIII 并执行滚动时域模拟。

```python
# -----------------------------------------------
# 2. 设计 MPCSimulator (中层循环)
# -----------------------------------------------
import pandas as pd
import numpy as np

class MPCSimulator:
    """
    执行滚动时域模拟，如 Collath et al. (2023) 和
    你的 p2_bi_model_ggdp.tex  中所述。
    """
    def __init__(self, optimizer_model: BESSOptimizerModelIII, 
                 full_data: pd.DataFrame, 
                 horizon_hours: int = 48, 
                 execution_hours: int = 24):
        
        self.optimizer = optimizer_model
        self.full_data = full_data
        self.horizon_steps = int(horizon_hours / optimizer_model.market_params['time_step_hours'])
        self.execution_steps = int(execution_hours / optimizer_model.market_params['time_step_hours'])
        self.total_steps = len(full_data)
        
        # 从你的模型中获取电池参数
        self.battery_params = optimizer_model.battery_params
        self.num_segments = optimizer_model.degradation_params['num_segments']
        self.segment_capacity = optimizer_model.degradation_params['segment_capacity_kwh']
        
        logger.info(f"MPCSimulator initialized: Horizon={horizon_hours}h, Execution={execution_hours}h")

    def _get_initial_segment_soc(self, total_soc_kwh: float) -> Dict[int, float]:
        """
        根据总 SOC 计算每个分段的 SOC。
        这是 Model (ii) 所需的 `initial_segment_soc`。
        假设从最深的分段 (J) 向上填充到最浅的分段 (1)。
        (这与 Xu et al. 的 LIFO 逻辑一致 [cite: 68-69])
        """
        segment_soc = {}
        remaining_soc = total_soc_kwh
        for j in range(self.num_segments, 0, -1): # 从 J 到 1
            energy_in_this_segment = min(remaining_soc, self.segment_capacity)
            segment_soc[j] = energy_in_this_segment
            remaining_soc -= energy_in_this_segment
        return segment_soc

    def run_full_simulation(self) -> Dict[str, Any]:
        """
        执行完整的年度滚动时域模拟 [cite: 130-135]。
        """
        # 初始化状态
        current_total_soc = self.battery_params['initial_soc'] * self.battery_params['capacity_kwh']
        
        # 存储结果
        all_profits = []
        all_cyclic_costs = []
        all_calendar_costs = []
        all_soc_profiles = []
        all_bids = {
            'p_ch': [], 'p_dis': [], 
            'p_afrr_pos_e': [], 'p_afrr_neg_e': [],
            'c_fcr': [], 'c_afrr_pos': [], 'c_afrr_neg': []
        }

        # 滚动循环 [cite: 131]
        for t_start in range(0, self.total_steps, self.execution_steps):
            t_end_horizon = min(t_start + self.horizon_steps, self.total_steps)
            t_end_execution = min(t_start + self.execution_steps, self.total_steps)
            
            if t_start >= t_end_execution:
                break # 到达终点

            logger.info(f"Running MPC step: {t_start} / {self.total_steps}")
            
            # 1. 准备数据和状态
            data_slice = self.full_data.iloc[t_start:t_end_horizon].reset_index(drop=True)
            initial_segment_soc = self._get_initial_segment_soc(current_total_soc)
            
            # 2. 构建并求解“内层”MILP
            # (这里假设 c_rate 是固定的，如果不是，你需要从配置中传入)
            c_rate = 0.5 # 示例
            model = self.optimizer.build_optimization_model(data_slice, c_rate, initial_segment_soc)
            results = self.optimizer.solve_model(model)
            
            if results['status'] not in ['optimal', 'feasible']:
                logger.error(f"Solver failed at step {t_start}. Stopping simulation.")
                break
                
            # 3. 记录执行时间段的结果 
            execution_indices = range(t_end_execution - t_start)
            
            # (这里需要你编写详细的利润/成本计算逻辑)
            # ...
            # all_profits.append(profit_for_this_execution_block)
            # all_cyclic_costs.append(cyclic_cost_for_this_block)
            # all_calendar_costs.append(calendar_cost_for_this_block)
            
            # (示例：记录 bids)
            # all_bids['p_ch'].append(results['p_ch'][:len(execution_indices)])
            # ...
            
            # 4. 更新下一个循环的状态 [cite: 133]
            # 获取执行周期的最后一步的 SOC
            last_execution_step = len(execution_indices) - 1
            current_total_soc = pyo.value(model.e_soc[last_execution_step])
            all_soc_profiles.append(current_total_soc)

        # 5. 聚合全年结果 [cite: 134]
        total_revenue = sum(all_profits)
        total_cyclic_cost = sum(all_cyclic_costs)
        total_calendar_cost = sum(all_calendar_costs)
        total_degradation_cost = total_cyclic_cost + total_calendar_cost

        logger.info(f"MPC Simulation finished for alpha={self.optimizer.degradation_params['alpha']}")
        logger.info(f"  Total Revenue: {total_revenue:.2f} EUR")
        logger.info(f"  Total Degradation Cost: {total_degradation_cost:.2f} EUR")

        return {
            "total_revenue": total_revenue,
            "total_degradation_cost": total_degradation_cost,
            "final_soc": current_total_soc,
            "bids": all_bids
        }
```

步骤 3：设计 MetaOptimizer 类（外层）这个类将运行 MPCSimulator 多次，以找到最佳的 $\alpha$ 值。

```python
# -----------------------------------------------
# 3. 设计 MetaOptimizer (外层循环)
# -----------------------------------------------

class MetaOptimizer:
    """
    执行“元优化”循环，以找到最大化 10 年期 ROI 的
    最优 alpha ( degradation price) 值。
    
    """
    def __init__(self, full_data: pd.DataFrame, 
                 country_config: Dict[str, Any], 
                 alpha_values: List[float]):
        
        self.full_data = full_data
        self.country_config = country_config # 包含 WACC, inflation 等
        self.alpha_values = alpha_values
        logger.info(f"MetaOptimizer initialized. Testing {len(alpha_values)} alpha values.")

    def _calculate_10_year_roi(self, annual_revenue: float, 
                             annual_degradation_cost: float,
                             total_investment: float) -> float:
        """
        计算 10 年期 ROI。
        
        **重要提示**: 根据项目描述 "whole_project_description.md" 的
        "Important Update"，我们必须假设一个 *固定的 10 年生命周期*，
        而不是使用 EoL (End-of-Life) 作为终点。
       
        
        这里的 "annual_degradation_cost" 只是一个评估指标 (占 30% 权重)，
        *而不是* 导致提前更换的因素。
        
        此处的 ROI 计算需要你根据比赛规则来精确定义。
        一个简化的 NPV (Net Present Value) 示例：
        """
        wacc = self.country_config['wacc']
        inflation = self.country_config['inflation']
        
        net_present_value = 0
        for m in range(1, 11): # 10 年
            # 假设收入和成本随通胀增长
            profit_in_year_m = annual_revenue * ((1 + inflation) ** (m - 1))
            # 按 WACC 贴现
            net_present_value += profit_in_year_m / ((1 + wacc) ** m)
            
        roi = (net_present_value - total_investment) / total_investment
        return roi

    def find_optimal_alpha(self) -> Dict[str, Any]:
        """
        执行元优化参数扫描 [cite: 142]。
        """
        best_alpha = None
        best_roi = -float('inf')
        simulation_results = []
        
        total_investment = self.battery_params['capacity_kwh'] * 200 # EUR 200/kWh

        for alpha in self.alpha_values:
            logger.info(f"--- META-OPTIMIZATION: Testing alpha = {alpha} ---")
            
            # 1. 创建 Model (iii) 实例
            optimizer_model = BESSOptimizerModelIII(alpha=alpha)
            
            # 2. 创建 MPC 模拟器实例
            simulator = MPCSimulator(optimizer_model, self.full_data)
            
            # 3. 运行完整的中层模拟
            annual_results = simulator.run_full_simulation()
            
            # 4. 计算 10 年期 ROI 
            # 注意：这里的 revenue 是已经减去了退化成本的净利润吗？
            # 你的模型  是 max(Revenue - alpha * Cost)，
            # 所以 annual_results['total_revenue'] 已经是净利润了。
            # (你需要根据你的结果输出来调整这里的 annual_profit)
            annual_profit = annual_results['total_revenue'] # 假设这是净利润
            annual_degradation_cost = annual_results['total_degradation_cost'] # 用于评估
            
            current_roi = self._calculate_10_year_roi(annual_profit, 
                                                    annual_degradation_cost, 
                                                    total_investment)
            
            result_summary = {
                "alpha": alpha,
                "roi_10_year": current_roi,
                "annual_profit_eur": annual_profit,
                "annual_degradation_cost_eur": annual_degradation_cost,
            }
            simulation_results.append(result_summary)
            
            # 5. 选择最佳结果 [cite: 145]
            if current_roi > best_roi:
                best_roi = current_roi
                best_alpha = alpha
                
        logger.info(f"Meta-Optimization finished.")
        logger.info(f"Optimal Alpha found: {best_alpha} with 10-Year ROI: {best_roi:.2%}")
        
        return {
            "best_alpha": best_alpha,
            "best_roi": best_roi,
            "full_results_sweep": simulation_results
        }
```

3. 给你的“Coding Agent”的行动计划实现 BESSOptimizerModelIII：在 optimizer.py 中创建 BESSOptimizerModelIII 类，使其继承自 BESSOptimizerModelII。修改 build_optimization_model 方法，添加你的 LaTeX 文件中 "Calendar Aging PWL Constraints" 7 所需的变量（lambda_cal, c_cal_cost_t）和约束（calendar_soc_con, calendar_cost_con, calendar_sos2_sum_con, calendar_sos2_set）。修改目标函数，减去 model.alpha * cost_calendar 8。重要： 修改 ModelII 和 ModelIII 的 build_optimization_model，使其接受 initial_segment_soc 字典作为参数，而不是 initial_soc 浮点数，并正确设置 e_soc_j[t=0, j] 的初始值。实现 MPCSimulator 类：创建一个新文件（例如 mpc_simulator.py）并导入你的 BESSOptimizerModelIII。实现 MPCSimulator 类（如上所示）。你需要特别注意 _get_initial_segment_soc 的逻辑（从深到浅填充还是从浅到深填充，这取决于你的分段定义）和 run_full_simulation 循环中记录结果的部分 9。实现 MetaOptimizer 类：在同一个新文件中（或一个主 run.py 文件中）实现 MetaOptimizer 类。注意： 你必须解决我发现的那个矛盾。你的 p2_bi_model_ggdp.tex 10 提到了基于 SOH 的电池更换，但你的 whole_project_description.md 的“重要更新”明确禁止了这一点，要求使用固定的 10 年寿命。建议： 严格遵守 whole_project_description.md 的规定。在 _calculate_10_year_roi 中，不要模拟电池更换。使用固定的 10 年项目期，并使用 annual_profit（来自你的目标函数）来计算 NPV。annual_degradation_cost 只是一个用于最终评估（占 30% 权重）的输出指标，而不是影响 ROI 计算中现金流的变量（除非它已经通过 $\alpha$ 从 annual_profit 中减去了）。这个框架为你提供了所需的滚动时域功能，并直接连接到你的元优化目标。