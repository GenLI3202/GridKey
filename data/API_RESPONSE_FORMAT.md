# API 响应格式说明

## 概述

GridKey Optimizer API 返回标准的 JSON 响应，包含优化结果、调度计划、SOC 轨迹等。

**响应结构：** 外层包装 + 内部 `data` 字段

---

## 完整响应结构

```json
{
    "status": "success | error",
    "data": {
        // OptimizationResult 内容
    }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `"success"` 或 `"error"` |
| `data` | object | 优化结果数据 (成功时) 或错误信息 (失败时) |

---

## data 字段详解

### 核心指标 (Core Metrics)

| 字段 | 类型 | 说明 |
|------|------|------|
| `objective_value` | float | 优化目标函数值 (EUR) |
| `net_profit` | float | 净利润 = objective_value - degradation_cost (EUR) |

### 收入明细 (Revenue Breakdown)

`revenue_breakdown` 是一个字典，包含各市场的收入：

| 字段 | 类型 | 说明 |
|------|------|------|
| `da` | float | 日前市场收入 (EUR) |
| `afrr_energy` | float | aFRR 能量市场收入 (EUR) |
| `fcr` | float | FCR 容量市场收入 (EUR) |
| `afrr_capacity` | float | aFRR 容量市场收入 (EUR) |
| `renewable_export` | float | 可再生能源出口收入 (EUR, 仅 Model III-renew) |

### 退化成本 (Degradation Costs)

| 字段 | 类型 | 说明 |
|------|------|------|
| `degradation_cost` | float | 总退化成本 = cyclic + calendar (EUR) |
| `cyclic_aging_cost` | float | 循环老化成本 (EUR) |
| `calendar_aging_cost` | float | 日历老化成本 (EUR) |

### 调度计划 (Schedule)

`schedule` 是一个数组，每个元素代表一个时间步的调度决策：

```json
{
    "timestamp": "2024-01-01T00:00:00",
    "action": "charge | discharge | idle",
    "power_kw": 1500.0,
    "market": "da",
    "soc_after": 0.55,
    "renewable_action": "self_consume",      // 可选
    "renewable_power_kw": 500.0             // 可选
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | string (ISO 8601) | 时间戳 |
| `action` | string | `"charge"`, `"discharge"`, 或 `"idle"` |
| `power_kw` | float | 充电/放电功率 (kW)，负值为充电，正值为放电 |
| `market` | string | 市场类型: `"da"`, `"fcr"`, `"afrr_cap"`, `"afrr_energy"` |
| `soc_after` | float | 该步结束后的 SOC (0-1) |
| `renewable_action` | string? | 可选: `"self_consume"`, `"export"`, `"curtail"` |
| `renewable_power_kw` | float? | 可选: 可再生能源功率 (kW) |

### SOC 轨迹 (SOC Trajectory)

`soc_trajectory` 是一个浮点数数组，表示每个时间步的 SOC：

```json
"soc_trajectory": [0.50, 0.55, 0.60, 0.45, 0.40, ...]
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `soc_trajectory` | array of float | SOC 时间序列，每个值在 [0, 1] 范围内 |
| 长度 | int | = time_horizon_hours × 4 (15-min 分辨率) |

### 可再生能源利用 (Renewable Utilization)

`renewable_utilization` 是可选字段，仅在 Model III-renew 时出现：

```json
{
    "total_generation_kwh": 8000.0,
    "self_consumption_kwh": 5000.0,
    "export_kwh": 2000.0,
    "curtailment_kwh": 1000.0,
    "utilization_rate": 0.875
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_generation_kwh` | float | 总发电量 (kWh) |
| `self_consumption_kwh` | float | 自消纳电量 (kWh) |
| `export_kwh` | float | 出口到电网电量 (kWh) |
| `curtailment_kwh` | float | 弃光电量 (kWh) |
| `utilization_rate` | float | 利用率 = (self + export) / total |

### 求解器元数据 (Solver Metadata)

| 字段 | 类型 | 说明 |
|------|------|------|
| `solve_time_seconds` | float | 求解时间 (秒) |
| `solver_name` | string | 求解器名称: `"highs"`, `"cplex"`, `"gurobi"` |
| `model_type` | string | 模型类型: `"I"`, `"II"`, `"III"`, `"III-renew"` |
| `status` | string | 求解状态: `"optimal"`, `"feasible"`, `"infeasible"`, `"timeout"` |
| `num_variables` | int? | 可选: 变量数量 (调试用) |
| `num_constraints` | int? | 可选: 约束数量 (调试用) |

---

## 响应状态说明

### status = "success"

优化成功完成，`data` 包含完整结果。

### status = "error"

优化失败，`data` 包含错误信息：

```json
{
    "status": "error",
    "data": {
        "detail": "Missing required market_prices keys: ['day_ahead']"
    }
}
```

常见错误：
| 错误信息 | 原因 | 解决方法 |
|----------|------|----------|
| `Missing required market_prices keys` | 缺少必需的价格字段 | 检查请求包含所有 6 个价格数组 |
| `time_horizon_hours exceeds maximum` | 时间范围超出限制 | HiGHS 最大 12h，商业求解器最大 48h |
| `Optimization failed: ...` | 求解器错误 | 检查输入数据有效性 |

---

## 与内部输出格式的区别

| 项目 | 内部格式 (`optimizer_output_template.json`) | API 格式 (`api_response_template.json`) |
|------|---------------------------------------------|----------------------------------------|
| **用途** | 调试、详细分析 | API 响应、前端展示 |
| **粒度** | 每步 40+ 字段 | 每步 5-7 字段 |
| **细分段** | segment_1 ~ segment_10 | 汇总为 cyclic_aging_cost |
| **二元变量** | y_ch, y_dis, y_fcr 等 | 无 |
| **功率细分** | p_ch, p_dis, p_afrr_* | 简化为 action + power |
| **价格字段** | 每步价格 | 已计算到 revenue |
| **SOC** | soc_kwh, soc_pct | 归一化的 soc_trajectory |

---

## 示例：完整响应

```json
{
    "status": "success",
    "data": {
        "objective_value": 450.0,
        "net_profit": 425.0,
        "revenue_breakdown": {
            "da": 350.0,
            "afrr_energy": 80.0,
            "fcr": 20.0
        },
        "degradation_cost": 25.0,
        "cyclic_aging_cost": 15.0,
        "calendar_aging_cost": 10.0,
        "schedule": [
            {"timestamp": "2024-01-01T00:00:00", "action": "discharge", "power_kw": 2236.0, "market": "da", "soc_after": 0.25}
        ],
        "soc_trajectory": [0.5, 0.25, 0.0, 0.17],
        "solve_time_seconds": 0.523,
        "solver_name": "highs",
        "model_type": "III-renew",
        "status": "optimal"
    }
}
```
