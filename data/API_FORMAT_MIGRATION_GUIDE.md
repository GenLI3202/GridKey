# API 输入格式变更说明

## 概述

原 `optimizer_input_template.json` 使用 **pandas DataFrame dict 格式**，已更新为 **FastAPI Pydantic 格式**。

**有两个 API 端点可用：**

| 端点 | 用途 | 时间范围 | 数组长度 |
|------|------|----------|----------|
| `/api/v1/optimize` | 灵活时间范围 | 1-12 小时 (HiGHS) | 15-min: 4-48, 4h块: 1-3 |
| `/api/v1/optimize-mpc` | 固定 12h MPC 滚动优化 | 固定 12 小时 | 15-min: 48, 4h块: 3 |

**重要：** 两个端点使用相同的字段名和格式，只是长度验证不同。

---

## ⚠️ 关键说明：模板文件与 API 请求的区别

### 两个模板文件

| 文件 | 用途 | 能否直接粘贴到 API |
|------|------|-------------------|
| `optimizer_input_template.json` | 完整文档（含元数据说明） | ❌ 否 |
| `optimizer_input_template_request_only.json` | 纯 API 请求 JSON | ✅ 是 |

### 为什么原模板不能直接粘贴？

`optimizer_input_template.json` 是**文档格式**，包含 API 不认识的元数据：

```json
{
    "_specification": {...},           // ← API 不认识（文档说明）
    "_notes": {...},                    // ← API 不认识（文档说明）
    "request_body": {                   // ← 多了一层包装！
        "location": "Munich",           //    实际请求字段在里面
        "country": "DE_LU",
        ...
    },
    "_field_descriptions": {...},       // ← API 不认识（文档说明）
    "_example_curl": {...}              // ← API 不认识（文档说明）
}
```

**API 期望的格式（根级别字段）：**

```json
{
    "location": "Munich",               // ← 直接在根级别
    "country": "DE_LU",
    "market_prices": {...},
    ...
}
```

### 正确的使用方式

**方式 1：使用 request_only 版本（推荐）**
```bash
# 直接复制这个文件的内容到 API
data/optimizer_input_template_request_only.json
```

**方式 2：从完整模板中提取**
```bash
# 打开 optimizer_input_template.json
# 只复制 "request_body" 里面的内容（不包括 "request_body" 这一行）
# 粘贴到 API
```

### 常见错误：422 Unprocessable Content

如果你看到 `422 Unprocessable Content` 错误，通常是因为：

| 错误原因 | 解决方法 |
|----------|----------|
| 粘贴了整个模板（含元数据） | 使用 `request_only` 版本 |
| `request_body` 包装层存在 | 只复制其内部内容 |
| 包含 `"_description"` 等字段 | 移除所有 `_` 开头的字段 |

---

## 格式对比

### 旧格式 (Invalid for current API)

```json
{
    "data_slice": {
        "price_day_ahead": {
            "0": 54.89,
            "1": 54.89,
            "2": 54.89,
            ...
        },
        "price_fcr": {
            "0": 38.52,
            "1": 38.52,
            ...
        }
    }
}
```

**特征：**
- 嵌套在 `data_slice` 对象内
- 字符串索引 (`"0"`, `"1"`, `"2"`, ...)
- 列名使用下划线前缀 (`price_day_ahead`, `price_fcr`)
- 包含元数据字段 (`_unit`, `_resolution`, `_description`)

### 新格式 (Valid for current API)

```json
{
    "market_prices": {
        "day_ahead": [54.89, 54.89, 54.89, ...],
        "afrr_energy_pos": [40.0, 40.0, ...],
        "fcr": [38.52, 42.30, 45.80]
    }
}
```

**特征：**
- 扁平结构，直接在请求体中
- 简单数组格式
- 字段名无下划线前缀
- 无元数据（值自解释）

---

## 字段映射表

| 旧字段名 | 新字段名 | 格式变化 |
|----------|----------|----------|
| `data_slice.price_day_ahead` | `market_prices.day_ahead` | dict → array |
| `data_slice.price_afrr_energy_pos` | `market_prices.afrr_energy_pos` | dict → array |
| `data_slice.price_afrr_energy_neg` | `market_prices.afrr_energy_neg` | dict → array |
| `data_slice.price_fcr` | `market_prices.fcr` | dict → array |
| `data_slice.price_afrr_pos` | `market_prices.afrr_capacity_pos` | dict → array |
| `data_slice.price_afrr_neg` | `market_prices.afrr_capacity_neg` | dict → array |
| `data_slice.p_renewable_forecast_kw` | `renewable_generation` | dict → array |

---

## 数组长度要求 (12 小时示例)

| 字段 | 旧格式 | 新格式 | 长度 |
|------|--------|--------|------|
| 15-min 数据 | 12 个键值对 | 48 个元素的数组 | 48 (12h × 4) |
| 4-h 块数据 | 3 个键值对 | 3 个元素的数组 | 3 (12h ÷ 4) |

---

## Agent 输出要求

### Market Price Agent 应输出：

**提取的 6 个价格数组：**

```python
{
    "day_ahead": [54.89, 54.89, 54.89, 54.89, ...],  # 48 values @ 15-min
    "afrr_energy_pos": [127.29, 101.87, 85.42, ...], # 48 values @ 15-min
    "afrr_energy_neg": [56.90, 52.40, 48.50, ...],   # 48 values @ 15-min
    "fcr": [38.52, 42.30, 45.80],                    # 3 values @ 4-hour blocks
    "afrr_capacity_pos": [14.45, 16.20, 18.50],     # 3 values @ 4-hour blocks
    "afrr_capacity_neg": [0.62, 0.58, 0.65]          # 3 values @ 4-hour blocks
}
```

### Renewable Generation Agent 应输出：

**15-min 分辨率数组：**

```python
[0.0, 0.0, 0.0, 0.0, 50.0, 150.0, ...]  # 48 values (kW)
```

**或按小时（由 API 层扩展）：**

```python
[0, 0, 0, 50, 300, 700, 950, 800, ...]  # 12 values (kW per hour)
```

### API 组装后的完整请求：

两个 Agent 的输出在 API 层组装为：

```json
{
    "location": "Munich",
    "country": "DE_LU",
    "model_type": "III-renew",
    "c_rate": 0.5,
    "alpha": 1.0,
    "market_prices": {
        "day_ahead": [54.89, 54.89, ...],        // 来自 Price Agent
        "afrr_energy_pos": [127.29, 101.87, ...],
        "afrr_energy_neg": [56.90, 52.40, ...],
        "fcr": [38.52, 42.30, 45.80],
        "afrr_capacity_pos": [14.45, 16.20, 18.50],
        "afrr_capacity_neg": [0.62, 0.58, 0.65]
    },
    "renewable_generation": [0.0, 0.0, 50.0, ...]  // 来自 Renewable Agent
}
```

---

## 完整 API 请求示例

### 端点 1: POST /api/v1/optimize (灵活时间范围)

```json
{
    "time_horizon_hours": 6,  // 可选: 1-12, 默认 6
    "model_type": "III-renew",
    "c_rate": 0.5,
    "alpha": 1.0,
    "market_prices": {
        "day_ahead": [54.89, ...],           // time_horizon_hours × 4 个值
        "afrr_energy_pos": [127.29, ...],
        "afrr_energy_neg": [56.90, ...],
        "fcr": [38.52, ...],                 // ceil(time_horizon_hours / 4) 个值
        "afrr_capacity_pos": [14.45, ...],
        "afrr_capacity_neg": [0.62, ...]
    },
    "renewable_generation": [0.0, ...]       // 可选, time_horizon_hours × 4 个值
}
```

### 端点 2: POST /api/v1/optimize-mpc (固定 12h)

```json
{
    "location": "Munich",
    "country": "DE_LU",
    "model_type": "III-renew",
    "c_rate": 0.5,
    "alpha": 1.0,
    "market_prices": {
        "day_ahead": [54.89, 54.89, ...],        // 48 values
        "afrr_energy_pos": [127.29, 101.87, ...], // 48 values
        "afrr_energy_neg": [56.90, 52.40, ...],   // 48 values
        "fcr": [38.52, 42.30, 45.80],             // 3 values
        "afrr_capacity_pos": [14.45, 16.20, 18.50],
        "afrr_capacity_neg": [0.62, 0.58, 0.65]
    },
    "renewable_generation": [0.0, 0.0, 50.0, ...]  // 48 values, optional
}
```

---

## 数据类型说明

| 字段 | 单位 | 允许值 |
|------|------|--------|
| `day_ahead` | EUR/MWh | float, 可为负值 |
| `afrr_energy_pos` | EUR/MWh | float, ≥ 0 |
| `afrr_energy_neg` | EUR/MWh | float, ≥ 0 |
| `fcr` | EUR/MW | float, ≥ 0 |
| `afrr_capacity_pos` | EUR/MW | float, ≥ 0 |
| `afrr_capacity_neg` | EUR/MW | float, ≥ 0 |
| `renewable_generation` | kW | float, ≥ 0 |

---

## 迁移检查清单

如果你的 Agent 目前输出旧格式：

- [ ] 将 dict 格式 (`"0": value, "1": value`) 改为数组格式 (`[value1, value2, ...]`)
- [ ] 移除 `data_slice` 包装层
- [ ] 移除元数据字段 (`_unit`, `_resolution` 等)
- [ ] 重命名字段（移除 `price_` 前缀）
- [ ] 确保 4-hour 块数据是 3 个元素的数组（不是 48 个重复值）

---

## 兼容性说明

**重要：** 旧格式是 pandas DataFrame 导出格式，用于内部数据传递。新格式是 FastAPI 的标准 JSON 请求格式。

如果需要保留旧格式支持，可以在 API 层添加转换逻辑。当前版本仅支持新格式。

---

## API 响应格式

### 响应模板文件

| 文件 | 用途 |
|------|------|
| `api_response_template.json` | API 响应示例 (可用于测试) |
| `API_RESPONSE_FORMAT.md` | 完整响应格式文档 |

### 响应结构概览

```json
{
    "status": "success",
    "data": {
        "objective_value": 450.0,
        "net_profit": 425.0,
        "revenue_breakdown": {"da": 350.0, "afrr_energy": 80.0, "fcr": 20.0},
        "degradation_cost": 25.0,
        "schedule": [{"timestamp": "...", "action": "charge", "power_kw": 1500.0, "soc_after": 0.55}],
        "soc_trajectory": [0.5, 0.55, 0.60, ...],
        "renewable_utilization": {...},
        "solve_time_seconds": 0.523,
        "solver_name": "highs",
        "model_type": "III-renew",
        "status": "optimal"
    }
}
```

详细字段说明请参考 `API_RESPONSE_FORMAT.md`。

---

## 文件清单

| 文件 | 类型 | 用途 |
|------|------|------|
| `optimizer_input_template.json` | 输入模板 | API 请求示例 (可直接使用) |
| `api_response_template.json` | 输出模板 | API 响应示例 |
| `API_FORMAT_MIGRATION_GUIDE.md` | 文档 | 输入格式迁移说明 |
| `API_RESPONSE_FORMAT.md` | 文档 | 输出格式完整说明 |
| `optimizer_output_template.json` | 内部格式 | 求解器详细输出 (调试用) |
| `optimizer_input_template_request_only.json` | 输入模板 | 纯净版本 (无元数据) |
