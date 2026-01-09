# Quarto 使用指南

本指南针对 GridKey 项目，提供 Quarto 的核心用法说明。

---

## 1. 前言：为什么引入 Quarto

**传统方法的局限：**

| 工具 | 优点 | 痛点 |
|------|------|------|
| LaTeX | 排版精美 | 语法繁琐，不支持代码执行 |
| Markdown | 语法简洁 | 不支持文献引用，格式单一 |
| Jupyter Notebook | 代码交互 | 输出格式有限，版本控制困难 |

**Quarto 的优势：**

- **一源多输出**：同一 `.qmd` 文件 → HTML / PDF / Slides
- **语法友好**：Markdown 正文 + LaTeX 数学公式
- **原生引用**：直接使用 BibTeX (`.bib`) 文献库
- **代码集成**：Python 代码块实时执行，结果嵌入文档
- **现代演示**：Revealjs 幻灯片，支持交互式图表

---

## 2. 安装与依赖

### 安装 Quarto

下载：https://quarto.org/docs/get-started/

验证安装：
```bash
quarto --version
```

### Python 依赖

```bash
pip install jupyter pyyaml pandas numpy plotly
```

### PDF 输出依赖（LaTeX）

| 系统 | 推荐发行版 |
|------|-----------|
| Windows | MiKTeX 或 TinyTeX |
| macOS | MacTeX 或 TinyTeX |
| Linux | TeX Live 或 TinyTeX |

快速安装 （示例）TinyTeX：
```bash
quarto install tinytex
```

---

## 3. 工作流概述

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  .qmd 源文件 │  →   │ Quarto 引擎  │  →   │  输出文件    │
│  (Markdown)  │      │  (Pandoc)   │      │ HTML/PDF/...│
└─────────────┘      └─────────────┘      └─────────────┘
       ↑                    ↑
   YAML Header        _quarto.yml
   (文档配置)           (项目配置)
```

### 核心命令（给AI渲染也ok）

```bash
quarto render doc.qmd          # 渲染单个文件
quarto render                  # 渲染整个项目
quarto preview doc.qmd         # 实时预览（自动刷新）
```

---

## 4. 核心概念：配置层级

Quarto 采用**两级配置**，文档配置可覆盖项目配置。

### 项目配置：`_quarto.yml`

定义**全局默认值**，适用于目录下所有 `.qmd` 文件：

```yaml
# _quarto.yml
bibliography: ../references.bib   # 文献库路径
format:
  html:
    theme: cosmo
    toc: true
```

### 文档配置：`.qmd` YAML Header

定义**单文档设置**，位于文件顶部 `---` 之间：

```yaml
---
title: "分析报告"
format:
  html:
    code-fold: true   # 仅此文档生效
---
```

### 继承与覆盖规则

```
┌────────────────────────────────────────────┐
│  _quarto.yml (项目级)                       │
│  • bibliography: ../references.bib  ✓ 继承  │
│  • theme: cosmo                     ✓ 继承  │
│  • toc: true                        ✗ 被覆盖│
└────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────┐
│  document.qmd YAML Header (文档级)          │
│  • toc: false                       ✓ 覆盖  │
│  • code-fold: true                  ✓ 新增  │
└────────────────────────────────────────────┘
```

**规则**：同名字段 → 文档覆盖项目；不同字段 → 合并生效

---

## 5. 输出格式详解

### 格式速查表（完整格式表参阅Quarto官方文档）

| 类型 | 格式 | 命令 | 用途 |
|------|------|------|------|
| 文档 | HTML | `--to html` | 交互式网页报告 |
| 文档 | PDF | `--to pdf` | 打印、正式文档 |
| 文档 | Word | `--to docx` | 可编辑文档，协作 |
| 演示 | **Revealjs** | `--to revealjs` | 网页幻灯片（推荐） |
| 演示 | PowerPoint | `--to pptx` | .pptx 幻灯片 |
| 演示 | Beamer | `--to beamer` | LaTeX PDF 幻灯片 |
| 高级 | Dashboard | `format: dashboard` | 交互式仪表板 |
| 高级 | Website | `type: website` | 多页面文档网站 |

### 5.1 HTML 报告

```yaml
format:
  html:
    toc: true              # 显示目录 (Table of Contents)
    toc-depth: 3           # 目录深度
    code-fold: true        # 代码块可折叠
    code-tools: true       # 显示代码工具栏
    embed-resources: true  # 自包含单文件（便于分享）
    self-contained: true   # 生成独立的 .html 文件，方便通过邮件/Slack 发送
    theme: cosmo           # 主题选择
    number-sections: true  # 章节自动编号
```

**常用主题**：`cosmo`（默认）、`flatly`（扁平）、`darkly`（暗色）、`journal`（学术）

### 5.2 PDF 文档

```yaml
format:
  pdf:
    documentclass: article    # 文档类型：article/report/book
    geometry:                 # 页面几何
      - top=2.5cm
      - bottom=2.5cm
      - left=2cm
      - right=2cm
    number-sections: true     # 章节自动编号
    toc: true                 # 显示目录
    colorlinks: true          # 彩色链接
    fig-pos: 'H'              # 强制图片不乱跑
    keep-tex: false           # 是否保留 .tex 中间文件
```

### 5.3 Revealjs 演示文稿

```yaml
format:
  revealjs:
    theme: serif               # 主题：serif/simple/night/moon
    slide-number: true         # 显示页码
    transition: slide          # 切换动画：slide/fade/convex/none
    chalkboard: true           # 启用画板（可在幻灯片上标注）
    scrollable: true           # 长内容页面可滚动
    incremental: true          # 列表项逐条显示
    center: false              # 内容垂直居中
    width: 1050                # 幻灯片宽度
    height: 700                # 幻灯片高度
    code-line-numbers: true    # 代码行号高亮
    footer: "GridKey Project"  # 页脚文字
```

**幻灯片分页**：使用 `##` 二级标题自动分页

**常用主题**：`serif`（衬线）、`simple`（简洁）、`night`（暗色）、`moon`（月光）

### 5.4 多格式同时输出

在 `_quarto.yml` 或文档 YAML Header 中配置多种格式：

```yaml
format:
  html:
    toc: true
    embed-resources: true
  pdf:
    toc: true
  revealjs:
    theme: serif
```

然后运行 `quarto render` 会同时输出所有配置的格式。

---

## 6. 代码执行控制

### 三种开发模式

| 模式 | 配置 | 速度 | 用途 |
|------|------|------|------|
| **开发模式** | `enabled: false` | 秒开 | 编辑文字、调整格式 |
| **测试模式** | `freeze: auto` | 较快 | 只重新执行修改的代码 |
| **发布模式** | 默认设置 | 完整 | 最终渲染 |

### 项目级（`_quarto.yml`）

```yaml
execute:
  echo: true       # 显示代码
  warning: true    # 显示警告
  freeze: auto     # 缓存：仅源码变更时重新执行
```

### 文档级（YAML Header）

```yaml
execute:
  enabled: false   # 跳过代码执行（开发模式，快速预览）
```

### 代码块级（Cell Options）

````markdown
```{python}
#| label: fig-soc           # 标签，用于交叉引用
#| fig-cap: "SOC 轨迹图"     # 图表标题
#| eval: false              # 不执行此代码块
#| echo: false              # 不显示代码，只显示结果
#| code-fold: true          # 代码可折叠
```
````

**开发技巧**：编辑文字时设置 `enabled: false` 跳过执行，完成后移除该行再渲染。

---

## 7. 文献引用与交叉引用

### 文献引用配置

```yaml
# _quarto.yml 或 .qmd YAML Header
bibliography: ../references.bib
csl: https://www.zotero.org/styles/ieee   # 引用格式
```

### 文献引用语法

```markdown
基于 @xuFactoringCycleAging2017 的退化模型...

多篇引用 [@collath2023; @xu2017]
```

### 参考文献列表

在文档末尾添加：

```markdown
## References

::: {#refs}
:::
```

### 交叉引用（Cross-References）

在代码块或元素上设置 `label`，然后用 `@` 引用：

| 类型 | 标签格式 | 引用语法 |
|------|----------|----------|
| 图表 | `#| label: fig-xxx` | `@fig-xxx` |
| 表格 | `#| label: tbl-xxx` | `@tbl-xxx` |
| 公式 | `{#eq-xxx}` | `@eq-xxx` |

**示例**：

````markdown
```{python}
#| label: fig-soc
#| fig-cap: "电池 SOC 轨迹"

import plotly.express as px
# 绑图代码...
```

如 @fig-soc 所示，SOC 呈周期性变化。
````

---

## 8. 场景化配置示例

### 场景 1：数据分析报告（给团队看）

```yaml
format:
  html:
    toc: true
    code-fold: true
    embed-resources: true  # 单文件，可发邮件
  docx: default            # Word 版本供编辑
```

### 场景 2：竞赛/会议演示文稿

```yaml
format:
  revealjs:
    theme: serif
    slide-number: true
    chalkboard: true       # 可以在幻灯片上画图
    transition: slide
```

### 场景 3：学术论文/技术报告

```yaml
format:
  pdf:
    documentclass: article
    number-sections: true
    toc: true
    colorlinks: true
```

### 场景 4：数据仪表板

```yaml
---
title: "BESS Performance Dashboard"
format: dashboard
---

## Row

```{python}
#| content: valuebox
#| title: "Total Profit"
dict(value="€125,430", icon="currency-euro")
```
```

---

## 9. 本项目配置速查

### 目录结构

```
doc/reports/
├── _quarto.yml          # 项目配置
├── _output/             # 输出目录 (gitignored)
├── *.qmd                # 源文件
└── QUARTO_GUIDE.md      # 本指南
```

### 当前 `_quarto.yml` 关键配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `output-dir` | `_output` | 输出文件目录 |
| `bibliography` | `../references.bib` | 文献库路径 |
| `csl` | IEEE | 引用格式 |
| `jupyter` | `python3` | Python 内核 |
| `execute.freeze` | `auto` | 代码缓存 |
| `format.html.theme` | `cosmo` | HTML 主题 |
| `format.html.embed-resources` | `true` | 自包含文件 |

---

## 快速参考

```bash
# 渲染命令
quarto render report.qmd              # 渲染单文件（所有配置格式）
quarto render report.qmd --to html    # 仅渲染 HTML
quarto render report.qmd --to pdf     # 仅渲染 PDF
quarto render report.qmd --to revealjs # 仅渲染幻灯片
quarto render                         # 渲染整个项目

# 预览命令
quarto preview report.qmd             # 实时预览（自动刷新）

# 创建项目模板
quarto create project default my_report
quarto create project revealjs my_slides
quarto create project dashboard my_dashboard
```
