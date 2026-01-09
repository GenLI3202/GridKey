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

## 2. 工作流概述

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  .qmd 源文件 │  →   │ Quarto 引擎  │  →   │  输出文件    │
│  (Markdown)  │      │  (Pandoc)   │      │ HTML/PDF/...│
└─────────────┘      └─────────────┘      └─────────────┘
       ↑                    ↑
   YAML Header        _quarto.yml
   (文档配置)           (项目配置)
```

**核心命令：**

```bash
quarto render doc.qmd          # 渲染单个文件
quarto render                  # 渲染整个项目
quarto preview doc.qmd         # 实时预览（自动刷新）
```

---

## 3. 核心概念：配置层级

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

## 4. 输出格式详解

### 4.1 HTML 报告

```yaml
format:
  html:
    toc: true              # 显示目录 (Table of Contents)
    toc-depth: 3           # 目录深度
    code-fold: true        # 代码块可折叠
    code-tools: true       # 显示代码工具栏
    embed-resources: true  # 自包含单文件（便于分享）
    theme: cosmo           # 主题：cosmo/flatly/darkly/...
```

**常用主题**：`cosmo`（默认）、`flatly`（扁平）、`darkly`（暗色）、`journal`（学术）

### 4.2 PDF 文档

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
    keep-tex: false           # 是否保留 .tex 中间文件
```

**注意**：PDF 输出需要安装 LaTeX 发行版（如 TinyTeX、MiKTeX）

### 4.3 Revealjs 演示文稿

```yaml
format:
  revealjs:
    theme: serif             # 主题：serif/simple/night/moon/...
    slide-number: true       # 显示页码
    transition: slide        # 切换动画：slide/fade/convex/none
    chalkboard: true         # 启用画板（可在幻灯片上标注）
    scrollable: true         # 长内容页面可滚动
    center: false            # 内容垂直居中
    width: 1050              # 幻灯片宽度
    height: 700              # 幻灯片高度
```

**幻灯片分页**：使用 `##` 二级标题自动分页

**常用主题**：`serif`（衬线）、`simple`（简洁）、`night`（暗色）、`moon`（月光）

---

## 5. 代码执行控制

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
#| eval: false    # 不执行此代码块
#| echo: false    # 不显示代码，只显示结果
#| code-fold: true
```
````

**开发技巧**：编辑文字时设置 `enabled: false` 跳过执行，完成后移除该行再渲染。

---

## 6. 文献引用

### 配置

```yaml
# _quarto.yml 或 .qmd YAML Header
bibliography: ../references.bib
csl: https://www.zotero.org/styles/ieee   # 引用格式
```

### 使用

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

---

## 7. 本项目配置速查

当前 `doc/reports/_quarto.yml` 关键配置：

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
# 渲染为 HTML
quarto render report.qmd --to html

# 渲染为 PDF
quarto render report.qmd --to pdf

# 渲染为 Revealjs 幻灯片
quarto render slides.qmd --to revealjs

# 实时预览
quarto preview report.qmd
```
