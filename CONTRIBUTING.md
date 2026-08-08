# 贡献指南（CONTRIBUTING）

感谢你对 factor-cuda 的兴趣！以下是贡献的完整流程。

## 项目定位

factor-cuda 是 **CUDA 因子截面分析加速**工具——GPU 加速量化因子截面分析（截面排序/相关性/IC/参数扫描）。
**本项目不做**因子计算（ml-quant-trading 职责）、数据获取（ashare-mcp 职责）、回测。

## 环境

- 构建/运行环境支持矩阵见 `docs/support_matrix.json`（单一真源：CUDA 13.3 / VS2026 MSVC 19.51 / Python 3.12.7 / compute capability 8.9）
- 构建：`pwsh -NoProfile -File _build_poc3.ps1 <target>`（CMake + Ninja）
- 测试（Git Bash）：`PYTHONIOENCODING=utf-8 python -m pytest tests/ -q`
- 测试（PowerShell）：`$env:PYTHONIOENCODING='utf-8'; python -m pytest tests/ -q`
  （本地 GPU 期望 126 passed / 3 skipped；GitHub Actions 无 GPU 跑 CPU/契约臂 75 passed）
- CUDA selfcheck：`build\poc3_*_selfcheck.exe`（每算子 ALL PASS）

## 代码约定（重要）

- **注释纯 ASCII**：`.cu/.cuh` 注释必须 ASCII-only（Windows GBK 解析会吞 UTF-8 中文尾字节，致编译失败）
- **Python 命令前置 `PYTHONIOENCODING=utf-8`**（Windows 中文输出）
- **构建用 pwsh**（`_build_poc3.ps1`），不用 cmd
- 内核修改后：`compute-sanitizer` 0 errors + selfcheck ALL PASS + pytest 无回归

## 提交流程

1. Fork + 分支（`fix/xxx` 或 `feat/xxx`）
2. 修改 + 本地验证（selfcheck + pytest）
3. **证据类改动**（性能/显存/数值）需：
   - 合成负例验证 fail-closed（伪造/异常数据必须被拒）
   - 内部审查（Workflow 对抗）+ 外部异后端审查（GPT-5.6-Sol via Codex）
   - 证据链 fail-closed（closure_status + provenance + 派生字段重算）
4. Commit message 遵循项目风格（`feat:`/`fix:`/`docs:` 前缀）
5. PR 说明改动 + 验证结果

## 审查要求

- **核心实现/证据类改动**：双后端审查闭合（内部 Workflow + 外部 GPT-5.6-Sol）——历史多次实证「我以为正确实际有误」的适配 bug 靠独立审查暴露
- **文档改动**：三语 README 校对（GPT-5.6-Sol）——术语一致性
- **发布改动**：PII 扫描（异后端独立确认零残留）+ 历史卫生

## Issue / PR

- Bug 报告：用 `.github/ISSUE_TEMPLATE/bug_report.md` 模板（含复现/环境/期望）
- 功能请求：用 `.github/ISSUE_TEMPLATE/feature_request.md`
- PR：用 `.github/PULL_REQUEST_TEMPLATE.md`

## 行为准则

贡献者须遵守 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。
