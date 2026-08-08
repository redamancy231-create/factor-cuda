# 安全政策（SECURITY）

## 报告漏洞

factor-cuda 是一个开源量化工具。如发现安全漏洞（如任意代码执行、敏感信息泄漏、数据完整性破坏），请**私有报告**（勿公开 Issue）：

- 通过 GitHub 私有 Issue / Security Advisory：https://github.com/redamancy231-create/factor-cuda/security
- 或在 Issue 中标记 `[SECURITY]`（如无法私有提交）

报告时请提供：
1. 漏洞描述（影响/危害）
2. 复现步骤（最小示例）
3. 受影响版本
4. 修复建议（可选）

## 响应时间

- 确认收到：24 小时内
- 初步评估：7 天内
- 修复计划：根据严重度（Critical/High 优先）

## 已知边界

- **非投资建议**：本项目是研究/教育工具，不构成投资建议
- **无认证/授权逻辑**：项目不处理用户凭证/密钥；如发现意外处理，属安全缺陷请报告
- **GPU 数据边界**：本项目只处理本地面板数据，不自动联网获取数据

## 依赖安全

- 依赖：Python 3.12 / CUDA Toolkit / VS MSVC / CMake / pybind11 / NumPy / CuPy（可选）
- 请保持依赖更新；发现依赖漏洞请同样私有报告
