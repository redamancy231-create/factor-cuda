# 支持（SUPPORT）

## 获取帮助

- **文档优先**：先读 `README.md`（三语：简体/English/繁體）、`docs/support_matrix.json`（构建环境）、`CLAUDE.md`（操作语义契约）、`CONTRIBUTING.md`（贡献）
- **Bug 报告**：用 `.github/ISSUE_TEMPLATE/bug_report.md` 模板开 Issue
- **功能请求**：用 `.github/ISSUE_TEMPLATE/feature_request.md`
- **讨论**：GitHub Discussions / Issue 评论区

## 不提供

- **投资建议**：本项目不构成任何投资建议，请勿据此做交易决策
- **商业支持**：本项目为个人开源维护，无 SLA/商业支持承诺
- **数据获取**：本项目不做数据获取（见 ashare-mcp）；如需行情数据请用数据源工具

## 常见问题

- **构建失败**：确认环境符合 `docs/support_matrix.json`（CUDA 13.3 / VS2026 MSVC 19.51 / Python 3.12.7 / CC 8.9）；Windows GBK 中文注释问题用 ASCII-only 注释
- **测试失败**：`PYTHONIOENCODING=utf-8 python -m pytest tests/ -q`；GPU 显存不足时 F=128 场景可能 vram-exhausted（物理余量薄）
- **Python 版本**：建议 3.12（oracle 冻结 NumPy 2.4.4 指纹）
