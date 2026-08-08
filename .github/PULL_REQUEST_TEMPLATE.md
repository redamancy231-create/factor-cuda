## 变更内容

<!-- 简述本次变更 -->

## 关联 Issue

<!-- Closes #xxx 或 Related #xxx -->

## 验证

- [ ] selfcheck ALL PASS
- [ ] pytest 无回归（`PYTHONIOENCODING=utf-8 python -m pytest tests/ -q`）
- [ ] compute-sanitizer 0 errors（如涉及内核）
- [ ] 证据类改动：fail-closed 合成负例验证 + 双后端审查（如适用）

## 检查清单

- [ ] 代码注释 ASCII-only（`.cu/.cuh`）
- [ ] CHANGELOG 登记
- [ ] 文档同步（README/support_matrix 如需更新）
- [ ] PII 无泄漏（绝对路径/个人标识）

## 备注

<!-- 需要审查者注意的点 -->
