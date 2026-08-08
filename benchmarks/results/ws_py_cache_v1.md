# ws_py_cache_v1 — 适配层自动缓存 Python 端证据

- generator: `ws_py_cache_v1.py` (schema 1.0.0)
- generated_at: 2026-08-08T21:23:50+08:00
- closure_status: `OK`
- git_head: `9f26012c44b3d22bb9c546cf4849a2c8a9801a58`
- git_dirty: False
- capture_sha256: `E7D7927B73BCBF6C0E14A5FF7130E366C13CAB42FE6152518AE9A2281A1F7EBA`
- env: python 3.12.7, NVIDIA GeForce RTX 4060 Laptop GPU (CC 8.9)

面板 `(T,N)=(1218,5000)`, seed 42。同一面板/输入/mask 仅缓存开关不同；11 次取中位。gate = speedup ≥ 1.2x。

| op | 无缓存 (ms) | 有缓存 (ms) | speedup | 位级一致 | 缓存复用 | 判定 |
|----|------------|------------|---------|---------|---------|------|
| rolling_ic | 62.6196 | 46.3362 | 1.3514x | PASS | PASS | BEATS |
| cs_rank | 37.3529 | 31.064 | 1.2024x | PASS | PASS | BEATS |

## 判定
**BEATS gate（测得的 op：rolling_ic、cs_rank）**：这些 op 在 corpus 面板 (T,N)=(1218,5000) 上收益 ≥ 1.2x，缓存路径结果位级一致且缓存实际复用。parameter_scan 未纳入性能门——其 C++ workspace 收益 1.18x 未达 1.2x（缓存对它是正确性收益而非性能门槛），详见备注。

## 备注
缓存收益来自消除 per-call 设备分配（C++ workspace 先例：cs_rank 16.4→9.17ms、rolling_ic 48.98→33.0ms；绝对收益 与 Python 端实测一致）。注意：Python 适配层存在加性开销（本产物 Python 端无缓存中位明显高于 C++ 绑定端先例，cs_rank 约 +14ms），并非≈0——缓存只消除设备分配，不影响适配层开销。仅测 rolling_ic 与 cs_rank 两 op（同面板 1218x5000/seed 42）；parameter_scan 未纳入性能门（其 C++ workspace 收益 1.18x 未达 1.2x，缓存对它是正确性收益）。
