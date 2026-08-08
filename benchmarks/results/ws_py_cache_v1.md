# ws_py_cache_v1 — 适配层自动缓存 Python 端证据

- generator: `ws_py_cache_v1.py` (schema 1.0.0)
- generated_at: 2026-08-08T20:09:46+08:00
- closure_status: `OK`
- git_head: `3784d2a216ef1579036ef311df4843c8e9b9b9d6`
- git_dirty: True
- capture_sha256: `328C98611858997B8E896D0900043D8AEA46E4C0A3BAC65C408A6A7312130D64`
- env: python 3.12.7, NVIDIA GeForce RTX 4060 Laptop GPU (CC 8.9)

面板 `(T,N)=(1218,5000)`, seed 42。同一面板/输入/mask 仅缓存开关不同；11 次取中位。gate = speedup ≥ 1.2x。

| op | 无缓存 (ms) | 有缓存 (ms) | speedup | 位级一致 | 缓存复用 | 判定 |
|----|------------|------------|---------|---------|---------|------|
| rolling_ic | 53.425 | 38.2007 | 1.3985x | PASS | PASS | BEATS |
| cs_rank | 30.3152 | 23.6233 | 1.2833x | PASS | PASS | BEATS |

## 判定
**BEATS gate**：全部 op 收益 ≥ 1.2x，缓存路径结果位级一致且缓存实际复用 → 自动缓存收益成立。

## 备注
Python 适配层 + 绑定层开销≈0（上会话实测）；缓存收益来自消除 per-call 设备分配（C++ workspace 先例：cs_rank 16.4→9.17ms、rolling_ic 48.98→33.0ms）。
