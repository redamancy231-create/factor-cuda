# PoC ② 公平基线性能报告（run `poc2_baseline_20260804c`）

- 生成命令: `python perf_bench_v1.py --render poc2_baseline_20260804c`
- corpus: `corpus_synth_v1` sha256=0A7D7AB5A6AD1F7E… T×N×F=1218×5000×12
- git_commit: `7b30bf4e88338ae36fa0afda1a171a00c3dd3229`
- 环境: Windows-11-10.0.26100-SP0 | NVIDIA GeForce RTX 4060 Laptop GPU | driver 610.88 | python 3.12.7 numpy 2.4.4 cupy 14.1.1 torch 2.13.0+cu132
- 三口径: warm e2e（判据）/ cold first-use / upload 全输入 H2D / resident（纯设备，corr 除外）

## 端到端 wall（ms）+ speedup（相对同语义最佳免费替代）

| 操作 | numpy | cupy | qgplearn | 同语义最佳 |
|---|---:|---:|---:|---:|
| cs_rank | 518.9 (0.05×) | 69.1 (0.40×) | 27.9 (1.00×) | 27.9 |
| cs_rank_desc | 538.6 (0.05×) | 70.0 (0.39×) | 27.5 (1.00×) | 27.5 |
| factor_corr | 5184.1 (0.60×) | 3087.0 (1.00×) | N/A | 3087.0 |
| rolling_ic | 1184.2 (0.13×) | 154.4 (1.00×) | 54.7 (非同语义) | 154.4 |
| parameter_scan(G=4) | 1906.9 (0.06×) | 250.4 (0.44×) | 110.7 (1.00×) | 110.7 |
| stock_corr(N=500) | 149.5 (0.29×) | 43.6 (1.00×) | N/A | 43.6 |
| stock_corr(N=2000) | 1418.6 (0.38×) | 537.5 (1.00×) | N/A | 537.5 |
| stock_corr(N=5000) | 8763.7 (0.29×) | 2558.9 (1.00×) | N/A | 2558.9 |

## GPU 臂三口径（ms）

| 操作 | backend | cold | upload | resident(gpu) | pure_device |
|---|---|---:|---:|---:|---:|
| cs_rank | cupy | 533.6 | 4.3 | 58.2 | 是 |
| cs_rank | qgplearn | 735.9 | 3.0 | 16.5 | 是 |
| cs_rank_desc | cupy | 71.0 | 5.3 | 58.3 | 是 |
| cs_rank_desc | qgplearn | 35.2 | 2.8 | 16.6 | 是 |
| factor_corr | cupy | 4753.2 | 46.0 | N/A | 否 |
| factor_corr | qgplearn | N/A | N/A | N/A | - |
| rolling_ic | cupy | 375.9 | 13.6 | 127.0 | 是 |
| rolling_ic | qgplearn | 145.0 | 7.6 | 42.5 | 是 |
| parameter_scan(G=4) | cupy | 291.6 | 4.4 | N/A | 否 |
| parameter_scan(G=4) | qgplearn | 126.7 | 2.9 | N/A | 否 |
| stock_corr(N=500) | cupy | 69.0 | 1.1 | N/A | 否 |
| stock_corr(N=500) | qgplearn | N/A | N/A | N/A | - |
| stock_corr(N=2000) | cupy | 630.4 | 4.5 | N/A | 否 |
| stock_corr(N=2000) | qgplearn | N/A | N/A | N/A | - |
| stock_corr(N=5000) | cupy | 3311.7 | 4.4 | N/A | 否 |
| stock_corr(N=5000) | qgplearn | N/A | N/A | N/A | - |
