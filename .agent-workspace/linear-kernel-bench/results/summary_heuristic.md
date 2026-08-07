# Aggregated oneDNN vs SGL heuristic (bucketed)

Grouped by (build_tag, cpu_binding_tag, dtype, linear_type, tp_size, M-bucket); decode = M<=256*tp, mid = 256*tp<M<=4096*tp, prefill = M>4096*tp. `geomean_speedup` is the geometric mean of per-case `sgl_ms/onednn_ms` ratios (>1 = oneDNN faster). `SGL win-rate` is the fraction of individual cases where SGL had lower latency (can diverge from the geomean's sign when a few large-margin cases skew the aggregate but most individual cases lean the other way); `recommendation` follows the win-rate, not the geomean. Cases where SGL was gated out (check_cpu_sgl_kernel=False) are excluded from these aggregates. `median AI` is the median arithmetic intensity (FLOPs per byte of naive, no-reuse memory traffic — see `bench_linear_kernels.py::arithmetic_intensity`) across cases in that bucket; bf16/fp16 share the same ratio since both are 2-byte types.

build_tag | cpu_binding_tag | dtype | linear | tp | M-bucket | n | SGL win-rate | geomean speedup (oneDNN/SGL) | recommendation | median AI (FLOPs/B)
|---|---|---|---|---|---|---|---|---|---|---|
O2-default | cpu192-222 | bf16 | down | 1 | decode | 70 | 0.63 | 1.081 | sgl | 31.67
O2-default | cpu192-222 | bf16 | down | 1 | mid | 40 | 0.12 | 1.608 | onednn | 768.00
O2-default | cpu192-222 | bf16 | down | 2 | decode | 80 | 0.57 | 0.975 | sgl | 56.89
O2-default | cpu192-222 | bf16 | down | 2 | mid | 40 | 0.00 | 1.328 | onednn | 772.83
O2-default | cpu192-222 | bf16 | down | 4 | decode | 63 | 0.71 | 0.949 | sgl | 61.75
O2-default | cpu192-222 | bf16 | down | 4 | mid | 28 | 0.18 | 1.123 | onednn | 988.69
O2-default | cpu192-222 | bf16 | down | 8 | decode | 70 | 0.76 | 0.925 | sgl | 83.64
O2-default | cpu192-222 | bf16 | down | 8 | mid | 28 | 0.25 | 1.071 | onednn | 955.73
O2-default | cpu192-222 | bf16 | gate_up | 1 | decode | 70 | 0.63 | 1.021 | sgl | 31.71
O2-default | cpu192-222 | bf16 | gate_up | 1 | mid | 40 | 0.18 | 1.158 | onednn | 796.44
O2-default | cpu192-222 | bf16 | gate_up | 2 | decode | 80 | 0.65 | 0.970 | sgl | 56.89
O2-default | cpu192-222 | bf16 | gate_up | 2 | mid | 40 | 0.00 | 1.180 | onednn | 1051.89
O2-default | cpu192-222 | bf16 | gate_up | 4 | decode | 90 | 0.64 | 0.929 | sgl | 62.38
O2-default | cpu192-222 | bf16 | gate_up | 4 | mid | 40 | 0.00 | 1.169 | onednn | 1146.88
O2-default | cpu192-222 | bf16 | gate_up | 8 | decode | 100 | 0.66 | 0.927 | sgl | 85.33
O2-default | cpu192-222 | bf16 | gate_up | 8 | mid | 40 | 0.03 | 1.172 | onednn | 1303.27
O2-default | cpu192-222 | bf16 | lm_head | 1 | decode | 70 | 0.51 | 1.117 | sgl | 31.72
O2-default | cpu192-222 | bf16 | lm_head | 2 | decode | 80 | 0.50 | 1.111 | onednn | 60.81
O2-default | cpu192-222 | bf16 | lm_head | 4 | decode | 90 | 0.40 | 1.095 | onednn | 62.53
O2-default | cpu192-222 | bf16 | lm_head | 8 | decode | 80 | 0.41 | 1.079 | onednn | 114.31
O2-default | cpu192-222 | bf16 | moe_gate | 1 | decode | 28 | 1.00 | 0.648 | sgl | 25.28
O2-default | cpu192-222 | bf16 | moe_gate | 1 | mid | 16 | 0.81 | 0.879 | sgl | 109.36
O2-default | cpu192-222 | bf16 | moe_gate | 2 | decode | 32 | 0.97 | 0.680 | sgl | 25.37
O2-default | cpu192-222 | bf16 | moe_gate | 2 | mid | 16 | 0.94 | 0.907 | sgl | 115.53
O2-default | cpu192-222 | bf16 | moe_gate | 4 | decode | 36 | 0.94 | 0.701 | sgl | 28.17
O2-default | cpu192-222 | bf16 | moe_gate | 4 | mid | 16 | 0.81 | 0.985 | sgl | 118.72
O2-default | cpu192-222 | bf16 | moe_gate | 8 | decode | 40 | 0.95 | 0.708 | sgl | 30.70
O2-default | cpu192-222 | bf16 | moe_gate | 8 | mid | 16 | 0.56 | 1.059 | sgl | 119.59
O2-default | cpu192-222 | bf16 | o_proj | 1 | decode | 84 | 0.76 | 0.945 | sgl | 31.51
O2-default | cpu192-222 | bf16 | o_proj | 1 | mid | 48 | 0.00 | 1.246 | onednn | 682.67
O2-default | cpu192-222 | bf16 | o_proj | 2 | decode | 96 | 0.80 | 0.923 | sgl | 51.20
O2-default | cpu192-222 | bf16 | o_proj | 2 | mid | 48 | 0.33 | 1.075 | onednn | 819.20
O2-default | cpu192-222 | bf16 | o_proj | 4 | decode | 108 | 0.81 | 0.902 | sgl | 59.36
O2-default | cpu192-222 | bf16 | o_proj | 4 | mid | 48 | 0.08 | 1.197 | onednn | 682.67
O2-default | cpu192-222 | bf16 | o_proj | 8 | decode | 120 | 0.87 | 0.866 | sgl | 56.11
O2-default | cpu192-222 | bf16 | o_proj | 8 | mid | 48 | 0.40 | 1.084 | onednn | 431.16
O2-default | cpu192-222 | bf16 | qkv | 1 | decode | 84 | 0.77 | 0.944 | sgl | 31.59
O2-default | cpu192-222 | bf16 | qkv | 1 | mid | 48 | 0.12 | 1.158 | onednn | 722.82
O2-default | cpu192-222 | bf16 | qkv | 2 | decode | 96 | 0.79 | 0.925 | sgl | 56.20
O2-default | cpu192-222 | bf16 | qkv | 2 | mid | 48 | 0.12 | 1.171 | onednn | 945.23
O2-default | cpu192-222 | bf16 | qkv | 4 | decode | 108 | 0.74 | 0.856 | sgl | 60.53
O2-default | cpu192-222 | bf16 | qkv | 4 | mid | 48 | 0.12 | 1.121 | onednn | 877.71
O2-default | cpu192-222 | bf16 | qkv | 8 | decode | 120 | 0.77 | 0.857 | sgl | 69.82
O2-default | cpu192-222 | bf16 | qkv | 8 | mid | 48 | 0.21 | 1.120 | onednn | 622.18
O2-default | cpu192-222 | fp16 | down | 1 | decode | 70 | 0.67 | 1.056 | sgl | 31.67
O2-default | cpu192-222 | fp16 | down | 1 | mid | 40 | 0.18 | 1.567 | onednn | 768.00
O2-default | cpu192-222 | fp16 | down | 2 | decode | 80 | 0.60 | 0.960 | sgl | 56.89
O2-default | cpu192-222 | fp16 | down | 2 | mid | 40 | 0.00 | 1.322 | onednn | 772.83
O2-default | cpu192-222 | fp16 | down | 4 | decode | 63 | 0.83 | 0.907 | sgl | 61.75
O2-default | cpu192-222 | fp16 | down | 4 | mid | 28 | 0.18 | 1.123 | onednn | 988.69
O2-default | cpu192-222 | fp16 | down | 8 | decode | 70 | 0.71 | 0.919 | sgl | 83.64
O2-default | cpu192-222 | fp16 | down | 8 | mid | 28 | 0.25 | 1.066 | onednn | 955.73
O2-default | cpu192-222 | fp16 | gate_up | 1 | decode | 70 | 0.69 | 1.003 | sgl | 31.71
O2-default | cpu192-222 | fp16 | gate_up | 1 | mid | 40 | 0.12 | 1.167 | onednn | 796.44
O2-default | cpu192-222 | fp16 | gate_up | 2 | decode | 80 | 0.65 | 0.971 | sgl | 56.89
O2-default | cpu192-222 | fp16 | gate_up | 2 | mid | 40 | 0.03 | 1.165 | onednn | 1051.89
O2-default | cpu192-222 | fp16 | gate_up | 4 | decode | 90 | 0.66 | 0.917 | sgl | 62.38
O2-default | cpu192-222 | fp16 | gate_up | 4 | mid | 40 | 0.00 | 1.158 | onednn | 1146.88
O2-default | cpu192-222 | fp16 | gate_up | 8 | decode | 100 | 0.72 | 0.903 | sgl | 85.33
O2-default | cpu192-222 | fp16 | gate_up | 8 | mid | 40 | 0.10 | 1.151 | onednn | 1303.27
O2-default | cpu192-222 | fp16 | lm_head | 1 | decode | 70 | 0.56 | 1.116 | sgl | 31.72
O2-default | cpu192-222 | fp16 | lm_head | 2 | decode | 80 | 0.53 | 1.109 | sgl | 60.81
O2-default | cpu192-222 | fp16 | lm_head | 4 | decode | 90 | 0.49 | 1.087 | onednn | 62.53
O2-default | cpu192-222 | fp16 | lm_head | 8 | decode | 80 | 0.45 | 1.066 | onednn | 114.31
O2-default | cpu192-222 | fp16 | moe_gate | 1 | decode | 28 | 0.96 | 0.668 | sgl | 25.28
O2-default | cpu192-222 | fp16 | moe_gate | 1 | mid | 16 | 0.88 | 0.858 | sgl | 109.36
O2-default | cpu192-222 | fp16 | moe_gate | 2 | decode | 32 | 0.94 | 0.679 | sgl | 25.37
O2-default | cpu192-222 | fp16 | moe_gate | 2 | mid | 16 | 0.81 | 0.912 | sgl | 115.53
O2-default | cpu192-222 | fp16 | moe_gate | 4 | decode | 36 | 0.94 | 0.699 | sgl | 28.17
O2-default | cpu192-222 | fp16 | moe_gate | 4 | mid | 16 | 0.69 | 0.999 | sgl | 118.72
O2-default | cpu192-222 | fp16 | moe_gate | 8 | decode | 40 | 0.95 | 0.727 | sgl | 30.70
O2-default | cpu192-222 | fp16 | moe_gate | 8 | mid | 16 | 0.50 | 1.073 | onednn | 119.59
O2-default | cpu192-222 | fp16 | o_proj | 1 | decode | 84 | 0.81 | 0.918 | sgl | 31.51
O2-default | cpu192-222 | fp16 | o_proj | 1 | mid | 48 | 0.02 | 1.245 | onednn | 682.67
O2-default | cpu192-222 | fp16 | o_proj | 2 | decode | 96 | 0.83 | 0.898 | sgl | 51.20
O2-default | cpu192-222 | fp16 | o_proj | 2 | mid | 48 | 0.40 | 1.076 | onednn | 819.20
O2-default | cpu192-222 | fp16 | o_proj | 4 | decode | 108 | 0.78 | 0.906 | sgl | 59.36
O2-default | cpu192-222 | fp16 | o_proj | 4 | mid | 48 | 0.06 | 1.199 | onednn | 682.67
O2-default | cpu192-222 | fp16 | o_proj | 8 | decode | 120 | 0.86 | 0.872 | sgl | 56.11
O2-default | cpu192-222 | fp16 | o_proj | 8 | mid | 48 | 0.46 | 1.096 | onednn | 431.16
O2-default | cpu192-222 | fp16 | qkv | 1 | decode | 84 | 0.76 | 0.929 | sgl | 31.59
O2-default | cpu192-222 | fp16 | qkv | 1 | mid | 48 | 0.15 | 1.159 | onednn | 722.82
O2-default | cpu192-222 | fp16 | qkv | 2 | decode | 96 | 0.84 | 0.907 | sgl | 56.20
O2-default | cpu192-222 | fp16 | qkv | 2 | mid | 48 | 0.10 | 1.170 | onednn | 945.23
O2-default | cpu192-222 | fp16 | qkv | 4 | decode | 108 | 0.83 | 0.848 | sgl | 60.53
O2-default | cpu192-222 | fp16 | qkv | 4 | mid | 48 | 0.06 | 1.139 | onednn | 877.71
O2-default | cpu192-222 | fp16 | qkv | 8 | decode | 120 | 0.74 | 0.851 | sgl | 69.82
O2-default | cpu192-222 | fp16 | qkv | 8 | mid | 48 | 0.19 | 1.120 | onednn | 622.18
O2-default | cpu192-223 | bf16 | down | 1 | decode | 70 | 0.69 | 1.057 | sgl | 31.67
O2-default | cpu192-223 | bf16 | down | 1 | mid | 40 | 0.15 | 1.640 | onednn | 768.00
O2-default | cpu192-223 | bf16 | down | 2 | decode | 80 | 0.64 | 0.965 | sgl | 56.89
O2-default | cpu192-223 | bf16 | down | 2 | mid | 40 | 0.03 | 1.389 | onednn | 772.83
O2-default | cpu192-223 | bf16 | down | 4 | decode | 63 | 0.62 | 0.965 | sgl | 61.75
O2-default | cpu192-223 | bf16 | down | 4 | mid | 28 | 0.21 | 1.143 | onednn | 988.69
O2-default | cpu192-223 | bf16 | down | 8 | decode | 70 | 0.63 | 0.953 | sgl | 83.64
O2-default | cpu192-223 | bf16 | down | 8 | mid | 28 | 0.04 | 1.178 | onednn | 955.73
O2-default | cpu192-223 | bf16 | gate_up | 1 | decode | 70 | 0.64 | 1.018 | sgl | 31.71
O2-default | cpu192-223 | bf16 | gate_up | 1 | mid | 40 | 0.07 | 1.162 | onednn | 796.44
O2-default | cpu192-223 | bf16 | gate_up | 2 | decode | 80 | 0.64 | 0.968 | sgl | 56.89
O2-default | cpu192-223 | bf16 | gate_up | 2 | mid | 40 | 0.00 | 1.150 | onednn | 1051.89
O2-default | cpu192-223 | bf16 | gate_up | 4 | decode | 90 | 0.64 | 0.923 | sgl | 62.38
O2-default | cpu192-223 | bf16 | gate_up | 4 | mid | 40 | 0.03 | 1.154 | onednn | 1146.88
O2-default | cpu192-223 | bf16 | gate_up | 8 | decode | 100 | 0.64 | 0.929 | sgl | 85.33
O2-default | cpu192-223 | bf16 | gate_up | 8 | mid | 40 | 0.10 | 1.154 | onednn | 1303.27
O2-default | cpu192-223 | bf16 | lm_head | 1 | decode | 70 | 0.51 | 1.112 | sgl | 31.72
O2-default | cpu192-223 | bf16 | lm_head | 2 | decode | 80 | 0.50 | 1.108 | onednn | 60.81
O2-default | cpu192-223 | bf16 | lm_head | 4 | decode | 90 | 0.44 | 1.081 | onednn | 62.53
O2-default | cpu192-223 | bf16 | lm_head | 8 | decode | 80 | 0.46 | 1.071 | onednn | 114.31
O2-default | cpu192-223 | bf16 | moe_gate | 1 | decode | 28 | 1.00 | 0.655 | sgl | 25.28
O2-default | cpu192-223 | bf16 | moe_gate | 1 | mid | 16 | 1.00 | 0.829 | sgl | 109.36
O2-default | cpu192-223 | bf16 | moe_gate | 2 | decode | 32 | 0.97 | 0.668 | sgl | 25.37
O2-default | cpu192-223 | bf16 | moe_gate | 2 | mid | 16 | 0.94 | 0.881 | sgl | 115.53
O2-default | cpu192-223 | bf16 | moe_gate | 4 | decode | 36 | 0.97 | 0.694 | sgl | 28.17
O2-default | cpu192-223 | bf16 | moe_gate | 4 | mid | 16 | 0.75 | 0.979 | sgl | 118.72
O2-default | cpu192-223 | bf16 | moe_gate | 8 | decode | 40 | 0.97 | 0.706 | sgl | 30.70
O2-default | cpu192-223 | bf16 | moe_gate | 8 | mid | 16 | 0.62 | 1.032 | sgl | 119.59
O2-default | cpu192-223 | bf16 | o_proj | 1 | decode | 84 | 0.81 | 0.944 | sgl | 31.51
O2-default | cpu192-223 | bf16 | o_proj | 1 | mid | 48 | 0.02 | 1.250 | onednn | 682.67
O2-default | cpu192-223 | bf16 | o_proj | 2 | decode | 96 | 0.77 | 0.929 | sgl | 51.20
O2-default | cpu192-223 | bf16 | o_proj | 2 | mid | 48 | 0.15 | 1.108 | onednn | 819.20
O2-default | cpu192-223 | bf16 | o_proj | 4 | decode | 108 | 0.73 | 0.919 | sgl | 59.36
O2-default | cpu192-223 | bf16 | o_proj | 4 | mid | 48 | 0.06 | 1.210 | onednn | 682.67
O2-default | cpu192-223 | bf16 | o_proj | 8 | decode | 120 | 0.84 | 0.879 | sgl | 56.11
O2-default | cpu192-223 | bf16 | o_proj | 8 | mid | 48 | 0.29 | 1.114 | onednn | 431.16
O2-default | cpu192-223 | bf16 | qkv | 1 | decode | 84 | 0.62 | 0.974 | sgl | 31.59
O2-default | cpu192-223 | bf16 | qkv | 1 | mid | 48 | 0.04 | 1.190 | onednn | 722.82
O2-default | cpu192-223 | bf16 | qkv | 2 | decode | 96 | 0.73 | 0.953 | sgl | 56.20
O2-default | cpu192-223 | bf16 | qkv | 2 | mid | 48 | 0.04 | 1.169 | onednn | 945.23
O2-default | cpu192-223 | bf16 | qkv | 4 | decode | 108 | 0.79 | 0.846 | sgl | 60.53
O2-default | cpu192-223 | bf16 | qkv | 4 | mid | 48 | 0.04 | 1.149 | onednn | 877.71
O2-default | cpu192-223 | bf16 | qkv | 8 | decode | 120 | 0.72 | 0.862 | sgl | 69.82
O2-default | cpu192-223 | bf16 | qkv | 8 | mid | 48 | 0.06 | 1.135 | onednn | 622.18
O2-default | cpu192-223 | fp16 | down | 1 | decode | 70 | 0.71 | 1.038 | sgl | 31.67
O2-default | cpu192-223 | fp16 | down | 1 | mid | 40 | 0.10 | 1.638 | onednn | 768.00
O2-default | cpu192-223 | fp16 | down | 2 | decode | 80 | 0.68 | 0.951 | sgl | 56.89
O2-default | cpu192-223 | fp16 | down | 2 | mid | 40 | 0.00 | 1.397 | onednn | 772.83
O2-default | cpu192-223 | fp16 | down | 4 | decode | 63 | 0.73 | 0.938 | sgl | 61.75
O2-default | cpu192-223 | fp16 | down | 4 | mid | 28 | 0.14 | 1.150 | onednn | 988.69
O2-default | cpu192-223 | fp16 | down | 8 | decode | 70 | 0.71 | 0.934 | sgl | 83.64
O2-default | cpu192-223 | fp16 | down | 8 | mid | 28 | 0.04 | 1.192 | onednn | 955.73
O2-default | cpu192-223 | fp16 | gate_up | 1 | decode | 70 | 0.67 | 1.006 | sgl | 31.71
O2-default | cpu192-223 | fp16 | gate_up | 1 | mid | 40 | 0.12 | 1.151 | onednn | 796.44
O2-default | cpu192-223 | fp16 | gate_up | 2 | decode | 80 | 0.65 | 0.968 | sgl | 56.89
O2-default | cpu192-223 | fp16 | gate_up | 2 | mid | 40 | 0.10 | 1.152 | onednn | 1051.89
O2-default | cpu192-223 | fp16 | gate_up | 4 | decode | 90 | 0.68 | 0.909 | sgl | 62.38
O2-default | cpu192-223 | fp16 | gate_up | 4 | mid | 40 | 0.07 | 1.158 | onednn | 1146.88
O2-default | cpu192-223 | fp16 | gate_up | 8 | decode | 100 | 0.72 | 0.907 | sgl | 85.33
O2-default | cpu192-223 | fp16 | gate_up | 8 | mid | 40 | 0.18 | 1.126 | onednn | 1303.27
O2-default | cpu192-223 | fp16 | lm_head | 1 | decode | 70 | 0.56 | 1.110 | sgl | 31.72
O2-default | cpu192-223 | fp16 | lm_head | 2 | decode | 80 | 0.55 | 1.100 | sgl | 60.81
O2-default | cpu192-223 | fp16 | lm_head | 4 | decode | 90 | 0.51 | 1.084 | sgl | 62.53
O2-default | cpu192-223 | fp16 | lm_head | 8 | decode | 80 | 0.46 | 1.065 | onednn | 114.31
O2-default | cpu192-223 | fp16 | moe_gate | 1 | decode | 28 | 0.93 | 0.692 | sgl | 25.28
O2-default | cpu192-223 | fp16 | moe_gate | 1 | mid | 16 | 1.00 | 0.831 | sgl | 109.36
O2-default | cpu192-223 | fp16 | moe_gate | 2 | decode | 32 | 0.94 | 0.687 | sgl | 25.37
O2-default | cpu192-223 | fp16 | moe_gate | 2 | mid | 16 | 1.00 | 0.876 | sgl | 115.53
O2-default | cpu192-223 | fp16 | moe_gate | 4 | decode | 36 | 0.97 | 0.701 | sgl | 28.17
O2-default | cpu192-223 | fp16 | moe_gate | 4 | mid | 16 | 0.75 | 0.982 | sgl | 118.72
O2-default | cpu192-223 | fp16 | moe_gate | 8 | decode | 40 | 0.97 | 0.720 | sgl | 30.70
O2-default | cpu192-223 | fp16 | moe_gate | 8 | mid | 16 | 0.62 | 1.050 | sgl | 119.59
O2-default | cpu192-223 | fp16 | o_proj | 1 | decode | 84 | 0.87 | 0.913 | sgl | 31.51
O2-default | cpu192-223 | fp16 | o_proj | 1 | mid | 48 | 0.04 | 1.260 | onednn | 682.67
O2-default | cpu192-223 | fp16 | o_proj | 2 | decode | 96 | 0.83 | 0.905 | sgl | 51.20
O2-default | cpu192-223 | fp16 | o_proj | 2 | mid | 48 | 0.19 | 1.116 | onednn | 819.20
O2-default | cpu192-223 | fp16 | o_proj | 4 | decode | 108 | 0.75 | 0.918 | sgl | 59.36
O2-default | cpu192-223 | fp16 | o_proj | 4 | mid | 48 | 0.08 | 1.245 | onednn | 682.67
O2-default | cpu192-223 | fp16 | o_proj | 8 | decode | 120 | 0.83 | 0.882 | sgl | 56.11
O2-default | cpu192-223 | fp16 | o_proj | 8 | mid | 48 | 0.31 | 1.135 | onednn | 431.16
O2-default | cpu192-223 | fp16 | qkv | 1 | decode | 84 | 0.73 | 0.953 | sgl | 31.59
O2-default | cpu192-223 | fp16 | qkv | 1 | mid | 48 | 0.02 | 1.206 | onednn | 722.82
O2-default | cpu192-223 | fp16 | qkv | 2 | decode | 96 | 0.81 | 0.928 | sgl | 56.20
O2-default | cpu192-223 | fp16 | qkv | 2 | mid | 48 | 0.04 | 1.178 | onednn | 945.23
O2-default | cpu192-223 | fp16 | qkv | 4 | decode | 108 | 0.82 | 0.829 | sgl | 60.53
O2-default | cpu192-223 | fp16 | qkv | 4 | mid | 48 | 0.04 | 1.163 | onednn | 877.71
O2-default | cpu192-223 | fp16 | qkv | 8 | decode | 120 | 0.72 | 0.849 | sgl | 69.82
O2-default | cpu192-223 | fp16 | qkv | 8 | mid | 48 | 0.12 | 1.129 | onednn | 622.18
O3-release | cpu192-222 | bf16 | down | 1 | decode | 70 | 0.66 | 1.072 | sgl | 31.67
O3-release | cpu192-222 | bf16 | down | 1 | mid | 40 | 0.20 | 1.562 | onednn | 768.00
O3-release | cpu192-222 | bf16 | down | 2 | decode | 80 | 0.57 | 0.960 | sgl | 56.89
O3-release | cpu192-222 | bf16 | down | 2 | mid | 40 | 0.00 | 1.309 | onednn | 772.83
O3-release | cpu192-222 | bf16 | down | 4 | decode | 63 | 0.75 | 0.924 | sgl | 61.75
O3-release | cpu192-222 | bf16 | down | 4 | mid | 28 | 0.18 | 1.115 | onednn | 988.69
O3-release | cpu192-222 | bf16 | down | 8 | decode | 70 | 0.77 | 0.923 | sgl | 83.64
O3-release | cpu192-222 | bf16 | down | 8 | mid | 28 | 0.36 | 1.050 | onednn | 955.73
O3-release | cpu192-222 | bf16 | gate_up | 1 | decode | 70 | 0.66 | 1.015 | sgl | 31.71
O3-release | cpu192-222 | bf16 | gate_up | 1 | mid | 40 | 0.12 | 1.152 | onednn | 796.44
O3-release | cpu192-222 | bf16 | gate_up | 2 | decode | 80 | 0.64 | 0.966 | sgl | 56.89
O3-release | cpu192-222 | bf16 | gate_up | 2 | mid | 40 | 0.03 | 1.160 | onednn | 1051.89
O3-release | cpu192-222 | bf16 | gate_up | 4 | decode | 90 | 0.63 | 0.928 | sgl | 62.38
O3-release | cpu192-222 | bf16 | gate_up | 4 | mid | 40 | 0.00 | 1.157 | onednn | 1146.88
O3-release | cpu192-222 | bf16 | gate_up | 8 | decode | 100 | 0.70 | 0.914 | sgl | 85.33
O3-release | cpu192-222 | bf16 | gate_up | 8 | mid | 40 | 0.05 | 1.154 | onednn | 1303.27
O3-release | cpu192-222 | bf16 | lm_head | 1 | decode | 70 | 0.53 | 1.116 | sgl | 31.72
O3-release | cpu192-222 | bf16 | lm_head | 2 | decode | 80 | 0.47 | 1.118 | onednn | 60.81
O3-release | cpu192-222 | bf16 | lm_head | 4 | decode | 90 | 0.43 | 1.093 | onednn | 62.53
O3-release | cpu192-222 | bf16 | lm_head | 8 | decode | 80 | 0.49 | 1.069 | onednn | 114.31
O3-release | cpu192-222 | bf16 | moe_gate | 1 | decode | 28 | 1.00 | 0.664 | sgl | 25.28
O3-release | cpu192-222 | bf16 | moe_gate | 1 | mid | 16 | 0.88 | 0.884 | sgl | 109.36
O3-release | cpu192-222 | bf16 | moe_gate | 2 | decode | 32 | 0.97 | 0.677 | sgl | 25.37
O3-release | cpu192-222 | bf16 | moe_gate | 2 | mid | 16 | 0.88 | 0.920 | sgl | 115.53
O3-release | cpu192-222 | bf16 | moe_gate | 4 | decode | 36 | 0.94 | 0.696 | sgl | 28.17
O3-release | cpu192-222 | bf16 | moe_gate | 4 | mid | 16 | 0.62 | 1.019 | sgl | 118.72
O3-release | cpu192-222 | bf16 | moe_gate | 8 | decode | 40 | 0.90 | 0.716 | sgl | 30.70
O3-release | cpu192-222 | bf16 | moe_gate | 8 | mid | 16 | 0.50 | 1.066 | onednn | 119.59
O3-release | cpu192-222 | bf16 | o_proj | 1 | decode | 84 | 0.79 | 0.941 | sgl | 31.51
O3-release | cpu192-222 | bf16 | o_proj | 1 | mid | 48 | 0.02 | 1.224 | onednn | 682.67
O3-release | cpu192-222 | bf16 | o_proj | 2 | decode | 96 | 0.79 | 0.915 | sgl | 51.20
O3-release | cpu192-222 | bf16 | o_proj | 2 | mid | 48 | 0.44 | 1.058 | onednn | 819.20
O3-release | cpu192-222 | bf16 | o_proj | 4 | decode | 108 | 0.76 | 0.912 | sgl | 59.36
O3-release | cpu192-222 | bf16 | o_proj | 4 | mid | 48 | 0.08 | 1.187 | onednn | 682.67
O3-release | cpu192-222 | bf16 | o_proj | 8 | decode | 120 | 0.87 | 0.856 | sgl | 56.11
O3-release | cpu192-222 | bf16 | o_proj | 8 | mid | 48 | 0.42 | 1.068 | onednn | 431.16
O3-release | cpu192-222 | bf16 | qkv | 1 | decode | 84 | 0.76 | 0.941 | sgl | 31.59
O3-release | cpu192-222 | bf16 | qkv | 1 | mid | 48 | 0.15 | 1.155 | onednn | 722.82
O3-release | cpu192-222 | bf16 | qkv | 2 | decode | 96 | 0.80 | 0.918 | sgl | 56.20
O3-release | cpu192-222 | bf16 | qkv | 2 | mid | 48 | 0.12 | 1.157 | onednn | 945.23
O3-release | cpu192-222 | bf16 | qkv | 4 | decode | 108 | 0.73 | 0.850 | sgl | 60.53
O3-release | cpu192-222 | bf16 | qkv | 4 | mid | 48 | 0.12 | 1.116 | onednn | 877.71
O3-release | cpu192-222 | bf16 | qkv | 8 | decode | 120 | 0.76 | 0.855 | sgl | 69.82
O3-release | cpu192-222 | bf16 | qkv | 8 | mid | 48 | 0.21 | 1.111 | onednn | 622.18
O3-release | cpu192-222 | fp16 | down | 1 | decode | 70 | 0.70 | 1.045 | sgl | 31.67
O3-release | cpu192-222 | fp16 | down | 1 | mid | 40 | 0.15 | 1.561 | onednn | 768.00
O3-release | cpu192-222 | fp16 | down | 2 | decode | 80 | 0.59 | 0.964 | sgl | 56.89
O3-release | cpu192-222 | fp16 | down | 2 | mid | 40 | 0.00 | 1.345 | onednn | 772.83
O3-release | cpu192-222 | fp16 | down | 4 | decode | 63 | 0.83 | 0.899 | sgl | 61.75
O3-release | cpu192-222 | fp16 | down | 4 | mid | 28 | 0.11 | 1.133 | onednn | 988.69
O3-release | cpu192-222 | fp16 | down | 8 | decode | 70 | 0.73 | 0.917 | sgl | 83.64
O3-release | cpu192-222 | fp16 | down | 8 | mid | 28 | 0.25 | 1.080 | onednn | 955.73
O3-release | cpu192-222 | fp16 | gate_up | 1 | decode | 70 | 0.69 | 1.001 | sgl | 31.71
O3-release | cpu192-222 | fp16 | gate_up | 1 | mid | 40 | 0.10 | 1.174 | onednn | 796.44
O3-release | cpu192-222 | fp16 | gate_up | 2 | decode | 80 | 0.64 | 0.968 | sgl | 56.89
O3-release | cpu192-222 | fp16 | gate_up | 2 | mid | 40 | 0.00 | 1.176 | onednn | 1051.89
O3-release | cpu192-222 | fp16 | gate_up | 4 | decode | 90 | 0.64 | 0.923 | sgl | 62.38
O3-release | cpu192-222 | fp16 | gate_up | 4 | mid | 40 | 0.00 | 1.174 | onednn | 1146.88
O3-release | cpu192-222 | fp16 | gate_up | 8 | decode | 100 | 0.72 | 0.904 | sgl | 85.33
O3-release | cpu192-222 | fp16 | gate_up | 8 | mid | 40 | 0.10 | 1.160 | onednn | 1303.27
O3-release | cpu192-222 | fp16 | lm_head | 1 | decode | 70 | 0.57 | 1.115 | sgl | 31.72
O3-release | cpu192-222 | fp16 | lm_head | 2 | decode | 80 | 0.53 | 1.106 | sgl | 60.81
O3-release | cpu192-222 | fp16 | lm_head | 4 | decode | 90 | 0.48 | 1.083 | onednn | 62.53
O3-release | cpu192-222 | fp16 | lm_head | 8 | decode | 80 | 0.47 | 1.063 | onednn | 114.31
O3-release | cpu192-222 | fp16 | moe_gate | 1 | decode | 28 | 1.00 | 0.691 | sgl | 25.28
O3-release | cpu192-222 | fp16 | moe_gate | 1 | mid | 16 | 0.94 | 0.861 | sgl | 109.36
O3-release | cpu192-222 | fp16 | moe_gate | 2 | decode | 32 | 0.94 | 0.681 | sgl | 25.37
O3-release | cpu192-222 | fp16 | moe_gate | 2 | mid | 16 | 0.75 | 0.910 | sgl | 115.53
O3-release | cpu192-222 | fp16 | moe_gate | 4 | decode | 36 | 0.97 | 0.701 | sgl | 28.17
O3-release | cpu192-222 | fp16 | moe_gate | 4 | mid | 16 | 0.62 | 0.996 | sgl | 118.72
O3-release | cpu192-222 | fp16 | moe_gate | 8 | decode | 40 | 0.93 | 0.736 | sgl | 30.70
O3-release | cpu192-222 | fp16 | moe_gate | 8 | mid | 16 | 0.38 | 1.076 | onednn | 119.59
O3-release | cpu192-222 | fp16 | o_proj | 1 | decode | 84 | 0.83 | 0.914 | sgl | 31.51
O3-release | cpu192-222 | fp16 | o_proj | 1 | mid | 48 | 0.04 | 1.238 | onednn | 682.67
O3-release | cpu192-222 | fp16 | o_proj | 2 | decode | 96 | 0.83 | 0.902 | sgl | 51.20
O3-release | cpu192-222 | fp16 | o_proj | 2 | mid | 48 | 0.23 | 1.090 | onednn | 819.20
O3-release | cpu192-222 | fp16 | o_proj | 4 | decode | 108 | 0.79 | 0.917 | sgl | 59.36
O3-release | cpu192-222 | fp16 | o_proj | 4 | mid | 48 | 0.06 | 1.225 | onednn | 682.67
O3-release | cpu192-222 | fp16 | o_proj | 8 | decode | 120 | 0.83 | 0.882 | sgl | 56.11
O3-release | cpu192-222 | fp16 | o_proj | 8 | mid | 48 | 0.50 | 1.092 | onednn | 431.16
O3-release | cpu192-222 | fp16 | qkv | 1 | decode | 84 | 0.79 | 0.925 | sgl | 31.59
O3-release | cpu192-222 | fp16 | qkv | 1 | mid | 48 | 0.08 | 1.171 | onednn | 722.82
O3-release | cpu192-222 | fp16 | qkv | 2 | decode | 96 | 0.83 | 0.903 | sgl | 56.20
O3-release | cpu192-222 | fp16 | qkv | 2 | mid | 48 | 0.10 | 1.173 | onednn | 945.23
O3-release | cpu192-222 | fp16 | qkv | 4 | decode | 108 | 0.81 | 0.846 | sgl | 60.53
O3-release | cpu192-222 | fp16 | qkv | 4 | mid | 48 | 0.06 | 1.128 | onednn | 877.71
O3-release | cpu192-222 | fp16 | qkv | 8 | decode | 120 | 0.77 | 0.851 | sgl | 69.82
O3-release | cpu192-222 | fp16 | qkv | 8 | mid | 48 | 0.21 | 1.123 | onednn | 622.18
O3-release | cpu192-223 | bf16 | down | 1 | decode | 70 | 0.69 | 1.054 | sgl | 31.67
O3-release | cpu192-223 | bf16 | down | 1 | mid | 40 | 0.12 | 1.621 | onednn | 768.00
O3-release | cpu192-223 | bf16 | down | 2 | decode | 80 | 0.65 | 0.962 | sgl | 56.89
O3-release | cpu192-223 | bf16 | down | 2 | mid | 40 | 0.03 | 1.362 | onednn | 772.83
O3-release | cpu192-223 | bf16 | down | 4 | decode | 63 | 0.63 | 0.967 | sgl | 61.75
O3-release | cpu192-223 | bf16 | down | 4 | mid | 28 | 0.14 | 1.136 | onednn | 988.69
O3-release | cpu192-223 | bf16 | down | 8 | decode | 70 | 0.70 | 0.926 | sgl | 83.64
O3-release | cpu192-223 | bf16 | down | 8 | mid | 28 | 0.07 | 1.140 | onednn | 955.73
O3-release | cpu192-223 | bf16 | gate_up | 1 | decode | 70 | 0.66 | 1.018 | sgl | 31.71
O3-release | cpu192-223 | bf16 | gate_up | 1 | mid | 40 | 0.10 | 1.153 | onednn | 796.44
O3-release | cpu192-223 | bf16 | gate_up | 2 | decode | 80 | 0.62 | 0.969 | sgl | 56.89
O3-release | cpu192-223 | bf16 | gate_up | 2 | mid | 40 | 0.00 | 1.144 | onednn | 1051.89
O3-release | cpu192-223 | bf16 | gate_up | 4 | decode | 90 | 0.68 | 0.923 | sgl | 62.38
O3-release | cpu192-223 | bf16 | gate_up | 4 | mid | 40 | 0.03 | 1.161 | onednn | 1146.88
O3-release | cpu192-223 | bf16 | gate_up | 8 | decode | 100 | 0.66 | 0.927 | sgl | 85.33
O3-release | cpu192-223 | bf16 | gate_up | 8 | mid | 40 | 0.12 | 1.135 | onednn | 1303.27
O3-release | cpu192-223 | bf16 | lm_head | 1 | decode | 70 | 0.51 | 1.110 | sgl | 31.72
O3-release | cpu192-223 | bf16 | lm_head | 2 | decode | 80 | 0.51 | 1.101 | sgl | 60.81
O3-release | cpu192-223 | bf16 | lm_head | 4 | decode | 90 | 0.44 | 1.097 | onednn | 62.53
O3-release | cpu192-223 | bf16 | lm_head | 8 | decode | 80 | 0.46 | 1.064 | onednn | 114.31
O3-release | cpu192-223 | bf16 | moe_gate | 1 | decode | 28 | 1.00 | 0.655 | sgl | 25.28
O3-release | cpu192-223 | bf16 | moe_gate | 1 | mid | 16 | 1.00 | 0.827 | sgl | 109.36
O3-release | cpu192-223 | bf16 | moe_gate | 2 | decode | 32 | 1.00 | 0.675 | sgl | 25.37
O3-release | cpu192-223 | bf16 | moe_gate | 2 | mid | 16 | 1.00 | 0.880 | sgl | 115.53
O3-release | cpu192-223 | bf16 | moe_gate | 4 | decode | 36 | 1.00 | 0.678 | sgl | 28.17
O3-release | cpu192-223 | bf16 | moe_gate | 4 | mid | 16 | 0.81 | 0.970 | sgl | 118.72
O3-release | cpu192-223 | bf16 | moe_gate | 8 | decode | 40 | 1.00 | 0.696 | sgl | 30.70
O3-release | cpu192-223 | bf16 | moe_gate | 8 | mid | 16 | 0.62 | 1.039 | sgl | 119.59
O3-release | cpu192-223 | bf16 | o_proj | 1 | decode | 84 | 0.81 | 0.945 | sgl | 31.51
O3-release | cpu192-223 | bf16 | o_proj | 1 | mid | 48 | 0.04 | 1.246 | onednn | 682.67
O3-release | cpu192-223 | bf16 | o_proj | 2 | decode | 96 | 0.78 | 0.917 | sgl | 51.20
O3-release | cpu192-223 | bf16 | o_proj | 2 | mid | 48 | 0.29 | 1.087 | onednn | 819.20
O3-release | cpu192-223 | bf16 | o_proj | 4 | decode | 108 | 0.76 | 0.919 | sgl | 59.36
O3-release | cpu192-223 | bf16 | o_proj | 4 | mid | 48 | 0.06 | 1.205 | onednn | 682.67
O3-release | cpu192-223 | bf16 | o_proj | 8 | decode | 120 | 0.88 | 0.873 | sgl | 56.11
O3-release | cpu192-223 | bf16 | o_proj | 8 | mid | 48 | 0.35 | 1.093 | onednn | 431.16
O3-release | cpu192-223 | bf16 | qkv | 1 | decode | 84 | 0.64 | 0.975 | sgl | 31.59
O3-release | cpu192-223 | bf16 | qkv | 1 | mid | 48 | 0.06 | 1.186 | onednn | 722.82
O3-release | cpu192-223 | bf16 | qkv | 2 | decode | 96 | 0.70 | 0.947 | sgl | 56.20
O3-release | cpu192-223 | bf16 | qkv | 2 | mid | 48 | 0.02 | 1.161 | onednn | 945.23
O3-release | cpu192-223 | bf16 | qkv | 4 | decode | 108 | 0.81 | 0.838 | sgl | 60.53
O3-release | cpu192-223 | bf16 | qkv | 4 | mid | 48 | 0.08 | 1.144 | onednn | 877.71
O3-release | cpu192-223 | bf16 | qkv | 8 | decode | 120 | 0.72 | 0.860 | sgl | 69.82
O3-release | cpu192-223 | bf16 | qkv | 8 | mid | 48 | 0.08 | 1.123 | onednn | 622.18
O3-release | cpu192-223 | fp16 | down | 1 | decode | 70 | 0.69 | 1.042 | sgl | 31.67
O3-release | cpu192-223 | fp16 | down | 1 | mid | 40 | 0.10 | 1.623 | onednn | 768.00
O3-release | cpu192-223 | fp16 | down | 2 | decode | 80 | 0.69 | 0.956 | sgl | 56.89
O3-release | cpu192-223 | fp16 | down | 2 | mid | 40 | 0.00 | 1.395 | onednn | 772.83
O3-release | cpu192-223 | fp16 | down | 4 | decode | 63 | 0.71 | 0.930 | sgl | 61.75
O3-release | cpu192-223 | fp16 | down | 4 | mid | 28 | 0.14 | 1.149 | onednn | 988.69
O3-release | cpu192-223 | fp16 | down | 8 | decode | 70 | 0.71 | 0.935 | sgl | 83.64
O3-release | cpu192-223 | fp16 | down | 8 | mid | 28 | 0.04 | 1.184 | onednn | 955.73
O3-release | cpu192-223 | fp16 | gate_up | 1 | decode | 70 | 0.67 | 1.010 | sgl | 31.71
O3-release | cpu192-223 | fp16 | gate_up | 1 | mid | 40 | 0.07 | 1.160 | onednn | 796.44
O3-release | cpu192-223 | fp16 | gate_up | 2 | decode | 80 | 0.66 | 0.971 | sgl | 56.89
O3-release | cpu192-223 | fp16 | gate_up | 2 | mid | 40 | 0.00 | 1.157 | onednn | 1051.89
O3-release | cpu192-223 | fp16 | gate_up | 4 | decode | 90 | 0.70 | 0.912 | sgl | 62.38
O3-release | cpu192-223 | fp16 | gate_up | 4 | mid | 40 | 0.03 | 1.157 | onednn | 1146.88
O3-release | cpu192-223 | fp16 | gate_up | 8 | decode | 100 | 0.70 | 0.910 | sgl | 85.33
O3-release | cpu192-223 | fp16 | gate_up | 8 | mid | 40 | 0.15 | 1.136 | onednn | 1303.27
O3-release | cpu192-223 | fp16 | lm_head | 1 | decode | 70 | 0.57 | 1.109 | sgl | 31.72
O3-release | cpu192-223 | fp16 | lm_head | 2 | decode | 80 | 0.55 | 1.101 | sgl | 60.81
O3-release | cpu192-223 | fp16 | lm_head | 4 | decode | 90 | 0.51 | 1.085 | sgl | 62.53
O3-release | cpu192-223 | fp16 | lm_head | 8 | decode | 80 | 0.47 | 1.067 | onednn | 114.31
O3-release | cpu192-223 | fp16 | moe_gate | 1 | decode | 28 | 1.00 | 0.673 | sgl | 25.28
O3-release | cpu192-223 | fp16 | moe_gate | 1 | mid | 16 | 0.94 | 0.834 | sgl | 109.36
O3-release | cpu192-223 | fp16 | moe_gate | 2 | decode | 32 | 0.97 | 0.696 | sgl | 25.37
O3-release | cpu192-223 | fp16 | moe_gate | 2 | mid | 16 | 1.00 | 0.879 | sgl | 115.53
O3-release | cpu192-223 | fp16 | moe_gate | 4 | decode | 36 | 1.00 | 0.714 | sgl | 28.17
O3-release | cpu192-223 | fp16 | moe_gate | 4 | mid | 16 | 0.75 | 0.978 | sgl | 118.72
O3-release | cpu192-223 | fp16 | moe_gate | 8 | decode | 40 | 0.97 | 0.723 | sgl | 30.70
O3-release | cpu192-223 | fp16 | moe_gate | 8 | mid | 16 | 0.50 | 1.041 | onednn | 119.59
O3-release | cpu192-223 | fp16 | o_proj | 1 | decode | 84 | 0.88 | 0.914 | sgl | 31.51
O3-release | cpu192-223 | fp16 | o_proj | 1 | mid | 48 | 0.04 | 1.265 | onednn | 682.67
O3-release | cpu192-223 | fp16 | o_proj | 2 | decode | 96 | 0.86 | 0.904 | sgl | 51.20
O3-release | cpu192-223 | fp16 | o_proj | 2 | mid | 48 | 0.17 | 1.120 | onednn | 819.20
O3-release | cpu192-223 | fp16 | o_proj | 4 | decode | 108 | 0.81 | 0.913 | sgl | 59.36
O3-release | cpu192-223 | fp16 | o_proj | 4 | mid | 48 | 0.02 | 1.242 | onednn | 682.67
O3-release | cpu192-223 | fp16 | o_proj | 8 | decode | 120 | 0.82 | 0.894 | sgl | 56.11
O3-release | cpu192-223 | fp16 | o_proj | 8 | mid | 48 | 0.21 | 1.145 | onednn | 431.16
O3-release | cpu192-223 | fp16 | qkv | 1 | decode | 84 | 0.71 | 0.957 | sgl | 31.59
O3-release | cpu192-223 | fp16 | qkv | 1 | mid | 48 | 0.04 | 1.212 | onednn | 722.82
O3-release | cpu192-223 | fp16 | qkv | 2 | decode | 96 | 0.81 | 0.921 | sgl | 56.20
O3-release | cpu192-223 | fp16 | qkv | 2 | mid | 48 | 0.06 | 1.176 | onednn | 945.23
O3-release | cpu192-223 | fp16 | qkv | 4 | decode | 108 | 0.81 | 0.832 | sgl | 60.53
O3-release | cpu192-223 | fp16 | qkv | 4 | mid | 48 | 0.06 | 1.151 | onednn | 877.71
O3-release | cpu192-223 | fp16 | qkv | 8 | decode | 120 | 0.71 | 0.851 | sgl | 69.82
O3-release | cpu192-223 | fp16 | qkv | 8 | mid | 48 | 0.06 | 1.135 | onednn | 622.18

## Build-flag sensitivity (O2 vs O3, same CPU binding)

For matching (cpu_binding_tag, dtype, linear_type, tp_size, M-bucket) keys, compare the geomean speedup across build_tags.

cpu_binding_tag | dtype | linear | tp | M-bucket | O2-default | O3-release
|---|---|---|---|---|---|---|
cpu192-222 | bf16 | down | 1 | decode | 1.081 | 1.072
cpu192-222 | bf16 | down | 1 | mid | 1.608 | 1.562
cpu192-222 | bf16 | down | 2 | decode | 0.975 | 0.960
cpu192-222 | bf16 | down | 2 | mid | 1.328 | 1.309
cpu192-222 | bf16 | down | 4 | decode | 0.949 | 0.924
cpu192-222 | bf16 | down | 4 | mid | 1.123 | 1.115
cpu192-222 | bf16 | down | 8 | decode | 0.925 | 0.923
cpu192-222 | bf16 | down | 8 | mid | 1.071 | 1.050
cpu192-222 | bf16 | gate_up | 1 | decode | 1.021 | 1.015
cpu192-222 | bf16 | gate_up | 1 | mid | 1.158 | 1.152
cpu192-222 | bf16 | gate_up | 2 | decode | 0.970 | 0.966
cpu192-222 | bf16 | gate_up | 2 | mid | 1.180 | 1.160
cpu192-222 | bf16 | gate_up | 4 | decode | 0.929 | 0.928
cpu192-222 | bf16 | gate_up | 4 | mid | 1.169 | 1.157
cpu192-222 | bf16 | gate_up | 8 | decode | 0.927 | 0.914
cpu192-222 | bf16 | gate_up | 8 | mid | 1.172 | 1.154
cpu192-222 | bf16 | lm_head | 1 | decode | 1.117 | 1.116
cpu192-222 | bf16 | lm_head | 2 | decode | 1.111 | 1.118
cpu192-222 | bf16 | lm_head | 4 | decode | 1.095 | 1.093
cpu192-222 | bf16 | lm_head | 8 | decode | 1.079 | 1.069
cpu192-222 | bf16 | moe_gate | 1 | decode | 0.648 | 0.664
cpu192-222 | bf16 | moe_gate | 1 | mid | 0.879 | 0.884
cpu192-222 | bf16 | moe_gate | 2 | decode | 0.680 | 0.677
cpu192-222 | bf16 | moe_gate | 2 | mid | 0.907 | 0.920
cpu192-222 | bf16 | moe_gate | 4 | decode | 0.701 | 0.696
cpu192-222 | bf16 | moe_gate | 4 | mid | 0.985 | 1.019
cpu192-222 | bf16 | moe_gate | 8 | decode | 0.708 | 0.716
cpu192-222 | bf16 | moe_gate | 8 | mid | 1.059 | 1.066
cpu192-222 | bf16 | o_proj | 1 | decode | 0.945 | 0.941
cpu192-222 | bf16 | o_proj | 1 | mid | 1.246 | 1.224
cpu192-222 | bf16 | o_proj | 2 | decode | 0.923 | 0.915
cpu192-222 | bf16 | o_proj | 2 | mid | 1.075 | 1.058
cpu192-222 | bf16 | o_proj | 4 | decode | 0.902 | 0.912
cpu192-222 | bf16 | o_proj | 4 | mid | 1.197 | 1.187
cpu192-222 | bf16 | o_proj | 8 | decode | 0.866 | 0.856
cpu192-222 | bf16 | o_proj | 8 | mid | 1.084 | 1.068
cpu192-222 | bf16 | qkv | 1 | decode | 0.944 | 0.941
cpu192-222 | bf16 | qkv | 1 | mid | 1.158 | 1.155
cpu192-222 | bf16 | qkv | 2 | decode | 0.925 | 0.918
cpu192-222 | bf16 | qkv | 2 | mid | 1.171 | 1.157
cpu192-222 | bf16 | qkv | 4 | decode | 0.856 | 0.850
cpu192-222 | bf16 | qkv | 4 | mid | 1.121 | 1.116
cpu192-222 | bf16 | qkv | 8 | decode | 0.857 | 0.855
cpu192-222 | bf16 | qkv | 8 | mid | 1.120 | 1.111
cpu192-222 | fp16 | down | 1 | decode | 1.056 | 1.045
cpu192-222 | fp16 | down | 1 | mid | 1.567 | 1.561
cpu192-222 | fp16 | down | 2 | decode | 0.960 | 0.964
cpu192-222 | fp16 | down | 2 | mid | 1.322 | 1.345
cpu192-222 | fp16 | down | 4 | decode | 0.907 | 0.899
cpu192-222 | fp16 | down | 4 | mid | 1.123 | 1.133
cpu192-222 | fp16 | down | 8 | decode | 0.919 | 0.917
cpu192-222 | fp16 | down | 8 | mid | 1.066 | 1.080
cpu192-222 | fp16 | gate_up | 1 | decode | 1.003 | 1.001
cpu192-222 | fp16 | gate_up | 1 | mid | 1.167 | 1.174
cpu192-222 | fp16 | gate_up | 2 | decode | 0.971 | 0.968
cpu192-222 | fp16 | gate_up | 2 | mid | 1.165 | 1.176
cpu192-222 | fp16 | gate_up | 4 | decode | 0.917 | 0.923
cpu192-222 | fp16 | gate_up | 4 | mid | 1.158 | 1.174
cpu192-222 | fp16 | gate_up | 8 | decode | 0.903 | 0.904
cpu192-222 | fp16 | gate_up | 8 | mid | 1.151 | 1.160
cpu192-222 | fp16 | lm_head | 1 | decode | 1.116 | 1.115
cpu192-222 | fp16 | lm_head | 2 | decode | 1.109 | 1.106
cpu192-222 | fp16 | lm_head | 4 | decode | 1.087 | 1.083
cpu192-222 | fp16 | lm_head | 8 | decode | 1.066 | 1.063
cpu192-222 | fp16 | moe_gate | 1 | decode | 0.668 | 0.691
cpu192-222 | fp16 | moe_gate | 1 | mid | 0.858 | 0.861
cpu192-222 | fp16 | moe_gate | 2 | decode | 0.679 | 0.681
cpu192-222 | fp16 | moe_gate | 2 | mid | 0.912 | 0.910
cpu192-222 | fp16 | moe_gate | 4 | decode | 0.699 | 0.701
cpu192-222 | fp16 | moe_gate | 4 | mid | 0.999 | 0.996
cpu192-222 | fp16 | moe_gate | 8 | decode | 0.727 | 0.736
cpu192-222 | fp16 | moe_gate | 8 | mid | 1.073 | 1.076
cpu192-222 | fp16 | o_proj | 1 | decode | 0.918 | 0.914
cpu192-222 | fp16 | o_proj | 1 | mid | 1.245 | 1.238
cpu192-222 | fp16 | o_proj | 2 | decode | 0.898 | 0.902
cpu192-222 | fp16 | o_proj | 2 | mid | 1.076 | 1.090
cpu192-222 | fp16 | o_proj | 4 | decode | 0.906 | 0.917
cpu192-222 | fp16 | o_proj | 4 | mid | 1.199 | 1.225
cpu192-222 | fp16 | o_proj | 8 | decode | 0.872 | 0.882
cpu192-222 | fp16 | o_proj | 8 | mid | 1.096 | 1.092
cpu192-222 | fp16 | qkv | 1 | decode | 0.929 | 0.925
cpu192-222 | fp16 | qkv | 1 | mid | 1.159 | 1.171
cpu192-222 | fp16 | qkv | 2 | decode | 0.907 | 0.903
cpu192-222 | fp16 | qkv | 2 | mid | 1.170 | 1.173
cpu192-222 | fp16 | qkv | 4 | decode | 0.848 | 0.846
cpu192-222 | fp16 | qkv | 4 | mid | 1.139 | 1.128
cpu192-222 | fp16 | qkv | 8 | decode | 0.851 | 0.851
cpu192-222 | fp16 | qkv | 8 | mid | 1.120 | 1.123
cpu192-223 | bf16 | down | 1 | decode | 1.057 | 1.054
cpu192-223 | bf16 | down | 1 | mid | 1.640 | 1.621
cpu192-223 | bf16 | down | 2 | decode | 0.965 | 0.962
cpu192-223 | bf16 | down | 2 | mid | 1.389 | 1.362
cpu192-223 | bf16 | down | 4 | decode | 0.965 | 0.967
cpu192-223 | bf16 | down | 4 | mid | 1.143 | 1.136
cpu192-223 | bf16 | down | 8 | decode | 0.953 | 0.926
cpu192-223 | bf16 | down | 8 | mid | 1.178 | 1.140
cpu192-223 | bf16 | gate_up | 1 | decode | 1.018 | 1.018
cpu192-223 | bf16 | gate_up | 1 | mid | 1.162 | 1.153
cpu192-223 | bf16 | gate_up | 2 | decode | 0.968 | 0.969
cpu192-223 | bf16 | gate_up | 2 | mid | 1.150 | 1.144
cpu192-223 | bf16 | gate_up | 4 | decode | 0.923 | 0.923
cpu192-223 | bf16 | gate_up | 4 | mid | 1.154 | 1.161
cpu192-223 | bf16 | gate_up | 8 | decode | 0.929 | 0.927
cpu192-223 | bf16 | gate_up | 8 | mid | 1.154 | 1.135
cpu192-223 | bf16 | lm_head | 1 | decode | 1.112 | 1.110
cpu192-223 | bf16 | lm_head | 2 | decode | 1.108 | 1.101
cpu192-223 | bf16 | lm_head | 4 | decode | 1.081 | 1.097
cpu192-223 | bf16 | lm_head | 8 | decode | 1.071 | 1.064
cpu192-223 | bf16 | moe_gate | 1 | decode | 0.655 | 0.655
cpu192-223 | bf16 | moe_gate | 1 | mid | 0.829 | 0.827
cpu192-223 | bf16 | moe_gate | 2 | decode | 0.668 | 0.675
cpu192-223 | bf16 | moe_gate | 2 | mid | 0.881 | 0.880
cpu192-223 | bf16 | moe_gate | 4 | decode | 0.694 | 0.678
cpu192-223 | bf16 | moe_gate | 4 | mid | 0.979 | 0.970
cpu192-223 | bf16 | moe_gate | 8 | decode | 0.706 | 0.696
cpu192-223 | bf16 | moe_gate | 8 | mid | 1.032 | 1.039
cpu192-223 | bf16 | o_proj | 1 | decode | 0.944 | 0.945
cpu192-223 | bf16 | o_proj | 1 | mid | 1.250 | 1.246
cpu192-223 | bf16 | o_proj | 2 | decode | 0.929 | 0.917
cpu192-223 | bf16 | o_proj | 2 | mid | 1.108 | 1.087
cpu192-223 | bf16 | o_proj | 4 | decode | 0.919 | 0.919
cpu192-223 | bf16 | o_proj | 4 | mid | 1.210 | 1.205
cpu192-223 | bf16 | o_proj | 8 | decode | 0.879 | 0.873
cpu192-223 | bf16 | o_proj | 8 | mid | 1.114 | 1.093
cpu192-223 | bf16 | qkv | 1 | decode | 0.974 | 0.975
cpu192-223 | bf16 | qkv | 1 | mid | 1.190 | 1.186
cpu192-223 | bf16 | qkv | 2 | decode | 0.953 | 0.947
cpu192-223 | bf16 | qkv | 2 | mid | 1.169 | 1.161
cpu192-223 | bf16 | qkv | 4 | decode | 0.846 | 0.838
cpu192-223 | bf16 | qkv | 4 | mid | 1.149 | 1.144
cpu192-223 | bf16 | qkv | 8 | decode | 0.862 | 0.860
cpu192-223 | bf16 | qkv | 8 | mid | 1.135 | 1.123
cpu192-223 | fp16 | down | 1 | decode | 1.038 | 1.042
cpu192-223 | fp16 | down | 1 | mid | 1.638 | 1.623
cpu192-223 | fp16 | down | 2 | decode | 0.951 | 0.956
cpu192-223 | fp16 | down | 2 | mid | 1.397 | 1.395
cpu192-223 | fp16 | down | 4 | decode | 0.938 | 0.930
cpu192-223 | fp16 | down | 4 | mid | 1.150 | 1.149
cpu192-223 | fp16 | down | 8 | decode | 0.934 | 0.935
cpu192-223 | fp16 | down | 8 | mid | 1.192 | 1.184
cpu192-223 | fp16 | gate_up | 1 | decode | 1.006 | 1.010
cpu192-223 | fp16 | gate_up | 1 | mid | 1.151 | 1.160
cpu192-223 | fp16 | gate_up | 2 | decode | 0.968 | 0.971
cpu192-223 | fp16 | gate_up | 2 | mid | 1.152 | 1.157
cpu192-223 | fp16 | gate_up | 4 | decode | 0.909 | 0.912
cpu192-223 | fp16 | gate_up | 4 | mid | 1.158 | 1.157
cpu192-223 | fp16 | gate_up | 8 | decode | 0.907 | 0.910
cpu192-223 | fp16 | gate_up | 8 | mid | 1.126 | 1.136
cpu192-223 | fp16 | lm_head | 1 | decode | 1.110 | 1.109
cpu192-223 | fp16 | lm_head | 2 | decode | 1.100 | 1.101
cpu192-223 | fp16 | lm_head | 4 | decode | 1.084 | 1.085
cpu192-223 | fp16 | lm_head | 8 | decode | 1.065 | 1.067
cpu192-223 | fp16 | moe_gate | 1 | decode | 0.692 | 0.673
cpu192-223 | fp16 | moe_gate | 1 | mid | 0.831 | 0.834
cpu192-223 | fp16 | moe_gate | 2 | decode | 0.687 | 0.696
cpu192-223 | fp16 | moe_gate | 2 | mid | 0.876 | 0.879
cpu192-223 | fp16 | moe_gate | 4 | decode | 0.701 | 0.714
cpu192-223 | fp16 | moe_gate | 4 | mid | 0.982 | 0.978
cpu192-223 | fp16 | moe_gate | 8 | decode | 0.720 | 0.723
cpu192-223 | fp16 | moe_gate | 8 | mid | 1.050 | 1.041
cpu192-223 | fp16 | o_proj | 1 | decode | 0.913 | 0.914
cpu192-223 | fp16 | o_proj | 1 | mid | 1.260 | 1.265
cpu192-223 | fp16 | o_proj | 2 | decode | 0.905 | 0.904
cpu192-223 | fp16 | o_proj | 2 | mid | 1.116 | 1.120
cpu192-223 | fp16 | o_proj | 4 | decode | 0.918 | 0.913
cpu192-223 | fp16 | o_proj | 4 | mid | 1.245 | 1.242
cpu192-223 | fp16 | o_proj | 8 | decode | 0.882 | 0.894
cpu192-223 | fp16 | o_proj | 8 | mid | 1.135 | 1.145
cpu192-223 | fp16 | qkv | 1 | decode | 0.953 | 0.957
cpu192-223 | fp16 | qkv | 1 | mid | 1.206 | 1.212
cpu192-223 | fp16 | qkv | 2 | decode | 0.928 | 0.921
cpu192-223 | fp16 | qkv | 2 | mid | 1.178 | 1.176
cpu192-223 | fp16 | qkv | 4 | decode | 0.829 | 0.832
cpu192-223 | fp16 | qkv | 4 | mid | 1.163 | 1.151
cpu192-223 | fp16 | qkv | 8 | decode | 0.849 | 0.851
cpu192-223 | fp16 | qkv | 8 | mid | 1.129 | 1.135

## CPU-binding sensitivity (192-223 vs 192-222, same build)

For matching (build_tag, dtype, linear_type, tp_size, M-bucket) keys, compare the geomean speedup across cpu_binding_tags. This isolates the effect of dropping one thread from the NUMA-node binding, independent of the O2/O3 build-flag axis above.

build_tag | dtype | linear | tp | M-bucket | cpu192-222 | cpu192-223
|---|---|---|---|---|---|---|
O2-default | bf16 | down | 1 | decode | 1.081 | 1.057
O2-default | bf16 | down | 1 | mid | 1.608 | 1.640
O2-default | bf16 | down | 2 | decode | 0.975 | 0.965
O2-default | bf16 | down | 2 | mid | 1.328 | 1.389
O2-default | bf16 | down | 4 | decode | 0.949 | 0.965
O2-default | bf16 | down | 4 | mid | 1.123 | 1.143
O2-default | bf16 | down | 8 | decode | 0.925 | 0.953
O2-default | bf16 | down | 8 | mid | 1.071 | 1.178
O2-default | bf16 | gate_up | 1 | decode | 1.021 | 1.018
O2-default | bf16 | gate_up | 1 | mid | 1.158 | 1.162
O2-default | bf16 | gate_up | 2 | decode | 0.970 | 0.968
O2-default | bf16 | gate_up | 2 | mid | 1.180 | 1.150
O2-default | bf16 | gate_up | 4 | decode | 0.929 | 0.923
O2-default | bf16 | gate_up | 4 | mid | 1.169 | 1.154
O2-default | bf16 | gate_up | 8 | decode | 0.927 | 0.929
O2-default | bf16 | gate_up | 8 | mid | 1.172 | 1.154
O2-default | bf16 | lm_head | 1 | decode | 1.117 | 1.112
O2-default | bf16 | lm_head | 2 | decode | 1.111 | 1.108
O2-default | bf16 | lm_head | 4 | decode | 1.095 | 1.081
O2-default | bf16 | lm_head | 8 | decode | 1.079 | 1.071
O2-default | bf16 | moe_gate | 1 | decode | 0.648 | 0.655
O2-default | bf16 | moe_gate | 1 | mid | 0.879 | 0.829
O2-default | bf16 | moe_gate | 2 | decode | 0.680 | 0.668
O2-default | bf16 | moe_gate | 2 | mid | 0.907 | 0.881
O2-default | bf16 | moe_gate | 4 | decode | 0.701 | 0.694
O2-default | bf16 | moe_gate | 4 | mid | 0.985 | 0.979
O2-default | bf16 | moe_gate | 8 | decode | 0.708 | 0.706
O2-default | bf16 | moe_gate | 8 | mid | 1.059 | 1.032
O2-default | bf16 | o_proj | 1 | decode | 0.945 | 0.944
O2-default | bf16 | o_proj | 1 | mid | 1.246 | 1.250
O2-default | bf16 | o_proj | 2 | decode | 0.923 | 0.929
O2-default | bf16 | o_proj | 2 | mid | 1.075 | 1.108
O2-default | bf16 | o_proj | 4 | decode | 0.902 | 0.919
O2-default | bf16 | o_proj | 4 | mid | 1.197 | 1.210
O2-default | bf16 | o_proj | 8 | decode | 0.866 | 0.879
O2-default | bf16 | o_proj | 8 | mid | 1.084 | 1.114
O2-default | bf16 | qkv | 1 | decode | 0.944 | 0.974
O2-default | bf16 | qkv | 1 | mid | 1.158 | 1.190
O2-default | bf16 | qkv | 2 | decode | 0.925 | 0.953
O2-default | bf16 | qkv | 2 | mid | 1.171 | 1.169
O2-default | bf16 | qkv | 4 | decode | 0.856 | 0.846
O2-default | bf16 | qkv | 4 | mid | 1.121 | 1.149
O2-default | bf16 | qkv | 8 | decode | 0.857 | 0.862
O2-default | bf16 | qkv | 8 | mid | 1.120 | 1.135
O2-default | fp16 | down | 1 | decode | 1.056 | 1.038
O2-default | fp16 | down | 1 | mid | 1.567 | 1.638
O2-default | fp16 | down | 2 | decode | 0.960 | 0.951
O2-default | fp16 | down | 2 | mid | 1.322 | 1.397
O2-default | fp16 | down | 4 | decode | 0.907 | 0.938
O2-default | fp16 | down | 4 | mid | 1.123 | 1.150
O2-default | fp16 | down | 8 | decode | 0.919 | 0.934
O2-default | fp16 | down | 8 | mid | 1.066 | 1.192
O2-default | fp16 | gate_up | 1 | decode | 1.003 | 1.006
O2-default | fp16 | gate_up | 1 | mid | 1.167 | 1.151
O2-default | fp16 | gate_up | 2 | decode | 0.971 | 0.968
O2-default | fp16 | gate_up | 2 | mid | 1.165 | 1.152
O2-default | fp16 | gate_up | 4 | decode | 0.917 | 0.909
O2-default | fp16 | gate_up | 4 | mid | 1.158 | 1.158
O2-default | fp16 | gate_up | 8 | decode | 0.903 | 0.907
O2-default | fp16 | gate_up | 8 | mid | 1.151 | 1.126
O2-default | fp16 | lm_head | 1 | decode | 1.116 | 1.110
O2-default | fp16 | lm_head | 2 | decode | 1.109 | 1.100
O2-default | fp16 | lm_head | 4 | decode | 1.087 | 1.084
O2-default | fp16 | lm_head | 8 | decode | 1.066 | 1.065
O2-default | fp16 | moe_gate | 1 | decode | 0.668 | 0.692
O2-default | fp16 | moe_gate | 1 | mid | 0.858 | 0.831
O2-default | fp16 | moe_gate | 2 | decode | 0.679 | 0.687
O2-default | fp16 | moe_gate | 2 | mid | 0.912 | 0.876
O2-default | fp16 | moe_gate | 4 | decode | 0.699 | 0.701
O2-default | fp16 | moe_gate | 4 | mid | 0.999 | 0.982
O2-default | fp16 | moe_gate | 8 | decode | 0.727 | 0.720
O2-default | fp16 | moe_gate | 8 | mid | 1.073 | 1.050
O2-default | fp16 | o_proj | 1 | decode | 0.918 | 0.913
O2-default | fp16 | o_proj | 1 | mid | 1.245 | 1.260
O2-default | fp16 | o_proj | 2 | decode | 0.898 | 0.905
O2-default | fp16 | o_proj | 2 | mid | 1.076 | 1.116
O2-default | fp16 | o_proj | 4 | decode | 0.906 | 0.918
O2-default | fp16 | o_proj | 4 | mid | 1.199 | 1.245
O2-default | fp16 | o_proj | 8 | decode | 0.872 | 0.882
O2-default | fp16 | o_proj | 8 | mid | 1.096 | 1.135
O2-default | fp16 | qkv | 1 | decode | 0.929 | 0.953
O2-default | fp16 | qkv | 1 | mid | 1.159 | 1.206
O2-default | fp16 | qkv | 2 | decode | 0.907 | 0.928
O2-default | fp16 | qkv | 2 | mid | 1.170 | 1.178
O2-default | fp16 | qkv | 4 | decode | 0.848 | 0.829
O2-default | fp16 | qkv | 4 | mid | 1.139 | 1.163
O2-default | fp16 | qkv | 8 | decode | 0.851 | 0.849
O2-default | fp16 | qkv | 8 | mid | 1.120 | 1.129
O3-release | bf16 | down | 1 | decode | 1.072 | 1.054
O3-release | bf16 | down | 1 | mid | 1.562 | 1.621
O3-release | bf16 | down | 2 | decode | 0.960 | 0.962
O3-release | bf16 | down | 2 | mid | 1.309 | 1.362
O3-release | bf16 | down | 4 | decode | 0.924 | 0.967
O3-release | bf16 | down | 4 | mid | 1.115 | 1.136
O3-release | bf16 | down | 8 | decode | 0.923 | 0.926
O3-release | bf16 | down | 8 | mid | 1.050 | 1.140
O3-release | bf16 | gate_up | 1 | decode | 1.015 | 1.018
O3-release | bf16 | gate_up | 1 | mid | 1.152 | 1.153
O3-release | bf16 | gate_up | 2 | decode | 0.966 | 0.969
O3-release | bf16 | gate_up | 2 | mid | 1.160 | 1.144
O3-release | bf16 | gate_up | 4 | decode | 0.928 | 0.923
O3-release | bf16 | gate_up | 4 | mid | 1.157 | 1.161
O3-release | bf16 | gate_up | 8 | decode | 0.914 | 0.927
O3-release | bf16 | gate_up | 8 | mid | 1.154 | 1.135
O3-release | bf16 | lm_head | 1 | decode | 1.116 | 1.110
O3-release | bf16 | lm_head | 2 | decode | 1.118 | 1.101
O3-release | bf16 | lm_head | 4 | decode | 1.093 | 1.097
O3-release | bf16 | lm_head | 8 | decode | 1.069 | 1.064
O3-release | bf16 | moe_gate | 1 | decode | 0.664 | 0.655
O3-release | bf16 | moe_gate | 1 | mid | 0.884 | 0.827
O3-release | bf16 | moe_gate | 2 | decode | 0.677 | 0.675
O3-release | bf16 | moe_gate | 2 | mid | 0.920 | 0.880
O3-release | bf16 | moe_gate | 4 | decode | 0.696 | 0.678
O3-release | bf16 | moe_gate | 4 | mid | 1.019 | 0.970
O3-release | bf16 | moe_gate | 8 | decode | 0.716 | 0.696
O3-release | bf16 | moe_gate | 8 | mid | 1.066 | 1.039
O3-release | bf16 | o_proj | 1 | decode | 0.941 | 0.945
O3-release | bf16 | o_proj | 1 | mid | 1.224 | 1.246
O3-release | bf16 | o_proj | 2 | decode | 0.915 | 0.917
O3-release | bf16 | o_proj | 2 | mid | 1.058 | 1.087
O3-release | bf16 | o_proj | 4 | decode | 0.912 | 0.919
O3-release | bf16 | o_proj | 4 | mid | 1.187 | 1.205
O3-release | bf16 | o_proj | 8 | decode | 0.856 | 0.873
O3-release | bf16 | o_proj | 8 | mid | 1.068 | 1.093
O3-release | bf16 | qkv | 1 | decode | 0.941 | 0.975
O3-release | bf16 | qkv | 1 | mid | 1.155 | 1.186
O3-release | bf16 | qkv | 2 | decode | 0.918 | 0.947
O3-release | bf16 | qkv | 2 | mid | 1.157 | 1.161
O3-release | bf16 | qkv | 4 | decode | 0.850 | 0.838
O3-release | bf16 | qkv | 4 | mid | 1.116 | 1.144
O3-release | bf16 | qkv | 8 | decode | 0.855 | 0.860
O3-release | bf16 | qkv | 8 | mid | 1.111 | 1.123
O3-release | fp16 | down | 1 | decode | 1.045 | 1.042
O3-release | fp16 | down | 1 | mid | 1.561 | 1.623
O3-release | fp16 | down | 2 | decode | 0.964 | 0.956
O3-release | fp16 | down | 2 | mid | 1.345 | 1.395
O3-release | fp16 | down | 4 | decode | 0.899 | 0.930
O3-release | fp16 | down | 4 | mid | 1.133 | 1.149
O3-release | fp16 | down | 8 | decode | 0.917 | 0.935
O3-release | fp16 | down | 8 | mid | 1.080 | 1.184
O3-release | fp16 | gate_up | 1 | decode | 1.001 | 1.010
O3-release | fp16 | gate_up | 1 | mid | 1.174 | 1.160
O3-release | fp16 | gate_up | 2 | decode | 0.968 | 0.971
O3-release | fp16 | gate_up | 2 | mid | 1.176 | 1.157
O3-release | fp16 | gate_up | 4 | decode | 0.923 | 0.912
O3-release | fp16 | gate_up | 4 | mid | 1.174 | 1.157
O3-release | fp16 | gate_up | 8 | decode | 0.904 | 0.910
O3-release | fp16 | gate_up | 8 | mid | 1.160 | 1.136
O3-release | fp16 | lm_head | 1 | decode | 1.115 | 1.109
O3-release | fp16 | lm_head | 2 | decode | 1.106 | 1.101
O3-release | fp16 | lm_head | 4 | decode | 1.083 | 1.085
O3-release | fp16 | lm_head | 8 | decode | 1.063 | 1.067
O3-release | fp16 | moe_gate | 1 | decode | 0.691 | 0.673
O3-release | fp16 | moe_gate | 1 | mid | 0.861 | 0.834
O3-release | fp16 | moe_gate | 2 | decode | 0.681 | 0.696
O3-release | fp16 | moe_gate | 2 | mid | 0.910 | 0.879
O3-release | fp16 | moe_gate | 4 | decode | 0.701 | 0.714
O3-release | fp16 | moe_gate | 4 | mid | 0.996 | 0.978
O3-release | fp16 | moe_gate | 8 | decode | 0.736 | 0.723
O3-release | fp16 | moe_gate | 8 | mid | 1.076 | 1.041
O3-release | fp16 | o_proj | 1 | decode | 0.914 | 0.914
O3-release | fp16 | o_proj | 1 | mid | 1.238 | 1.265
O3-release | fp16 | o_proj | 2 | decode | 0.902 | 0.904
O3-release | fp16 | o_proj | 2 | mid | 1.090 | 1.120
O3-release | fp16 | o_proj | 4 | decode | 0.917 | 0.913
O3-release | fp16 | o_proj | 4 | mid | 1.225 | 1.242
O3-release | fp16 | o_proj | 8 | decode | 0.882 | 0.894
O3-release | fp16 | o_proj | 8 | mid | 1.092 | 1.145
O3-release | fp16 | qkv | 1 | decode | 0.925 | 0.957
O3-release | fp16 | qkv | 1 | mid | 1.171 | 1.212
O3-release | fp16 | qkv | 2 | decode | 0.903 | 0.921
O3-release | fp16 | qkv | 2 | mid | 1.173 | 1.176
O3-release | fp16 | qkv | 4 | decode | 0.846 | 0.832
O3-release | fp16 | qkv | 4 | mid | 1.128 | 1.151
O3-release | fp16 | qkv | 8 | decode | 0.851 | 0.851
O3-release | fp16 | qkv | 8 | mid | 1.123 | 1.135
