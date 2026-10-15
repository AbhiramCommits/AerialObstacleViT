| backend | resolution | batch | mean ms | p50 ms | p95 ms | FPS | device |
|---|---|---|---|---|---|---|---|
| pytorch-eager | 224x224 | 1 | 12.94 | 4.90 | 5.22 | 77.26 | apple-mps (Metal Performance Shaders) |
| torchscript | 224x224 | 1 | 4.04 | 4.01 | 4.16 | 247.69 | apple-mps (Metal Performance Shaders) |
| pytorch-eager | 224x224 | 1 | 32.57 | 32.07 | 38.47 | 30.70 | cpu |
| torchscript | 224x224 | 1 | 27.47 | 27.44 | 28.23 | 36.41 | cpu |
| onnxruntime | 224x224 | 1 | 51.92 | 50.99 | 57.73 | 19.26 | cpu (CPUExecutionProvider) |
| pytorch-eager | 224x224 | 8 | 6.87 | 6.36 | 7.54 | 1164.02 | apple-mps (Metal Performance Shaders) |
| torchscript | 224x224 | 8 | 6.07 | 5.58 | 8.37 | 1317.22 | apple-mps (Metal Performance Shaders) |
| pytorch-eager | 224x224 | 8 | 223.60 | 219.44 | 255.00 | 35.78 | cpu |
| torchscript | 224x224 | 8 | 215.06 | 212.33 | 228.60 | 37.20 | cpu |
| onnxruntime | 224x224 | 8 | 448.71 | 447.69 | 461.92 | 17.83 | cpu (CPUExecutionProvider) |
| pytorch-eager | 320x320 | 1 | 10.03 | 5.55 | 8.37 | 99.69 | apple-mps (Metal Performance Shaders) |
| torchscript | 320x320 | 1 | 4.62 | 4.63 | 4.79 | 216.40 | apple-mps (Metal Performance Shaders) |
| pytorch-eager | 320x320 | 1 | 54.32 | 54.09 | 55.45 | 18.41 | cpu |
| torchscript | 320x320 | 1 | 52.14 | 52.09 | 53.14 | 19.18 | cpu |
| onnxruntime | 320x320 | 1 | 124.00 | 122.91 | 130.98 | 8.06 | cpu (CPUExecutionProvider) |
| pytorch-eager | 320x320 | 8 | 8.08 | 5.59 | 5.82 | 990.43 | apple-mps (Metal Performance Shaders) |
| torchscript | 320x320 | 8 | 6.53 | 5.64 | 10.04 | 1225.77 | apple-mps (Metal Performance Shaders) |
| pytorch-eager | 320x320 | 8 | 440.27 | 426.88 | 497.95 | 18.17 | cpu |
| torchscript | 320x320 | 8 | 494.12 | 489.04 | 526.24 | 16.19 | cpu |
| onnxruntime | 320x320 | 8 | 1171.86 | 1156.64 | 1238.83 | 6.83 | cpu (CPUExecutionProvider) |

**Onboard feasibility at 10.0 Hz target:** best backend `torchscript` at 224x224 reaches 247.69 FPS -> **feasible**.

