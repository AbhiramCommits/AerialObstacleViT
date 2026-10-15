| scope | mIoU before | mIoU control | mIoU after | Δ mIoU vs before | Δ mIoU vs control | Δ mAP@0.5 vs before | Δ mAP@0.5 vs control |
|---|---|---|---|---|---|---|---|
| **overall** | 0.6063 | 0.6335 | 0.6388 | 0.0325 | 0.0053 | 0.0235 | -0.0105 |
| cluster 0 (low-contrast/haze) | 0.6107 | 0.6428 | 0.6425 | 0.0318 | -0.0003 | 0.0088 | -0.0370 |
| cluster 1 (dense-objects) | 0.6039 | 0.6256 | 0.6360 | 0.0320 | 0.0104 | 0.0251 | -0.0024 |

_control = identical warm-start and epochs, trained **without** the mined hard frames; `vs control` therefore isolates the effect of the hard examples themselves. 24 hard frames were added, up-weighted x3._

