| method | trainable params | total params | trainable % | mIoU | landing-zone IoU | mAP@0.5 | rare-class mAP@0.5 | ms/img | device |
|---|---|---|---|---|---|---|---|---|---|
| DINOv2 (frozen) + linear head | 2,083,733 | 24,140,309 | 8.632 % | 0.4967 | 0.8835 | 0.3658 | 0.0000 | 88.3 | apple-mps (Metal Performance Shaders) |
| owlvit (zero-shot) | 0 | 153,231,879 | 0.000 % | 0.0751 | 0.6658 | 0.1853 | 0.0050 | 273.3 | apple-mps (Metal Performance Shaders) |
| clip (zero-shot) | 0 | 151,277,313 | 0.000 % | 0.0755 | 0.4350 | 0.0002 | 0.0000 | 789.3 | apple-mps (Metal Performance Shaders) |
