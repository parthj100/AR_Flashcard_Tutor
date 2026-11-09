# YOLOv8 comparative benchmark

- device: `mps`
- imgsz: `640`
- conf: `0.25`
- warmup: `5`
- classes: `microscope, calculator, backpack, periodic_table_poster, globe_model, safety_goggles`
- total images: `554`
- timestamp: `20260511_152022`

## Overall

| model | images | wall s | mean ms | p50 ms | p95 ms | p99 ms | FPS | mean conf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `yolov8n.pt` | 554 | 24.02 | 43.04 | 31.45 | 64.57 | 283.18 | 23.23 | 51.47% |
| `yolov8s.pt` | 554 | 30.93 | 55.53 | 46.2 | 89.23 | 297.91 | 18.01 | 55.34% |

## Per-class mean latency (ms)

| model | backpack | calculator | globe_model | microscope | periodic_table_poster | safety_goggles |
|---|---|---|---|---|---|---|
| `yolov8n.pt` | 81.37 | 40.16 | 37.2 | 33.04 | 35.27 | 29.41 |
| `yolov8s.pt` | 77.46 | 55.33 | 52.82 | 48.1 | 53.73 | 45.39 |

## Per-class FPS

| model | backpack | calculator | globe_model | microscope | periodic_table_poster | safety_goggles |
|---|---|---|---|---|---|---|
| `yolov8n.pt` | 12.29 | 24.9 | 26.88 | 30.27 | 28.35 | 34.0 |
| `yolov8s.pt` | 12.91 | 18.07 | 18.93 | 20.79 | 18.61 | 22.03 |
