# Gate 3 — анализ данных 1000 видео (инсайты из report1000)

Скрипт: `analyze_gate3.py` (venv с pandas+mpl). Графики: `figures/*.png` (копии в `deepdive_outbox` → VK).

## 1. Куда уходит время (per-video wall, p50)
`figures/gate3_component_time.png`

| Компонент | p50, с | доля | движок |
|---|---|---|---|
| core_optical_flow | 63.4 | **20.4%** | RAFT (Triton) |
| scene_classification | 60.6 | **19.5%** | resnet50 (inprocess) |
| cut_detection | 56.0 | **18.0%** | CV+эвристики (CPU) |
| core_clip | 48.9 | 15.7% | CLIP (Triton) |
| core_depth_midas | 43.4 | 13.9% | MiDaS (Triton) |
| video_pacing | 18.4 | 5.9% | CPU |
| segmenter | 12.4 | 4.0% | CPU |
| download | 5.6 | 1.8% | I/O |
| uniqueness | 2.4 | 0.8% | CPU (поверх clip) |

**Вывод:** 5 компонентов (flow+scene+cut+clip+depth) = **~87%** времени. Рычаги оптимизации — именно они;
download/segment/pacing/uniqueness дёшевы, их трогать смысла нет. Самый дорогой — **optical_flow (RAFT)**.

## 2. Пайплайн CPU-bound (не GPU)
`figures/gate3_cpu_vs_gpu.png`

- **CPU% p50 ≈ 56%** по torch/CV-компонентам, **GPU util mean ≈ 4%** (макс среди компонентов).
- GPU почти простаивает даже у Triton-компонентов — узкое место CPU + import-tax + I/O, не GPU-инференс.
- **Подтверждает стратегию:** дешёвая GPU + высокий N (много ядер) окупаются; дорогая карта — нет.
  На 48-ядерном поде N=8 дал ~92 видео/ч (×3.4 к N=4); есть запас поднять N ещё (GPU/VRAM не лимит).

## 3. Распределение времени/видео
`figures/gate3_video_wall_hist.png` — p50=306с, p95=505с, mean=316с (n=1000). Разброс отражает
разнообразие corpus1000 (длительность 4-893с): короткие клипы ~150-250с, длинные до ~500-600с.

## 4. Надёжность
`figures/gate3_ok_fail.png` — из 9000 (видео×компонент): **ok=8379, fail(краш rc≠0)=528, absent=93**.
- **scene: 497 краш** = L13 (cuDNN на RTX 2000 Ada) — единственная крупная проблема, фикс готов+валидирован.
- Остальные краши малы и транзиентны: uniq 13, cut 5, flow 4, depth 3, clip 2, pacing 4 (rc=120 таймаут /
  rc=2 — обычно битые/пустые входы отдельных видео).
- **absent=93** (нет выхода): депх/флоу/кат/клип по ~15-25 — каскад от ~видео с проблемным
  download/segment или Triton-таймаутов; не краши компонента.

## Итоговые рекомендации (из данных)
1. **Оптимизировать в порядке цены:** optical_flow (RAFT) → scene → cut_detection → clip → depth.
   Для flow/clip/depth (Triton) — проверить batch-size/разрешение; cut (CPU) — профилировать эвристики.
2. **Import-tax (Phase 4, venv-на-SSD)** — по-прежнему главный системный рычаг: срезает фикс. ~45с/импорт
   с КАЖДОГО per-video subprocess КАЖДОГО компонента; при 9 компонентах × 1000 видео это доминирующая
   доля «фикса». Даст больше, чем точечная оптимизация любого одного компонента.
3. **cuDNN-guard (L13) обязателен** для всех GPU-inprocess torch-компонентов до следующего scale-теста.
4. **N можно поднять** (>8) на многоядерном поде — GPU/VRAM не лимит; упор в CPU-ядра.
