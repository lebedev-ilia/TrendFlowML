# Final Benchmark Results Summary

Aggregated results from multiple benchmark runs.

Format: `mean_value (outlier1, outlier2, ...)` - mean excludes outliers, outliers shown in parentheses.

## Component: core_clip

| Model | Triton model 1 | Triton model 2 | Triton Preprocess | Triton Batch | Frames cnt | Runs | Duration (s) | Image Inf (s) | Text Inf (s) | Peak CPU % | Peak GPU % | Triton Delta RAM (MB) | Triton Delta VRAM (MB) | Component Delta VRAM (MB) | Component Delta RAM (MB) | Summary Delta RAM | Summary Delta VRAM |
|--------------|--------|------|-------------|---------------|------------|---------------|------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| ViT-B/32 vunknown | clip_image_224 | clip_text | preprocess_clip_image_224 | 1 | 1 | 3 | 47 | 0.650 | 34 | 100 | 4 | 1636 | 685 | 0 | 642 | 2278 | 685 |
| ViT-B/32 vunknown | clip_image_224 | clip_text | preprocess_clip_image_224 | 1 | 2 | 3 | 46 | 0.655 | 34 | 100 | 1 | 1692 | 679 | 5 | 413 | 2105 | 685 |
| ViT-B/32 vunknown | clip_image_224 | clip_text | preprocess_clip_image_224 | 1 | 5 | 3 | 46 | 0.813 | 34 | 100 | 4 | 1690 | 688 | 0 | 334 | 2025 | 687 |
| ViT-B/32 vunknown | clip_image_224 | clip_text | preprocess_clip_image_224 | 1 | 50 | 3 | 51 | 3.000 | 34 | 99 | 7 | 1713 | 678 | 22 | 162 | 1876 | 700 |
| ViT-B/32 vunknown | clip_image_224 | clip_text | preprocess_clip_image_224 | 1 | 131 | 3 | 57 | 4.762 | 34 | 100 | 6 | 1748 | 664 | 5 | 611 | 2359 | 670 |
| ViT-B/32 vunknown | clip_image_224 | clip_text | preprocess_clip_image_224 | 2 | 1 | 3 | 46 | 0.627 | 34 | 100 | 1 | 1832 | 680 | 4 | 166 | 1998 | 684 |
| ViT-B/32 vunknown | clip_image_224 | clip_text | preprocess_clip_image_224 | 2 | 2 | 3 | 47 | 0.637 | 34 | 99 | 13 | 1547 | 680 | 8 | 446 | 1993 | 689 |
| ViT-B/32 vunknown | clip_image_224 | clip_text | preprocess_clip_image_224 | 2 | 131 | 3 | 57 | 6.188 | 34 | 100 | 6 | 1588 | 682 | 7 | 704 | 2292 | 689 |
| ViT-B/32 vunknown | clip_image_224 | clip_text | preprocess_clip_image_224 | 4 | 131 | 3 | 57 | 5.642 | 34 | 100 | 5 | 1631 | 678 | 1 | 712 | 2343 | 680 |
| ViT-B/32 vunknown | clip_image_224 | clip_text | preprocess_clip_image_224 | 16 | 131 | 3 | 56 | 5.189 | 34 | 100 | 5 | 1379 | 664 | 4 | 660 | 2039 | 668 |
| ViT-B/32 vunknown | clip_image_336 | clip_text | preprocess_clip_image_336 | 1 | 1 | 3 |  |  |  |  |  |  |  |  |  |  |  |
| ViT-B/32 vunknown | clip_image_336 | clip_text | preprocess_clip_image_336 | 1 | 131 | 3 |  |  |  |  |  |  |  |  |  |  |  |
| ViT-B/32 vunknown | clip_image_336 | clip_text | preprocess_clip_image_336 | 16 | 1 | 3 | 47 | 0.661 | 34 | 100 | 2 | 1599 | 685 | 3 | 741 | 2340 | 689 |
| ViT-B/32 vunknown | clip_image_336 | clip_text | preprocess_clip_image_336 | 16 | 131 | 3 | 66 | 10.738 | 37 | 100 | 9 | 1740 | 680 | 10 | 668 | 2408 | 690 |
| ViT-B/32 vunknown | clip_image_448 | clip_text | preprocess_clip_image_448 | 1 | 1 | 3 |  |  |  |  |  |  |  |  |  |  |  |
| ViT-B/32 vunknown | clip_image_448 | clip_text | preprocess_clip_image_448 | 1 | 131 | 3 |  |  |  |  |  |  |  |  |  |  |  |
| ViT-B/32 vunknown | clip_image_448 | clip_text | preprocess_clip_image_448 | 16 | 1 | 3 |  |  |  |  |  |  |  |  |  |  |  |
| ViT-B/32 vunknown | clip_image_448 | clip_text | preprocess_clip_image_448 | 16 | 131 | 3 |  |  |  |  |  |  |  |  |  |  |  |
---

## Навигация

[README](README.md) · [Vault](../docs/MAIN_INDEX.md)
