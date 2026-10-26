# Resume and application wording

[← Portfolio](../README.md)

Use this link for the combined showcase: **https://github.com/pranjalsinggh/kaggle-achievements**

## Compact achievements section

- **Kaggle — Petals to the Metal:** Achieved **12th of 50 teams**, scoring **0.94693 macro F1** with an EfficientNetV2S/EfficientNetB3 ensemble for 104-class flower classification (ongoing leaderboard snapshot, October 2026).
- **Kaggle — Connect X:** Achieved **41st of 255 teams** with a native C++ Connect Four agent using bitboards, iterative deepening, alpha-beta pruning, and transposition caching (ongoing leaderboard snapshot, October 2026).

## Project section: computer vision

**Flower Classification Ensemble | Python, TensorFlow/Keras, NumPy, pandas, scikit-learn**

- Built a 104-class image classifier with ImageNet-pretrained EfficientNetV2S and EfficientNetB3, mixed-precision training on two T4 GPUs, image augmentation, and four-view test-time averaging.
- Trained on 12,753 images and used 3,712 validation images for checkpoint and blend selection; a 50/50 probability blend achieved 0.95479 validation macro F1 and 0.94693 on Kaggle, reaching 12/50 teams in the October 25, 2026 snapshot.
- Validated prediction alignment and all 7,382 test IDs before generating the competition submission.

## Project section: game AI and systems

**Connect X Search Agent | C++17, Python, ctypes, Linux, Zig**

- Built and deployed a native bitboard Connect Four agent with iterative-deepening alpha-beta search, tactical pruning, a 2²¹-entry transposition table, and a 1.45-second move budget; reached 41/255 teams in the October 25, 2026 snapshot.
- Passed 308 generated board checks against Kaggle helpers and an official-loader self-play game; won 8/8 games against each of random and built-in negamax in small seeded benchmarks.
- Cross-compiled a Linux shared library and integrated it with Python; one sampled Kaggle invocation searched 5.75 million nodes to depth 18 in 1.479 seconds.

## Short combined line

**Kaggle competition projects:** Petals to the Metal — 12/50 teams, 0.94693 macro F1; Connect X — 41/255 teams. October 25, 2026 ongoing leaderboard snapshot. [Code and evidence](https://github.com/pranjalsinggh/kaggle-achievements).

The percentages derived from the screenshot are 24.0% and 16.1% (rank divided by team count). Prefer the exact ranks in applications. These records do not establish a final competition placement, medal, prize, or top-10 result.
