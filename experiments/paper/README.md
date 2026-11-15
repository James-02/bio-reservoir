# Paper configs

The configs of every run and study the paper reports. Each file is a sweep
([Sweeps](../../README.md#sweeps)) that expands into named runs or studies; a run writes
`results/runs/<name>/` and a study `results/optuna/optuna-<name>.db`, the names the paper
scripts in [scripts/paper](../../scripts/paper) read.

| File | Paper section | Names |
|---|---|---|
| [models/](models) | The design and the comparisons, as single configs | `array`, `array-forest`, `isolated-traps`, `esn-mean`, `esn-image` (each with a `-forest` reading) |
| [classification.yaml](classification.yaml) | Classification Performance | `balanced-array-s<seed>`, `balanced-raw-<readout>`, `readout-two-images-<readout>-s<seed>` |
| [tasks.yaml](tasks.yaml) | Other Tasks and Validation | `unbalanced-array-s<seed>`, `binary-isolated-traps-s<seed>`, `validation-1-array-s<seed>` |
| [test.yaml](test.yaml) | Other Tasks and Validation (test split, run once) | `test-array`, `test-raw-ridge` |
| [comparisons.yaml](comparisons.yaml) | Comparisons and Ablations | `balanced-esn-image-s<seed>`, `random-features`, `surrogate-two-images-s<seed>`, `ablation-no-delay` |
| [robustness.yaml](robustness.yaml) | Robustness and Heterogeneity | `robustness-noise`, `heterogeneity-density`, `layout-gaussian` |
| [array-size.yaml](array-size.yaml) | Robustness and Heterogeneity (array size) | `array-size`, `array-size-16384-s<seed>`, `array-size-ideal-4096` |
| [washout-map.yaml](washout-map.yaml) | Colony Dynamics (Figure 2D) | `washout-map-0.3` |
| [searches.yaml](searches.yaml) | Design Studies | `search-array-a`, `selection-esn-image` |

Reporting seeds are 6348-6352, selection seeds 6368-6372 and the search seed 6337.

```bash
python -m training.run sweep experiments/paper/classification.yaml --list
python -m training.run sweep experiments/paper/classification.yaml --only model=array --processes N
python -m training.run resolve experiments/paper/tasks.yaml --only task=binary model=array seed=6348
python -m training.run study-table experiments/paper/robustness.yaml --group
```

Run `classification.yaml` before `comparisons.yaml`: the readouts and the linear surrogates
refit the saved features of `balanced-array-s<seed>` and `balanced-array-forest-s<seed>`.
Run `test.yaml` once, last, with `--final-test`.
