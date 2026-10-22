# Seoul LST Modelling · Urban Form, Land Surface Temperature and Heat Resilience

Predicting Seoul's land surface temperature from urban form at three grid resolutions (120 m / 240 m / 480 m), comparing 2016 and 2023, and using a Spatial Durbin Model for interpretation.

**Target variables**: `nor_` (normal day), `ext_` (extreme day), `hr_` (heat resilience = extreme − normal).

---

## Pipeline

```
work flow 0  air temperature → identify extreme heat days (apparent temperature > 33 °C)
work flow 1  environment setup and grid preparation
work flow 2  GBDT hyperparameter search (200 candidates × 20 folds)
work flow 3  480 m grids, cleaning (lst_ratio ≥ 0.8)
  3plus_GBDT  R² comparison, feature importance, partial dependence
  3plus_SDM   Spatial Durbin Model, exports SDM_effects.xlsx
work flow 4  GBDT retrained at 120 m / 240 m / 480 m and on 2023
```

Notebooks in order:

| File | Purpose |
|---|---|
| `seoul_work_flow_0_air_tem.ipynb` | Heat-day selection |
| `seoul_work_flow_1.ipynb` | Grid preparation |
| `seoul_work_flow_2.ipynb` | GBDT random search |
| `seoul_work_flow_3_480m_based.ipynb` | 480 m dataset |
| `seoul_work_flow_3plus_GBDT_tailored.ipynb` | Tuned GBDT + PDP |
| `seoul_work_flow_3plus_SDM.ipynb` | Spatial Durbin Model |
| `seoul_work_flow_4_2023.ipynb` | 2023 replication |
| `seoul_work_flow_4_unit_120m.ipynb` / `_240m` | Resolution comparison |

## Features

BCR, BHV, SVF, NDVI, EV, water ratio, distance to water / park / mountain.

## Stack

Python 3 · Jupyter · geopandas · scikit-learn (`GradientBoostingRegressor`, `RandomizedSearchCV`, `PartialDependenceDisplay`, `RepeatedKFold`) · libpysal + spreg (`ML_Lag`) · matplotlib · openpyxl.

## Data — not included

All inputs are read from absolute local paths (`D:\seoul\grids\...`, `E:\seoul\480_based`). No shapefiles, grids or station tables are committed, and there is no `requirements.txt`. To run: supply the gridded LST and urban-form layers, then update the paths at the top of each notebook.

Notebook outputs are retained, including an early kernel-start error cell in `work flow 1`; that cell is environmental, not part of the analysis.

## Notes

- `seoul_work_flow_0` ends with GBDT import boilerplate — the modelling itself starts at `work flow 2`
- Results are per-resolution; don't compare R² across the 120 / 240 / 480 m tables without checking the sample sizes

## License

Not declared.
