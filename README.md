# Flood susceptibility of the Indus floodplain in Layyah District, Pakistan

Code and data for a machine learning flood susceptibility study of the Indus floodplain in Layyah District, Punjab. Models are trained on the Indus corridor (land within 15 km of the main stem) of six districts (Mianwali, Bhakkar, Muzaffargarh, Dera Ghazi Khan, Rajanpur and Rahim Yar Khan) and tested on Layyah, which was withheld from training, and on land first flooded after the training period.

Flood labels come from the JRC Global Surface Water monthly history (2000 to 2021), extended to 2025 with Landsat 8 and 9. Predictors are terrain, drainage, soil, rainfall and dry-season Landsat indices; flood-season water indices are excluded because they record the floods themselves.

## Main results

| Test (Layyah) | Logistic regression AUC | Distance to the Indus alone | Difference (95% block-bootstrap interval) |
|---|---|---|---|
| Floods of 2000-2019 | 0.936 | 0.887 | 0.049 (0.026 to 0.075) |
| Land first flooded in 2020-2021 | 0.881 | 0.878 | 0.003 (-0.029 to 0.036) |

The two most susceptible of five classes (40% of the corridor) hold 87% of land flooded from 2000 to 2019; the 40% of the corridor nearest the river holds 80%.

## Contents

- `src/fs_common.py`: shared Earth Engine definitions (corridor, flood rule, predictors)
- `src/qgis_layout.py`: map layout helpers (QGIS 3.40)
- `scripts/01` to `09`: input checks, sampling, training and evaluation, diagnostics, mapping, exposure and figures
- `scripts/11_revision_ee.py`, `scripts/12_revision_local.py`: block bootstrap, ablation, subset tests, distance bands, Open Buildings check
- `data/`: district and tehsil boundaries (FAO GAUL; OCHA)
- `outputs/samples/`: training and test points with predictor values
- `outputs/*.json`, `outputs/revision/`: all reported statistics
- `outputs/layyah_score.tif`: 30 m susceptibility score for Layyah's corridor (quintile cut points in `outputs/layyah_map_stats.json`)
- `outputs/layyah_inventory.tif`: flood record used for the maps
- `outputs/model_logistic.joblib`: the final model

## Running

Earth Engine scripts need a service account key, referenced by `EE_SERVICE_ACCOUNT_KEY` in a local `.env` file (not included). Local scripts need Python 3.12 with `earthengine-api`, `python-dotenv`, `pandas`, `scikit-learn`, `xgboost`, `joblib` and `numpy`, for example:

```
uv run --with earthengine-api --with python-dotenv --with pandas --with scikit-learn --with xgboost --with joblib --with numpy python scripts/12_revision_local.py
```

## Licence and citation

Code under the MIT licence. Input datasets keep their own licences. Please cite the archived release (DOI on the Zenodo record) and the accompanying article once published.
