# Wellbeing ML

Machine-learning models that estimate wellbeing indicators from lifestyle and survey data,
served through a FastAPI backend with a web frontend.

> **Disclaimer:** educational project. Predictions are statistical estimates, not a medical
> or psychological diagnosis.

## Datasets (Kaggle)
| Dataset | Target | Status |
|---|---|---|
| Sleep Health and Lifestyle | Stress level | in progress |
| Student Depression | Depression (yes/no) | planned |
| Student Stress Factors | Stress level | planned |
| OSMI Mental Health in Tech | Treatment seeking | planned |

Cite each dataset and check its license on the Kaggle page.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/download_data.sh                        # needs ~/.kaggle/kaggle.json
python -m ml.datasets.sleep.train
```
Outputs go to `models/` (model file + metrics report).
