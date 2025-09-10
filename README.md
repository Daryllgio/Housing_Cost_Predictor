# **House Cost Prediction**

## **Overview**

A practical machine-learning workflow that predicts **median house value** using the classic **California Housing** dataset. The notebook walks through **data loading → cleaning → exploratory data analysis (EDA) → encoding → model training & evaluation** with **Linear Regression** and **Random Forest** baselines.

> Notebook: `HouseCostPrediction.ipynb`
> Data: `housing.csv` (columns include: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `median_house_value`, `ocean_proximity`)

---

## **Highlights**

* **Straightforward preprocessing** — handle missing values; one-hot encode `ocean_proximity`.
* **EDA first** — distributions, correlations, and a geo scatter to understand location effects.
* **Two baseline models** — fast, interpretable **Linear Regression** and non-linear **Random Forest Regressor**.

---

## **Results (example run)**

Using an 80/20 split with `random_state=42`:

| Model             |  R² (test) | RMSE (USD) |
| ----------------- | ---------: | ---------: |
| Linear Regression | **\~0.65** |  **≈ 69k** |
| Random Forest     | **\~0.83** |  **≈ 49k** |

> Numbers will vary slightly with preprocessing choices and seeds; keep this table in sync with the notebook’s final metrics.

---

## **Exploratory Data Analysis (EDA)**

**Distributions (numeric features)**

<p align="center">
  <img src="https://github.com/user-attachments/assets/a859b302-9127-44d3-8712-f700f2965248" alt="Feature distributions (histograms)" width="920">
</p>

**Correlation matrix**

<p align="center">
  <img src="https://github.com/user-attachments/assets/cb668055-1fa7-4e8a-a4fc-81e108fc07af" alt="Correlation heatmap" width="920">
</p>

**Geospatial view**

<p align="center">
  <img src="https://github.com/user-attachments/assets/bc4f343a-1f79-49c5-9f46-a0ebd429c600" alt="Geo scatter colored by median house value" width="920">
</p>

---

## **Project Structure**

```
.
├─ HouseCostPrediction.ipynb   # end-to-end workflow
├─ housing.csv                 # California housing dataset
└─ README.md
```

---

## **Environment**

* Python 3.9+
* pandas, numpy
* scikit-learn
* matplotlib, seaborn
* Jupyter

Quick install:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## **Quick Start**

```bash
# 1) Clone and enter
git clone <your-repo-url>
cd <repo>

# 2) (Optional) create & activate a venv
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

# 3) Install deps
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 4) Run the notebook
jupyter notebook HouseCostPrediction.ipynb
```

---

## **Modeling Details**

* **Preprocessing**

  * Handle missing values (e.g., drop rows with NA in `total_bedrooms` or impute).
  * One-hot encode `ocean_proximity` with `pd.get_dummies`.
* **Split**

  * `train_test_split(X, y, test_size=0.2, random_state=42)`
* **Models**

  * `LinearRegression()`
  * `RandomForestRegressor(random_state=42)`
* **Scoring**

  * Primary: **R²**
  * Also: **RMSE** (`mean_squared_error(..., squared=False)`)

---

## **What You’ll Learn**

* Turning raw tabular data into a **model-ready matrix** (encoding, NA handling).
* Reading patterns from **EDA visuals**.
* Comparing **linear** vs. **non-linear** baselines.
* Reporting results clearly (R², RMSE, plots).

---

## **Roadmap**

* **Imputation & scaling** with a `ColumnTransformer`.
* **Feature engineering**: `rooms_per_household`, `bedrooms_per_room`, `population_per_household`, interactions with latitude/longitude.
* **Modeling**: hyperparameter tuning (`GridSearchCV`); try Gradient Boosting / XGBoost / LightGBM.
* **Evaluation**: k-fold CV, residual analysis, log-transform of target to reduce heteroscedasticity.
* **Packaging**: export model with `joblib`, add a small `predict.py` or FastAPI endpoint.

