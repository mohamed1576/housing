# California Housing Price Prediction

An end-to-end Machine Learning pipeline utilizing `RandomForestRegressor` to predict California housing prices. The project implements optimized feature scaling and engineering via a custom data pipeline, with hyperparameters tuned using `GridSearchCV` to achieve a highly accurate and robust baseline.

---

## 📌 Project Overview
Predicting real estate values requires a solid understanding of structural and geographic features. This repository hosts a clean, production-ready Machine Learning workflow that transforms raw California housing data, handles missing values, applies scaling, and optimizes a ensemble regressor to deliver high-quality price predictions.

### Key Highlights:
* **Model Choice:** Trained a Robust `RandomForestRegressor` optimized via `GridSearchCV`.
* **Performance Metric:** Secured a final Root Mean Squared Error (RMSE) of **$50,415**.
* **Data Pipelines:** Bundled preprocessing and scaling steps into an isolated transformer pipeline to entirely prevent data leakage.

---

## 📁 Repository Structure
* `Housing_exploration.ipynb` - Main Jupyter notebook covering Exploratory Data Analysis (EDA), statistical visualization, pipeline building, and model tuning.
* `my_data_pipeline.pkl` - The serialized feature engineering and preprocessing pipeline.
* `my_random_forest_model.pkl` - The final, optimized trained Random Forest model ready for inference.
* `.gitignore` - Standard configuration file to exclude temporary system and Jupyter files from tracking.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Environment:** Jupyter Notebook (100%)
* **ML Libraries:** Scikit-Learn, Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib

---

## ⚙️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mohamed1576/housing.git](https://github.com/mohamed1576/housing.git)
   cd housing

   🧠 Methodology & Pipeline
1. Exploratory Data Analysis (EDA): Visualized geographical data maps, analyzed feature correlations, and inspected target attribute skewness.

2. Data Cleansing & Preprocessing: Handled missing continuous variables and treated statistical outliers to ensure safe model convergence.

3. Feature Engineering & Leakage Prevention: Built and saved a clean transformation pipeline (my_data_pipeline.pkl) separating training transformations from validation data.

4. Hyperparameter Tuning: Conducted grid search validation on multiple forest estimators to minimize final prediction variance.

5. Evaluation: Evaluated the optimal estimator against the unseen test set, resulting in a final model error validation of RMSE: $50,415.

   
