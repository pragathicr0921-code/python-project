# Notebook: house_price_regression.ipynb
# Requirements:
# pip install pandas numpy matplotlib scikit-learn seaborn openpyxl

import numpy as np
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv("train.csv")  # path to Kaggle 'train.csv' from House Prices competition

# Quick look
print("Shape:", df.shape)
display(df.head())

# 2. Target distribution
target = "SalePrice"
plt.figure(figsize=(6,4))
sns.histplot(df[target], kde=True)
plt.title("SalePrice distribution")
plt.show()

# If skewed, log-transform target
print("Skewness:", df[target].skew())
df["LogSalePrice"] = np.log1p(df[target])

plt.figure(figsize=(6,4))
sns.histplot(df["LogSalePrice"], kde=True)
plt.title("Log(SalePrice) distribution")
plt.show()

# Use LogSalePrice as target
y = df["LogSalePrice"]

# 3. Select features (example subset suitable for linear regression)
# We'll pick a mix of numeric and categorical features commonly useful.
features = [
    "OverallQual", "GrLivArea", "GarageCars", "GarageArea",
    "TotalBsmtSF", "1stFlrSF", "FullBath", "HalfBath",
    "TotRmsAbvGrd", "YearBuilt", "YearRemodAdd",
    "Neighborhood", "MSZoning", "Exterior1st", "BldgType"
]

# Keep only rows where target is present
X = df[features].copy()

# Quick missing value check
print("Missing values per feature:")
print(X.isna().sum())

# 4. Feature engineering: create 'HouseAge' and 'RemodAge' and drop YearBuilt/YearRemodAdd afterwards.
X["HouseAge"] = df["YrSold"] - X["YearBuilt"]
X["RemodAge"] = df["YrSold"] - X["YearRemodAdd"]
# drop YearBuilt and YearRemodAdd
X = X.drop(columns=["YearBuilt", "YearRemodAdd"])

# Update numeric and categorical lists
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
# treat 'Neighborhood', 'MSZoning','Exterior1st','BldgType' as categoricals
categorical_features = ["Neighborhood", "MSZoning", "Exterior1st", "BldgType"]

# Safety: remove categorical from numeric list if present
numeric_features = [c for c in numeric_features if c not in categorical_features]
print("Numerics:", numeric_features)
print("Categoricals:", categorical_features)

# 5. Build preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
], remainder="drop")

# 6. Build modeling pipeline with LinearRegression
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 7. Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Cross-validation baseline
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = -cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
print("CV RMSE (log-scale):", cv_scores.mean(), "±", cv_scores.std())

# 9. Fit on full training set
model.fit(X_train, y_train)

# 10. Evaluate on validation set
y_pred_log = model.predict(X_valid)
rmse_log = mean_squared_error(y_valid, y_pred_log, squared=False)
r2 = r2_score(y_valid, y_pred_log)
print(f"Validation RMSE (log scale): {rmse_log:.4f}, R2: {r2:.4f}")

# Convert back to original scale to interpret RMSE in dollars (approx)
y_valid_orig = np.expm1(y_valid)
y_pred_orig = np.expm1(y_pred_log)
rmse_orig = mean_squared_error(y_valid_orig, y_pred_orig, squared=False)
print(f"Validation RMSE (original dollars): {rmse_orig:.2f}")

# 11. Inspect coefficients (approx) — after ONEHOT it's harder to map directly; get feature names
ohe = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
ohe_feat_names = ohe.get_feature_names_out(categorical_features).tolist()
feature_names = numeric_features + ohe_feat_names
coefs = model.named_steps["regressor"].coef_
coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
display(coef_df.sort_values(by="coef", key=abs, ascending=False).head(20))

# 12. Try Ridge for regularization to reduce multicollinearity impact
ridge_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("ridge", Ridge(alpha=1.0))
])
ridge_cv = -cross_val_score(ridge_pipeline, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
print("Ridge CV RMSE (log):", ridge_cv.mean())

# Fit ridge and evaluate
ridge_pipeline.fit(X_train, y_train)
y_pred_ridge_log = ridge_pipeline.predict(X_valid)
rmse_ridge_log = mean_squared_error(y_valid, y_pred_ridge_log, squared=False)
print(f"Ridge validation RMSE (log): {rmse_ridge_log:.4f}")

# 13. Save the best pipeline
import joblib
joblib.dump(ridge_pipeline, "house_price_ridge_pipeline.pkl")
print("Saved model to house_price_ridge_pipeline.pkl")
