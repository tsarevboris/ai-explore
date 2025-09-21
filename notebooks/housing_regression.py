# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Housing Regression

# %% [markdown]
# ## Read the Data

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# %% [markdown]
# ### Read File

# %%
train_data_file = "../data/raw/house-prices-advanced-regression-techniques/train.csv"
train_data = pd.read_csv(train_data_file)
train_data.head()

# %% [markdown]
# ### Prepare Train Data

# %%
features = ["GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
X = train_data[features]
y = train_data["SalePrice"]

# %% [markdown]
# ## Fit the Model

# %% [markdown]
# ### Linear Regression

# %%
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print(str.format("Predictions: {}", y_pred[:5].round(0)))
print(str.format("Actual:      {}", y.values[:5].round(0)))

# %% [markdown]
# ### Measure Model Performance

# %%
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y, y_pred))
print(str.format("RMSE: {}", rmse.round(0)))

# %%
from sklearn.model_selection import cross_val_score

cross_rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10))
pd.Series(cross_rmse).describe()

# %% [markdown]
# ## Save Predictions

# %%
# Read test data
test_data_file = "../data/raw/house-prices-advanced-regression-techniques/test.csv"
test_data = pd.read_csv(test_data_file)

X_test = test_data[features]
X_test_preprocessed = X_test.fillna({
    "GrLivArea": 0,
    "OverallQual": 0,
    "GarageCars": 0,
    "TotalBsmtSF": 0,
    "FullBath": 0,
    "YearBuilt": X_test["YearBuilt"].median()
})

y_test_pred = model.predict(X_test_preprocessed)

submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": y_test_pred
})
submission.to_csv("../data/processed/house_prices_predictions.csv", index=False)

# %%
