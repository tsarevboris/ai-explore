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
# # Polynomial Regression

# %% [markdown]
# Add powers of each feature as a new feature and train as linear model

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Test Data

# %%
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 1 * X**2 + X - 12 + np.random.randn(m, 1)

# Plotting the data
plt.scatter(X, y, s=18)  # slightly smaller points
plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# %% [markdown]
# ## Add Power Features

# %%
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# %% [markdown]
# ## Regression

# %%
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Print model parameters
print("Intercept (real is -12):", lin_reg.intercept_)
print("Coefficients (real is [1, 1]):", lin_reg.coef_)

# Plot predictions
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.scatter(X, y, s=18)
plt.plot(X_new, y_new, color='r', linewidth=2)
plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.grid(True, linestyle='--', alpha=0.5)

# set limits for better visualization
plt.xlim(-3.5, 3.5)
plt.ylim(-14, 0)

plt.show()

# %% [markdown]
# ## Learning Curves

# %% [markdown]
# Plot of training error and validation error against training iteration 

# %% [markdown]
# ### Underfitting Model

# %%
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(LinearRegression(), X, y, train_sizes=np.linspace(0.1, 1.0, 80), cv=5, scoring='neg_mean_squared_error')
train_errors = -train_scores.mean(axis=1)
val_errors = -val_scores.mean(axis=1)

plt.plot(train_sizes, train_errors, "r-+", linewidth=1, label="Training error")
plt.plot(train_sizes, val_errors, "b-", linewidth=2, label="Validation error")
plt.xlabel("Training set size")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# %% [markdown]
# ### Overfitting Model

# %%
from sklearn.pipeline import make_pipeline

polynomial_pipeline = make_pipeline(PolynomialFeatures(degree=10, include_bias=False), LinearRegression())
train_sizes, train_scores, val_scores = learning_curve(polynomial_pipeline, X, y, train_sizes=np.linspace(0.1, 1.0, 80), cv=5, scoring='neg_mean_squared_error')
train_errors = -train_scores.mean(axis=1)
val_errors = -val_scores.mean(axis=1)

plt.plot(train_sizes, train_errors, "r-+", linewidth=1, label="Training error")
plt.plot(train_sizes, val_errors, "b-", linewidth=2, label="Validation error")
plt.xlabel("Training set size")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0, 3)
plt.show()

# %% [markdown]
# ### Good Fitting Model

# %%
square_pipeline = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
train_sizes, train_scores, val_scores = learning_curve(square_pipeline, X, y, train_sizes=np.linspace(0.1, 1.0, 80), cv=5, scoring='neg_mean_squared_error')
train_errors = -train_scores.mean(axis=1)
val_errors = -val_scores.mean(axis=1)

plt.plot(train_sizes, train_errors, "r-+", linewidth=1, label="Training error")
plt.plot(train_sizes, val_errors, "b-", linewidth=2, label="Validation error")
plt.xlabel("Training set size")
plt.ylabel("Mean Squared Error")   
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0, 3)
plt.show()

# %% [markdown]
# ## Bias/Variance Trade-Off
#
# - **Bias** - wrong assumprions, high-bias leads to underfitting
# - **Variance** - model's excessive sensitivity, high variance leads to overfitting
# - **Irreducible error** - noisinesss of the data 
