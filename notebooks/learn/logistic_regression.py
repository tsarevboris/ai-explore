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
# # Logistic Regression
# Using regression algorithm for classification using **sigmoid** function

# %% [markdown]
# Sigmoid:
# $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
#
# Estimated probability (vectorized):
# $$p = \sigma(X\theta)$$
#
# Prediction with threshold τ ∈ (0,1) (typically τ = 0.5):
# $$\hat{y} = \mathbf{1}[\,p \ge \tau\,]$$
#
# Cost (negative log-likelihood):
# $$J(\theta) = -\frac{1}{m}\left[y^\top \log p + (1 - y)^\top \log(1 - p)\right]$$

# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = load_iris(as_frame=True)
X = iris.data.iloc[:, :2].values  # Use only first two features for easy visualization
y = iris.target.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Test set accuracy: {accuracy:.2f}")


# %% [markdown]
#
