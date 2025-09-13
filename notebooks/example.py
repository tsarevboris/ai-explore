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

# %%
import numpy as np

import matplotlib.pyplot as plt

# Sample numeric sequence
nums = list(range(10))
squares = [n**2 for n in nums]

plt.figure(figsize=(10,4))

# Line plot
plt.subplot(1, 2, 1)
plt.plot(nums, squares, marker='o')
plt.title('Squares')
plt.xlabel('n')
plt.ylabel('n^2')
plt.grid(True)

# Sine curve
x = np.linspace(0, 2 * np.pi, 300)
plt.subplot(1, 2, 2)
plt.plot(x, np.sin(x), color='crimson')
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)

plt.tight_layout()
plt.show()
