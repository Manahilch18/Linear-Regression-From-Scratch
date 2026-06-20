"""
============================================================
Linear Regression from Scratch + Gradient Descent Visualizer
============================================================
Beginner-friendly project.
No sklearn used for the actual math -- we build everything
using only NumPy. sklearn is used ONLY at the end, to check
that our scratch model gives a similar answer.

Topics practiced: NumPy, Pandas, Matplotlib, basic calculus
(gradients), and the core idea behind every ML model:
    "guess -> measure error -> adjust -> repeat"
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for the 3D bonus plot

# Make results repeatable (same random numbers every run)
np.random.seed(42)


# ------------------------------------------------------------
# STEP 1: CREATE A SIMPLE DATASET (Study Hours -> Marks)
# ------------------------------------------------------------
# We create FAKE data on purpose, so we already roughly know
# the answer (helps you trust that the algorithm is working).
#
# Real relationship we hide inside the data: marks = 5*hours + 10 (+ noise)

n_samples = 60
hours = np.random.uniform(1, 10, n_samples)          # study hours: 1 to 10
noise = np.random.normal(0, 5, n_samples)             # random noise
marks = 5 * hours + 10 + noise                         # true relationship + noise

# Put it in a Pandas DataFrame (good practice -- real datasets come this way)
df = pd.DataFrame({"hours": hours, "marks": marks})
print("First 5 rows of our dataset:")
print(df.head())
print("\nDataset shape:", df.shape)


# ------------------------------------------------------------
# STEP 2: PREPARE THE DATA FOR MATH (NumPy arrays)
# ------------------------------------------------------------
X = df["hours"].values   # input feature
y = df["marks"].values   # target we want to predict

# Feature scaling (normalization) -- makes gradient descent
# converge faster and more reliably. This is standard ML practice.
X_mean, X_std = X.mean(), X.std()
X_norm = (X - X_mean) / X_std


# ------------------------------------------------------------
# STEP 3: DEFINE THE MODEL
# ------------------------------------------------------------
# Our model is the simplest possible line:
#       prediction = w * x + b
# w (weight/slope) and b (bias/intercept) are the two numbers
# the algorithm will learn.

def predict(X, w, b):
    return w * X + b


# ------------------------------------------------------------
# STEP 4: DEFINE THE COST FUNCTION (Mean Squared Error)
# ------------------------------------------------------------
# This tells us "how wrong" our current line is.
# Lower cost = better fit.

def compute_cost(X, y, w, b):
    n = len(y)
    predictions = predict(X, w, b)
    errors = predictions - y
    cost = (1 / (2 * n)) * np.sum(errors ** 2)
    return cost


# ------------------------------------------------------------
# STEP 5: GRADIENT DESCENT (the heart of the project)
# ------------------------------------------------------------
# Gradients tell us which direction to move w and b to REDUCE
# the cost. We take small steps (learning_rate) repeatedly.

def gradient_descent(X, y, w, b, learning_rate, epochs):
    n = len(y)
    cost_history = []
    w_history = []
    b_history = []

    for i in range(epochs):
        predictions = predict(X, w, b)
        errors = predictions - y

        # Partial derivatives of cost w.r.t. w and b
        dw = (1 / n) * np.sum(errors * X)
        db = (1 / n) * np.sum(errors)

        # Update step: move opposite to the gradient
        w = w - learning_rate * dw
        b = b - learning_rate * db

        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)
        w_history.append(w)
        b_history.append(b)

        # print progress every 100 steps
        if i % 100 == 0:
            print(f"Epoch {i:4d} | cost = {cost:8.3f} | w = {w:.3f} | b = {b:.3f}")

    return w, b, cost_history, w_history, b_history


# ------------------------------------------------------------
# STEP 6: TRAIN THE MODEL
# ------------------------------------------------------------
w_init, b_init = 0.0, 0.0       # start with a random/zero guess
learning_rate = 0.1
epochs = 1000

print("\nTraining started...\n")
w_final, b_final, cost_history, w_hist, b_hist = gradient_descent(
    X_norm, y, w_init, b_init, learning_rate, epochs
)
print(f"\nTraining finished!")
print(f"Final learned line (on normalized X): marks = {w_final:.3f} * x_norm + {b_final:.3f}")


# ------------------------------------------------------------
# STEP 7: VISUALIZATION 1 -- Regression line fit on real data
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(X, y, color="steelblue", label="Actual data")

# convert normalized-space predictions back to real hours for plotting
X_line = np.linspace(X.min(), X.max(), 100)
X_line_norm = (X_line - X_mean) / X_std
y_line = predict(X_line_norm, w_final, b_final)

plt.plot(X_line, y_line, color="red", linewidth=2, label="Fitted line (our model)")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression from Scratch: Study Hours vs Marks")
plt.legend()
plt.tight_layout()
plt.savefig(" plot1_regression_fit.png", dpi=120)
plt.close()
print("\nSaved: plot1_regression_fit.png")


# ------------------------------------------------------------
# STEP 8: VISUALIZATION 2 -- Loss curve (cost vs epochs)
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(range(epochs), cost_history, color="darkorange")
plt.xlabel("Epoch (iteration)")
plt.ylabel("Cost (MSE)")
plt.title("How the Cost Decreases as the Model Learns")
plt.tight_layout()
plt.savefig(" plot2_loss_curve.png", dpi=120)
plt.close()
print("Saved: plot2_loss_curve.png")


# ------------------------------------------------------------
# STEP 9: VISUALIZATION 3 (BONUS) -- 3D Cost Surface + descent path
# ------------------------------------------------------------
# This shows the "bowl shape" of the cost function and the path
# gradient descent took to reach the bottom (minimum cost).

w_range = np.linspace(min(w_hist) - 2, max(w_hist) + 2, 60)
b_range = np.linspace(min(b_hist) - 2, max(b_hist) + 2, 60)
W, B = np.meshgrid(w_range, b_range)

Cost_surface = np.zeros(W.shape)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Cost_surface[i, j] = compute_cost(X_norm, y, W[i, j], B[i, j])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(W, B, Cost_surface, cmap="viridis", alpha=0.6)

# Plot the path gradient descent actually took
ax.plot(w_hist, b_hist, cost_history, color="red", linewidth=2, label="Gradient Descent Path")
ax.scatter(w_hist[0], b_hist[0], cost_history[0], color="black", s=40, label="Start")
ax.scatter(w_hist[-1], b_hist[-1], cost_history[-1], color="lime", s=40, label="End (minimum)")

ax.set_xlabel("w (weight)")
ax.set_ylabel("b (bias)")
ax.set_zlabel("Cost")
ax.set_title("3D Cost Surface & Gradient Descent Path")
ax.legend()
plt.tight_layout()
plt.savefig("plot3_3d_cost_surface.png", dpi=120)
plt.close()
print("Saved: plot3_3d_cost_surface.png")


# ------------------------------------------------------------
# STEP 10: COMPARE WITH SKLEARN (sanity check only)
# ------------------------------------------------------------
from sklearn.linear_model import LinearRegression

sk_model = LinearRegression()
sk_model.fit(X.reshape(-1, 1), y)
sk_pred = sk_model.predict(X.reshape(-1, 1))

# our model's predictions on the original (un-normalized) X
our_pred = predict(X_norm, w_final, b_final)

from sklearn.metrics import mean_squared_error, r2_score
print("\n--- Comparison: Our Scratch Model vs sklearn ---")
print(f"Our model     -> MSE: {mean_squared_error(y, our_pred):.3f} | R2: {r2_score(y, our_pred):.3f}")
print(f"sklearn model -> MSE: {mean_squared_error(y, sk_pred):.3f} | R2: {r2_score(y, sk_pred):.3f}")
print(f"\nsklearn coefficients -> slope: {sk_model.coef_[0]:.3f}, intercept: {sk_model.intercept_:.3f}")
print("(Note: our w,b are in normalized space, so they won't look the same")
print(" as sklearn's raw slope/intercept -- but the MSE and R2 should match closely.)")

print("\nAll done! Check the  linreg_project folder for 3 PNG plots.")
