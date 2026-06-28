# 📈 Linear Regression from Scratch — Pure NumPy

> Building Machine Learning from the ground up — no shortcuts, just math and code.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Mathematical%20Core-orange?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualizations-red?logo=matplotlib)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-Verification%20Only-green?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 🧠 What Is This Project?

This project implements **Linear Regression completely from scratch** using only NumPy — no ML libraries for training.

The dataset simulates the relationship between **study hours and exam marks**, with realistic noise added to reflect real-world data.

The goal is simple:
> Understand *how* Machine Learning models actually learn — not just *use* them.

---

## 🚀 What I Built

```
Raw Data  →  Normalization  →  Gradient Descent  →  Learned Line  →  Validation vs Sklearn
```

| Component | Description |
|-----------|-------------|
| 📐 **Prediction** | `ŷ = w·x + b` |
| 📉 **Cost Function** | Mean Squared Error (MSE) |
| ⚙️ **Optimizer** | Gradient Descent from scratch |
| 📊 **Evaluation** | MSE + R² Score |
| ✅ **Verification** | Side-by-side check with Sklearn |

---

## 📊 Sample Training Output

```
Training started...

Epoch 0    | Cost = 592.436
Epoch 100  | Cost = 9.683
...
Training finished!

Final Learned Line:
marks = 13.066 × x_norm + 35.611
```

---

## 📉 Visualizations

### Regression Fit
<img width="840" height="600" alt="Regression Fit" src="https://github.com/user-attachments/assets/89527eda-47d9-4dd1-99a3-2e6997bce3ea" />

### Loss Curve (Cost vs Epochs)
<img width="840" height="600" alt="Loss Curve" src="https://github.com/user-attachments/assets/1294571f-c828-429a-b1a7-917abd292ddb" />

### 3D Cost Surface + Gradient Descent Path
<img width="960" height="720" alt="3D Cost Surface" src="https://github.com/user-attachments/assets/a3062d4d-e394-4b8e-8eee-7e69551a70a9" />

---

## 🔑 Core ML Concepts

### Linear Regression
```
y = w·x + b
```
- `w` → weight (slope) — how much marks change per study hour
- `b` → bias (intercept) — baseline marks

### Mean Squared Error
```
MSE = (1/n) × Σ(y - ŷ)²
```
Lower MSE = better predictions.

### Gradient Descent
```
w = w - α × (∂L/∂w)
b = b - α × (∂L/∂b)
```
Repeated until the cost stops decreasing.

**The Loop:**
```
Predict → Measure Error → Compute Gradients → Update Weights → Repeat
```

---

## 🗂️ Project Structure

```
Linear-Regression-From-Scratch/
│
├── linear_regression_scratch.py   # Full implementation
├── plot1_regression_fit.png       # Best-fit line visualization
├── plot2_loss_curve.png           # Cost vs Epochs
├── plot3_3d_cost_surface.png      # 3D gradient descent path
└── README.md
```

---

## ⚡ How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/Linear-Regression-From-Scratch.git
cd Linear-Regression-From-Scratch

# Install dependencies
pip install numpy pandas matplotlib scikit-learn

# Run the project
python linear_regression_scratch.py
```

---

## 🛠️ Tech Stack

| Library | Role |
|---------|------|
| **NumPy** | Math, arrays, gradient computations |
| **Pandas** | Dataset creation and handling |
| **Matplotlib** | All visualizations (2D + 3D) |
| **Scikit-Learn** | Final verification only — not used for training |

---

## 📚 What I Learned

- How gradient descent actually minimizes loss — step by step
- Why feature normalization matters before training
- The math behind MSE and why we differentiate it
- How to visualize the loss surface in 3D
- That Sklearn's `LinearRegression` is doing exactly this under the hood

---

## 🗺️ My Learning Journey

```
Python → NumPy → Pandas → Matplotlib → Math for AI → ML from Scratch
```

This project is **Step 1** of building ML from the ground up.

**Next steps planned:**
- [ ] Multiple Linear Regression
- [ ] Polynomial Regression
- [ ] Logistic Regression from Scratch
- [ ] L1 / L2 Regularization
- [ ] Neural Network — one neuron at a time

---

## 👩‍💻 Author

**Manahil**
- 🐙 GitHub:  https://github.com/Manahilch18 
- 💼 LinkedIn:  
www.linkedin.com/in/manahil-ishfaq-673439322

---

> *"The best way to understand a machine learning algorithm is to build it yourself."*

⭐ If this helped you, consider giving it a star — it means a lot!
