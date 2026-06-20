# 📈 Linear Regression from Scratch using NumPy

## Overview

This project implements Linear Regression completely from scratch using NumPy without relying on machine learning libraries for training.

The goal is to understand how Machine Learning models learn by manually implementing:

* Linear Regression
* Mean Squared Error (MSE)
* Gradient Descent Optimization
* Model Evaluation
* Data Visualization

The project uses a synthetic dataset representing the relationship between study hours and exam marks.

---

## Features

### Data Generation

* Generate study hours and marks dataset
* Add realistic noise to simulate real-world data

### Data Processing

* DataFrame creation using Pandas
* Feature normalization
* NumPy array operations

### Machine Learning from Scratch

* Prediction function
* Cost Function (MSE)
* Gradient Descent algorithm
* Weight and bias updates

### Visualizations

* Scatter Plot of Study Hours vs Marks
* Linear Regression Best Fit Line
* Loss Curve (Cost vs Epochs)
* 3D Cost Surface Visualization
* Gradient Descent Path Visualization

### Model Validation

* Compare results with sklearn LinearRegression
* Calculate MSE
* Calculate R² Score

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn (only for verification)

---

## Project Structure

```text
Linear-Regression-From-Scratch/
│
├── linear_regression_scratch.py
├── plot1_regression_fit.png
├── plot2_loss_curve.png
├── plot3_3d_cost_surface.png
├── README.md
```

---

## Machine Learning Concepts Learned

### Linear Regression

Linear Regression finds the best fitting straight line:

y = wx + b

Where:

* w = weight (slope)
* b = bias (intercept)

---

### Cost Function

Mean Squared Error (MSE) measures prediction error.

Lower cost indicates a better model.

---

### Gradient Descent

Gradient Descent repeatedly updates model parameters to minimize error.

Process:

1. Predict
2. Calculate Error
3. Compute Gradients
4. Update Parameters
5. Repeat

---

## Sample Output

Training started...

Epoch 0 | Cost = 592.436

Epoch 100 | Cost = 9.683

Training finished!

Final learned line:

marks = 13.066 × x_norm + 35.611

---

## Learning Outcomes

Through this project I learned:

* NumPy mathematical operations
* Data preprocessing
* Feature scaling
* Gradient Descent optimization
* Cost functions
* Linear Regression mathematics
* Data visualization with Matplotlib
* Model evaluation techniques

---

## Why This Project Matters

This project helped me understand the mathematics behind Machine Learning instead of simply using pre-built libraries.

Building Linear Regression from scratch provides a strong foundation for:

* Machine Learning
* Deep Learning
* Neural Networks
* Artificial Intelligence

---

## Future Improvements

* Multiple Linear Regression
* Polynomial Regression
* Logistic Regression
* L1/L2 Regularization
* Interactive GUI Dashboard

---

## Author

Manahil

Learning Journey:

Python → NumPy → Pandas → Matplotlib → Mathematics for AI → Machine Learning

---

⭐ If you find this project useful, consider giving it a star.
