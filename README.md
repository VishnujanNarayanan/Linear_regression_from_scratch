#  Linear Regression from Scratch

This project demonstrates how to build, train, and visualize a linear regression model **from scratch using only NumPy** — no machine learning libraries like `scikit-learn`.

The goal is to show deep understanding by:

- Building both single-variable and multivariable linear regression
- Implementing gradient descent manually
- Comparing with closed-form solution (normal equation)
- Validating against `scikit-learn`
- Adding strong visualizations and clean code design
- Wrapping it into a reusable Python class

---

##  Project Structure

```
Linear_regression_scratch/
├── Linear_regression_class.py            # Reusable linear regression class (OOP)
├── Linear_regression_from_scratch.ipynb  # Single-variable regression notebook
├── multivariable_regression.ipynb        # Multivariable regression + visuals
├── env/                                  # Virtual environment (optional)
```

---

##  What I learnt

 Gradient Descent (step-by-step)  
 Mean Squared Error (MSE) loss  
 Closed-form Normal Equation  
 Vectorization tricks  
 How to visualize training performance  

---

## Features Implemented

-  **Single-variable regression**
-  **Multivariable regression (3 features)**
-  **Gradient descent implementation**
-  **Normal Equation (closed-form)**
-  **Comparison with `scikit-learn`**
-  **Loss curve and residuals**
-  **Reusable class for future use**
-  **Modular structure**

---

##  Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/<your-username>/Linear_regression_scratch.git
cd Linear_regression_scratch
```

### 2. Set Up Environment

Use your existing virtual environment or create one:

```bash
python -m venv env
env\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Run the Notebooks

Open either:

- `Linear_regression_from_scratch.ipynb` → single-variable regression  
- `multivariable_regression.ipynb` → multivariable regression

---

##  Example Class Usage

```python
from Linear_regression_class import LinearRegressionScratch

model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)
model.plot_loss()
model.plot_predictions()
model.plot_coefficients()
```

---

##  Future Add-ons (Optional)

- Add regularization (Ridge, Lasso)
- Polynomial features
- Learning rate comparison
- CLI tool for quick training
- Deploy as a microservice (Flask)

---

##  Acknowledgements

Inspired by Géron’s "Hands-On ML" and traditional ML coursework, but restructured for clarity, visuals, and reusability.

---

##  Author

**Vishnujan Narayanan**  
[GitHub](https://github.com/VishnujanNarayanan) • [Substack](https://substack.com/@vishnujannarayanan) • [LinkedIn](www.linkedin.com/in/vishnujan-narayanan)
