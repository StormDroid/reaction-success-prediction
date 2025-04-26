# Reaction Success Prediction 🚀

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://choosealicense.com/licenses/mit/)

This project predicts whether a chemical reaction will succeed based on reaction conditions like temperature, pH, concentration, and pressure using machine learning models.

---

## 📊 Project Workflow

- Load and explore reaction data
- Handle missing or non-numeric values
- Visualize feature distributions with histograms
- Scale features with StandardScaler
- Split data into training and testing sets
- Train and evaluate:
  - Logistic Regression
  - Support Vector Classifier (SVC)
  - XGBoost Classifier
- Plot confusion matrices for model evaluation

---

## 🛠 Technologies Used

- Python 3.11
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

## 🚀 How to Run the Project

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/reaction-success-prediction.git
    cd reaction-success-prediction
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Python script:
    ```bash
    python reaction_success_prediction.py
    ```

---

## 📂 Dataset Overview

The dataset `reaction_data.csv` contains:
- **Temperature** (°C)
- **pH** (acidity/basicity)
- **Concentration** (mol/L)
- **Pressure** (atm)
- **Catalyst** (binary 0 or 1)
- **Reaction_Success** (target: 1 = success, 0 = failure)

---

## ✅ Project Results

| Model                | Accuracy  |
|----------------------|-----------|
| Logistic Regression  | 100%      |
| SVC                  | 100%      |
| XGBoost              | 100%      |

> *Note: Due to small dataset size (10 samples), models perform perfectly. With larger datasets, additional tuning would be needed.*

---

## 🌱 Future Improvements

- Add more reaction data for better model generalization
- Use GridSearchCV for hyperparameter tuning
- Deploy the best model using Flask or FastAPI
- Build a small web app to predict reaction outcomes

---

## 📜 License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

---
