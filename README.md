# Customer Churn Prediction in Bank

Live Demo:- https://customer-churn-prediction-in-bank-5kfammx7axesifdqds4ayj.streamlit.app/


This project predicts whether a **bank customer is likely to churn** using an **Artificial Neural Network (ANN)** trained on historical customer data.  
It also includes an **interactive Streamlit web application** for real-time predictions.

---

## 📌 Project Overview

Customer churn is a critical business problem in the banking sector.  
This project formulates churn prediction as a **binary classification problem**:

- `1` → Customer **churns**
- `0` → Customer **does not churn**

The solution helps banks:
- Identify **high-risk customers**
- Improve **customer retention strategies**
- Reduce **revenue loss**
- Enable **data-driven decision-making**

---

## 📊 Dataset Information

- **Dataset**: Bank Customer Churn Dataset  
- **Source**: Kaggle  
- **Total Records**: ~10,000  
- **Target Variable**: `Exited`

### 📁 Feature Description

| Feature | Description |
|------|------------|
| CreditScore | Customer credit score |
| Geography | Customer country |
| Gender | Customer gender |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products |
| HasCrCard | Credit card ownership |
| IsActiveMember | Active membership status |
| EstimatedSalary | Estimated annual salary |
| Exited | Target variable (Churn) |

---

## 🧹 Data Preprocessing

### ✔️ Feature Removal
Removed non-informative identifiers:
- `RowNumber`
- `CustomerId`
- `Surname`

### ✔️ Encoding
- **Gender** → Label Encoding
- **Geography** → One-Hot Encoding

### ✔️ Feature Scaling
- Applied **StandardScaler** to numerical features
- Ensures faster convergence during ANN training

All preprocessing objects are **saved and reused** during inference:
- `scaler.pkl`
- `label_encoder_gender.pkl`
- `onehot_encoder_geo.pkl`

---

## 🔀 Train-Test Split

- **Training Set**: 80%
- **Testing Set**: 20%
- **Random State**: 42

---

## 🧠 Model Architecture (ANN)

The model is implemented using **TensorFlow / Keras**.

### 🔧 Network Structure

| Layer | Details |
|-----|--------|
| Input Layer | Number of features after encoding |
| Hidden Layer 1 | 64 neurons, ReLU |
| Hidden Layer 2 | 32 neurons, ReLU |
| Output Layer | 1 neuron, Sigmoid |

### ⚙️ Compilation Details

- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: Binary Cross-Entropy
- **Metric**: Accuracy

### ⏹️ Regularization Strategy
- **EarlyStopping** with patience = 10
- Prevents overfitting and restores best weights

---

## 🏋️ Model Training

- **Epochs**: Up to 100
- **Callbacks**:
  - EarlyStopping
  - TensorBoard (for training visualization)

The final trained model is saved as:
```text
model.h5
````

## 🔮 Model Inference Pipeline

During prediction, the following steps are executed:

1. User inputs customer details via the **Streamlit UI**
2. Categorical features are **encoded** using saved encoders
3. Numerical features are **scaled** using `StandardScaler`
4. The ANN model outputs a **churn probability**
5. The probability is **thresholded at 0.5** to generate the final decision

---

## 🌐 Streamlit Web Application

The Streamlit application provides an interactive interface that allows users to:

- Input customer details
- Obtain **real-time churn probability**
- Interpret churn risk in a simple and intuitive manner

---

## 🧩 Application Features

- Interactive sliders and dropdown menus
- Probability-based prediction output
- Clean, minimal, and user-friendly UI

---

## 📈 Output Interpretation

- **Churn Probability > 0.5** → ⚠️ Customer is **likely to churn**
- **Churn Probability ≤ 0.5** → ✅ Customer is **unlikely to churn**


---

## 🧠 Why ANN Works Well for This Problem

- Captures **non-linear relationships** in customer behavior
- Handles **complex feature interactions**
- Performs well with **standardized numerical inputs**
- Produces **probabilistic outputs**, useful for churn risk assessment

---

## 🚀 Future Improvements

- Handle class imbalance using **SMOTE**
- Add evaluation metrics such as **ROC-AUC** and **Recall**
- Experiment with advanced models (XGBoost, TabNet)
- Introduce **model explainability** using SHAP
- Deploy using **Docker** for production readiness
