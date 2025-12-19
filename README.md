# Customer Churn Prediction in Bank

Live Demo:- https://customer-churn-prediction-in-bank-5kfammx7axesifdqds4ayj.streamlit.app/

This project uses **Machine Learning / Deep Learning** to **predict whether a bank customer will churn** — i.e., stop being a customer — based on historical customer data. Predicting churn helps banks identify at-risk customers and take proactive steps to retain them.

---

## 📌 1. Project Objective

The main goal is to **build a model that predicts customer churn** based on features like age, balance, credit score, tenure, and more.

**Typical Workflow:**

1. Load dataset
2. Exploratory Data Analysis (EDA)
3. Clean and preprocess data
4. Encode categorical variables
5. Scale numerical features
6. Train ML or DL model
7. Evaluate performance
8. Save model and make predictions

---

## 📁 2. Repository Structure

| **File / Folder**          | **Description**                                                                                |
| -------------------------- | ---------------------------------------------------------------------------------------------- |
| `.gitignore`               | Files and folders to ignore in Git.                                                            |
| `README.md`                | Project description (this file).                                                               |
| `app.py`                   | Python app (Streamlit/Flask) to input customer data and predict churn using the trained model. |
| `code.ipynb`               | Main notebook: data loading, preprocessing, training, and evaluation.                          |
| `data.csv`                 | Dataset containing customer features and target column (`Exited`).                             |
| `label_encoder_gender.pkl` | Label Encoder for the gender feature.                                                          |
| `onehot_encoder_geo.pkl`   | One-Hot Encoder for geography/country feature.                                                 |
| `scaler.pkl`               | Scaler for numeric features (e.g., StandardScaler).                                            |
| `model.h5`                 | Trained Keras/TensorFlow deep learning model.                                                  |
| `pred.ipynb`               | Notebook for running predictions on new customer data.                                         |
| `requirements.txt`         | Python dependencies required for the project.                                                  |

---

## 🧠 3. Machine Learning / Deep Learning Concepts

**Model Type:**

* Keras/TensorFlow neural network (binary classifier)
* Architecture (typical):
  * Input layer
  * 1–3 Dense hidden layers with ReLU activation
  * Output layer with Sigmoid activation
* Loss function: Binary Cross-Entropy

**Preprocessing:**

* Label Encoding (`gender`)
* One-Hot Encoding (`geography`)
* Feature Scaling (`scaler.pkl`)

**Validation:**

* Validation accuracy: **0.8640** (~86.4% correct predictions)

---

## 🧪 5. Results Interpretation

**Validation Accuracy:** 0.8640  
Indicates the model predicts churn correctly **86.4%** of the time.

**Notes:**

* Accuracy is only one metric; consider also:
  * Precision
  * Recall
  * F1-score
  * ROC-AUC
* Model performance can improve with:
  * Hyperparameter tuning
  * Class imbalance handling (SMOTE, undersampling, etc.)
  * Alternative models (Random Forest, XGBoost, ensemble methods)

---

## 🏁 6. Using the App (`app.py`)

The app allows users to input features such as:

* Age
* Credit score
* Geography
* Balance
* Number of products

Then, it runs:

```python
model = load_model('model.h5')
gender_encoder = pickle.load(...)
geo_encoder = pickle.load(...)
scaler = pickle.load(...)

output = model.predict(processed_input)
```

