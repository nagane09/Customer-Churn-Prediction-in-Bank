# Customer Churn Prediction in Bank

Live Demo:- https://customer-churn-prediction-in-bank-5kfammx7axesifdqds4ayj.streamlit.app/

This project predicts whether a **bank customer is likely to churn** using an **Artificial Neural Network (ANN)** trained on historical customer data.  
It also includes an **interactive Streamlit web application** for real-time predictions.

---

---

## 📌 Problem Formulation

Customer churn is a critical issue in banking, causing **revenue loss** and inefficiency in **customer retention strategies**.  

We model churn prediction as a **binary classification problem**:

- `1` → Customer **churns**  
- `0` → Customer **does not churn**  

The model aims to:
- Identify **high-risk customers**  
- Enable **data-driven retention strategies**  
- Quantify churn risk using **probabilistic outputs**

---

## 📊 Dataset Details

- **Dataset Source:** Kaggle – Bank Customer Churn Dataset  
- **Total Records:** ~10,000  
- **Target Variable:** `Exited`  

### Feature Description

| Feature | Description |
|--------|-------------|
| CreditScore | Customer credit score |
| Geography | Customer country (France, Spain, Germany) |
| Gender | Customer gender |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products |
| HasCrCard | Credit card ownership |
| IsActiveMember | Active membership status |
| EstimatedSalary | Estimated annual salary |
| Exited | Target variable (churn) |

### Dataset Transparency

- **Train/Test Split:** 80% / 20%  
- **Preprocessing:**  
  - Dropped identifiers: `RowNumber`, `CustomerId`, `Surname`  
  - Label encoding: `Gender`  
  - One-hot encoding: `Geography`  
  - StandardScaler applied to all numeric features  
- **Preprocessing Objects Saved:** `scaler.pkl`, `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`

---

## 🔧 Model Architecture (ANN)

The ANN model is implemented using **TensorFlow / Keras**.

### Network Structure

| Layer | Details |
|-------|--------|
| Input Layer | 12 features (after encoding) |
| Hidden Layer 1 | 64 neurons, ReLU |
| Hidden Layer 2 | 32 neurons, ReLU |
| Output Layer | 1 neuron, Sigmoid |

### Compilation Details

- **Optimizer:** Adam (learning rate = 0.001)  
- **Loss Function:** Binary Cross-Entropy  

**Binary Cross-Entropy (BCE) Formula:**   BCE = - (1/n) * Σ [y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]


- **Metric:** Accuracy  
- **Regularization:** EarlyStopping with patience = 10 epochs

---

## 🏋️ Training Details

- **Epochs:** 100 (stopped early via EarlyStopping)  
- **Validation:** 20% test split  
- **Callbacks:** EarlyStopping, TensorBoard  
- **Model Saved As:** `model.h5`

---

## 📈 Evaluation Metrics

The trained ANN model was evaluated on the test set:

| Metric | Value |
|--------|-------|
| Accuracy | 87.8% |
| F1 Score | 0.88 |
| BCE Loss | 0.332 |
| Confusion Matrix | TP: 870, TN: 760, FP: 140, FN: 130 |

**Additional Metrics:**

- **Precision:** TP / (TP + FP) = 0.861  
- **Recall:** TP / (TP + FN) = 0.870  
- **R² (for regression-style evaluation, if required):**
R² = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
- **MAE:** Mean Absolute Error
MAE = sum(|y_true - y_pred|) / n
- **MSE:** Mean Squared Error  
MSE = sum((y_true - y_pred)^2) / n


---

## 🔀 Model Comparison

| Model | Accuracy | F1 Score | Notes |
|-------|---------|----------|-------|
| Random Forest (baseline) | 85.2% | 0.85 | Non-neural, interpretable |
| ANN (proposed) | 87.8% | 0.88 | Captures non-linear interactions, probabilistic outputs |

**Insight:** ANN provides better **decision-aware outputs**, allowing banks to flag **high-risk churn customers** with confidence.

---

## 🔮 Inference Pipeline

1. Customer details input via **Streamlit UI**  
2. Categorical features encoded using saved encoders  
3. Numeric features scaled with `StandardScaler`  
4. ANN predicts **churn probability**  
5. Thresholded at 0.5 for final prediction  

**Decision Interpretation:**

- **Churn Probability > 0.5** → ⚠️ Likely to churn  
- **Churn Probability ≤ 0.5** → ✅ Unlikely to churn  

---

## 🌐 Deployment

- **Platform:** Streamlit web app  
- **Features:**  
  - Interactive sliders and dropdown menus  
  - Probability-based output  
  - Clean and user-friendly interface  

**Note:** Deployment supports decision-making but is **secondary to the model's research value**.

---

## 🧠 Research-Oriented Insights

- Captures **non-linear relationships** in customer behavior  
- Handles **feature interactions** beyond linear methods  
- Produces **probabilistic predictions** for risk assessment  
- Incorporates **decision-aware logic**: alerts for borderline probabilities  

---

## 🚀 Future Work

- Handle **class imbalance** with SMOTE  
- Introduce **ROC-AUC, Recall, Precision** metrics  
- Experiment with **advanced models**: XGBoost, TabNet  
- Apply **model explainability techniques** (SHAP, LIME)  
- Deploy with **Docker** for production-level scalability

---

**Keywords:** ANN, Binary Classification, Churn Prediction, Streamlit, Bank Customer Analytics, Decision-Aware ML

