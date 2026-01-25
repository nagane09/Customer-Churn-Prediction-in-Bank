# Customer Churn Prediction Using Artificial Neural Networks  
**A Comparative and Robustness-Oriented Study**

## Abstract
Customer churn prediction is a critical problem in the banking sector, where accurately identifying customers likely to leave can significantly reduce revenue loss. This project presents a comprehensive machine learning pipeline for customer churn prediction using an Artificial Neural Network (ANN). The study systematically evaluates data preprocessing strategies, neural network architectures, decision thresholds, robustness under noise, class imbalance handling, and comparative performance against classical machine learning models. Experimental results demonstrate that the proposed ANN achieves competitive performance with an ROC-AUC of **0.855**, outperforming Logistic Regression and closely matching Random Forest performance, while offering better flexibility for cost-sensitive optimization.

---

## 1. Introduction
Customer retention is more cost-effective than customer acquisition, making churn prediction a central task in customer relationship management. Traditional statistical models often struggle with nonlinear feature interactions present in real-world banking data. Deep learning models, particularly Artificial Neural Networks (ANNs), provide a powerful alternative by learning complex representations from data.

This project investigates:
- Whether ANN-based models outperform classical baselines
- How feature importance, threshold tuning, and class imbalance affect churn detection
- The robustness of the model under noisy and reduced-data scenarios

---

## 2. Dataset Description
The dataset consists of bank customer records with demographic, financial, and behavioral attributes.

### Original Features
- CreditScore  
- Geography  
- Gender  
- Age  
- Tenure  
- Balance  
- NumOfProducts  
- HasCrCard  
- IsActiveMember  
- EstimatedSalary  
- Exited (Target variable)

### Preprocessing Decisions
- Dropped identifiers: `RowNumber`, `CustomerId`, `Surname`
- Label Encoding for `Gender`
- One-Hot Encoding for `Geography`
- Feature scaling using `StandardScaler`

Encoders and scalers are serialized using `pickle` to ensure inference-time consistency.

---

## 3. Methodology

### 3.1 Data Preprocessing Pipeline
1. Categorical encoding:
   - Gender → LabelEncoder
   - Geography → OneHotEncoder
2. Feature normalization using `StandardScaler`
3. Train-test split (80% training / 20% testing)

---

### 3.2 Model Architecture (ANN)
The primary ANN architecture is defined as:

- Input layer: 13 features
- Hidden Layer 1: 64 neurons (ReLU)
- Hidden Layer 2: 32 neurons (ReLU)
- Output Layer: 1 neuron (Sigmoid)

**Optimizer:** Adam (learning rate = 0.001)  
**Loss Function:** Binary Cross-Entropy  
**Regularization:** Early stopping (patience = 10)

---

## 4. Experimental Setup

### 4.1 Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix
- Business-oriented cost function

---

### 4.2 Baseline Models
To contextualize ANN performance, the following classical models were implemented:
- Logistic Regression
- Random Forest (100 trees)

---

## 5. Results and Analysis

### 5.1 ANN Performance
| Metric | Value |
|------|------|
| Accuracy | **0.862** |
| ROC-AUC | **0.855** |
| Precision (Churn) | 0.73 |
| Recall (Churn) | 0.47 |


---

### 5.2 Model Comparison
| Model | Accuracy |
|-----|---------|
| ANN | 0.862 |
| Logistic Regression | 0.811 |
| Random Forest | **0.866** |

The ANN significantly outperforms Logistic Regression and achieves performance comparable to Random Forest.

---

### 5.3 Feature Importance Analysis
Feature importance computed using Random Forest highlights:
- Age
- Balance
- Number of Products
- Geography-related features

This supports the ANN’s capacity to model nonlinear interactions without explicit feature selection.

---

## 6. Robustness Studies

### 6.1 Feature Ablation
Removing `EstimatedSalary`:
- Accuracy: **0.8635**

This indicates that salary contributes limited predictive power.

---

### 6.2 Reduced Training Data
Training with only 50% of the training data:
- Accuracy: **0.8535**

The small performance drop demonstrates strong generalization.

---

### 6.3 Noise Injection
Gaussian noise (σ = 0.01) added to test data:
- Accuracy: **0.862**

The model remains stable under mild perturbations.

---

## 7. Threshold Sensitivity Analysis
The impact of varying classification thresholds:

| Threshold | Precision (Churn) | Recall (Churn) |
|--------|------------------|---------------|
| 0.3 | 0.60 | 0.64 |
| 0.5 | 0.75 | 0.46 |
| 0.7 | 0.86 | 0.31 |

Lower thresholds improve recall at the cost of increased false positives.

---

## 8. Cost-Sensitive Optimization
A business-oriented cost function was defined:

Cost = 10 × False Negatives + 1 × False Positives


Lower thresholds significantly reduce costly false negatives, making them preferable in churn-prevention scenarios.

---

## 9. Handling Class Imbalance
Class weighting was applied during training to emphasize churned customers. This increased recall for the minority class while slightly reducing overall accuracy, highlighting a common trade-off in imbalanced classification tasks.

---

## 10. Limitations
- Single dataset without cross-industry validation
- Limited hyperparameter optimization
- Reduced interpretability compared to tree-based models

---

## 11. Future Work
- Hyperparameter tuning using Bayesian optimization
- Model explainability with SHAP or LIME
- Deployment as a real-time decision-support tool
- Integration with Streamlit for interactive inference

---

## 12. How to Run the Project

### 12.1 Clone the Repository
```bash
git clone <your-repository-url>
cd <project-directory>
```
# 12.2 Install Dependencies
```bash
pip install -r requirements.txt
```

# 12.3 Run the Application (Streamlit)
```bash
streamlit run app.py
```
