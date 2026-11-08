
# Steel Plates Fault Classification

## 1. Problem Statement

The objective of this project is to implement and compare five machine learning classification algorithms for predicting fault types in steel plates. The implemented models are Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM).

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

The Steel Plates Faults dataset from the UCI Machine Learning Repository is used for this project.

The dataset contains 1,941 instances and 27 input features. The original dataset represents seven fault types using seven binary target columns:

- Pastry
- Z_Scratch
- K_Scatch
- Stains
- Dirtiness
- Bumps
- Other_Faults

Since each record belongs to exactly one fault category, the seven binary target columns were converted into a single multiclass target variable.

The final classification problem therefore contains 27 input features and 7 target classes.

## 3. GitHub Repository Link

[GitHub Repository](https://github.com/2026ak05345-dot/steel-plates-fault-classification-ml)

## 4. Models Used

The following five classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes
5. Support Vector Machine

### Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7275 | 0.9066 | 0.7329 | 0.7275 | 0.7287 | 0.6487 |
| Decision Tree | 0.7429 | 0.8283 | 0.7455 | 0.7429 | 0.7429 | 0.6688 |
| KNN | 0.7275 | 0.8941 | 0.7318 | 0.7275 | 0.7245 | 0.6548 |
| Naive Bayes | 0.4524 | 0.7955 | 0.5741 | 0.4524 | 0.4003 | 0.3789 |
| SVM | 0.7506 | 0.9204 | 0.7578 | 0.7506 | 0.7517 | 0.6765 |

## 5. Model Performance Observations

### Logistic Regression

Logistic Regression provided balanced and consistent performance across the seven classes. It achieved good AUC and moderate overall classification performance.

### Decision Tree

Decision Tree achieved strong accuracy, precision, F1 score and MCC. Its AUC was lower than the AUC obtained by Logistic Regression, KNN and SVM.

### KNN

KNN produced performance comparable to Logistic Regression. It achieved a strong AUC, although its F1 score and MCC were slightly lower than those of the best-performing models.

### Naive Bayes

Naive Bayes showed the lowest overall performance among the five models. Its accuracy, precision, recall, F1 score and MCC were substantially lower than those of the other models.

### SVM

SVM achieved the best overall performance. It obtained the highest Accuracy, AUC, Precision, Recall, F1 Score and MCC among the five evaluated models.

### Overall Winner

Based on the evaluation results, **SVM is the overall best-performing model** for the Steel Plates Faults dataset.

## 6. Streamlit Application

The Streamlit application provides:

- CSV test-data upload
- Machine learning model selection
- Evaluation metrics
- Confusion matrix
- Classification report

The application loads the pre-trained models and does not retrain models using the uploaded test data.

## 7. Project Files

The repository contains:

- `app.py` - Streamlit application
- `requirements.txt` - Required Python packages
- `README.md` - Project documentation
- `test_data.csv` - Test dataset used for evaluation
- `model/` - Saved trained classification models
- `training_code.py` - Model training and evaluation code

## 8. Streamlit App Link

[Live Streamlit App](https://steel-plates-fault-classification-ml-db98thenuccpqgbgew2kaq.streamlit.app/)

## 9. BITS Virtual Lab Screenshot

The final assignment PDF includes the required screenshot demonstrating execution in the BITS Virtual Lab.
