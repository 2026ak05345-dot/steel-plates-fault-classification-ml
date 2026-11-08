
# ============================================================
# Machine Learning Assignment-2
# Steel Plates Fault Classification
# ============================================================

from ucimlrepo import fetch_ucirepo
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


# ------------------------------------------------------------
# 1. Load Steel Plates Faults dataset
# ------------------------------------------------------------

steel_plates = fetch_ucirepo(id=198)

X = steel_plates.data.features
original_targets = steel_plates.data.targets


# ------------------------------------------------------------
# 2. Convert seven binary target columns into one target
# ------------------------------------------------------------

y = original_targets.idxmax(axis=1)

print("Dataset shape:", X.shape)
print("Number of classes:", y.nunique())

print("\nClass distribution:")
print(y.value_counts())


# ------------------------------------------------------------
# 3. Train-test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 4. Define five classification models
# ------------------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "Naive Bayes": GaussianNB(),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            probability=True,
            random_state=42
        ))
    ])
}


# ------------------------------------------------------------
# 5. Train models and calculate metrics
# ------------------------------------------------------------

results = []

os.makedirs("model", exist_ok=True)

model_files = {
    "Logistic Regression":
        "model/logistic_regression.joblib",

    "Decision Tree":
        "model/decision_tree.joblib",

    "KNN":
        "model/knn.joblib",

    "Naive Bayes":
        "model/naive_bayes.joblib",

    "SVM":
        "model/svm.joblib"
}


for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Probability predictions for AUC
    y_proba = model.predict_proba(X_test)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_proba,
        multi_class="ovr",
        average="weighted"
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    results.append({
        "ML Model Name": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    # Save trained model
    joblib.dump(
        model,
        model_files[model_name]
    )

    print(
        f"{model_name} saved successfully."
    )


# ------------------------------------------------------------
# 6. Create model comparison table
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.round(4).to_string(index=False))


# ------------------------------------------------------------
# 7. Save comparison results
# ------------------------------------------------------------

results_df.to_csv(
    "model_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# 8. Create test_data.csv
# ------------------------------------------------------------

test_data = X_test.copy()
test_data["target"] = y_test.values

test_data.to_csv(
    "test_data.csv",
    index=False
)


print("\nTraining completed successfully.")
print("Saved files:")
print("- model_comparison.csv")
print("- test_data.csv")
print("- model/*.joblib")
