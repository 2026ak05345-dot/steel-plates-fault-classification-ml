
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Steel Plates Fault Classification",
    page_icon="🔧",
    layout="wide"
)

st.title("Steel Plates Fault Classification")
st.write(
    "Interactive evaluation of five machine learning classification "
    "models trained on the Steel Plates Faults dataset."
)


# ---------------------------------------------------------
# MODEL FILES
# ---------------------------------------------------------

model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib"
}


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# ---------------------------------------------------------
# DATASET UPLOAD
# ---------------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload the test CSV file containing the 27 input features and target column",
    type=["csv"]
)


if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.success("Test dataset uploaded successfully.")

    st.write("Dataset shape:", test_data.shape)

    with st.expander("Preview uploaded data"):
        st.dataframe(test_data.head())

    # -----------------------------------------------------
    # CHECK TARGET COLUMN
    # -----------------------------------------------------

    if "target" not in test_data.columns:

        st.error(
            "The uploaded CSV must contain a column named 'target'."
        )

        st.stop()

    # Separate features and target
    X_test_app = test_data.drop(columns=["target"])
    y_test_app = test_data["target"]

    # -----------------------------------------------------
    # MODEL SELECTION
    # -----------------------------------------------------

    st.header("2. Select Classification Model")

    selected_model_name = st.selectbox(
        "Choose a model:",
        list(model_files.keys())
    )

    selected_model = load_model(
        model_files[selected_model_name]
    )

    st.success(
        f"{selected_model_name} loaded successfully."
    )

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    y_pred = selected_model.predict(X_test_app)

    # Probability values for AUC
    y_proba = selected_model.predict_proba(X_test_app)

    # -----------------------------------------------------
    # CALCULATE METRICS
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test_app,
        y_pred
    )

    auc = roc_auc_score(
        y_test_app,
        y_proba,
        multi_class="ovr",
        average="weighted"
    )

    precision = precision_score(
        y_test_app,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test_app,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test_app,
        y_pred,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test_app,
        y_pred
    )

    # -----------------------------------------------------
    # DISPLAY METRICS
    # -----------------------------------------------------

    st.header("3. Evaluation Metrics")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col4, metric_col5, metric_col6 = st.columns(3)

    metric_col1.metric("Accuracy", f"{accuracy:.4f}")
    metric_col2.metric("AUC", f"{auc:.4f}")
    metric_col3.metric("Precision", f"{precision:.4f}")

    metric_col4.metric("Recall", f"{recall:.4f}")
    metric_col5.metric("F1 Score", f"{f1:.4f}")
    metric_col6.metric("MCC", f"{mcc:.4f}")

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    st.header("4. Confusion Matrix")

    class_names = sorted(y_test_app.unique())

    cm = confusion_matrix(
        y_test_app,
        y_pred,
        labels=class_names
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(cm)

    ax.set_title(
        f"Confusion Matrix - {selected_model_name}"
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)

    fig.tight_layout()

    st.pyplot(fig)

    # -----------------------------------------------------
    # CLASSIFICATION REPORT
    # -----------------------------------------------------

    st.header("5. Classification Report")

    report = classification_report(
        y_test_app,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

else:

    st.info(
        "Please upload the test_data.csv file to begin model evaluation."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Machine Learning Assignment-2 | Steel Plates Fault Classification"
)
