seed = 42

# Import Blocks

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

# ____________________________________________________________________________________________________

# Data Exploration

df = pd.read_csv("Telco-Customer-Churn.csv")

info = {
    "Description": df.describe(),
    "First 5 Rows": df.head(),
    "Rows and Columns": df.shape,
    "Features": df.drop("Churn", axis=1).columns,
    "Missing Data Sum": df.isna().sum(),
}


pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

for item in info:
    print(f"{item}\n{"_"*20}\n{info[item]}\n\n")

# ____________________________________________________________________________________________________

# Label Mapping


three_values_cols = [
    "TechSupport",
    "MultipleLines",
    "OnlineSecurity",
    "DeviceProtection",
    "StreamingTV",
    "StreamingMovies",
]

df[three_values_cols] = df[three_values_cols].replace(
    {"No internet service": "No", "No phone service": "No"}
)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# ____________________________________________________________________________________________________

# Feature Engineering

df["ChargePerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)

# ____________________________________________________________________________________________________

# Feature-Target Split

X = df.drop(["customerID", "gender", "Churn"], axis=1)
y = df["Churn"]

# ____________________________________________________________________________________________________

# Splitting Data

X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, random_state=seed, train_size=0.7, stratify=y
)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_validation, y_validation, random_state=seed, test_size=0.5, stratify=y_validation
)
# ____________________________________________________________________________________________________

# Encoding

# _____assigning columns to encode
y_n_cols = [
    "Partner",
    "Dependents",
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "MultipleLines",
]

str_cols = ["Contract", "PaymentMethod", "InternetService"]

# _____fitting
ordinal_encoder = OrdinalEncoder().fit(X_train[y_n_cols])
one_hot_encoder = (
    OneHotEncoder(sparse_output=False)
    .set_output(transform="pandas")
    .fit(X_train[str_cols])
)

# _____transforming
X_sets = [X_train, X_validation, X_test]
for index, i in enumerate(X_sets):

    i[y_n_cols] = ordinal_encoder.transform(i[y_n_cols])
    str_cols_encoded = one_hot_encoder.transform(i[str_cols])
    i[str_cols_encoded.columns] = str_cols_encoded

    X_sets[index] = X_sets[index].drop(str_cols, axis=1)

X_train, X_validation, X_test = X_sets

# ____________________________________________________________________________________________________

# Feature Selection

xgbc = XGBClassifier(
    n_estimators=150,
    random_state=seed,
    max_depth=3,
    min_child_weight=0.1,
    subsample=0.8,
)
xgbc.fit(X_train, y_train)

importances = pd.Series(xgbc.feature_importances_, index=X_train.columns).sort_values(
    ascending=False
)

for i in range(len(X_sets)):
    X_sets[i] = X_sets[i][importances.head(11).index.tolist()]
X_train, X_validation, X_test = X_sets

# ____________________________________________________________________________________________________

# Training The Model

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        max_depth=3,
        min_samples_leaf=5,
        min_samples_split=20,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=150,
        random_state=seed,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=0.1,
        subsample=0.8,
        reg_alpha=1,
    ),
}
score = 0

for technique in models:
    model = models[technique]
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_validation)
    roc_auc = roc_auc_score(y_validation, y_prob[:, 1])

    print(roc_auc)

    if roc_auc > score:
        score = roc_auc
        best_model = technique

# ____________________________________________________________________________________________________

# Choosing the best model

print(f""" ____________________________
|                            |
|  Best Model is:  ({best_model}) |
|____________________________|\n\n""")

THRESHOLD = 0.26
y_prob = models[best_model].predict_proba(X_test)
y_pred = np.array(y_prob >= THRESHOLD, dtype=int)[:, 1]

metrics = {
    "Accuracy Score": np.round(accuracy_score(y_test, y_pred), 3),
    "Classification Report": classification_report(y_test, y_pred),
    "Confusion Matrix": confusion_matrix(y_test, y_pred),
    "ROC AUC Score": np.round(roc_auc_score(y_test, y_prob[:, 1]), 3),
}

for metric in metrics:
    print(f"{metric} for {best_model}\n{"_"*30}\n{metrics[metric]}\n\n")

ConfusionMatrixDisplay(metrics["Confusion Matrix"]).plot(cmap="Blues")
plt.show()
