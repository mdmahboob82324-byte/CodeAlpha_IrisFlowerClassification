# CodeAlpha Data Science Internship - Task 1: Iris Flower Classification
# Put Iris.csv (Kaggle: saurabh00007/iriscsv) in the same folder as this file.
# Run:  python CodeAlpha_Iris_Flower_Classification.py

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

sns.set_theme(style="whitegrid")

# ---------------- 1. LOAD DATA ----------------
if os.path.exists("Iris.csv"):
    df = pd.read_csv("Iris.csv")
    print("Loaded Iris.csv from Kaggle")
else:
    from sklearn.datasets import load_iris
    raw = load_iris(as_frame=True)
    df = raw.frame.copy()
    df.columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "target"]
    df["Species"] = df["target"].map(dict(enumerate(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])))
    df.insert(0, "Id", range(1, len(df) + 1))
    df = df.drop(columns="target")
    print("Iris.csv not found -> using Iris data from Scikit-learn")

# ---------------- 2. EXPLORE ----------------
print("\nShape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nSummary statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())
print("\nClass distribution:\n", df["Species"].value_counts())

# ---------------- 3. VISUALIZE ----------------
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

sns.pairplot(df.drop(columns="Id"), hue="Species", palette="Set2")
plt.savefig("pairplot.png", dpi=100, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature correlation")
plt.savefig("correlation.png", dpi=100, bbox_inches="tight")
plt.show()

# ---------------- 4. PREPARE DATA ----------------
X = df[features]
y = df["Species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("\nTrain samples:", len(X_train), "| Test samples:", len(X_test))

# ---------------- 5. TRAIN & COMPARE MODELS ----------------
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
    "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    rows.append({
        "Model": name,
        "Test Accuracy": accuracy_score(y_test, model.predict(X_test)),
        "CV Mean Accuracy": cross_val_score(model, X_train, y_train, cv=cv).mean(),
    })
results = pd.DataFrame(rows).sort_values("CV Mean Accuracy", ascending=False).reset_index(drop=True)
print("\nModel comparison:\n", results.round(4))

# ---------------- 6. EVALUATE BEST MODEL ----------------
best_name = results.loc[0, "Model"]
best_model = models[best_name]
y_pred = best_model.predict(X_test)

print("\nBest model:", best_name)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
ConfusionMatrixDisplay(cm, display_labels=best_model.classes_).plot(cmap="Blues", xticks_rotation=20)
plt.title(f"Confusion matrix - {best_name}")
plt.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
plt.show()

# ---------------- 7. PREDICT NEW FLOWERS ----------------
new_flowers = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2], [6.0, 2.9, 4.5, 1.5], [6.9, 3.1, 5.8, 2.3]],
    columns=features)
new_flowers["Predicted Species"] = best_model.predict(new_flowers[features])
print("\nPredictions for new flowers:\n", new_flowers)
