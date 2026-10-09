# 🌸 CodeAlpha_IrisFlowerClassification

**CodeAlpha Data Science Internship · Task 1: Iris Flower Classification**

A machine learning project that predicts the species of an Iris flower (**setosa, versicolor or virginica**) from four simple measurements, built with Python and Scikit-learn.

**Language:** Python 3.8+ &nbsp;|&nbsp; **Library:** Scikit-learn &nbsp;|&nbsp; **Status:** Completed &nbsp;|&nbsp; **Best test accuracy:** 96.7% (SVM)

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Key Concepts](#-key-concepts)
4. [Project Workflow](#-project-workflow)
5. [Models Used](#-models-used)
6. [Visualizations](#-visualizations)
7. [Results](#-results)
8. [Project Structure](#-project-structure)
9. [How to Run](#-how-to-run)
10. [What I Learned](#-what-i-learned)
11. [Future Improvements](#-future-improvements)
12. [Author](#-author)

---

## 🎯 Project Overview

**Problem:** Given the measurements of an Iris flower, can a computer tell which species it is?

**Approach:** This is a **supervised classification** problem. The model is shown many flowers together with their correct species, learns the patterns, and is then tested on flowers it has never seen.

```
[ 5.1, 3.5, 1.4, 0.2 ]  ──►  MODEL  ──►  Iris-setosa
     INPUT (X)                            OUTPUT (y)
```

**Task objectives (from CodeAlpha):**

- ✅ Use measurements of Iris flowers as input data
- ✅ Train a machine learning model to classify the species
- ✅ Use Scikit-learn for dataset access and model building
- ✅ Evaluate accuracy and performance using test data
- ✅ Understand basic classification concepts

---

## 📊 Dataset

- **Source:** [Iris CSV on Kaggle](https://www.kaggle.com/datasets/saurabh00007/iriscsv)
- **Size:** 150 flowers (rows)
- **Classes:** 3 species with 50 flowers each (a balanced dataset)
- **Missing values:** none

| Column | Description | Role |
|---|---|---|
| `Id` | Row number (dropped before training) | - |
| `SepalLengthCm` | Length of the sepal in cm | Input feature |
| `SepalWidthCm` | Width of the sepal in cm | Input feature |
| `PetalLengthCm` | Length of the petal in cm | Input feature |
| `PetalWidthCm` | Width of the petal in cm | Input feature |
| `Species` | Iris-setosa, Iris-versicolor or Iris-virginica | **Target (output)** |

> **Note:** the species names are the **output** the model predicts, not the input. The input is the four measurements.

**Key observations from the data**

- *Setosa* is clearly separated from the other two species, mainly by petal length and petal width.
- *Versicolor* and *virginica* overlap slightly, which is where the model can make mistakes.
- Petal length and petal width are strongly correlated and are the most useful features.

---

## 📚 Key Concepts

| Concept | Simple explanation |
|---|---|
| **Classification** | Predicting a category (a species name) instead of a number |
| **Supervised learning** | Learning from data that already has correct answers (labels) |
| **Features (X) and label (y)** | Features are the inputs; the label is the answer to predict |
| **Train/test split** | Learn on one part of the data, check performance on a part the model has never seen |
| **Stratified split** | Keeps the same proportion of each species in the train and test sets |
| **Feature scaling** | Puts all measurements on a similar scale, which helps models like SVM and KNN |
| **Cross-validation** | Tests the model several times on different parts of the data for a more reliable score |
| **Overfitting** | The model memorizes training data and performs badly on new data |
| **Data leakage** | Test data accidentally influencing training; avoided by putting the scaler inside a pipeline |

---

## 🔄 Project Workflow

1. **Load the data** - read `Iris.csv` (falls back to Scikit-learn's built-in Iris data if the file is missing).
2. **Explore the data** - `head()`, `info()`, `describe()`, missing values and class counts.
3. **Visualize** - pair plot, box plots and a correlation heatmap.
4. **Prepare the data** - choose the 4 features as `X` and `Species` as `y`; split **80% train / 20% test** (`random_state=42`, `stratify=y`).
5. **Train models** - train five classifiers; scale features inside a `Pipeline` where needed.
6. **Compare models** - use test accuracy and **5-fold stratified cross-validation**.
7. **Evaluate the best model** - accuracy, classification report and confusion matrix on the test set.
8. **Predict new flowers** - classify new sets of measurements with the best model.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## 🤖 Models Used

| Model | Notes |
|---|---|
| Logistic Regression | Simple, fast linear baseline |
| K-Nearest Neighbors (k=5) | Classifies by the closest training flowers |
| Decision Tree | Rule-based model that is easy to interpret |
| **Support Vector Machine (RBF)** | Finds the best boundary between species; **best model here** |
| Random Forest (200 trees) | Ensemble of many decision trees |

---

## 📈 Visualizations

### Pair plot of all features
Setosa (small petals) is clearly separate; versicolor and virginica overlap a little.

<img width="1263" height="1123" alt="pairplot" src="https://github.com/user-attachments/assets/fee9693b-4db7-4238-a4ec-082d026756ed" />

### Feature correlation
Petal length and petal width are strongly correlated and are the most useful features.

<img width="679" height="615" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/21ea2e96-bb31-4c78-8f03-7abb61228278" />

### Model comparison
All five models score well; the SVM is best.

<img width="828" height="423" alt="model_comparison" src="https://github.com/user-attachments/assets/0086c04e-c4d1-40ab-aa3b-c2802705b4da" />


### Confusion matrix (best model: SVM)
Rows are the true species and columns are the predicted species. Only one versicolor was predicted as virginica.

<img width="588" height="470" alt="confusion_matrix" src="https://github.com/user-attachments/assets/554d0496-f6b7-4b35-816b-88d3259e5866" />


---

## 🏆 Results

### Model comparison

| Model | Test Accuracy | CV Mean Accuracy |
|---|---|---|
| **SVM (RBF)** | **0.967** | **0.967** |
| Logistic Regression | 0.933 | 0.958 |
| K-Nearest Neighbors | 0.933 | 0.958 |
| Decision Tree | 0.933 | 0.950 |
| Random Forest | 0.900 | 0.950 |

The best model was chosen using **cross-validation**, not a single split, because the test set has only 30 flowers.

### Best model: SVM (RBF) on the test set

The model classified **29 of 30** flowers correctly (**96.7% accuracy**).

| Species | Correct | Result |
|---|---|---|
| Iris-setosa | 10 / 10 | Perfect |
| Iris-versicolor | 9 / 10 | One predicted as virginica |
| Iris-virginica | 10 / 10 | Perfect |

**Metric meanings**

- **Accuracy** - correct predictions divided by total predictions.
- **Precision** - when the model says "virginica", how often is it right?
- **Recall** - of all real virginica flowers, how many did the model find?
- **F1-score** - one number combining precision and recall.
- **Confusion matrix** - shows exactly which species were mixed up.

> The only error is between *versicolor* and *virginica*, which matches what the data exploration showed: those two species overlap slightly.

> Small differences between runs can appear with a different Scikit-learn version, but results stay in the same range.

---

## 📁 Project Structure

```
CodeAlpha_IrisFlowerClassification/
│
├── CodeAlpha_Iris_Flower_Classification.ipynb   # Jupyter notebook with outputs and charts
├── CodeAlpha_Iris_Flower_Classification.py      # Complete code in one file
├── Iris.csv                                     # Dataset from Kaggle
├── images/                                      # Charts shown in this README
│   ├── pairplot.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   └── confusion_matrix.png
└── README.md                                    # Project documentation
```

---

## ▶️ How to Run

**1. Clone the repository**

```bash
git clone https://github.com/mdmahboob82324-byte/CodeAlpha_IrisFlowerClassification.git
cd CodeAlpha_IrisFlowerClassification
```

**2. Install the libraries**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn notebook
```

**3. Run it - choose one option**

*Option A: Python script*

```bash
python CodeAlpha_Iris_Flower_Classification.py
```

*Option B: Jupyter notebook*

```bash
jupyter notebook
```

Open `CodeAlpha_Iris_Flower_Classification.ipynb` and choose **Run → Run All Cells**.

> Keep `Iris.csv` in the same folder as the code. If it is missing, the code automatically uses the same Iris data bundled with Scikit-learn.

---

## 💡 What I Learned

- How a supervised classification project works from raw data to predictions
- Why data must be split into training and test sets, and why to stratify
- How to compare several models fairly using cross-validation
- How to read accuracy, precision, recall, F1-score and the confusion matrix
- How to avoid data leakage by using Scikit-learn pipelines
- How to present a data science project clearly on GitHub

---

## 🚀 Future Improvements

- Tune hyperparameters with `GridSearchCV`
- Try more models such as Gradient Boosting or a small neural network
- Plot decision boundaries to show how the model separates the species
- Build a small web app where a user enters measurements and gets the species

---

## 👤 Author

**Md Chinna Mahaboob**
Data Science Intern at **CodeAlpha**

🔗 LinkedIn: [Md Chinna Mahaboob](https://www.linkedin.com/in/m-d-mahaboob-006527380)
💻 GitHub: [mdmahboob82324-byte](https://github.com/mdmahboob82324-byte)

---

*This project was completed as part of the CodeAlpha Data Science Internship (Task 1).*
