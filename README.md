# Dr. House: Predicting Hepatocellular Carcinoma Survival

![Python](https://img.shields.io/badge/Python-3-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)

A machine learning project that predicts whether a patient with **hepatocellular carcinoma (HCC)**, the most common type of liver cancer, will survive one year after diagnosis. It has two parts:

1. **An analysis notebook** covering data exploration, cleaning, and a comparison of four classifiers.
2. **A desktop app (PyQt5)** where you enter a patient's 49 clinical values and get a "Patient Will Survive" / "Patient Will NOT Survive" prediction from a Gradient Boosting model.

> **Educational project only.** This is a university course project built on a very small dataset. It is not a medical device and must not be used for clinical decisions.

<!-- Add a screenshot of the app here: ![Dr. House GUI](docs/screenshot.png) -->

---

## Dataset

The data is the **HCC Survival dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/423/hcc+survival), collected at a university hospital in Portugal (Coimbra Hospital and University Centre).

| | |
|---|---|
| Patients | 165 |
| Features | 49 (23 quantitative, 26 qualitative) |
| Target (`Class`) | Survival at 1 year: `Lives` (102) or `Dies` (63) |
| Missing data | About 10% of values, marked with `?` |

Dataset reference: Santos, M. S. et al. *A new cluster-based oversampling method for improving survival prediction of hepatocellular carcinoma patients.* Journal of Biomedical Informatics, 58, 49–59, 2015.

---

## Repository structure

```
.
├── DrHouseApp.py                # Desktop application (PyQt5)
├── Dr_House_Analysis.ipynb      # Exploration, cleaning and model comparison
├── data/
│   ├── hcc_dataset.csv          # Raw dataset (the app trains on this file)
│   └── patient_data_clean.csv   # Cleaned dataset exported by the notebook
├── graphs/                      # Figures shown by the app's "Show Graphs" button
├── icons/                       # Button and title-bar icons
├── sound/                       # happy.wav and womp_womp.wav
└── font/                        # PoetsenOne-Regular.ttf
```

---

## The notebook

`Dr_House_Analysis.ipynb` follows this pipeline:

**1. Cleaning numerical variables (23)**
- Replace `?` with `NaN` and fill with the column **mean**.
- Cast columns to `int` or `float` as appropriate.
- Detect outliers with the **IQR rule** (bounds at Q1 − 1.75·IQR and Q3 + 1.75·IQR) and replace them with the largest in-range value.
- Compare distributions before and after cleaning with histograms and bar plots.

**2. Cleaning categorical variables (26)**
- Replace `?` with `NaN` and fill with the column **mode**.

**3. Modelling**
- The cleaned data is exported to `patient_data_clean.csv`, categorical features are one-hot encoded, and `Lives`/`Dies` is mapped to 1/0.
- Four classifiers are trained and evaluated (classification report, MAE, confusion matrix, ROC curve and AUC):

| Model | Configuration |
|---|---|
| Decision Tree | `max_depth=10`, then tuned with `GridSearchCV` (5-fold) over `max_depth`, `min_samples_split`, `min_samples_leaf` |
| K-Nearest Neighbours | `n_neighbors=3` |
| Gradient Boosting | 100 estimators, learning rate 0.1 |
| Random Forest | 100 estimators |

All splits use `random_state=20` (70/30 for the Decision Tree, 67/33 for the others).

### Results

Approximate results from running the notebook pipeline on the cleaned data (one train/test split, so 50–55 test patients per model):

| Model | Accuracy | AUC* |
|---|---|---|
| Decision Tree (`max_depth=10`) | 0.68 | 0.66 |
| Decision Tree (grid search) | 0.66 | 0.63 |
| K-Nearest Neighbours | 0.64 | 0.57 |
| **Gradient Boosting** | **0.76** | **0.73** |
| **Random Forest** | **0.76** | **0.73** |

\*Computed from hard class predictions, as in the notebook. The Decision Tree has no fixed seed, so its numbers can vary slightly between runs.

Gradient Boosting and Random Forest tie on this split and clearly beat the single tree and KNN. Gradient Boosting was chosen for the app. With only ~50 test patients, small differences between models are not statistically meaningful.

---

## The app

`DrHouseApp.py` is a desktop GUI built with **PyQt5**. On launch it trains a `GradientBoostingClassifier` (100 estimators, learning rate 0.1) on `data/hcc_dataset.csv` and prints a classification report to the console.

**Inputs.** 49 fields: 26 dropdowns for categorical variables (e.g. `Gender`, `Cirrhosis`, `PS`, `Ascites`) and 23 text boxes for numerical ones (e.g. `Age`, `AFP`, `Hemoglobin`, `Albumin`).

**Buttons**
- **Show Ranges:** lists the accepted range for each numerical variable, based on the notebook's outlier analysis.
- **Submit:** predicts and shows *Patient Will Survive* or *Patient Will NOT Survive*, with a short sound for each outcome.
- **Show Graphs:** opens the figures from `graphs/` (raw vs. cleaned distributions and the effect of outlier treatment).

---

## Getting started

**Requirements:** Python 3 and

```bash
pip install PyQt5 pandas numpy scikit-learn matplotlib seaborn jupyter
```

**Run the app** (from the repository root, since asset paths are relative):

```bash
python DrHouseApp.py
```

**Run the notebook**

```bash
jupyter notebook Dr_House_Analysis.ipynb
```

Before running, update the path in the "Import Data" cell to `data/hcc_dataset.csv`.

---

## Known limitations

- **Range validation is not applied on Submit.** `preprocess_input()` implements the "Value Not Accepted" check, but the Submit button calls `show_selections()`, which skips it.
- **The app does not use the notebook's cleaning.** It trains directly on the raw CSV, where numeric columns containing `?` are read as text and one-hot encoded. As a result, typed numeric values only influence the prediction if they exactly match a value seen in training, and some inputs (such as `Age`) are effectively ignored.
- **Evaluation is a single small split.** Imputation and outlier capping are computed on the full dataset before splitting, which can leak information into the test set. Cross-validation would give more reliable estimates.

## Possible improvements

- Share one preprocessing pipeline (imputation, encoding, outlier handling) between the notebook and the app, e.g. with a scikit-learn `Pipeline`.
- Use cross-validation and report class-wise recall, since missing a patient who will not survive matters more than overall accuracy.
- Show a predicted probability alongside the label.

---

## Authors

Elmano Vaz, Pedro Ferreira and Miguel Lopes, as the final project for the IACD course.

Originally developed in the team repository: [Elbro1234/Dr.-House-IACD](https://github.com/Elbro1234/Dr.-House-IACD).
