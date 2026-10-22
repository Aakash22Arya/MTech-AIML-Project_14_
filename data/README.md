# Dataset Documentation

## Dataset Name

**Breast Cancer Wisconsin (Diagnostic) Dataset**

## Description

This project uses the Breast Cancer Wisconsin (Diagnostic) dataset provided by scikit-learn. It is a binary classification dataset used to investigate the effectiveness of feature selection in reducing input dimensionality while maintaining predictive performance.

The dataset contains measurements derived from cell nuclei characteristics and classifies samples as malignant or benign.

## Dataset Information

| Property       | Description                          |
| -------------- | ------------------------------------ |
| Dataset        | Breast Cancer Wisconsin (Diagnostic) |
| Source         | scikit-learn                         |
| Total samples  | 569                                  |
| Input features | 30                                   |
| Target classes | Malignant and benign                 |
| Feature type   | Numeric                              |
| Task           | Binary classification                |

## Data Source

The dataset is loaded using the `load_breast_cancer()` function from scikit-learn.

Official documentation:  
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

Original dataset source:  
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## Data Loading

The dataset can be loaded directly in Python:

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
df = data.frame
```

To save the dataset as a CSV file:

```python
df.to_csv("data/sample_data.csv", index=False)
```

Run the export command from the project root directory, after creating the `data` folder.

## Role in the Project

The dataset is used to compare Logistic Regression models trained with:

- All 30 features
- The top 20 selected features
- The top 15 selected features
- The top 10 selected features
- The top 5 selected features

Feature selection is performed using the ANOVA F-test through scikit-learn's `SelectKBest`.

The dataset supports the evaluation of Accuracy, Macro-F1, training time, and inference time.

## Notes

- The notebook can load the dataset directly from scikit-learn; downloading a separate dataset file is not required to run the experiment.
- `sample_data.csv` is an optional local copy included to support the prescribed repository structure.
- The dataset should not be modified manually before running the experiment.
