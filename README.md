# Assignment 17: Feature Selection for Classification

**Course:** Modern AI (AI26501)  
**Programme:** M.Tech – I Semester  
**Semester:** Spring 2026  
**Group:** 14  
**Team Members:** Aakash Arya, Tinku Raj

## 1. Project Overview

This project investigates whether feature selection can reduce the input dimensionality and computational cost of a classification model without substantially sacrificing predictive performance.

The experiment compares Logistic Regression trained using all available features against models trained using progressively smaller feature subsets selected through the ANOVA F-test.

The **Breast Cancer Wisconsin (Diagnostic)** dataset is used to evaluate the trade-off between predictive performance and computational efficiency.

### Research Question

Can ANOVA F-test-based feature selection reduce input dimensionality and computational cost while maintaining classification performance?

### Hypotheses

- **H1 — Efficiency:** Using fewer features may reduce model training and inference time.
- **H2 — Predictive Performance:** A smaller feature subset may preserve Accuracy and Macro-F1 close to the all-features baseline.

Both hypotheses are evaluated experimentally.

## 2. Objectives

- Establish a baseline using all 30 input features.
- Apply a feature-selection method to identify informative features.
- Compare configurations using 30, 20, 15, 10, and 5 features.
- Evaluate Accuracy and Macro-F1.
- Measure training and inference time.
- Examine confusion matrices and classification errors.
- Use repeated stratified cross-validation to assess performance stability.
- Investigate the trade-off between dimensionality, predictive performance, and computational efficiency.

## 3. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

The dataset is loaded using scikit-learn's `load_breast_cancer()` function.

| Property | Description |
|---|---|
| Number of samples | 569 |
| Number of input features | 30 |
| Feature type | Numeric |
| Classification task | Binary classification |
| Target classes | Malignant and benign |
| Baseline configuration | All 30 features |
| Reduced configurations | Top 20, Top 15, Top 10, Top 5 |

The features represent measurements derived from cell-nuclei characteristics.

Dataset documentation:  
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

## 4. Methodology

### 4.1 Data Loading and Splitting

The dataset is divided into a feature matrix `X` and target vector `y`. An 80:20 stratified train/test split is used, with a fixed random seed of 42.

The same split is used for all feature configurations.

### 4.2 Feature Scaling

`StandardScaler` standardizes the numeric features. The scaler is fitted on training data only and subsequently applied to test data to avoid data leakage.

### 4.3 Feature Selection

The experiment uses `SelectKBest` with the ANOVA F-test (`f_classif`) to select features according to their statistical relationship with the target class.

The following configurations are evaluated:

1. All 30 features — baseline
2. Top 20 features
3. Top 15 features
4. Top 10 features
5. Top 5 features

Feature selection is fitted using training data only.

### 4.4 Classification Model

Logistic Regression is used for every configuration. The maximum iteration count is set to 5,000 to support model convergence.

Keeping the classifier and data split consistent makes the comparison more meaningful.

### 4.5 Computational Efficiency

Training and inference times are measured using repeated runs after a warm-up call. The median of 200 timing measurements is used to reduce the effect of individual timing fluctuations.

The experiment also measures feature-selection transformation time to help distinguish classifier prediction time from the additional cost of transforming the input features.

### 4.6 Cross-Validation

Repeated stratified five-fold cross-validation is performed with 10 repeats, resulting in 50 evaluation folds per configuration.

A scikit-learn `Pipeline` ensures that scaling, feature selection, and classification are fitted within each training fold, reducing the risk of data leakage.

## 5. Evaluation Metrics

| Metric | Purpose |
|---|---|
| Number of Features | Measures input dimensionality |
| Accuracy | Measures the proportion of correctly classified samples |
| Macro-F1 | Evaluates both classes with equal weighting |
| Training Time | Measures model-fitting time |
| Inference Time | Measures prediction time |
| Selection Transform Time | Measures the feature-selection transformation cost |
| Confusion Matrix | Shows correct and incorrect predictions by class |

The experiment also examines false negatives, where malignant samples are predicted as benign, and false positives, where benign samples are predicted as malignant.

## 6. Experimental Results

The initial test-set results reported in the project presentation are shown below. These values should be checked against the final notebook outputs before submission.

| Configuration | Features | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| All features | 30 | 0.982456 | 0.981151 |
| Top-20 | 20 | 0.973684 | 0.971863 |
| Top-15 | 15 | 0.964912 | 0.962302 |
| Top-10 | 10 | 0.947368 | 0.943990 |
| Top-5 | 5 | 0.938596 | 0.934947 |

### Initial Observations

- The all-features configuration achieved the highest recorded Accuracy and Macro-F1.
- Reduced-feature configurations require fewer input features.
- The initial results indicate a trade-off between feature reduction and predictive performance.
- Computational efficiency must be evaluated using verified training and inference measurements.

## 7. Outputs and Visualizations

The notebook generates experimental results and visualizations in the `results/` directory.

Expected outputs include:

- `results.csv` — test-set metrics and timing measurements.
- `comparison_table.csv` — comparative results across configurations.
- `cv_results.csv` — repeated cross-validation results, if generated by the notebook.
- `feature_scores.csv` — ANOVA F-scores, if exported.
- `selected_features.csv` — selected feature names, if exported.
- `figures/` — generated performance plots, feature-score plots, and confusion matrices.

Only files actually generated by the notebook should be included in the final repository.

## 8. Project Structure

The repository follows the assignment's prescribed structure, adapted for a notebook-based implementation without a separate source-code directory.

```text
MTech-AIML-Project_14_Feature_Selection/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── sample_data.csv
├── notebooks/
│   └── experiment.ipynb
├── results/
│   ├── results.csv
│   ├── comparison_table.csv
│   ├── cv_results.csv
│   └── figures/
└── presentation/
    └── Group14_Feature_Selection.pptx
```

The notebook loads the dataset directly from scikit-learn, so a local dataset file is not required to run the experiment. The `data/` directory is included in the structure to follow the assignment instructions; its contents should reflect the files actually maintained in the repository.

The results directory may contain additional CSV files generated by the notebook.

## 9. Installation and Execution

### Prerequisites

- Python 3.10 or another compatible Python version
- pip
- Jupyter Notebook or JupyterLab

### Install Dependencies

Create a `requirements.txt` file containing:

```text
numpy
pandas
matplotlib
scikit-learn
jupyter
ipykernel
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### Run the Experiment

1. Clone or download the repository.
2. Open a terminal in the project root.
3. Launch Jupyter Notebook:

```bash
jupyter notebook
```

4. Open `notebooks/experiment.ipynb`.
5. Execute all cells in order.

The notebook generates the experimental results and visualizations in the configured output directories.

## 10. Limitations

- Only one dataset and one classification algorithm are evaluated.
- ANOVA F-test evaluates features individually and does not explicitly account for feature redundancy.
- Test-set results depend on the selected train/test split.
- Very short training and inference times can vary with hardware and system load.
- Fewer features do not necessarily guarantee lower end-to-end inference time.
- Generalization to other datasets has not been established.
- Final timing results should be consistent across the notebook, exported CSV files, and presentation.

## 11. Conclusion

This project evaluates whether feature selection can reduce the dimensionality of a classification task while maintaining predictive performance and improving computational efficiency.

Using the Breast Cancer Wisconsin (Diagnostic) dataset, Logistic Regression is evaluated with all 30 features and four reduced-feature configurations.

The initial test results show that the all-features model achieved the highest predictive scores. The final assessment of efficiency should be based on the verified timing measurements and cross-validation results.

The experiment highlights the importance of evaluating both predictive performance and computational cost when determining whether feature selection is beneficial.

## 12. References

1. Scikit-learn Developers. *Breast Cancer Wisconsin (Diagnostic) Dataset*.  
   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

2. Scikit-learn Developers. *SelectKBest*.  
   https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

3. Scikit-learn Developers. *ANOVA F-value for Classification*.  
   https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html

4. Scikit-learn Developers. *LogisticRegression*.  
   https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
