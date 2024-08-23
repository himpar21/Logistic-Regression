# Logistic Regression on Breast Cancer Dataset

This project demonstrates how to implement logistic regression using the Breast Cancer dataset. The goal is to classify whether a tumor is benign (class 2) or malignant (class 4).

## Dataset

The dataset contains the following features:
- `Clump Thickness`
- `Uniformity of Cell Size`
- `Uniformity of Cell Shape`
- `Marginal Adhesion`
- `Single Epithelial Cell Size`
- `Bare Nuclei`
- `Bland Chromatin`
- `Normal Nucleoli`
- `Mitoses`

The target variable is:
- `Class`: 2 (Benign) or 4 (Malignant)

## Project Setup

### Prerequisites

Ensure you have the following installed:
- Python 3.7+
- `pip` for package management

### Libraries

Install the necessary libraries with the following command:

```bash
pip install -r requirements.txt
```

Content of requirements.txt:
```txt
pandas
scikit-learn
matplotlib
seaborn
```
###  Running the Model
#### 1) Clone the repository:
```bash
git clone https://github.com/himpar21/Logistic-Regression
cd Logistic-Regression
```

#### 2) Place the dataset (breast_cancer.csv) in the project directory.

#### 3) Run the Python script:
```bash
python logisticregression.py
```

This will output:

- The model's accuracy on the test set
- A classification report with precision, recall, and F1-scores
- A confusion matrix
- A plot showing the importance of each feature in the logistic regression model.
