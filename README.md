# Titanic Survival Classification

## Project Overview

This project is a supervised machine learning model that predicts whether a passenger survived the Titanic disaster using key features such as passenger class, gender, age, fare, and embarkation point.

The model demonstrates a complete data science workflow including data preprocessing, feature engineering, model training, and evaluation using accuracy metrics.

---

## Dataset

The project uses the built-in Titanic dataset from the Seaborn library.

---

## Features Used

The following features were selected for prediction:

* `pclass` → Passenger class
* `sex` → Gender
* `age` → Age of passenger
* `fare` → Ticket fare
* `embarked` → Port of embarkation

Target variable:

* `survived` → 0 = Did not survive, 1 = Survived

---

## Data Preprocessing

The following preprocessing steps were applied:

* Handled missing values:

  * Age → filled with median
  * Embarked → filled with mode
* Encoded categorical variables:

  * Sex → male = 0, female = 1
  * Embarked → S = 0, C = 1, Q = 2
* Removed remaining missing values using `dropna()`

---

## Model Used

* Decision Tree Classifier (from scikit-learn)

---

## Workflow

1. Load dataset using Seaborn
2. Select relevant features
3. Clean missing values
4. Encode categorical variables
5. Split data into training and testing sets
6. Train Decision Tree model
7. Make predictions
8. Evaluate accuracy

---

## Results

The model outputs an accuracy score on the test dataset, showing how well it can predict survival outcomes.

Example output:

```
Titanic Model Accuracy: 0.77
```

---

## Technologies Used

* Python
* Pandas
* Seaborn
* Scikit-learn

---

## How to Run

1. Install dependencies:

```bash
pip install pandas seaborn scikit-learn
```

2. Run the script:

```bash
python code.py
```

---

## Project Structure

```
Project-2-Titanic-Classification/
│
├── code.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Author

Seif Mohamed

AI Virtual Internship Project
