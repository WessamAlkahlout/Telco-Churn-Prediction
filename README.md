# Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer will leave (churn). This is my first ML project, built after learning on my own for under six months.

## Dataset

The Telco Customer Churn dataset (IBM sample data, found on Kaggle): 7,043 customers and 21 columns, including contract type, tenure, services used, and monthly charges. The target is `Churn` (Yes/No).

Download it and save it as `Telco-Customer-Churn.csv` in the same folder as the script.

## What I did

1. **Exploration:** looked at the raw data first, which showed that `TotalCharges` is stored as text.
2. **Cleaning:**
   - Converted `TotalCharges` to numbers and filled the blank values with 0.
   - Merged "No internet service" and "No phone service" with "No".
   - Mapped `Churn` to 1 (Yes) and 0 (No).
3. **Feature engineering:** added `ChargePerMonth = TotalCharges / (tenure + 1)`.
4. **Dropped columns:** `customerID` and `gender`.
5. **Splitting:** 70% train, 15% validation, 15% test, stratified so the churn rate (about 27%) is the same in every split.
6. **Encoding:** Yes/No columns with `OrdinalEncoder`, and `Contract`, `PaymentMethod`, `InternetService` with `OneHotEncoder`. Both encoders were fit on the training set only, to avoid data leakage.
7. **Feature selection:** kept the 11 most important features according to an XGBoost model trained on the training set.
8. **Model comparison:** trained Random Forest and XGBoost, both with settings that limit overfitting, and picked the better one by validation ROC AUC.
9. **Final test:** scored the chosen model once on the test set.

## Results

| Model | Validation ROC AUC |
|---|---|
| Random Forest | 0.841 |
| XGBoost | 0.849 |

XGBoost scored slightly higher on validation, so it was chosen.

Final test results for XGBoost, using a 0.26 probability cutoff:

| Metric | Value |
|---|---|
| ROC AUC | 0.842 |
| Accuracy | 0.746 |
| Churn precision | 0.51 |
| Churn recall | 0.80 |
| Churn F1 | 0.63 |

Confusion matrix (test set, 1,057 customers, 281 of whom churned):

|  | Predicted stay | Predicted churn |
|---|---|---|
| **Actually stayed** | 562 | 214 |
| **Actually churned** | 55 | 226 |

The model catches 226 of the 281 customers who actually left (80%), and misses 55.

## Why a 0.26 cutoff?

The default cutoff is 0.5. I lowered it to 0.26 because missing a customer who leaves usually costs a company more than a retention offer sent by mistake. I chose the cutoff using recall and F1 score on the validation set only, with the goal of missing as less as possible. The cost of this choice is more false alarms: about half of the customers flagged as likely to leave would actually have stayed, and accuracy is lower than it would be with a higher cutoff. Accuracy is not the main number here, since a model that predicts "nobody churns" would score about 73% accuracy while catching no one.

## Limitations

- I chose the 0.26 cutoff by hand, by looking at recall and F1 on the validation set. It was not tuned in a more systematic way, and the test set was used only once, for the final scores.
- The dataset is small (7,043 rows), and the validation and test sets have about 1,000 customers each, so scores can shift by a point or two with a different split.
- Many customers leave for reasons this data does not capture, such as a competitor's offer, so some churners cannot be predicted from these columns.

## How to run

1. Install Python 3.12 or newer (the script uses nested quotes inside f-strings, which need 3.12+).
2. Install the libraries:

   ```
   pip install -r requirements.txt
   ```

3. Put `Telco-Customer-Churn.csv` next to the script and run it:

   ```
   python churn_model.py
   ```
 

## Tools

Python, pandas, NumPy, Matplotlib, scikit-learn, XGBoost.
