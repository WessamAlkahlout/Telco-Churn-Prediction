# Telco Customer Churn Prediction

A machine learning project that predicts whether a telecom customer will leave (churn). This is my first ML project, built after learning on my own for under six months.

## Dataset

The Telco Customer Churn dataset (IBM sample data, commonly found on Kaggle): 7,043 customers and 21 columns, including contract type, tenure, services used, and monthly charges. The target is `Churn` (Yes/No).

Download it and save it as `Telco-Customer-Churn.csv` in the same folder as the script.

## What I did

1. **Exploration:** looked at the raw data first, which showed that `TotalCharges` is stored as text, and there are several Yes/No columns I have to handle later on.
2. **Cleaning:**
   - Converted `TotalCharges` to numbers and filled the blank values with 0.
   - Merged "No internet service" and "No phone service" into "No".
   - Mapped `Churn` to 1 (Yes) and 0 (No).
3. **Feature engineering:** added `ChargePerMonth = TotalCharges / (tenure + 1)`.
4. **Dropped columns:** `customerID` and `gender` because they have no effect.
5. **Splitting:** 70% train, 15% validation, 15% test.
6. **Encoding:** Yes/No columns with `OrdinalEncoder`, and `Contract`, `PaymentMethod`, `InternetService` with `OneHotEncoder`. Both encoders were fit on the training set only, to avoid data leakage.
7. **Feature selection:** kept the 11 most important features according to a Random Forest trained on the training set.
8. **Model comparison:** trained Random Forest and XGBoost, both with settings that limit overfitting, and picked the better one by validation ROC AUC.
9. **Final test:** scored the chosen model once on the test set.

## Results

| Model | Validation ROC AUC |
|---|---|
| Random Forest | 0.853 |
| XGBoost | 0.855 |

The two models are very close, so the difference is within normal noise.

Final test results for XGBoost, using a 0.25 probability cutoff:

| Metric | Value |
|---|---|
| ROC AUC | 0.856 |
| Accuracy | 0.763 |
| Churn precision | 0.55 |
| Churn recall | 0.82 |
| Churn F1 | 0.66 |

Confusion matrix (test set, 1,057 customers):

|  | Predicted stay | Predicted churn |
|---|---|---|
| **Actually stayed** | 562 | 199 |
| **Actually churned** | 52 | 244 |

The model catches 196 of the 296 customers who actually left.

## Why a 0.25 cutoff?

The default cutoff is 0.5. I used 0.25 so the model flags more possible churners, because missing a customer who leaves usually costs a company more than a retention offer sent by mistake. The cost is more false alarms.

## Limitations

- I did not use stratified splitting, so the churn rate in each split is close to the overall rate but not forced to match.
- I looked at test scores while experimenting early on, so the test AUC may be slightly optimistic.
- The 0.35 cutoff was chosen by hand and not tuned.
- The XGBoost `subsample=0.01` setting is unusually low. It gave the best validation score, and other values I tried scored 0.02 or more lower. I only compared them on one validation split, so I can't say how stable the gap is across different splits.
- The dataset is small (7,043 rows), so scores can shift by a point or two with a different split.

## How to run

1. Install Python 3.12 or newer (the script uses nested quotes inside f-strings, which need 3.12+).
2. Install the libraries:

   ```
   pip install pandas numpy matplotlib scikit-learn xgboost
   ```

3. Put `Telco-Customer-Churn.csv` next to the script and run it:

   ```
   python churn_model.py
   ```

   Replace `churn_model.py` with the name of your script file.

## Tools

Python, pandas, NumPy, Matplotlib, scikit-learn, XGBoost.
