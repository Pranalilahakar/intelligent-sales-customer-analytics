# Phase 6: Customer Purchase Propensity Model Evaluation

## 1. Project Objective

The objective of Phase 6 is to develop and evaluate a machine learning model that identifies customers who are more likely to make another purchase during a defined future period.

The project includes:

- Preparing customer-level features and a future purchase target.
- Comparing baseline and machine learning models.
- Evaluating model performance using standard classification metrics.
- Performing a chronological backtest.
- Comparing different classification thresholds.
- Saving model artifacts, predictions, and evaluation results.

**Important:** This is a repeat-purchase propensity prediction task. It is not a lead conversion prediction task.

## 2. Dataset and Target Definition

### Dataset Overview

- Source: Historical retail transaction dataset.
- Valid transaction rows: 392,692.
- Unique customers across the complete dataset: 4,338.
- Available date range: December 1, 2010 to December 9, 2011.

### Initial Model Periods

- Feature period: December 1, 2010 to September 30, 2011.
- Outcome period: October 1, 2011 to December 9, 2011.
- Customers in the feature dataset: 3,616.
- Customers who purchased during the outcome period: 1,838.
- Customers without an observed purchase during the outcome period: 1,778.

### Target Definition

The target column is `Purchased_Future_Window`.

- `1` = The customer made at least one observed purchase during the outcome period.
- `0` = No purchase was observed for that customer during the outcome period.

A customer labelled `0` is not necessarily permanently churned. The label only describes purchasing behavior during the specified observation window.

## 3. Feature Engineering

The following customer-level features are used by the initial purchase propensity model.

| Feature | Description |
|---|---|
| `Recency_Days` | Days since the customer's most recent purchase within the feature period |
| `Total_Invoices` | Number of distinct invoices associated with the customer |
| `Total_Revenue` | Total recorded revenue during the feature period |
| `Total_Transactions` | Number of transaction records associated with the customer |
| `Average_Quantity` | Average quantity per transaction record |
| `Average_Order_Value` | Average recorded invoice value |

Only historical behavior available within the feature period should be used to calculate the model inputs. Future outcome-period purchases must not be included in the features.

## 4. Initial Model Comparison

Three approaches were evaluated using a stratified random train-test split.

The dataset contained 3,616 customers, with 2,892 assigned to training and 724 assigned to testing.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.5083 | 0.5083 | 1.0000 | 0.6740 | 0.5000 | 0.5083 | 0.2499 |
| Logistic Regression | 0.6837 | 0.7294 | 0.6005 | 0.6587 | 0.7612 | 0.7847 | 0.1986 |
| Random Forest | 0.6823 | 0.7078 | 0.6386 | 0.6714 | 0.7424 | 0.7734 | 0.2045 |

### Interpretation

- Logistic Regression achieved a ROC-AUC of 0.7612, indicating useful ranking ability on the random test split.
- Random Forest achieved a slightly higher F1-score of 0.6714 compared with 0.6587 for Logistic Regression.
- Logistic Regression was selected for the initial saved model artifact.
- The baseline provides a reference point for evaluating whether the machine learning models offer useful predictive performance.

The Logistic Regression selection is provisional. The preferred model may change depending on the business objective, the cost of false positives and false negatives, and performance on later data.

## 5. Chronological Backtest

A separate chronological evaluation was performed to assess the model using an earlier training period and a later test period.

### Backtest Design

**Training snapshot**
- Feature data through June 30, 2011.
- Future purchase outcome window: July 1 to August 31, 2011.
- Customers in the training snapshot: 2,960.

**Test snapshot**
- Feature data through August 31, 2011.
- Future purchase outcome window: September 1 to October 31, 2011.
- Customers in the test snapshot: 3,317.

The customer cohorts may overlap because customers can appear in both historical snapshots. Each snapshot uses its own feature cutoff and subsequent outcome window.

### Chronological Test Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.5526 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.4474 | 0.2512 |
| Logistic Regression | 0.6862 | 0.7573 | 0.4394 | 0.5561 | 0.7459 | 0.7297 | 0.2037 |

### Interpretation

- Logistic Regression achieved a ROC-AUC of 0.7459 on the chronological test cohort.
- Its precision was 0.7573, while recall was 0.4394 at the evaluated classification threshold.
- The baseline predicted no positive customers at its default decision threshold, resulting in zero precision and recall.
- The chronological results are separate from the initial random-split results and should be reported independently.

The chronological evaluation provides additional evidence of ranking performance on this historical period. However, one backtest does not establish reliable performance on current or future customer populations.

## 6. Classification Threshold Analysis

The threshold analysis compares how different decision thresholds affect the number of customers selected for targeting.

A lower threshold generally selects more customers, increasing recall while potentially increasing false positives. A higher threshold generally selects fewer customers and can increase precision while reducing recall.

| Threshold | Customers Targeted | Precision | Recall | F1-score |
|---:|---:|---:|---:|---:|
| 0.30 | 1,739 | 0.6072 | 0.7116 | 0.6553 |
| 0.40 | 1,200 | 0.6992 | 0.5654 | 0.6252 |
| 0.50 | 861 | 0.7573 | 0.4394 | 0.5561 |
| 0.60 | 632 | 0.8149 | 0.3470 | 0.4868 |
| 0.70 | 468 | 0.8568 | 0.2702 | 0.4109 |

### Threshold Interpretation

- **0.30:** Targets the largest group and captures more future purchasers, but has lower precision.
- **0.40:** Provides a provisional balance between targeting volume and precision.
- **0.50:** Targets fewer customers with higher precision than the lower thresholds.
- **0.60:** Targets a smaller, more selective group.
- **0.70:** Has the highest precision among the evaluated thresholds but the lowest recall.

A threshold of `0.40` is a provisional starting point for further investigation, not a universally optimal choice. The appropriate threshold depends on campaign costs, the value of repeat purchases, and the business's tolerance for missed customers.

The threshold analysis uses the same historical test cohort for all thresholds. It is a comparison of decision rules, not an independent validation of the chosen threshold.

## 7. Evaluation Metrics Explained

| Metric | Meaning |
|---|---|
| Accuracy | Proportion of all predictions that are correct |
| Precision | Proportion of predicted purchasers who actually purchased |
| Recall | Proportion of actual purchasers identified by the model |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ability to rank positive cases above negative cases across thresholds |
| PR-AUC | Area under the precision-recall curve |
| Brier Score | Mean squared error of predicted probabilities against binary outcomes; lower is better |

A strong result on one metric does not guarantee that the model is suitable for every business objective. Model selection should consider several metrics and the costs of different errors.

## 8. Generated Project Artifacts

### Datasets and Evaluation Results

- `data/processed/customer_purchase_propensity_dataset.csv`
- `data/processed/customer_purchase_propensity_metrics.csv`
- `data/processed/customer_purchase_propensity_test_predictions.csv`
- `data/processed/customer_purchase_propensity_backtest_metrics.csv`
- `data/processed/customer_purchase_propensity_backtest.csv`
- `data/processed/customer_purchase_propensity_threshold_analysis.csv`

### Model Artifact

- `models/customer_purchase_propensity_model.joblib`

The saved model artifact contains the selected model and associated feature metadata, as produced by the training script.

### Scripts

- `customer_purchase_propensity.py`
- `validate_customer_purchase_propensity.py`
- `backtest_customer_purchase_propensity.py`
- `evaluate_purchase_thresholds.py`

## 9. Limitations and Risks

1. **Historical data:** The dataset ends in December 2011. Its customer behavior may not represent current market conditions.
2. **Repeat purchase versus lead conversion:** The target measures repeat purchasing, not whether a marketing lead becomes a customer.
3. **Random-split limitation:** A random split does not prove that the model generalizes to future time periods.
4. **Limited chronological testing:** The backtest covers one historical test window and should be extended to additional periods.
5. **Probability calibration:** The scores have not been demonstrated to be calibrated probabilities of purchase.
6. **Threshold selection:** The evaluated thresholds do not account for campaign costs, customer lifetime value, or incremental marketing impact.
7. **Unobserved behavior:** A customer without a purchase in the outcome window may purchase later.
8. **Production readiness:** Successful script execution and artifact validation do not establish production reliability.
9. **Data quality:** Future use requires checks for missing values, unusual transactions, duplicates, and consistency in the source data.
10. **Feature consistency:** Any future inference workflow must calculate features using the same definitions and compatible preprocessing as the trained model.

## 10. Conclusion

Phase 6 establishes an initial customer purchase propensity modeling workflow with baseline comparison, Logistic Regression and Random Forest evaluation, a chronological backtest, threshold analysis, and saved artifacts.

The initial Logistic Regression model achieved a ROC-AUC of 0.7612 on a random test split. In the separate chronological backtest, it achieved a ROC-AUC of 0.7459. These results suggest useful ranking ability on the evaluated historical data, but they do not establish performance on present-day customers.

The model is suitable for continued experimentation and development. Before business deployment, it requires additional chronological validation, evaluation on more recent data, probability calibration if probabilities will be interpreted literally, and testing against actual business outcomes.

**Phase 6 status:** Initial modeling, artifact validation, chronological backtesting, and threshold analysis have been completed. Final workflow verification and review of the project deliverables remain necessary before declaring the phase complete.