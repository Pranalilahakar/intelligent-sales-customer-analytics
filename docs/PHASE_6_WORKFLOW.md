# Phase 6: Reproducible Training and Evaluation Workflow

## 1. Prerequisites

Before running Phase 6, ensure that:

- The project virtual environment is activated.
- Required Python dependencies are installed.
- The cleaned dataset exists at `data/processed/sales_analysis.csv`.
- Commands are executed from the project root directory.

## 2. Train and Compare Purchase Propensity Models

Run:

```powershell
python .\customer_purchase_propensity.py
```

This script prepares customer-level features, defines the future purchase target, compares the baseline and machine learning models, and saves the selected model and evaluation outputs.

## 3. Validate Generated Artifacts

Run:

```powershell
python .\validate_customer_purchase_propensity.py
```

This script checks the generated datasets, prediction scores, evaluation metrics, feature metadata, and saved model artifact.

## 4. Run Chronological Backtesting

Run:

```powershell
python .\backtest_customer_purchase_propensity.py
```

This evaluates a model using an earlier training period and a later test period. Keep these results separate from the random train-test split results.

## 5. Evaluate Classification Thresholds

Run:

```powershell
python .\evaluate_purchase_thresholds.py
```

This compares thresholds of 0.30, 0.40, 0.50, 0.60, and 0.70. Review the number of customers targeted, precision, recall, F1-score, and confusion-matrix counts.

A threshold of 0.40 is only a provisional starting point. The final choice should depend on business costs and campaign outcomes.

## 6. Important Output Files

### Datasets and Evaluation Results

- `data/processed/customer_purchase_propensity_dataset.csv`
- `data/processed/customer_purchase_propensity_metrics.csv`
- `data/processed/customer_purchase_propensity_test_predictions.csv`
- `data/processed/customer_purchase_propensity_backtest_metrics.csv`
- `data/processed/customer_purchase_propensity_backtest.csv`
- `data/processed/customer_purchase_propensity_threshold_analysis.csv`

### Saved Model

- `models/customer_purchase_propensity_model.joblib`

### Documentation

- `docs/PHASE_6_MODEL_EVALUATION.md`
- `docs/PHASE_6_WORKFLOW.md`

## 7. Interpretation and Limitations

- The model predicts repeat purchasing during a defined future observation window. It does not predict whether a marketing lead will convert.
- The source data ends in December 2011, so it may not represent current customer behavior.
- Random-split and chronological backtest metrics must be reported separately.
- The scores have not been proven to be calibrated purchase probabilities.
- Threshold selection requires business validation.
- Successful script execution does not by itself establish production readiness.

## 8. Phase 6 Completion Checklist

- [x] Initial model training and comparison completed.
- [x] Basic model artifact validation completed.
- [x] Chronological backtest completed.
- [x] Classification threshold analysis completed.
- [ ] Evaluation document formatting reviewed.
- [ ] Workflow document saved and verified.
- [ ] Final Git changes reviewed before committing.
