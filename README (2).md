# Credit Risk Prediction & Expected Loss Analysis

An end-to-end data science portfolio project combining **credit card default prediction**, **explainable statistical analysis**, **expected loss estimation**, and **Monte Carlo portfolio credit-risk simulation**.

The project uses the **Default of Credit Card Clients** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), covering 30,000 Taiwanese credit card clients and their repayment history in 2005.

> **Purpose:** Educational and portfolio demonstration only. This model is not validated for lending decisions, regulatory capital calculation, or production risk management.

## Project objectives

- Investigate demographic, credit-limit, billing, and repayment patterns associated with next-month payment default.
- Compare classification algorithms and estimate each client's **Probability of Default (PD)**.
- Assess discrimination, recall, probability calibration, and classification thresholds.
- Estimate illustrative **Expected Loss (EL)** using explicit assumptions about **Loss Given Default (LGD)** and **Exposure at Default (EAD)**.
- Explore risk segmentation, loss concentration, scenario sensitivity, and simulated tail losses.

## Dataset

**Source:** [UCI — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)  
**Observations:** 30,000  
**Predictors:** 23 original features (plus engineered features)  
**Target:** `DEFAULT` — 1 = defaulted on next month's payment; 0 = did not default  
**Class distribution:** 22.12% default / 77.88% no default

| Feature group | Variables |
| --- | --- |
| Credit limit | `LIMIT_BAL` |
| Demographics | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` |
| Repayment status | `PAY_0`, `PAY_2`–`PAY_6` |
| Monthly statement balances | `BILL_AMT1`–`BILL_AMT6` |
| Previous payments | `PAY_AMT1`–`PAY_AMT6` |
| Target | `DEFAULT` |

The imported file initially contained an extra header row. This was corrected, and the resulting dataset contained **30,000 records with no missing values in the original variables**. Undocumented categorical codes were investigated rather than silently assumed to have conventional meanings.

## Methodology

1. **Exploratory Data Analysis:** Data quality, target imbalance, credit limits, ages, bill amounts, payments, repayment delays, credit utilization, demographic subgroups, correlations, and outliers.
2. **Statistical testing:** Chi-square tests, Mann–Whitney U tests, Spearman correlations, and rank-biserial effect sizes.
3. **Feature engineering:** Average bill, average payment, average credit utilization, months with delays, maximum repayment-delay status, and payment-to-bill ratio.
4. **Preprocessing:** 80/20 stratified train/test split; numeric median imputation and scaling; categorical imputation and one-hot encoding in a `scikit-learn` pipeline.
5. **Model comparison:** DummyClassifier, Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost using stratified cross-validation.
6. **Optimization and validation:** Randomized hyperparameter search, probability calibration assessment, training-only threshold selection, and a held-out test evaluation.
7. **Financial risk analysis:** PD-based expected loss, segmented losses, stress scenarios, loss concentration, and Monte Carlo loss simulation.

## Key EDA findings

- Repayment behavior was a leading risk indicator: `MONTHS_DELAYED` had a **Spearman correlation of approximately 0.39** with default; September repayment status (`PAY_0`) had approximately **0.29**.
- Median credit limits were **90,000 NTD** for defaulting clients versus **150,000 NTD** for non-defaulting clients.
- Median average utilization was approximately **46%** for defaulters versus **24%** for non-defaulters.
- The variables showed skewness and extreme financial values. Outliers were not automatically removed because they may reflect legitimate exposures.

## Machine learning results

Initial stratified cross-validation results (rounded):

| Model | ROC-AUC | PR-AUC (AP) | Recall at 0.50 | F1 at 0.50 | Brier score |
| --- | ---: | ---: | ---: | ---: | ---: |
| DummyClassifier | 0.50 | 0.22 | 0.00 | 0.00 | 0.17 |
| Logistic Regression | 0.76 | 0.52 | 0.30 | 0.41 | 0.14 |
| Random Forest | 0.78 | 0.56 | 0.37 | 0.48 | 0.13 |
| XGBoost | 0.79 | 0.56 | 0.37 | 0.48 | 0.13 |
| LightGBM | 0.78 | 0.56 | 0.38 | 0.48 | 0.13 |
| CatBoost | 0.79 | 0.56 | 0.37 | 0.48 | 0.13 |

**Selected candidate:** Tuned XGBoost, with a search-selected cross-validation PR-AUC of approximately **0.57**. Its original probability estimates were retained after sigmoid calibration did not demonstrate a clear benefit.

### Final held-out test evaluation

**Test sample:** 6,000 clients  
**Preselected classification threshold:** 0.25 (chosen using training out-of-fold predictions)

| Metric | Final test result |
| --- | ---: |
| Accuracy | 0.778 |
| Precision (default) | 0.499 |
| Recall (default) | 0.606 |
| F1-score (default) | 0.547 |
| ROC-AUC | ~0.78 |
| PR-AUC (AP) | ~0.56 |
| Brier score | ~0.13 |
| Log loss | ~0.43 |

Confusion matrix:

| Actual / Predicted | No default | Default |
| --- | ---: | ---: |
| No default | **3,866** (TN) | **807** (FP) |
| Default | **523** (FN) | **804** (TP) |

The model detected **804 of 1,327** test-set defaults. Lowering the threshold improved recall at the cost of more false positives. Discrimination and calibration should be assessed separately from the decision threshold.

## Expected loss analysis

For each account, illustrative expected loss was calculated as:

\[
EL_i = PD_i \times LGD_i \times EAD_i
\]

The UCI dataset **does not contain observed LGD, realized losses, or actual EAD**. For demonstration, the base scenario assumes **LGD = 45%** and **EAD = 50% of the recorded credit limit**. This is not an observed or validated EAD estimate.

### Base portfolio results (6,000 test clients)

| Metric | Result |
| --- | ---: |
| Total assumed EAD | 503.89 million NTD |
| Average predicted PD | 22.18% |
| EAD-weighted predicted PD | 17.18% |
| Total expected loss | **38.96 million NTD** |
| Average expected loss per client | **6,494 NTD** |
| Expected loss / EAD | **7.73%** |

### Risk-segment contribution

| Segment | Clients | Mean PD | Share of portfolio EL |
| --- | ---: | ---: | ---: |
| Low | 1,769 | 6.68% | 15.86% |
| Moderate | 2,620 | 15.71% | **33.76%** |
| High | 893 | 34.93% | 21.56% |
| Very high | 718 | 68.11% | 28.82% |

The **top 10% of accounts ranked by individual expected loss** account for **38.63%** of total estimated portfolio EL.

### Sensitivity scenarios

| Scenario | Assumed LGD | Assumed EAD / limit | Total expected loss |
| --- | ---: | ---: | ---: |
| Optimistic | 30% | 30% | 15.58 million NTD |
| Base | 45% | 50% | 38.96 million NTD |
| Stress | 65% | 70% | 78.79 million NTD |

These scenarios **hold PD constant** while varying LGD and EAD; they are sensitivity tests, not a full macroeconomic stress model.

## Monte Carlo simulation

A simulation of **10,000 portfolio outcomes** modeled default events as independent Bernoulli draws, conditional on the predicted PDs, with fixed assumed LGD and EAD.

| Risk measure | Result |
| --- | ---: |
| Analytical expected loss | 38,961,942.99 NTD |
| Simulated mean loss | 38,968,140.40 NTD |
| Simulated loss standard deviation | 1,183,525.66 NTD |
| Credit VaR 95% | 40,927,698.90 NTD |
| Credit VaR 99% | **41,759,505.72 NTD** |
| Expected Shortfall 95% | 41,427,938.48 NTD |
| Expected Shortfall 99% | **42,102,477.00 NTD** |

Simulation validation showed the simulated mean deviated by only **0.02%** from the analytical EL and the simulated standard deviation by **−0.31%** from its analytical counterpart. This checks numerical consistency under the model assumptions, not real-world predictive validity.

## Business insights

- Monitor repeated repayment delays as an early-warning signal.
- Consider both **PD and EAD** when prioritizing portfolio monitoring; a moderate-risk group can contribute the largest aggregate expected loss.
- Review concentration: a relatively small share of accounts accounts for a disproportionate share of estimated EL.
- Use sensitivity analysis to communicate the dependence of results on LGD and exposure assumptions.
- Evaluate fairness, review costs, and regulatory requirements before considering any real lending use.

## Limitations

- Historical Taiwan 2005 data may not generalize to today's borrowers or geographies.
- The outcome represents **next-month payment default**, not necessarily a bank's formal regulatory default definition.
- LGD and EAD are **assumptions**, not measured quantities, so EL and tail-risk outputs are illustrative.
- Independent default draws and constant LGD/EAD omit common economic shocks, correlated defaults, and recovery uncertainty, potentially understating extreme losses.
- The same training data were used for model/parameter selection and subsequent threshold development; the held-out test is the principal final assessment.
- Demographic variables may raise fairness and legal concerns in actual credit decisions.

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the completed `.ipynb` notebook and any dataset file it requires, or use the notebook's UCI data-loading code.
3. Install any missing packages, for example:

   ```python
   %pip install ucimlrepo xgboost lightgbm catboost scikit-learn scipy seaborn matplotlib pandas numpy
   ```

4. Execute notebook cells **from top to bottom**. Check that preprocessing is fitted only on training data and the held-out test set is not used to tune parameters.

> **Notebook path:** Add a link here after uploading your completed notebook to the GitHub repository, e.g. `[Open analysis notebook](./Credit_Risk_Analysis.ipynb)`.

## Tech stack

Python · Google Colab / Jupyter · pandas · NumPy · matplotlib · seaborn · SciPy · scikit-learn · XGBoost · LightGBM · CatBoost

## Possible extensions

- SHAP explanations and model fairness analysis.
- More rigorous nested cross-validation and time-based / external validation.
- Better-specified exposure and recovery models using data with actual observed EAD and LGD.
- Correlated-default and macroeconomic-factor simulations for more realistic tail risk.

## Reference

Yeh, I.-C. & Lien, C.-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications. Dataset hosted by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
