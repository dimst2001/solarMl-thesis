 What's Next — In Order
 (finished steps are in alreadydone.MD)


Notebook 2 — Feature Selection (finish the exogenous configuration)

Choose the datetime features for the exogenous config (e.g. hour, day of year, month; cyclical sin/cos encoding)

Implement them in build_feature_frame and set "exogenous" to enabled in FEATURE_CONFIGS

Re-run notebook 2 -> data/features/group_4_exogenous/ (same 80/20 chronological split + MinMaxScaler)



Notebook 3 — Regression Models (run for both configs: endogenous and exogenous)

Load the train/test files listed in artifacts/metrics/system_4_feature_configs.csv

Linear Regression (baseline)

Decision Tree Regressor

Random Forest Regressor

MLP Regressor

Multi random state cross validation on the training set (repeat with several seeds, report mean ± std)

Each model: train → predict → evaluate



Notebook 4 — Evaluation & Comparison

MAE, RMSE, R² for all models, per config (endogenous vs exogenous)

Training time + model size comparison

Predicted vs actual plots

Feature importance (Random Forest)

Identify best model (accuracy + efficiency)



Notebook 5 — Results Summary

Final comparison table

Best model conclusion

Charts ready for thesis document
