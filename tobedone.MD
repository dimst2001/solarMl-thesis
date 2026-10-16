 What's Next — In Order
 (finished steps are in alreadydone.MD)


Notebook 2 — Feature Selection (finish the exogenous configuration)

Choose the datetime features for the exogenous config (e.g. hour, day of year, month; cyclical sin/cos encoding)

Add them to FEATURE_CONFIGS["exogenous"]["datetime_features"] and implement them in build_feature_frame (section 7)

Uncomment pipeline 8B (8B.1-8B.7) and re-run notebook 2 -> data/features/group_4_exogenous/ (same 80/20 chronological split + MinMaxScaler)



Notebook 3B — Training Regressor Models (Exogenous)

Run notebooks/3B_Training_Regressor_Models_Exogenous.ipynb once pipeline 8B has exported the exogenous data
(same structure as 3A: 3B.1 Linear Regression ... 3B.5 XGBoost, 3B.6 comparison summary)



Notebook 4 — Evaluation & Comparison

Load artifacts/metrics/system_4_<config>_model_metrics.csv and artifacts/experiments/system_4_<config>_test_predictions.csv from 3A and 3B

MAE, RMSE, R² for all models, per config (endogenous vs exogenous)

Training time + model size comparison

Predicted vs actual plots

Feature importance (Random Forest, XGBoost)

Identify best model (accuracy + efficiency)



Notebook 5 — Results Summary

Final comparison table

Best model conclusion

Charts ready for thesis document
