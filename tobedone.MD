 What's Next — In Order
 (finished steps are in alreadydone.MD)


Notebook 4 — Evaluation & Comparison (new; reads saved files only, no retraining)

Load artifacts/metrics/system_4_<config>_model_metrics.csv and artifacts/experiments/system_4_<config>_test_predictions.csv for endogenous and exogenous

4.1 Metrics table: MAE, RMSE, R² and CV mean ± std per model, endogenous vs exogenous

4.2 Efficiency: fit time, predict time and model size against accuracy

4.3 Predicted vs actual: scatter per model + time-series overlay for sample days (clear, cloudy, winter)

4.4 Residuals: error by irradiance level and by month / season

4.5 Feature importance: Random Forest + XGBoost built-in, permutation importance on a test sample

4.6 Identify best model (accuracy + efficiency)



Notebook 5 — Results Summary

Final comparison table

Best model conclusion

Charts ready for thesis document



Optional

Light hyperparameter tuning (small time-series grid search on the CV subsample)
