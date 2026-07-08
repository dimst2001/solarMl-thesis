# Solar PV Power Prediction — IEEE Thesis
**Author:** Raul
**Dataset:** NREL PVDAQ (data.openei.org/submissions/4568)

## Project Structure
notebooks/          → Jupyter notebooks (analysis + ML models)
metrics/            → Sensor metadata per system
scripts/            → Data download and processing scripts
data/sample/        → Small data samples for demonstration

## Setup
pip install pandas pyarrow scikit-learn matplotlib seaborn xgboost jupyterlab

## Data Download
aws s3 cp --no-sign-request s3://oedi-data-lake/pvdaq/parquet/...
