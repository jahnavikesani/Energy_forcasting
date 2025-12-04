"""Train all forecasting models and save optional metrics.

Usage:
    python -m backend.scripts.train_all --processed sample.parquet

Outputs (written under backend/app/models/):
    arima_model.pkl
    xgb_model.bst
    xgb_features.joblib
    lstm_final.h5
    lstm_scaler.joblib
    model_metrics.joblib   # dict of validation RMSE for weighting ensemble
"""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np

from .train_arima import train_arima
from .train_xgb import train_xgb
from .train_lstm import train_lstm
from app.utils import load_processed
from sklearn.metrics import mean_squared_error

MODEL_DIR = Path(__file__).resolve().parents[1] / "app" / "models"


def eval_arima(model, df):
    # Use last 120 points for quick eval
    ts = df.set_index('timestamp')['power'].asfreq('T').fillna(method='ffill')
    tail = ts.iloc[-180:]
    split = int(len(tail) * 0.7)
    train, test = tail.iloc[:split], tail.iloc[split:]
    fitted = model.apply(train)
    # Forecast length of test
    fc = fitted.get_forecast(steps=len(test)).predicted_mean
    # Align lengths
    rmse = float(mean_squared_error(test.values[: len(fc)], fc.values[: len(test)], squared=False))
    return rmse


def main(processed_file: str):
    df = load_processed(processed_file)
    metrics = {}

    print("Training ARIMA...")
    arima_model = train_arima(processed_file)
    try:
        metrics['arima'] = eval_arima(arima_model, df)
    except Exception as exc:
        print("ARIMA metric failed:", exc)

    print("Training XGBoost...")
    xgb_model = train_xgb(processed_file)
    # train_xgb already prints RMSE; recompute for metrics from its held-out set if desired not necessary
    # For simplicity we skip duplicate evaluation.

    print("Training LSTM...")
    lstm_model = train_lstm(processed_file)
    # LSTM RMSE printed (scaled). We could add inverse scaling if needed; omit for now.

    metrics_path = MODEL_DIR / "model_metrics.joblib"
    joblib.dump(metrics, metrics_path)
    print("Saved metrics to", metrics_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="sample.parquet", help="Processed parquet filename")
    args = parser.parse_args()
    main(args.processed)