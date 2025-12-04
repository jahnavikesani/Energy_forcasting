from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional heavy imports � fall back gracefully if unavailable
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
PARQUET_PATH = DATA_DIR / "sample.parquet"
CSV_PATH = DATA_DIR / "sample.csv"
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
RECENT_DEFAULT_POINTS = 120

app = FastAPI(title="Energy Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model placeholders
arima_model = None
xgb_model = None
xgb_features: Optional[List[str]] = None
lstm_model = None
lstm_scaler = None


def _bootstrap_series(length: int = 180) -> pd.DataFrame:
    now = datetime.utcnow().replace(second=0, microsecond=0)
    rows = []
    for idx in range(length):
        ts = now - timedelta(minutes=length - idx - 1)
        base = 0.8 + 0.2 * math.sin(idx / 6.0)
        noise = random.uniform(-0.05, 0.05)
        rows.append({"timestamp": ts, "power": max(0.0, base + noise)})
    return pd.DataFrame(rows)


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["lag_1"] = df["power"].shift(1).fillna(df["power"])
    df["rolling_3"] = df["power"].rolling(window=3, min_periods=1).mean()
    return df


def _read_processed_df() -> pd.DataFrame:
    if PARQUET_PATH.exists() and PARQUET_PATH.stat().st_size > 0:
        try:
            return pd.read_parquet(PARQUET_PATH)
        except Exception as exc:  # pragma: no cover - diagnostics only
            print("Failed to read parquet:", exc)
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        try:
            return pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
        except Exception as exc:  # pragma: no cover - diagnostics only
            print("Failed to read CSV:", exc)
    return _bootstrap_series()


def _write_processed_df(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(PARQUET_PATH, index=False)
    except Exception as exc:  # pragma: no cover - diagnostics only
        print("Failed to write parquet:", exc)
    try:
        df.to_csv(CSV_PATH, index=False)
    except Exception as exc:  # pragma: no cover - diagnostics only
        print("Failed to write CSV:", exc)


def ensure_fresh_processed(max_minutes: int = 60, use_real_data: bool = True) -> pd.DataFrame:
    """Load processed data, optionally using real meter data if available"""
    
    # Try to load real meter data first if enabled
    if use_real_data:
        real_data_files = [
            DATA_DIR / "real_meter_data.csv",
            DATA_DIR / "manual_readings.csv",
            DATA_DIR / "green_button_data.csv"
        ]
        
        for file_path in real_data_files:
            if file_path.exists():
                try:
                    df_real = pd.read_csv(file_path, parse_dates=["timestamp"])
                    if not df_real.empty and "timestamp" in df_real.columns and "power" in df_real.columns:
                        df_real = _ensure_feature_columns(df_real)
                        print(f"✓ Using real meter data from {file_path.name}")
                        return df_real
                except Exception as e:
                    print(f"Warning: Could not load {file_path.name}: {e}")
    
    # Fall back to sample data
    df = _ensure_feature_columns(_read_processed_df())
    if df.empty:
        df = _ensure_feature_columns(_bootstrap_series())

    now = datetime.utcnow().replace(second=0, microsecond=0)
    last_ts = df["timestamp"].iloc[-1]
    if last_ts.tzinfo is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)
        last_ts = last_ts.tz_convert(None)

    gap_minutes = int(max(0, (now - last_ts).total_seconds() // 60))
    gap_minutes = min(gap_minutes, max_minutes)
    appended_rows = []
    current_ts = last_ts
    last_power = float(df["power"].iloc[-1]) if not df.empty else 1.0
    for i in range(gap_minutes):
        current_ts = current_ts + timedelta(minutes=1)
        idx = len(df) + i
        trend = 0.0005 * idx
        wave = 0.2 * math.sin(idx / 10.0)
        noise = random.uniform(-0.05, 0.05)
        next_power = max(0.0, last_power + trend + wave + noise)
        appended_rows.append({"timestamp": current_ts, "power": next_power})
        last_power = next_power

    if appended_rows:
        df = pd.concat([df, pd.DataFrame(appended_rows)], ignore_index=True)
        df = _ensure_feature_columns(df)
        if len(df) > 2000:
            df = df.iloc[-2000:].reset_index(drop=True)
        _write_processed_df(df)

    return df


def _recent_series(df: pd.DataFrame, n: int) -> Tuple[List[str], List[float]]:
    tail = df.tail(n)
    timestamps = tail["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    values = tail["power"].astype(float).tolist()
    return timestamps, values


def load_models() -> None:
    global arima_model, xgb_model, xgb_features, lstm_model, lstm_scaler
    try:
        arima_model = joblib.load(MODEL_DIR / "arima_model.pkl")
        print("Loaded ARIMA model")
    except Exception as exc:
        print("ARIMA not loaded:", exc)
    try:
        if xgb is not None:
            xgb_model = xgb.Booster()
            xgb_model.load_model(str(MODEL_DIR / "xgb_model.bst"))
            xgb_features = joblib.load(MODEL_DIR / "xgb_features.joblib")
            print("Loaded XGBoost model")
    except Exception as exc:
        print("XGBoost not loaded:", exc)
    try:
        if load_model is not None:
            lstm_model = load_model(str(MODEL_DIR / "lstm_final.h5"))
            lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.joblib")
            print("Loaded LSTM model")
    except Exception as exc:
        print("LSTM not loaded:", exc)


load_models()


class ForecastRequest(BaseModel):
    horizon: int = 60
    recent_window: Optional[List[float]] = None
    recent_timestamps: Optional[List[str]] = None
    debug: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: ForecastRequest):
    if req.recent_window is None or len(req.recent_window) < 1:
        df = ensure_fresh_processed()
        ts, vals = _recent_series(df[["timestamp", "power"]], RECENT_DEFAULT_POINTS)
        req.recent_window = vals
        req.recent_timestamps = ts

    preds = {"arima": None, "xgb": None, "lstm": None}

    try:
        if arima_model is not None:
            steps = req.horizon
            res = arima_model.get_forecast(steps=steps)
            preds["arima"] = float(res.predicted_mean[-1])
    except Exception:
        preds["arima"] = None

    try:
        if xgb is not None and xgb_model is not None and xgb_features is not None:
            # Iteratively predict for horizon steps
            history = list(req.recent_window)
            ts_list = req.recent_timestamps or []
            
            if ts_list and len(ts_list) == len(req.recent_window):
                try:
                    current_ts = datetime.fromisoformat(ts_list[-1])
                except ValueError:
                    current_ts = datetime.utcnow()
            else:
                current_ts = datetime.utcnow()
            
            # Store predictions at different intervals for debugging
            debug_preds = []
            
            # Predict step by step for horizon minutes
            for step in range(req.horizon):
                # Calculate features for current step
                current_ts = current_ts + timedelta(minutes=1)
                last = float(history[-1])
                prev = float(history[-2]) if len(history) > 1 else last
                rolling_vals = history[-3:] if len(history) >= 3 else history
                rolling = float(np.mean(rolling_vals)) if rolling_vals else last
                
                hour = current_ts.hour + current_ts.minute / 60.0
                dow = current_ts.weekday()  # 0=Monday, 6=Sunday
                
                feature_map = {
                    "sin_hour": float(np.sin(2 * np.pi * hour / 24.0)),
                    "cos_hour": float(np.cos(2 * np.pi * hour / 24.0)),
                    "dow": float(dow),
                    "lag_1": prev,
                    "rolling_3": rolling,
                }
                
                X_vec = [[feature_map.get(name, last) for name in xgb_features]]
                dmat = xgb.DMatrix(np.array(X_vec), feature_names=xgb_features)
                pred_val = float(xgb_model.predict(dmat)[0])
                
                # Add prediction to history for next iteration
                history.append(pred_val)
                
                # Store every 30 steps for debug
                if step % 30 == 0 or step == req.horizon - 1:
                    debug_preds.append({
                        "step": step + 1,
                        "hour": round(hour, 2),
                        "dow": dow,
                        "pred": round(pred_val, 4)
                    })
            
            # Return the final prediction
            preds["xgb"] = history[-1]
            if req.debug and debug_preds:
                preds["xgb_debug"] = debug_preds
    except Exception as e:
        preds["xgb"] = None
        if req.debug:
            preds["xgb_error"] = str(e)

    try:
        from tensorflow.keras.models import load_model as _load_model
        import joblib as _joblib

        model_path = MODEL_DIR / "lstm_final.h5"
        scaler_path = MODEL_DIR / "lstm_scaler.joblib"
        if model_path.exists():
            tmp_model = _load_model(str(model_path))
            tmp_scaler = None
            if scaler_path.exists():
                tmp_scaler = _joblib.load(scaler_path)
            
            # Get current timestamp for features
            ts_list = req.recent_timestamps or []
            if ts_list and len(ts_list) == len(req.recent_window):
                try:
                    current_ts = datetime.fromisoformat(ts_list[-1])
                except ValueError:
                    current_ts = datetime.utcnow()
            else:
                current_ts = datetime.utcnow()
            
            # Start with recent history - need to build full feature arrays
            power_history = list(req.recent_window[-60:])
            if len(power_history) < 60:
                power_history = [power_history[0]] * (60 - len(power_history)) + power_history
            
            # Iteratively predict for horizon steps
            for step in range(req.horizon):
                current_ts = current_ts + timedelta(minutes=1)
                
                # Build feature matrix for last 60 steps (6 features each)
                feature_matrix = []
                for i, power_val in enumerate(power_history[-60:]):
                    # Calculate time-based features for each historical point
                    ts_offset = current_ts - timedelta(minutes=60-i)
                    hour = ts_offset.hour + ts_offset.minute / 60.0
                    dow = ts_offset.weekday()
                    
                    # Get lag and rolling features
                    if i == 0:
                        lag_1 = power_val
                        rolling_3 = power_val
                    elif i < 3:
                        lag_1 = power_history[-60:][i-1] if i > 0 else power_val
                        rolling_3 = float(np.mean(power_history[-60:][:i+1]))
                    else:
                        lag_1 = power_history[-60:][i-1]
                        rolling_3 = float(np.mean(power_history[-60:][i-3:i+1]))
                    
                    feature_matrix.append([
                        power_val,
                        float(np.sin(2 * np.pi * hour / 24.0)),
                        float(np.cos(2 * np.pi * hour / 24.0)),
                        float(dow),
                        lag_1,
                        rolling_3
                    ])
                
                feature_matrix = np.array(feature_matrix)
                
                # Scale the features
                try:
                    if tmp_scaler is not None:
                        scaled = tmp_scaler.transform(feature_matrix)
                        # X is features 1-5 (excluding power), shape (60, 5)
                        X_input = scaled[:, 1:].reshape(1, 60, 5)
                    else:
                        X_input = feature_matrix[:, 1:].reshape(1, 60, 5)
                except Exception as e:
                    print(f"LSTM scaling error: {e}")
                    X_input = feature_matrix[:, 1:].reshape(1, 60, 5)
                
                # Predict (returns scaled power value)
                p = tmp_model.predict(X_input, verbose=0)
                pred_scaled = float(p.flatten()[0])
                
                # Inverse transform to get actual power value
                # Create dummy array with scaled prediction in power column
                if tmp_scaler is not None:
                    dummy = np.zeros((1, 6))
                    dummy[0, 0] = pred_scaled
                    # Use mean values for other features (they don't matter for inverse transform)
                    dummy[0, 1:] = scaled[-1, 1:]
                    pred_val = tmp_scaler.inverse_transform(dummy)[0, 0]
                else:
                    pred_val = pred_scaled
                
                # Add prediction to history for next iteration
                power_history.append(float(pred_val))
            
            # Return the final prediction
            preds["lstm"] = power_history[-1]
    except Exception as exc:
        print("LSTM on-demand load/predict failed:", exc)
        import traceback
        traceback.print_exc()

    available = [v for v in preds.values() if v is not None]
    if not available:
        last = float(req.recent_window[-1])
        models_status = {
            "arima": arima_model is not None,
            "xgb": xgb_model is not None,
            "lstm": lstm_model is not None or (MODEL_DIR / "lstm_final.h5").exists(),
        }
        return {
            "pred_arima": None,
            "pred_xgb": None,
            "pred_lstm": None,
            "ensemble": last,
            "models": models_status,
        }

    ensemble = float(np.mean(available))
    models_status = {
        "arima": arima_model is not None,
        "xgb": xgb_model is not None,
        "lstm": lstm_model is not None or (MODEL_DIR / "lstm_final.h5").exists(),
    }
    return {
        "pred_arima": preds["arima"],
        "pred_xgb": preds["xgb"],
        "pred_lstm": preds["lstm"],
        "ensemble": ensemble,
        "models": models_status,
    }


@app.get("/predict_lstm_demo")
def predict_lstm_demo():
    try:
        from tensorflow.keras.models import load_model as _load_model
        import joblib as _joblib

        data_path = PARQUET_PATH if PARQUET_PATH.exists() else CSV_PATH
        if not data_path.exists():
            return {"error": f"data file not found: {data_path}"}
        df = ensure_fresh_processed()
        seq = df["power"].astype(float).tolist()[-60:]
        import numpy as _np

        seq = _np.array(seq)
        if seq.shape[0] < 60:
            seq = _np.pad(seq, (60 - seq.shape[0], 0), mode="edge")
        model_path = MODEL_DIR / "lstm_final.h5"
        scaler_path = MODEL_DIR / "lstm_scaler.joblib"
        if not model_path.exists():
            return {"error": "lstm model not found"}
        tmp_model = _load_model(str(model_path))
        tmp_scaler = None
        if scaler_path.exists():
            tmp_scaler = _joblib.load(scaler_path)
        if tmp_scaler is not None:
            scaled = tmp_scaler.transform(seq.reshape(-1, 1)).reshape(1, 60, 1)
        else:
            scaled = seq.reshape(1, 60, 1)
        p = tmp_model.predict(scaled)
        return {"pred_lstm": float(p.flatten()[-1])}
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/models/status")
def models_status():
    status = {
        "arima_loaded": arima_model is not None,
        "xgb_loaded": xgb_model is not None,
        "lstm_loaded": lstm_model is not None or (MODEL_DIR / "lstm_final.h5").exists(),
        "paths": {
            "arima": str(MODEL_DIR / "arima_model.pkl"),
            "xgb": str(MODEL_DIR / "xgb_model.bst"),
            "lstm": str(MODEL_DIR / "lstm_final.h5"),
        },
    }
    return status


@app.get("/recent")
def recent(n: int = 60):
    if n <= 0:
        raise HTTPException(status_code=400, detail="n must be positive")
    df = ensure_fresh_processed()
    if df.empty:
        raise HTTPException(status_code=404, detail="No processed data available")
    timestamps, values = _recent_series(df[["timestamp", "power"]], n)
    return {"timestamps": timestamps, "recent": values}
