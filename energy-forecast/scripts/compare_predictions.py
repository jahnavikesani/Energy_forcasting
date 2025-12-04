"""
Compare real meter readings with model predictions
Shows accuracy metrics and visualization
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
import json

API_URL = "http://127.0.0.1:8000"

def load_meter_data():
    """Load real meter data from processed folder"""
    data_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    
    # Try different file names
    for filename in ["real_meter_data.csv", "manual_readings.csv", "green_button_data.csv"]:
        file_path = data_dir / filename
        if file_path.exists():
            print(f"✓ Loading meter data from {filename}")
            df = pd.read_csv(file_path, parse_dates=["timestamp"])
            return df
    
    print("✗ No meter data found. Run fetch_utility_data.py first.")
    return None

def get_model_prediction(horizon=60):
    """Get prediction from the model API"""
    try:
        payload = {"horizon": horizon}
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        if response.ok:
            return response.json()
        else:
            print(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Failed to get prediction: {e}")
        return None

def compare_predictions(meter_df, horizons=[1, 15, 30, 60, 120]):
    """Compare model predictions with actual meter readings"""
    
    if meter_df is None or len(meter_df) == 0:
        print("No meter data to compare")
        return
    
    # Get latest meter reading
    latest = meter_df.sort_values("timestamp").iloc[-1]
    latest_time = latest["timestamp"]
    latest_power = latest["power"]
    
    print("\n" + "="*80)
    print("PREDICTION vs ACTUAL COMPARISON")
    print("="*80)
    print(f"\nCurrent Reading:")
    print(f"  Time:  {latest_time}")
    print(f"  Power: {latest_power:,.2f} watts ({latest_power/1000:.2f} kW)")
    print(f"\n{'Horizon':<12} {'Model Pred':<15} {'Actual':<15} {'Error':<15} {'Error %'}")
    print("-"*80)
    
    results = []
    
    for horizon in horizons:
        # Get model prediction
        pred_data = get_model_prediction(horizon)
        
        if pred_data is None:
            continue
        
        model_pred = pred_data.get("ensemble")
        
        # Find actual reading at that horizon (if available)
        target_time = latest_time + timedelta(minutes=horizon)
        
        # Find closest reading within 5 minutes
        time_diffs = abs((meter_df["timestamp"] - target_time).dt.total_seconds() / 60)
        closest_idx = time_diffs.idxmin()
        
        if time_diffs[closest_idx] <= 5:  # Within 5 minutes
            actual = meter_df.loc[closest_idx, "power"]
            error = model_pred - actual
            error_pct = (error / actual) * 100
            
            print(f"{horizon} min{'':<6} {model_pred:>10,.2f} W  {actual:>10,.2f} W  {error:>10,.2f} W  {error_pct:>6.2f}%")
            
            results.append({
                "horizon": horizon,
                "model_pred": model_pred,
                "actual": actual,
                "error": error,
                "error_pct": error_pct
            })
        else:
            print(f"{horizon} min{'':<6} {model_pred:>10,.2f} W  {'N/A':<15} (no data)")
    
    if results:
        # Summary statistics
        results_df = pd.DataFrame(results)
        mae = results_df["error"].abs().mean()
        rmse = np.sqrt((results_df["error"] ** 2).mean())
        mape = results_df["error_pct"].abs().mean()
        
        print("\n" + "="*80)
        print("ACCURACY METRICS")
        print("="*80)
        print(f"Mean Absolute Error (MAE):       {mae:,.2f} watts ({mae/1000:.2f} kW)")
        print(f"Root Mean Square Error (RMSE):   {rmse:,.2f} watts ({rmse/1000:.2f} kW)")
        print(f"Mean Absolute Percentage Error:  {mape:.2f}%")
        
        if mape < 5:
            print("\n✓ Excellent accuracy (< 5% error)")
        elif mape < 10:
            print("\n✓ Good accuracy (< 10% error)")
        elif mape < 15:
            print("\n⚠ Acceptable accuracy (< 15% error)")
        else:
            print("\n✗ Poor accuracy (> 15% error) - consider retraining")
        
        return results_df
    else:
        print("\n✗ No actual readings found for comparison")
        return None

def analyze_individual_models(meter_df, horizon=60):
    """Compare individual model performance"""
    
    if meter_df is None or len(meter_df) == 0:
        return
    
    pred_data = get_model_prediction(horizon)
    if pred_data is None:
        return
    
    latest = meter_df.sort_values("timestamp").iloc[-1]
    latest_time = latest["timestamp"]
    target_time = latest_time + timedelta(minutes=horizon)
    
    # Find actual reading
    time_diffs = abs((meter_df["timestamp"] - target_time).dt.total_seconds() / 60)
    closest_idx = time_diffs.idxmin()
    
    if time_diffs[closest_idx] > 5:
        print(f"\n✗ No actual data found near {target_time}")
        return
    
    actual = meter_df.loc[closest_idx, "power"]
    
    print(f"\n" + "="*80)
    print(f"INDIVIDUAL MODEL COMPARISON (Horizon: {horizon} minutes)")
    print("="*80)
    print(f"Actual reading at {target_time}: {actual:,.2f} watts ({actual/1000:.2f} kW)\n")
    print(f"{'Model':<12} {'Prediction':<15} {'Error':<15} {'Error %':<12} {'Status'}")
    print("-"*80)
    
    models = {
        "ARIMA": pred_data.get("pred_arima"),
        "XGBoost": pred_data.get("pred_xgb"),
        "LSTM": pred_data.get("pred_lstm"),
        "Ensemble": pred_data.get("ensemble")
    }
    
    for model_name, prediction in models.items():
        if prediction is not None:
            error = prediction - actual
            error_pct = (error / actual) * 100
            
            if abs(error_pct) < 5:
                status = "✓ Excellent"
            elif abs(error_pct) < 10:
                status = "✓ Good"
            elif abs(error_pct) < 15:
                status = "⚠ Fair"
            else:
                status = "✗ Poor"
            
            print(f"{model_name:<12} {prediction:>10,.2f} W  {error:>10,.2f} W  {error_pct:>6.2f}%     {status}")
        else:
            print(f"{model_name:<12} {'N/A':<15} {'N/A':<15} {'N/A':<12} ✗ Failed")

def live_monitoring_mode():
    """Continuous monitoring - compare predictions as new data comes in"""
    import time
    
    print("\n" + "="*80)
    print("LIVE MONITORING MODE")
    print("="*80)
    print("Checking predictions every 60 seconds. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            meter_df = load_meter_data()
            if meter_df is not None:
                latest = meter_df.sort_values("timestamp").iloc[-1]
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
                print(f"Latest meter: {latest['power']/1000:.2f} kW at {latest['timestamp']}")
                
                pred = get_model_prediction(horizon=1)
                if pred:
                    print(f"1-min forecast: {pred['ensemble']/1000:.2f} kW")
                    print(f"Models: ARIMA={pred.get('pred_arima',0)/1000:.2f}, "
                          f"XGB={pred.get('pred_xgb',0)/1000:.2f}, "
                          f"LSTM={pred.get('pred_lstm',0)/1000:.2f} kW")
            
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")

if __name__ == "__main__":
    print("="*80)
    print("MODEL vs METER COMPARISON TOOL")
    print("="*80)
    
    print("\nSelect mode:")
    print("1. Compare predictions at multiple horizons")
    print("2. Analyze individual model performance")
    print("3. Live monitoring (continuous)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    meter_df = load_meter_data()
    
    if choice == "1":
        horizons = [1, 15, 30, 60, 120, 180]
        compare_predictions(meter_df, horizons)
    
    elif choice == "2":
        horizon = int(input("Enter horizon in minutes (default 60): ") or "60")
        analyze_individual_models(meter_df, horizon)
    
    elif choice == "3":
        live_monitoring_mode()
    
    else:
        print("Invalid choice")
