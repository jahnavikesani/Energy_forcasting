# Using Real Meter Data - Quick Start Guide for Florida

## Step 1: Get Your Meter Data

### Option A: Manual Entry (Easiest - Start Here!)
```powershell
cd C:\Users\jahna\OneDrive\Desktop\overhead\energy-forecast
.\.venv\Scripts\python.exe scripts\fetch_utility_data.py
# Select option 3 (Manual Entry)
```

**Enter your readings like this:**
```
2025-12-03 08:00, 22.5
2025-12-03 08:15, 23.1
2025-12-03 08:30, 22.8
2025-12-03 08:45, 22.3
```
(Press Enter twice when done)

### Option B: Download from Your Utility Website

**FPL (Florida Power & Light):**
1. Go to https://www.fpl.com
2. Login → Energy Dashboard → Export Data
3. Download as CSV or Green Button XML
4. Save to `data/processed/real_meter_data.csv`

**Duke Energy:**
1. Go to https://www.duke-energy.com
2. Login → My Usage → Download Usage Data
3. Save to `data/processed/real_meter_data.csv`

**Format your CSV like this:**
```csv
timestamp,power
2025-12-03 08:00:00,22500
2025-12-03 08:15:00,23100
2025-12-03 08:30:00,22800
```
(power is in watts, not kW)

### Option C: Green Button XML
```powershell
cd C:\Users\jahna\OneDrive\Desktop\overhead\energy-forecast
.\.venv\Scripts\python.exe scripts\fetch_utility_data.py
# Select option 2, then enter path to downloaded XML file
```

---

## Step 2: Restart Backend (It Will Auto-Detect Real Data)

Your backend is already running and will automatically use real meter data if it finds:
- `data/processed/real_meter_data.csv`
- `data/processed/manual_readings.csv`
- `data/processed/green_button_data.csv`

Just restart it:
```powershell
# Stop current backend (Ctrl+C in the terminal)
# Then restart:
cd C:\Users\jahna\OneDrive\Desktop\overhead\energy-forecast
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see: `✓ Using real meter data from real_meter_data.csv`

---

## Step 3: Compare Predictions with Reality

```powershell
cd C:\Users\jahna\OneDrive\Desktop\overhead\energy-forecast
.\.venv\Scripts\python.exe scripts\compare_predictions.py
```

**Select option 1** to see comparison at multiple time horizons:
```
Horizon      Model Pred      Actual          Error           Error %
--------------------------------------------------------------------------------
1 min        22,450.00 W     22,500.00 W     -50.00 W       -0.22%
15 min       22,100.00 W     22,300.00 W    -200.00 W       -0.90%
60 min       21,800.00 W     21,900.00 W    -100.00 W       -0.46%
```

**Select option 2** to analyze individual models:
```
INDIVIDUAL MODEL COMPARISON (Horizon: 60 minutes)
--------------------------------------------------------------------------------
Model        Prediction      Error           Error %      Status
ARIMA        22,820.00 W     920.00 W        4.20%        ✓ Excellent
XGBoost      21,702.00 W     -198.00 W      -0.90%        ✓ Excellent
LSTM         21,129.00 W     -771.00 W      -3.52%        ✓ Excellent
Ensemble     21,884.00 W     -16.00 W       -0.07%        ✓ Excellent
```

**Select option 3** for live monitoring (updates every 60 seconds):
```
[2025-12-03 14:30:00]
Latest meter: 22.50 kW at 2025-12-03 14:29:00
1-min forecast: 22.45 kW
Models: ARIMA=22.82, XGB=21.70, LSTM=21.13 kW
```

---

## Step 4: Using Real Data in Streamlit

Open your Streamlit app: http://localhost:8501

The "Get Forecast" button now uses your real meter data automatically!

---

## Quick Florida Utility Tips

### FPL Customers:
- Smart meter data updates every 15-30 minutes
- Can request hourly data through Energy Dashboard
- API access requires business account

### Duke Energy Customers:
- Interval data available (15-minute readings)
- Download via "My Usage" portal
- Green Button supported

### TECO Customers:
- Hourly data available online
- Energy Manager tool shows real-time usage
- Export available in CSV format

---

## Troubleshooting

**"No meter data found"**
- Make sure your CSV is in `data/processed/` folder
- Check filename: must be `real_meter_data.csv`, `manual_readings.csv`, or `green_button_data.csv`
- Verify CSV has columns: `timestamp`, `power`

**"Power values seem wrong"**
- Check if you're using kW or watts (models expect watts)
- 1 kW = 1000 watts
- Typical home: 2-5 kW average, 20-30 kW with AC/heat pump

**"Predictions don't match"**
- Make sure you have at least 60 data points
- Check timestamps are continuous (1-minute intervals preferred)
- Retrain models with your data: `python backend/scripts/train_all.py`

---

## Converting Your Data to Right Format

If your utility gives you kWh per hour:
```python
# kWh per hour is average kW during that hour
# Example: 1.5 kWh in 1 hour = 1.5 kW average = 1500 watts
power_watts = kwh_value * 1000
```

If you have 15-minute intervals:
```python
# 0.375 kWh in 15 minutes
# = 0.375 kWh / 0.25 hours = 1.5 kW average
# = 1500 watts
power_watts = (kwh_value / hours) * 1000
```

---

## Next Steps

1. **Start with manual entry** - Enter 10-20 readings to test
2. **Compare accuracy** - Run compare_predictions.py
3. **Download more data** - Get historical data from your utility
4. **Retrain models** - Use your data for better accuracy
5. **Set up automation** - Schedule data fetches if using API

**Need help?** Check the logs in:
- Backend: Terminal where uvicorn is running
- Streamlit: Terminal where streamlit is running
