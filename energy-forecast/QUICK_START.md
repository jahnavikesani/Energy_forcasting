# Florida Meter Data - Quick Start Guide

## Method 1: Get Data from Your Utility Website (RECOMMENDED)

### FPL Customers:
```
1. Visit: https://www.fpl.com/login
2. Login → Energy Dashboard → My Usage
3. Select "Hourly" view
4. Click "Download" → Save CSV
5. Move file to: energy-forecast\data\processed\
6. Rename to: real_meter_data.csv
```

### Duke Energy Customers:
```
1. Visit: https://www.duke-energy.com/myaccount
2. Login → My Usage → Usage History
3. Select date range (last 2-3 days)
4. Click "Export Data" → Download CSV
5. Move file to: energy-forecast\data\processed\
6. Rename to: real_meter_data.csv
```

### TECO Customers:
```
1. Visit: https://www.tampaelectric.com/myaccount
2. Login → Energy Manager → Usage Data
3. Select "Hourly" view
4. Click "Download" → Export CSV
5. Move file to: energy-forecast\data\processed\
6. Rename to: real_meter_data.csv
```

---

## Method 2: Manual Entry (If No Online Access)

Run this command:
```powershell
.\.venv\Scripts\python.exe scripts\fetch_utility_data.py
```

Then select **Option 3: Manual Entry**

### What You Need:
- Your last 2-3 days of meter readings
- Check your meter physically OR use monthly bill

### Bill Example:
```
December 2025 Bill:
- Dec 1: 450 kWh used
- Dec 2: 475 kWh used  
- Dec 3: 462 kWh used (so far)
```

### How to Convert kWh → kW:
```
Average kW = kWh ÷ 24 hours

Dec 1: 450 kWh ÷ 24 = 18.75 kW average
Dec 2: 475 kWh ÷ 24 = 19.79 kW average
Dec 3: 462 kWh ÷ 24 = 19.25 kW average
```

### Enter in Script Like This:
```
Timestamp: 2025-12-01 12:00
Power (kW): 18.75

Timestamp: 2025-12-02 12:00
Power (kW): 19.79

Timestamp: 2025-12-03 12:00
Power (kW): 19.25
```

---

## Method 3: Smart Meter Direct Reading

If you have a smart meter with LCD display:

1. **Go to your electric meter** (usually outside)
2. **Press the display button** (if it has one)
3. **Look for "Current Demand" or "kW"**
   - Shows instantaneous power usage
   - Example: `18.5 kW` or `18500 W`
4. **Record time and value**
5. **Repeat every few hours** for 2-3 days

---

## CSV Format Required

Your CSV file must have these columns:

```csv
timestamp,power
2025-12-03 08:00:00,18500
2025-12-03 09:00:00,19200
2025-12-03 10:00:00,20100
```

**Important:**
- `timestamp`: YYYY-MM-DD HH:MM:SS format
- `power`: In WATTS (not kW)
  - If you have kW, multiply by 1000
  - Example: 18.5 kW = 18500 watts

---

## After Getting Data

### Option A: If you downloaded CSV from utility:
```powershell
# Copy file to project
Copy-Item "C:\Downloads\your_meter_data.csv" ".\data\processed\real_meter_data.csv"

# Run comparison
.\.venv\Scripts\python.exe scripts\compare_predictions.py
```

### Option B: If you used manual entry:
```powershell
# Just run comparison (file already saved)
.\.venv\Scripts\python.exe scripts\compare_predictions.py
```

---

## Troubleshooting

### "I don't have online access"
- Call your utility customer service
- Ask them to enable "Online Account Management"
- Takes 24-48 hours to activate

### "My meter doesn't have a display"
- Use manual entry from monthly bills
- Estimate hourly usage patterns:
  - Morning (6-9 AM): Higher usage (cooking, AC)
  - Midday (12-3 PM): Peak usage (AC, appliances)
  - Evening (6-9 PM): High usage (cooking, lights, AC)
  - Night (10 PM-6 AM): Lower usage (baseline)

### "CSV has wrong format"
Run the fetch script to convert:
```powershell
.\.venv\Scripts\python.exe scripts\fetch_utility_data.py
# Select Option 2: Process existing CSV
```

---

## Quick Test

To verify your data is working:

```powershell
# Check if file exists
Test-Path .\data\processed\real_meter_data.csv

# See first few lines
Get-Content .\data\processed\real_meter_data.csv -Head 5

# Run backend (it will auto-detect real data)
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Look for this message in backend logs:
```
✓ Using real meter data from real_meter_data.csv
```

---

## Contact Your Utility

**FPL:** 1-800-226-3545  
**Duke Energy:** 1-800-228-8485  
**TECO:** 1-877-588-1010

Ask for: "Access to my smart meter hourly usage data"
