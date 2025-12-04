"""
Fetch real-time energy data from Florida utility APIs
Supports: FPL, Duke Energy, TECO, and generic Green Button data
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

# Configuration
CONFIG = {
    "utility": "FPL",  # Options: "FPL", "Duke", "TECO", "GreenButton"
    "account_number": "",  # Your account number
    "api_key": "",  # Get from utility website
    "username": "",
    "password": "",
}

class UtilityDataFetcher:
    """Fetch energy data from utility smart meters"""
    
    def __init__(self, config):
        self.config = config
        self.utility = config["utility"]
        
    def fetch_fpl_data(self, start_date, end_date):
        """
        Fetch FPL data via their API
        Note: Requires FPL account and API access
        Documentation: https://www.fpl.com/api-documentation
        """
        url = "https://api.fpl.com/energy/usage"
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
        params = {
            "account": self.config["account_number"],
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "interval": "hour"  # or "15min" for more granular data
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data["readings"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["power"] = df["kwh"] * 1000  # Convert kWh to watts
            return df[["timestamp", "power"]]
        except Exception as e:
            print(f"FPL API Error: {e}")
            return None
    
    def fetch_duke_data(self, start_date, end_date):
        """
        Fetch Duke Energy data
        Documentation: https://www.duke-energy.com/myaccount/api
        """
        url = "https://api.duke-energy.com/usage/interval"
        headers = {
            "X-API-Key": self.config["api_key"],
            "Authorization": f"Basic {self.config['username']}:{self.config['password']}"
        }
        params = {
            "accountNumber": self.config["account_number"],
            "startDate": start_date.strftime("%Y%m%d"),
            "endDate": end_date.strftime("%Y%m%d")
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data["intervalData"])
            df["timestamp"] = pd.to_datetime(df["readTime"])
            df["power"] = df["usage"] * 1000  # Convert kWh to watts
            return df[["timestamp", "power"]]
        except Exception as e:
            print(f"Duke Energy API Error: {e}")
            return None
    
    def fetch_green_button_data(self, xml_file):
        """
        Parse Green Button XML file
        Most utilities support Green Button standard
        Download from your utility's website
        """
        import xml.etree.ElementTree as ET
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            readings = []
            for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
                timestamp_elem = entry.find(".//{http://naesb.org/espi}start")
                value_elem = entry.find(".//{http://naesb.org/espi}value")
                
                if timestamp_elem is not None and value_elem is not None:
                    timestamp = datetime.fromtimestamp(int(timestamp_elem.text))
                    power = float(value_elem.text) * 1000  # Convert to watts
                    readings.append({"timestamp": timestamp, "power": power})
            
            df = pd.DataFrame(readings)
            return df
        except Exception as e:
            print(f"Green Button parsing error: {e}")
            return None
    
    def fetch_data(self, days_back=7):
        """Fetch data based on configured utility"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        if self.utility == "FPL":
            return self.fetch_fpl_data(start_date, end_date)
        elif self.utility == "Duke":
            return self.fetch_duke_data(start_date, end_date)
        else:
            print(f"Utility {self.utility} not implemented. Use Green Button XML instead.")
            return None
    
    def save_to_processed(self, df, filename="real_meter_data.csv"):
        """Save data to processed folder"""
        if df is None:
            return
        
        data_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = data_dir / filename
        df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(df)} readings to {output_path}")
        return output_path

def manual_entry_mode():
    """
    Manual mode: Enter your readings manually
    Use this if you don't have API access
    """
    print("\n" + "="*60)
    print("MANUAL METER READING ENTRY")
    print("="*60)
    print("\nEnter your meter readings (press Enter twice when done):")
    print("Format: YYYY-MM-DD HH:MM, kW_value")
    print("Example: 2025-12-03 08:00, 22.5")
    print()
    
    readings = []
    while True:
        entry = input("Reading: ").strip()
        if not entry:
            break
        
        try:
            date_str, kw_str = entry.split(",")
            timestamp = datetime.strptime(date_str.strip(), "%Y-%m-%d %H:%M")
            power = float(kw_str.strip()) * 1000  # Convert kW to watts
            readings.append({"timestamp": timestamp, "power": power})
            print(f"  ✓ Added: {timestamp} - {power:.0f} watts")
        except Exception as e:
            print(f"  ✗ Invalid format: {e}")
    
    if readings:
        df = pd.DataFrame(readings)
        df = df.sort_values("timestamp")
        
        data_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / "manual_readings.csv"
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved {len(readings)} readings to {output_path}")
        return df
    else:
        print("No readings entered.")
        return None

if __name__ == "__main__":
    print("="*60)
    print("UTILITY METER DATA FETCHER")
    print("="*60)
    
    print("\nSelect mode:")
    print("1. API Mode (requires utility API credentials)")
    print("2. Green Button XML (download from utility website)")
    print("3. Manual Entry (type readings manually)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        # Configure your utility here
        print("\nEdit CONFIG section in this file with your credentials")
        fetcher = UtilityDataFetcher(CONFIG)
        df = fetcher.fetch_data(days_back=7)
        if df is not None:
            fetcher.save_to_processed(df)
    
    elif choice == "2":
        xml_file = input("Enter path to Green Button XML file: ").strip()
        fetcher = UtilityDataFetcher(CONFIG)
        df = fetcher.fetch_green_button_data(xml_file)
        if df is not None:
            fetcher.save_to_processed(df, "green_button_data.csv")
    
    elif choice == "3":
        df = manual_entry_mode()
    
    else:
        print("Invalid choice")
