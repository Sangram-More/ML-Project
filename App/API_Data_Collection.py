import requests
import pandas as pd
import os

# ── FRED API Key ──────────────────────────────────────────────────────────────
# Register for a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
# Set it as an environment variable:  FRED_API_KEY=your_key_here
# Or replace the os.getenv() call below with your key directly (do NOT commit it).
# ─────────────────────────────────────────────────────────────────────────────

def api_data_retrival(series_id, name):

    api_key = os.getenv("FRED_API_KEY", "YOUR_FRED_API_KEY_HERE")

    # FRED API URL
    url = "https://api.stlouisfed.org/fred/series/observations"

    # Parameters for API request
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }

    # Making API request
    response = requests.get(url, params=params)

    # Checking if the response is successful
    if response.status_code == 200:
        data = response.json()
        observations = data.get("observations", [])
        df = pd.DataFrame(observations)
        df.to_csv(f"{name}.csv", index=False)
        print(f"Data saved to '{name}.csv'")
    else:
        print(f"Failed to retrieve data: {response.status_code} - {response.text}")


# ----- Series IDs for each FRED dataset ----------

FEDRates                 = "FEDFUNDS"
UnemployemenrRate        = "UNRATE"
GDP                      = "GDP"
RealGDP                  = "GDPC1"
RealPotentialGDP         = "GDPPOT"
RealGDPPerCapita         = "A939RX0Q048SBEA"
InflationConsumerPrice   = "FPCPITOTLZGUSA"
ConsumerPriceIndexAllItems = "CPALTT01USM657N"
MedianConsumerPriceIndex = "MEDCPIM158SFRBCLE"

# --------- Fetch all datasets ----------
api_data_retrival(FEDRates,                  "FEDRates")
api_data_retrival(UnemployemenrRate,         "UnemployemenrRate")
api_data_retrival(GDP,                       "GDP")
api_data_retrival(RealGDP,                   "RealGDP")
api_data_retrival(RealPotentialGDP,          "RealPotentialGDP")
api_data_retrival(RealGDPPerCapita,          "RealGDPPerCapita")
api_data_retrival(InflationConsumerPrice,    "InflationConsumerPrice")
api_data_retrival(ConsumerPriceIndexAllItems,"ConsumerPriceIndexAllItems")
api_data_retrival(MedianConsumerPriceIndex,  "MedianConsumerPriceIndex")
