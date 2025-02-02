import requests
import pandas as pd


def api_data_retrival(series_id, name):

    # API Key
    api_key = "d45a04206227ded2f814f8046869de4f"

    # Series ID for Dataset
    series_id = series_id

    # FRED API URL
    url = f"https://api.stlouisfed.org/fred/series/observations"

    # Parameters for API request
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",  # JSON response
    }

    # Making API request
    response = requests.get(url, params=params)

    # Checking if the response is successful
    if response.status_code == 200:
        # Parsing JSON data
        data = response.json()
        observations = data.get("observations", [])
        
        # Convert to DataFrame
        df = pd.DataFrame(observations)
        
        # Save to CSV
        df.to_csv(f"{name}.csv", index=False)
        print(f"Data saved to '{name}.csv'")
    else:
        print(f"Failed to retrieve data: {response.status_code} - {response.text}")



# ----- Initialing Series ID to fetch different datasets ----------

FEDRates = "FEDFUNDS"
UnemployemenrRate = "UNRATE"
GDP = "GDP"
RealGDP = "GDPC1"
RealPotentialGDP = "GDPPOT"
RealGDPPerCapita = "A939RX0Q048SBEA"
InflationConsumerPrice = "FPCPITOTLZGUSA"
ConsumerPriceIndexAllItems = "CPALTT01USM657N"
MedianConsumerPriceIndex = "MEDCPIM158SFRBCLE"

# --------- Calling the function to use API ----------
api_data_retrival(FEDRates, "FEDRates")
api_data_retrival(UnemployemenrRate, "UnemployemenrRate")
api_data_retrival(GDP, "GDP")
api_data_retrival(RealGDP, "RealGDP")
api_data_retrival(RealPotentialGDP, "RealPotentialGDP")
api_data_retrival(RealGDPPerCapita, "RealGDPPerCapita")
api_data_retrival(InflationConsumerPrice, "InflationConsumerPrice")
api_data_retrival(ConsumerPriceIndexAllItems, "ConsumerPriceIndexAllItems")
api_data_retrival(MedianConsumerPriceIndex, "MedianConsumerPriceIndex")