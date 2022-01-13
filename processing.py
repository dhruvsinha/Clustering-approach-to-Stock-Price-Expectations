import json
from urllib.request import urlopen
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from yahoofinancials import YahooFinancials

###########################################################
# Section 1 - Getting Stock Data
###########################################################

# Getting list of company tickers for S&P500 companies
data = pd.read_csv('D:/ML/book1.csv', header=0)
tickers = data['Ticker']

# Retrieving stock price data
# df = web.DataReader(tickers, 'yahoo', '2016-01-31', '2019-01-31')

# Storing the data as multiple time series for later use
# df['Close'].to_csv('D:/tickerdata.csv')

###########################################################
# Section 2 - Getting Data for Financial Ratios
###########################################################

# List of financial ratios to get
ratios = ['priceToOperatingCashFlowsRatio', 'priceToSalesRatio', 'enterpriseValueMultiple', 'ebitperRevenue', 'netProfitMargin', 'returnOnAssets', 'returnOnEquity', 'returnOnCapitalEmployed',
          'receivablesTurnover', 'payablesTurnover', 'inventoryTurnover', 'quickRatio', 'cashRatio', 'currentRatio', 'debtRatio', 'debtEquityRatio', 'totalDebtToCapitalization',
           'dividendPayoutRatio']

# Array to store results
ratioData = np.zeros((len(tickers), len(ratios)))

# Function to parse JSON data provided by API
def get_jsonparsed_data(url):
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)


for i in range(1):

    # Setting URL as API's webpage for that ticker
    comp = 'UAA'
    URL = "https://financialmodelingprep.com/api/v3/financial-ratios/{}".format(comp)

    # Returns dictionary of ratios with dates
    data = get_jsonparsed_data(URL)

    try:
        for j in range(len(ratios)):

            # Returns investment value ratios
            for k in data['ratios'][0]['investmentValuationRatios'].keys():
                if k == ratios[j]:
                    if data['ratios'][0]['investmentValuationRatios'][k] == '':
                        pass
                    else:
                        ratioData[i, j] = data['ratios'][0]['investmentValuationRatios'][k]

            # Returns profitability indicator ratios
            for k in data['ratios'][0]['profitabilityIndicatorRatios'].keys():
                if k == ratios[j]:
                    if data['ratios'][0]['profitabilityIndicatorRatios'][k] == '':
                        pass
                    else:
                        ratioData[i, j] = data['ratios'][0]['profitabilityIndicatorRatios'][k]

            # Returns operation performance ratios
            for k in data['ratios'][0]['operatingPerformanceRatios'].keys():
                if k == ratios[j]:
                    if data['ratios'][0]['operatingPerformanceRatios'][k] == '':
                        pass
                    else:
                        ratioData[i, j] = data['ratios'][0]['operatingPerformanceRatios'][k]
            
            # Returns liquidity measure ratios
            for k in data['ratios'][0]['liquidityMeasurementRatios'].keys():
                if k == ratios[j]:
                    if data['ratios'][0]['liquidityMeasurementRatios'][k] == '':
                        pass
                    else:
                        ratioData[i, j] = data['ratios'][0]['liquidityMeasurementRatios'][k]

            # Returns debt-oriented ratios      
            for k in data['ratios'][0]['debtRatios'].keys():
                if k == ratios[j]:
                    if data['ratios'][0]['debtRatios'][k] == '':
                        pass
                    else:
                        ratioData[i, j] = data['ratios'][0]['debtRatios'][k]
            
            # Returns cash flow indicators
            for k in data['ratios'][0]['cashFlowIndicatorRatios'].keys():
                if k == ratios[j]:
                    if data['ratios'][0]['cashFlowIndicatorRatios'][k] == '':
                        pass
                    else:
                        ratioData[i, j] = data['ratios'][0]['cashFlowIndicatorRatios'][k]
        
        print("Done for {}".format(comp))

    except KeyError:
        print("Missing values for {} for company {}".format(ratios[j], comp))


# Saving results
pd.DataFrame(ratioData, index=tickers, columns=ratios).to_csv('D:/ML/ratios.csv')