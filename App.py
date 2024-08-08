import streamlit as st
import spacy
import yfinance as yf
from transformers import LlamaForSequenceClassification, LlamaTokenizer

nlp = spacy.load("en_core_web_sm")
llama_tokenizer = LlamaTokenizer.from_pretrained("llama")
llama_model = LlamaForSequenceClassification.from_pretrained("llama")

st.title("Investment Advisor")
investment_preferences = st.text_area("Enter your investment preferences:")
def extract_parameters(user_input):
    # Initialize an empty dictionary to store the extracted parameters
    parameters = {}

    # Process the user input using Spacy
    doc = nlp(user_input)

    # Extract investment goal
    for ent in doc.ents:
        if ent.label_ == "GOAL":
            parameters["goal"] = ent.text

    # Extract risk tolerance
    for ent in doc.ents:
        if ent.label_ == "RISK":
            parameters["risk_tolerance"] = ent.text

    # Extract investment amount
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            parameters["investment_amount"] = ent.text

    # Extract investment horizon
    for ent in doc.ents:
        if ent.label_ == "DATE":
            parameters["investment_horizon"] = ent.text

    # Extract preferred sectors
    for ent in doc.ents:
        if ent.label_ == "SECTOR":
            parameters["preferred_sectors"] = ent.text

    # Extract volatility tolerance
    for ent in doc.ents:
        if ent.label_ == "VOLATILITY":
            parameters["volatility_tolerance"] = ent.text

    return parameters

def handle_default_values(parameters):
    # Assign default values if not provided
    if "goal" not in parameters:
        parameters["goal"] = "medium-term"
    if "risk_tolerance" not in parameters:
        parameters["risk_tolerance"] = "medium"
    if "investment_amount" not in parameters:
        parameters["investment_amount"] = 10000
    if "investment_horizon" not in parameters:
        parameters["investment_horizon"] = "1 year"
    if "preferred_sectors" not in parameters:
        parameters["preferred_sectors"] = ["technology", "finance"]
    if "volatility_tolerance" not in parameters:
        parameters["volatility_tolerance"] = 0.5

    return parameters

def fetch_top_stocks(preferred_sectors):
    # Fetch top 10 stocks from yfinance
    top_stocks = []
    for sector in preferred_sectors:
        stocks = yf.download(sector, start="2020-01-01", end="2022-02-26")["Adj Close"]
        top_stocks.extend(stocks.nlargest(10).index.tolist())

    return top_stocks

def predict_stock_prices(stocks, investment_horizon):
    # TO DO: Implement a time series model to predict future stock prices
    # For now, just return a random prediction
    import random
    prediction = {stock: random.uniform(0, 100) for stock in stocks}
    return prediction

def calculate_volatility(stocks, investment_horizon):
    # Calculate the daily returns of each stock
    daily_returns = {}
    for stock in stocks:
        data = yf.download(stock, start="2020-01-01", end="2022-02-26")["Adj Close"]
        daily_returns[stock] = data.pct_change()

    # Calculate the volatility of each stock
    volatility = {}
    for stock, returns in daily_returns.items():
        volatility[stock] = returns.std() * np.sqrt(252)

    return volatility

import pandas as pd

# Define the stocks to analyze
stocks = ["AAPL", "GOOG", "MSFT"]

# Retrieve the historical stock data
data = {}
for stock in stocks:
    ticker = yf.Ticker(stock)
    hist = ticker.history(period="1y")
    data[stock] = hist

# Perform technical analysis
def calculate_moving_averages(data):
    ma_50 = data["Close"].rolling(window=50).mean()
    ma_200 = data["Close"].rolling(window=200).mean()
    return ma_50, ma_200

def calculate_rsi(data):
    delta = data["Close"].diff(1)
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean().abs()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

# Calculate the moving averages and RSI for each stock
analysis = {}
for stock, hist in data.items():
    ma_50, ma_200 = calculate_moving_averages(hist)
    rsi = calculate_rsi(hist)
    analysis[stock] = {"MA 50": ma_50, "MA 200": ma_200, "RSI": rsi}

# Perform fundamental analysis
def calculate_eps(data):
    eps = data["EPS"]
    return eps

# Calculate the EPS for each stock
eps_data = {}
for stock in stocks:
    ticker = yf.Ticker(stock)
    eps = calculate_eps(ticker.info)
    eps_data[stock] = eps

# Combine the technical and fundamental analysis
results = {}
for stock in stocks:
    results[stock] = {"Technical Analysis": analysis[stock], "Fundamental Analysis": eps_data[stock]}

print(results)

def generate_advice(parameters, prediction, volatility):
    # Initialize an empty list to store the investment advice
    advice = []

    # Loop through each stock and generate advice
    for stock, prediction in prediction.items():
        if prediction > parameters["investment_amount"] * (1 + parameters["volatility_tolerance"]):
            advice.append(f"Buy {stock} with a predicted price of ${prediction:.2f} and volatility of {volatility[stock]:.2f}%")
        elif prediction < parameters["investment_amount"] * (1 - parameters["volatility_tolerance"]):
            advice.append(f"Sell {stock} with a predicted price of ${prediction:.2f} and volatility of {volatility[stock]:.2f}%")
        else:
            advice.append(f"Hold {stock} with a predicted price of ${prediction:.2f} and volatility of {volatility[stock]:.2f}%")

    return advice

def generate_advice(summary):
    input_ids = tokenizer.encode(summary, return_tensors="pt")
    output = model.generate(input_ids, max_length=512)
    advice = tokenizer.decode(output[0], skip_special_tokens=True)
    return advice

st.header("Investment Advice")
advice = generate_advice(parameters, prediction, volatility)
for item in advice:
    st.write(item)
    