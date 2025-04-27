# Simple Financial Forecasting Model Based on Pretrained GPT-2 Backbone

In this project, we build a financial forecasting model based on a pretrained GPT-2 backbone. Given a time series dataset consisting of OHLCV observations, the objective is to forecast the future direction of price change.

The model learns a mapping:

$$
f: \mathcal{X} \rightarrow \mathcal{Y}
$$

where:

- $$X \in \mathcal{X} \subset \mathbb{R}^{1024 \times F}$$ represents a sequence of length 1024 (i.e., 1024 time steps) with $$F$$ features at each time step. $$F$$ consists of appropriately-transformed OHLCV features concatenated with a collection of derived features (see **Data Augmentations** below).
- $$Y \in \{0, 1\}$$ is the binary label, where $$Y = 1$$ indicates that the closing price at time $$t+1$$ is greater than the closing price at time $$t$$, i.e., $$\text{close}[t+1] > \text{close}[t]$$.

In this setup, the model predicts the direction of price change based on a sliding window of past observations.


## Training Data
The dataset used for experiments consists of minute bars of SPY ETF observations, from 2005 to 2024. The minute bars are aggregated to 3-hourly windows.


## Data Augmentations
Talib library is used to augment the raw data with some common financial indicators:
- Short and long-term Simple-Moving Averages (SMA) of the close price, with explicit crossover SMA(Short)>SMA(Long) included as an indicator variable taking values in $$0,1$$.
- Average True Range (ATR) values of the price series, with varying window length.
- Bollinger Bands.

In some sense, we can think of these derived indicators as providing more signal regarding the price momentum which could be used to anticipate subsequent price behaviour.



## Training
The training is feasible on a single 4090.


# Further avenues of exploration:

- Scaling to higher n GPU, with more data                    []
- Verifying effect of smaller time window                    []
- Integrating with backtest                                  []
