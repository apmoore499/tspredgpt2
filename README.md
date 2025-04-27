Simple implementation of a financial forecasting model based on a pretrained GPT-2 backbone.

Given a time series dataset consisting of OHLCV observations, the goal is to anticipate the future direction of price change.

Simply, the model learns a mapping:


$f: X \rightarrow Y, X\in[1024,F],Y\in\{0,1}$

Using pytorch / lightning.


