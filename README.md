Simple financial forecasting model based on a pretrained GPT-2 backbone.

Given a time series dataset consisting of OHLCV observations, we consider a forecasting objective to anticipate the future direction of price change.

Each

Simply, the model learns a mapping:


$f: \mathcal{X} \rightarrow \mathcal{Y}$ 

Where each $X\in\mathcal{X} \subset [1024,F]

And $Y\in\{0,1}$


Where 

Using pytorch / lightning.


