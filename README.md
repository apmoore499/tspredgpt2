# Financial Forecasting Model Based on Pretrained GPT-2 Backbone

In this project, we build a financial forecasting model based on a pretrained GPT-2 backbone. Given a time series dataset consisting of OHLCV observations, the objective is to forecast the future direction of price change.

The model learns a mapping:

$$
f: \mathcal{X} \rightarrow \mathcal{Y}
$$

where:

- \( X \in \mathcal{X} \subset \mathbb{R}^{1024 \times F} \) represents a sequence of length 1024 (i.e., 1024 time steps) with \( F \) features at each time step.
- \( Y \in \{0, 1\} \) is the binary label, where \( Y = 1 \) indicates that the closing price at time \( t+1 \) is greater than the closing price at time \( t \), i.e., \( \text{close}[t+1] > \text{close}[t] \).

In this setup, the model predicts the direction of price change based on a sliding window of past observations.

The training is feasible on a single 4090.



# List of tasks to do:

- Causal attention mask                      []
- Scaling to higher n GPU                    []
- Verifying effect of smaller time window    []
- Integrating with backtest                  []
