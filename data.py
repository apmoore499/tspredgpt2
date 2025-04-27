
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from scipy.stats import zscore


import talib
from talib import MA_Type

import tqdm



def generate_synthetic_data(n_samples=20000, seq_length=512):
    time = np.linspace(0, 100, seq_length)

    data = []
    targets = []

    for i in range(n_samples):
        close = np.cumsum(np.random.normal(0, 1, seq_length)) + time * np.random.uniform(-0.1, 0.1)
        open_price = close + np.random.normal(0, 2, seq_length)
        high = np.maximum(open_price, close) + np.abs(np.random.normal(0, 1, seq_length))
        low = np.minimum(open_price, close) - np.abs(np.random.normal(0, 1, seq_length))

        series = np.stack([close, open_price, high, low], axis=0)
        target = 1 if close[-1] > close[0] else 0

        data.append(series)
        targets.append(target)

    return np.array(data), np.array(targets)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, seq_length):
        self.data = torch.FloatTensor(data)

        # Normalize each channel
        #for i in range(self.data.shape[1]):
        #    mean = self.data[:,i,:].mean()
        #    std = self.data[:,i,:].std()
        #    self.data[:,i,:] = (self.data[:,i,:] - mean) / std

        self.targets = torch.LongTensor(targets)
        self.seq_length = seq_length

        #self.data=torch.nan_to_num(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]






def add_fourier_features(x, max_freq=10, n_bands=6):
    """
    Adds Fourier features along the sequence length.

    Args:
        x: Tensor of shape [B, T, C]
        max_freq: Max frequency for sinusoidal embedding
        n_bands: Number of frequency bands

    Returns:
        Tensor of shape [B, T, C + 2*n_bands]
    """
    B, T, C = x.shape
    freqs = torch.linspace(1.0, max_freq, n_bands, device=x.device)  # Shape: [n_bands]
    pos = torch.linspace(0, 1, T, device=x.device).unsqueeze(1)       # Shape: [T, 1]

    # [T, n_bands]
    sin = torch.sin(2 * torch.pi * freqs * pos)
    cos = torch.cos(2 * torch.pi * freqs * pos)

    # Concatenate sin/cos → [T, 2 * n_bands]
    fourier_feats = torch.cat([sin, cos], dim=1)  # [T, 2 * n_bands]
    fourier_feats = fourier_feats.unsqueeze(0).repeat(B, 1, 1)  # [B, T, 2 * n_bands]

    return torch.cat([x, fourier_feats], dim=-1)  # [B, T, C + 2 * n_bands]




#-------------------------------
#synthetic data for testing........



# synthetic_data_config = {
#     'batch_size': 32,
#     'num_epochs': 10,
#     'learning_rate': 1e-4,  # Lower learning rate for transformer
#     'num_classes': 2,
#     'seq_length': 1024,
#     'num_channels': cohl.shape[-1]
# }

# print("Generating synthetic data...")
# train_data, train_targets = generate_synthetic_data(n_samples=16000, seq_length=synthetic_data_config['seq_length'])
# val_data, val_targets = generate_synthetic_data(n_samples=4000, seq_length=synthetic_data_config['seq_length'])
#-------------------------------

import lmdb
import pickle
import torch
import tqdm
#from tqdm import tqdm
from torch.utils.data import DataLoader




class FastLMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.env = None  # will be initialized per worker

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,   # VERY important: don't load entire DB into RAM
                meminit=False,
                max_readers=126,   # More readers = better for many workers
            )

    def __getitem__(self, idx):
        self._init_env()
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(str(idx).encode())
        sample = pickle.loads(byteflow)
        return sample

    def __len__(self):
        self._init_env()
        with self.env.begin() as txn:
            return txn.stat()['entries']



def get_lmdb_path(prefix):

    return f"/media/krillman/DISK3_1TB/llama_install/transformer_fx_prediction/code_gpt2_ts/data/{prefix}_dataset.lmdb"



def save_lmdb_dset(dataset,prefix):



    # ---- Your dataset
    #dataset = train_dataset  # replace this
    # ----------------------

    # Loader (batch_size=1 to get single samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Create LMDB environment
    map_size = int(2e9)  # Adjust as needed
    lmdb_path=get_lmdb_path(prefix)
    env = lmdb.open(lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for idx, batch in tqdm.tqdm(enumerate(loader), total=len(dataset)):
            # batch is a tuple: (inputs, labels)
            # Each element has shape (1, ...)
            batch = tuple(b.squeeze(0) for b in batch)  # Remove batch dim

            txn.put(str(idx).encode(), pickle.dumps(batch))

    print(f"Saved {len(dataset)} samples to {lmdb_path}")




#---------------------- SPY data for testing...

N_WINDOWS=100 #just take 20 * seq_length for training data

def return_formatted_SPY_data(seq_length,resample_interval='10min',data_fn='pricedata_SPY.feather',**kwargs):


    df=pd.read_feather(data_fn)
    df['datetime']=df.index

    # Convert timestamp to datetime if not already
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    #df has columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df.columns=['open','high','low','close','volume']

    # Resample to time interval with the following rules:
    df = df.resample(resample_interval).agg({
        'open': 'first',      # First price of the day
        'high': 'max',        # Highest price of the day
        'low': 'min',         # Lowest price of the day
        'close': 'last',      # Last price of the day
        'volume': 'sum'       # Sum of volume for the day
    })

    #adding some indicators
    #bolinger bands
    upper, middle, lower = talib.BBANDS(df.close.ffill(), matype=MA_Type.T3)

    df['upper']=upper
    df['middle']=middle
    df['lower']=lower


    #sma fast and slow
    sma_25=talib.SMA(df.close.ffill(),5*5) #sma 250 min (~4 hour)
    sma_150=talib.SMA(df.close.ffill(),30*5) #sma 1500 min (~24 hour)

    #atr 1 day, 10 day
    atr_1D=talib.ATR(df.high.ffill(),df.low.ffill(),df.close.ffill(),timeperiod=6*24) #6 * 10min * 24 hours
    atr_10D=talib.ATR(df.high.ffill(),df.low.ffill(),df.close.ffill(),timeperiod=6*24*10) #6 * 10min * 24 hours * 10 day


    df['sma_25']=sma_25
    df['sma_150']=sma_150
    df['atr_1D']=atr_1D
    df['atr_10D']=atr_10D


    #sma_cross=(df['sma_25']>df['sma_150'])*1.0
    #breakpoint()
    df['sma_cross']=df['sma_25']>df['sma_150'] #sma cross indicator
    df['sma_cross']=df['sma_cross']*1.0

    #outcome variable for prediction
    #predicting if close[t+1]>close[t]
    next_close_increase=(df.close.diff(1)>0)*1.0

    df=df.ffill() # ffill missing prices

    #seq_length=1024

    df=df[df.volume!=0]#.iloc[seq_length:] #start from first 1024 samples
    ts=np.array([o.total_seconds()/60/10 for o in df.index.diff(1).tolist()])
    df=df.pct_change()#.dropna()

    df['time']=ts
    df['next_close_increase']=next_close_increase
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df=df.dropna()

    print('example df with indicators and next_close_increase outcome variable')
    print(df.head(5))



    df=df.reset_index(drop=True).dropna()

    #breakpoint()


    #30*5
    df=df.iloc[30*5:]
    #df=df.iloc[:seq_length+seq_length*N_WINDOWS] #subset the data, otherwise too muchh to fit on single gpu


    features_shape=df.iloc[:seq_length,:-1].values.shape

    features_array=np.zeros((df.shape[0]-seq_length,*features_shape))#.astype(np.float16)
    labels_array=np.zeros(df.shape[0]-seq_length)#.astype(np.float16)

    zscore_cols=['open', 'high', 'low', 'close', 'volume', 'upper', 'middle', 'lower',
       'sma_25', 'sma_150', 'atr_1D', 'atr_10D']
    df=df.copy()
    df[zscore_cols]=df[zscore_cols].apply(zscore)


    for i in tqdm.tqdm(range(df.shape[0]-seq_length)):
        feats=df.iloc[i:i+seq_length,:-1].values#.astype(np.float16)
        labels=df.iloc[i+seq_length:i+seq_length+1,-1].values#.astype(np.int32)
        features_array[i]=feats
        labels_array[i]=labels


    features=torch.from_numpy(features_array)
    labels=torch.from_numpy(labels_array)


    print(f'features shape: {features.shape}')
    print(f'labels shape: {labels.shape}')


    one_idx=np.where(labels)[0].flatten()
    zero_idx=np.array([i for i in np.arange(labels.shape[0]) if i not in one_idx]).flatten()

    n_one=len(one_idx)
    n_zero=len(zero_idx)
    to_sel=min(n_one,n_zero)
    together=np.hstack([one_idx[:to_sel],zero_idx[:to_sel]]).flatten()
    together.sort()

    features=features[together]
    labels=labels[together]

    N_obs=len(features)

    prop_train=0.7
    prop_val=0.15
    prop_test=0.15

    N_train=int(prop_train*N_obs)
    N_val=int(prop_val*N_obs)
    N_test=N_obs-N_train-N_val



    # randperm split - do not use this for time series cross validation partitions
    # train_idx,val_idx,test_idx=torch.randperm(N_obs).split((N_train,N_val,N_test))
    # nb use contiguous non-overlapping time samples to prevent data leakage
    train_idx,val_idx,test_idx=torch.arange(N_obs).split((N_train,N_val,N_test))


    features=features.float()
    labels=labels.long()

    #breakpoint()

    train_dataset = TimeSeriesDataset(features[train_idx], labels[train_idx], seq_length)
    val_dataset = TimeSeriesDataset(features[val_idx], labels[val_idx], seq_length)
    test_dataset = TimeSeriesDataset(features[test_idx], labels[test_idx], seq_length)

    return train_dataset,val_dataset,test_dataset



