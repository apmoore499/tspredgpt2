# main training script
# AM 


#---------------------
# usage:
# python train.py
# open tensorboard instance in script dir to view training progress
#---------------------

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import pytorch_lightning as pl
from transformers import GPT2Model, GPT2Config
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('medium') #go faster on 4090


import numpy as np
import pandas as pd

import pathlib
import yaml




#----------------- custom
from data import return_formatted_SPY_data,FastLMDBDataset
from models import GPT2TimeSeriesModel
#-----------------


if __name__=='__main__':

    SEQUENCE_LENGTH=1024 #for gpt2


    # config for experiment
    config = {
        'batch_size': 16,
        'num_epochs': 30,
        'learning_rate': 5e-4,
        'num_classes': 2, #binary outcome
        'seq_length': SEQUENCE_LENGTH, # Your target sequence length
        'num_layers_to_unfreeze' : 0,  # do not unfreeze any of the layers in gpt - seems to work better
        'resample_interval':'3h',
    }

    train_dataset,val_dataset,test_dataset=return_formatted_SPY_data(**config)




    from data import save_lmdb_dset,get_lmdb_path

    save_lmdb_dset(train_dataset,'train')
    save_lmdb_dset(val_dataset,'val')
    save_lmdb_dset(test_dataset,'test')

    #LMDBDatasetlmdb_path=get_lmdb_path(prefix)
    #breakpoint()

    train_loader = DataLoader(FastLMDBDataset(get_lmdb_path('train')), batch_size=16, shuffle=True, num_workers=4,pin_memory=True)
    val_loader = DataLoader(FastLMDBDataset(get_lmdb_path('val')), batch_size=16, shuffle=False, num_workers=4,pin_memory=True)
    test_loader = DataLoader(FastLMDBDataset(get_lmdb_path('test')), batch_size=16, shuffle=False, num_workers=0)



    features_sample=list(train_loader)[0][0]
    print('first item in train loader shape: ')
    print(features_sample.shape)


    config['num_channels']=features_sample.shape[-1]


    model_save_fn=f'gpt2-timeseries-'

        # Setup logging and checkpointing
    logger = TensorBoardLogger('lightning_logs', name='gpt2_timeseries')

    val_loss_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename=model_save_fn+'-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )


    # Initialize model instance using the modified class
    model = GPT2TimeSeriesModel(
        num_classes=config['num_classes'],
        learning_rate=config['learning_rate'],
        num_layers_to_unfreeze=config['num_layers_to_unfreeze'],
        config=config # Pass the config
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        logger=logger,
        callbacks=[val_loss_callback],
        # Use 'gpu' if available, fallback to 'cpu'
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1, # Specify number of devices
        precision='16-mixed', #Save memory
        #gradient_clip_val=1.0, # Clip gradients for simple example
        accumulate_grad_batches=2 # Accumulate gradients over batches
    )


    #train
    trainer.fit(model,train_loader,val_loader)

    #test
    trainer.test(model,test_loader)



