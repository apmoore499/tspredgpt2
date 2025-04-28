# main training script
# AM 


#---------------------
# usage:
# python train.py
# open tensorboard instance in script dir to view training progress
#---------------------

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2Model

torch.set_float32_matmul_precision('medium') #go faster on 4090

from data import GPT2TS_DIR


import pathlib

import numpy as np
import pandas as pd
import yaml

#----------------- custom
from data import (
    FastLMDBDataset,
    get_lmdb_path,
    return_formatted_SPY_data,
    save_lmdb_dset,
)
from models import GPT2TimeSeriesModel

#-----------------


if __name__=='__main__':

    SEQUENCE_LENGTH=1024 #for gpt2, this is default
    m_idx=0

    for resample_interval in ['10min']:#:'1h','20min']:
        for num_layers_to_unfreeze in [2]:
            m_idx+=1

            # config for experiment
            config = {
                'batch_size': 16,
                'num_epochs': 100,
                'learning_rate': 5e-4,
                'num_classes': 2, #binary outcome
                'seq_length': SEQUENCE_LENGTH, # Your target sequence length
                'num_layers_to_unfreeze' : num_layers_to_unfreeze,  # do not unfreeze any of the layers in gpt - seems to work better
                'resample_interval':resample_interval,
                'shuffle_train':True, #normally True...
            }

            train_dataset,val_dataset,test_dataset=return_formatted_SPY_data(**config)



            #delete old dir...
            import os
            import shutil
            #ddir='/data/'
            ddir=GPT2TS_DIR.joinpath('data/').as_posix()
            if os.path.exists(ddir):
                shutil.rmtree(ddir)
                print('purged old data dir')
            os.mkdir(ddir)
            print('create fresh empty data dir')

            save_lmdb_dset(train_dataset,'train')
            save_lmdb_dset(val_dataset,'val')
            save_lmdb_dset(test_dataset,'test')

            #LMDBDatasetlmdb_path=get_lmdb_path(prefix)
            #breakpoint()

            train_loader = DataLoader(FastLMDBDataset(get_lmdb_path('train')), batch_size=config['batch_size'], shuffle=config['shuffle_train'], num_workers=12,pin_memory=True,persistent_workers=True)
            val_loader = DataLoader(FastLMDBDataset(get_lmdb_path('val')), batch_size=config['batch_size'], shuffle=False, num_workers=4,pin_memory=True)
            test_loader = DataLoader(FastLMDBDataset(get_lmdb_path('test')), batch_size=config['batch_size'], shuffle=False, num_workers=0)



            features_sample=list(train_loader)[0][0]
            print('first item in train loader shape: ')
            print(features_sample.shape)


            config['num_channels']=features_sample.shape[-1]


            model_save_fn=f'gpt2-timeseries-'

                # Setup logging and checkpointing
            logger = TensorBoardLogger('lightning_logs', name='gpt2_timeseries')



            #import shutil

            #shutil.rmtree('checkpoints')


            val_loss_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath=f'checkpoints_{m_idx}',
                filename=model_save_fn+'-{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                mode='min',
            )

            #clean the directory...


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
                gradient_clip_val=1.0, # Clip gradients for simple example
                accumulate_grad_batches=2 # Accumulate gradients over batches
            )


            #get the class labels in each....

            #breakpoint()

            #trainer.test(model,val_loader)


            #train
            trainer.fit(model,train_loader,val_loader)

            #test
            trainer.test(model,test_loader)


            #@TODO auprc / confusion matrix
            #@TODO get labels and counts and predictions for train/val/test dsets [y]
            #@TODO resnet backbone insteadd
            #@TODO double check that label generation and indexing is correct.   [y]



