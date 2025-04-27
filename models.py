# models.py

# torch model definitions


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import pytorch_lightning as pl
from transformers import GPT2Model, GPT2Config
import torch.nn.functional as F
from data import add_fourier_features


# --- Define which pretrained model to use ---
# Choose one of the standard GPT-2 models.
# 'gpt2' has n_embd=768
# 'gpt2-medium' has n_embd=1024
# 'gpt2-large' has n_embd=1280
# 'gpt2-xl' has n_embd=1600
PRETRAINED_MODEL_ID = 'gpt2'
class GPT2TimeSeriesModel(pl.LightningModule):
    def __init__(self, num_classes, config=None, learning_rate=1e-4,num_layers_to_unfreeze=0):
        super().__init__()
        # Saves num_classes, config, learning_rate as hyperparameters
        self.save_hyperparameters()#(ignore=['config']) # ignore config itself if it's complex/mutable

        self.learning_rate = learning_rate
        self.num_layers_to_unfreeze=num_layers_to_unfreeze
        # Get input dimensions from config, with defaults
        input_proj_dim = config.get('num_channels', 5) if config else 5
        target_seq_length = config.get('seq_length', 512) if config else 512


        # --- Load the base GPT-2 config and modify it for your sequence length ---
        # This gets the standard config for the pretrained model (e.g., gpt2)
        try:
            gpt2_base_config = GPT2Config.from_pretrained(PRETRAINED_MODEL_ID)
        except Exception as e:
            print(f"Error loading pretrained config {PRETRAINED_MODEL_ID}: {e}")
            # Fallback to a default config if loading fails
            gpt2_base_config = GPT2Config()


        # --- IMPORTANT: Match n_embd to the pretrained model ---
        # The embedding dimension (n_embd) MUST match the pretrained model.
        # We will use the n_embd from the loaded base config.
        pretrained_n_embd = gpt2_base_config.n_embd
        # Update the config hyperparameter saved by save_hyperparameters
        if self.hparams.config is None:
             self.hparams.config = {} # Initialize if None
        self.hparams.config['n_embd'] = pretrained_n_embd # Update config hp to match pretrained model

        # Now, update the max sequence length in the config that will be used by from_pretrained
        gpt2_base_config.n_positions = target_seq_length # Set to 1024
        gpt2_base_config.n_ctx = target_seq_length     # Set to 1024

        # You can keep other config parameters (n_layer, n_head, etc.) from the pretrained model

        # --- Initialize GPT-2 model with pre-trained weights ---
        # The from_pretrained method will load the weights from PRETRAINED_MODEL_ID
        # and use the provided config (which has the updated sequence length).
        # The library handles the positional embeddings mismatch: it will initialize
        # new positional embeddings for positions beyond the original trained length.
        # You might see a warning about shape mismatch in positional embeddings during loading,
        # which is expected and usually okay as they will be fine-tuned.
        print(f"Loading pretrained model: {PRETRAINED_MODEL_ID} with config n_positions={gpt2_base_config.n_positions}, n_ctx={gpt2_base_config.n_ctx}")
        try:
            self.gpt2 = GPT2Model.from_pretrained(PRETRAINED_MODEL_ID, config=gpt2_base_config)

            self.gpt2.requires_grad_(False)
            print("Pretrained model loaded successfully.")
        except Exception as e:
            print(f"Error loading pretrained model {PRETRAINED_MODEL_ID}: {e}")
            print("Initializing model with base config instead (no pretrained weights loaded).")
            self.gpt2 = GPT2Model(gpt2_base_config)


        # --- Freeze the weights of the pretrained GPT-2 model ---
        print(f"Freezing all weights for the {PRETRAINED_MODEL_ID} model...")
        for param in self.gpt2.parameters():
            param.requires_grad = False


        # Adjusted input projection layer to match extra Fourier features
        n_fourier_feats = 8 * 2  # sin & cos for 8 bands
        input_proj_dim = config.get('num_channels', 1) + n_fourier_feats
        self.input_projection = nn.Linear(input_proj_dim, pretrained_n_embd)

        # Using SiLU
        # Use nonlinearity since input space is changed
        self.input_projection = nn.Sequential(
            nn.Linear(input_proj_dim, pretrained_n_embd),
            nn.SiLU(),
            nn.LayerNorm(pretrained_n_embd),
            nn.Linear(pretrained_n_embd, pretrained_n_embd)
            )

        # --- Adapt the input projection layer ---

        # --- Adapt the decoder layer ---
        # It must take the pretrained model's embedding size (e.g., 768) as input
        self.decoder = nn.Sequential(
            nn.Linear(pretrained_n_embd, 512), # Start with the pretrained embedding size
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
            )

        self.input_decoder_params=list(self.input_projection.parameters())+list(self.decoder.parameters())


        print(f"Unfreezing final {num_layers_to_unfreeze} layers for the {PRETRAINED_MODEL_ID} model...")

        # Unfreeze final 2 layers of gpt for fine tuning
        #num_layers_to_unfreeze = 2
        for i in range(1, num_layers_to_unfreeze + 1):
            for param in self.gpt2.h[-i].parameters():
                param.requires_grad = True


    # Add after unfreezing to verify
    def print_trainable_parameters(self):
        trainable_params = 0
        all_params = 0
        for name, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"Trainable layer: {name}")
                print(f'Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of all params)')


    def forward(self, x):
        # x shape expected: [batch_size, sequence_length, channels] -> [32, 1024, n_features]
        # NOTE: Your data loading needs to provide this shape.
        # The previous script output [batch_size, channels, sequence_length] before transpose.
        # Let's adjust this forward pass to match the expected output from the data script.
        # Assumed input x shape from data script: [batch_size, channels, sequence_length] -> [32, n_features, 1024]


        # print("Input shape to forward:", x.shape) # Uncomment for debugging
        # Transpose from [batch_size, channels, sequence_length] to [batch_size, sequence_length, channels]
        # Required shape for nn.Linear to project the last dimension (channels)
        #x = x.transpose(1, 2) # Shape becomes [32, 1024, 5]
        # print("Shape after transpose:", x.shape) # Uncomment for debugging
        batch_size, seq_length, _ = x.shape
        # Add Fourier features
        x = add_fourier_features(x, max_freq=20, n_bands=8) #x<-[x+fourier_features]
        # Project to GPT-2 embedding space
        x = self.input_projection(x)

        attention_mask = (x != 0).any(dim=2).long() #nb attention mask should be causal by default
                                                    # https://discuss.pytorch.org/t/huggingfaces-gpt2-implement-causal-attention/74984


        gpt2_output = self.gpt2(inputs_embeds=x, attention_mask=attention_mask)
        last_data_indices = attention_mask.sum(dim=1) - 1
        last_data_indices = torch.clamp(last_data_indices, min=0)
        batch_indices = torch.arange(x.size(0), device=x.device)
        last_hidden_state = gpt2_output.last_hidden_state[batch_indices, last_data_indices, :]
        return self.decoder(last_hidden_state)



    def training_step(self, batch, batch_idx):
        x, y = batch        
        #print(x)
        # Check your data for NaN/inf values (useful for debug)
        #print(torch.isnan(x).any())
        #print(torch.isinf(y).any())
        y_hat = self(x)
        # Make sure y is LongTensor for cross_entropy
        y = y.long()
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True) # Add prog_bar=True for visibility
        self.log('train_acc', acc, prog_bar=True,on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Make sure y is LongTensor for cross_entropy
        y = y.long()
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True) # Add prog_bar=True for visibility
        self.log('val_acc', acc, prog_bar=True,on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Make sure y is LongTensor for cross_entropy
        y = y.long()
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('test_loss', loss, prog_bar=True) # Add prog_bar=True for visibility
        self.log('test_acc', acc, prog_bar=True,on_epoch=True)
        return loss


    def configure_optimizers(self):
        # AdamW optimizer is recommended for transformers
        # Separate parameter groups for different learning rates

        optimizer = AdamW([
            {
            'params': self.input_decoder_params,  # Custom heads
            'lr': self.learning_rate,
            'weight_decay': 0.01
            },
            {
            'params': [p for i in range(self.num_layers_to_unfreeze) 
                      for p in self.gpt2.h[-i-1].parameters()],  # Unfrozen GPT-2 layers
            'lr': self.learning_rate * 0.1,  # Lower learning rate for pretrained layers
            'weight_decay': 0.01
            }
            ], betas=(0.9, 0.999), eps=1e-8)

        # Learning rate scheduler
        # Use total training steps for CosineAnnealingLR if possible
        # trainer.estimated_stepping_batches is available after setup completes
        # You might need to move optimizer/scheduler setup to setup() if called before trainer.fit
        # For simplicity, keeping it here and using estimated_stepping_batches which PL resolves
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches, # Use total steps
            eta_min=1e-6
            )

        return {
        "optimizer": optimizer,
        "lr_scheduler": {
        "scheduler": scheduler,
            "interval": "step", # Apply scheduler per step
            "frequency": 1,
            "monitor": "val_loss", # Monitor validation loss (optional for step scheduler)
            "strict": False # Don't crash if monitor key is not found immediately
            }
            }
