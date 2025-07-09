from argparse import ArgumentParser
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModel
import datasets


class XLMRobertaMLMModule(pl.LightningModule):

    def __init__(
        self, 
        model_name="xlm-roberta-base",
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=1000,
        max_steps=10000
    ):
        super().__init__() # See lighting_example, but not sure what this does
        self.save_hyperparameters() # same...
        
        self.encoder = AutoModel.from_pretrained(model_name) #We use AutoModel and create a cutom head later.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mlm_head = self._create_mlm_head() #Our own head
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def _create_mlm_head(self):
        """Implementation inspired by the Bert MLM Prediction head from the transformers module. 
        The ReLU was choosen for the sake of simplicity"""
        hidden_size = self.encoder.config.hidden_size
        vocab_size = self.tokenizer.vocab_size
        
        mlm_head =  torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_size, eps=self.encoder.config.layer_norm_eps),
            torch.nn.Linear(hidden_size, vocab_size)
        )
        torch.nn.init.zeros_(mlm_head[-1].bias) # ? How does this bias work? 
    
        return mlm_head
    
    """
    Relevant snippet of the transformer library:https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
    Around likne 700

    class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]  # Usually GELU
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
    """

    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the model - returns only logits"""
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get sequence output (last hidden states)
        sequence_output = encoder_outputs.last_hidden_state
        
        # Pass through MLM head to get predictions
        prediction_scores = self.mlm_head(sequence_output)
        
        return prediction_scores

    def training_step(self, batch, batch_idx):
        """Training step for MLM"""
        # Forward pass to get logits
        logits = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # Calculate loss
        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size), 
            batch['labels'].view(-1)
        )
        
        self.log(
            "train_loss", 
            loss, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True, 
            logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for MLM"""
        # Forward pass to get logits
        logits = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # Calculate loss
        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size), 
            batch['labels'].view(-1)
        )
        
        self.log_dict(
            {"val_loss": loss}, 
            on_step=False, 
            on_epoch=True, 
            prog_bar=True, 
            logger=True
        )

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Combine parameters from both encoder and MLM head
        all_parameters = list(self.encoder.parameters()) + list(self.mlm_head.parameters())
        
        # AdamW optimizer (commonly used for transformer models)
        optimizer = torch.optim.AdamW(
            all_parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Linear warmup + cosine decay scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.max_steps
        )
        
        return [optimizer], [scheduler]


def main(hparams):
    """Main training function"""
    # Initialize logging (optional for now)
    wandb_logger = None
    if hasattr(hparams, 'use_wandb') and hparams.use_wandb:
        try:
            wandb_logger = WandbLogger(
                log_model=False, 
                project="XLM-RoBERTa-Continual-Pretraining", 
                name="xlm-roberta-bilingual-mlm"
            )
        except Exception as e:
            print(f"Warning: Could not initialize WandB logger: {e}")
            wandb_logger = None

    # Initialize the model
    module = XLMRobertaMLMModule(
        learning_rate=hparams.learning_rate,
        max_steps=hparams.max_steps
    )

    # TODO: Initialize data module (next step)
    # datamodule = BilingualC4DataModule()

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        max_steps=hparams.max_steps,
        logger=wandb_logger,
        gradient_clip_val=1.0,  # Gradient clipping for stability
        accumulate_grad_batches=hparams.accumulate_grad_batches,
    )

    print("✅ XLM-RoBERTa MLM module initialized successfully!")
    print(f"Model: {module.encoder.config.name_or_path}")
    print(f"Vocabulary size: {module.tokenizer.vocab_size}")
    print(f"Hidden size: {module.encoder.config.hidden_size}")
    print(f"MLM head: Custom 2-layer head with GELU activation")
    
    # TODO: Add training loop (next step)
    # trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="auto", help="Accelerator type")
    parser.add_argument("--devices", default="auto", help="Number of devices")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use WandB for logging")
    
    args = parser.parse_args()
    main(args)
