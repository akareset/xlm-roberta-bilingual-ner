from argparse import ArgumentParser
import lightning.pytorch as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModel, DataCollatorForLanguageModeling,  get_linear_schedule_with_warmup
import datasets
from datasets import load_dataset, interleave_datasets
import random
from typing import Optional, Dict, Any
from lightning.pytorch.callbacks import ModelCheckpoint


class XLMRobertaMLMModule(L.LightningModule):

    def __init__(
        self,
        model_name="xlm-roberta-base",
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=1000,
        max_steps=10000
    ):
        super().__init__()  # See lighting_example, but not sure what this does
        self.save_hyperparameters()  # same...

        # We use AutoModel and create a cutom head later.
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mlm_head = self._create_mlm_head()  # Our own head

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

        mlm_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),  # See: https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1105
            torch.nn.LayerNorm(
                hidden_size, eps=self.encoder.config.layer_norm_eps),
            torch.nn.Linear(hidden_size, vocab_size)
        )
        torch.nn.init.zeros_(mlm_head[-1].bias)  # ? How does this bias work?

        return mlm_head

    def forward(self, input_ids, attention_mask=None):
        # TODO
        """Forward pass through the model"""
        encoder_outputs = self.encoder(
            # forward pass thorugh the model
            input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = encoder_outputs.last_hidden_state  # Get the last output
        # Pass through MLM head to get predictions
        prediction_scores = self.mlm_head(sequence_output)
        return prediction_scores

    def training_step(self, batch, batch_idx):
        """Training step for MLM"""
        logits = self(
            # Call to the forward method
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = F.cross_entropy(logits.view(-1, self.tokenizer.vocab_size),
                               batch['labels'].view(-1))  # Calculate loss
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for MLM"""
        logits = self(input_ids=batch['input_ids'],
                      attention_mask=batch['attention_mask'])
        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size), batch['labels'].view(-1))
        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """Test step for MLM"""
        logits = self(input_ids=batch['input_ids'],
                      attention_mask=batch['attention_mask'])
        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size), batch['labels'].view(-1))
        self.log("test_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # Combine parameters from both encoder and MLM head
        all_parameters = list(self.encoder.parameters()) + \
            list(self.mlm_head.parameters())
        
        optimizer = torch.optim.AdamW(all_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)

        # based on the paper https://arxiv.org/pdf/1901.07291, 5.1 Training details
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
                "frequency": 1
            }
        }

class BilingualC4DataModule(L.LightningDataModule):
    """Data module for bilingual C4 dataset (English and Yoruba) with MLM masking"""

    def __init__(
        self,
        tokenizer_name: str = "xlm-roberta-base",  # ? Duplicate tokenizer
        max_length: int = 512,
        batch_size: int = 8,
        num_workers: int = 4,
        mlm_probability: float = 0.15,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        streaming: bool = True,
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mlm_probability = mlm_probability
        self.train_size = train_size
        self.val_size = val_size
        self.streaming = streaming

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

       # ! Masking
        """Randomly selects 15% of tokens for masking
        Replaces 80% of selected tokens with [MASK]
        Replaces 10% with random tokens
        Leaves 10% unchanged
        Creates the labels tensor with -100 for non-masked positions (ignored in loss calculation)"""
    
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability,
        )

    def prepare_data(self):
        print("Loading English C4 dataset...")
        load_dataset("allenai/c4", "en", streaming=self.streaming)
        print("Loading Yoruba C4 dataset...")
        load_dataset("allenai/c4", "yo", streaming=self.streaming)

    def setup(self, stage: Optional[str] = None): # ?  What is stage? 
        """Set up the datasets for training, validation, and testing"""
        if stage == "fit" or stage is None:
            print("Setting up training datasets...")
            en_dataset = load_dataset(
                "allenai/c4", "en", split="train", streaming=self.streaming)
            # TODO
            # Yoruba has only 46,214 sentences? Validation split is also very small
            # Should we use checkpoint averaging (end of the 6-th lecture) or some overssamplimg strrategies?

            yo_dataset = load_dataset(
                "allenai/c4", "yo", split="train", streaming=self.streaming)

            # Got a problem with loading dataset, therefore remove columns before interleaving
            en_dataset = en_dataset.remove_columns(["timestamp", "url"])
            yo_dataset = yo_dataset.remove_columns(["timestamp", "url"])

            # Interleave the datasets to mix English and Yoruba samples
            # Use type: ignore to handle the type checker issue
            self.train_dataset = interleave_datasets( # <-- actually quite a huge oversampling of Yoruba, as the probabilities are then 50/50  
                [en_dataset, yo_dataset],  # type: ignore
                probabilities=[0.7, 0.3],  # TODO think of a good probabilities, as after bi-specialization we need to fine-tune the model on English NER, catasrphical forgetting is still a thing 
                stopping_strategy="all_exhausted") # <--- we need to use oversampling strategy, otherwise the dataset will have size of yoruba  

            # For validation, we'll use a smaller portion
            en_val = load_dataset("allenai/c4", "en",
                                  split="validation", streaming=self.streaming)
            yo_val = load_dataset("allenai/c4", "yo",
                                  split="validation", streaming=self.streaming)
            
            # Also remove columns from validation sets
            en_val = en_val.remove_columns(["timestamp", "url"])
            yo_val = yo_val.remove_columns(["timestamp", "url"])

            self.val_dataset = interleave_datasets(
                [en_val, yo_val])  # type: ignore

            # Apply preprocessing
            self.train_dataset = self.train_dataset.map(
                self._preprocess_function,
                batched=True,
                remove_columns=["text"] #Remove redundant collumns, see below:
            )

            """# Original C4 dataset structure:
            {
                "text": "Hello world, this is some text...",
                "timestamp": "2023-01-01T00:00:00Z", 
                "url": "https://example.com"

                After _preprocess_function (tokenization):
                # Dataset now contains BOTH original and tokenized data:
            {
                "text": "Hello world, this is some text...",      # Original text
                "timestamp": "2023-01-01T00:00:00Z",          # Original metadata
                "url": "https://example.com",                 # Original metadata
                "input_ids": [0, 31414, 232, 45, 16, ...],   # NEW: Tokenized text
                "attention_mask": [1, 1, 1, 1, 1, ...],      # NEW: Attention mask
                "special_tokens_mask": [1, 0, 0, 0, 0, ...]  # NEW: Special token mask
            }
            After remove_columns=["text", "timestamp", "url"]:
            # Dataset keeps only the tokenized data needed for training:
            {
                "input_ids": [0, 31414, 232, 45, 16, ...],   # Tokenized text (KEPT)
                "attention_mask": [1, 1, 1, 1, 1, ...],      # Attention mask (KEPT)
                "special_tokens_mask": [1, 0, 0, 0, 0, ...]  # Special token mask (KEPT)
            }
            }"""

            self.val_dataset = self.val_dataset.map(
                self._preprocess_function,
                batched=True,
                remove_columns=["text", "timestamp", "url"]
            )

            # Limit dataset size if specified
            if self.train_size:
                self.train_dataset = self.train_dataset.take(self.train_size)
            if self.val_size:
                self.val_dataset = self.val_dataset.take(self.val_size)

    def _preprocess_function(self, examples):
        """Tokenize and prepare the text data"""
        # Tokenize the texts
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # We'll pad in the data collator
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )
        return tokenized

    def train_dataloader(self):
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
        )


def main(
    accelerator: str = "auto",
    devices: str | int = "auto",
    learning_rate: float = 5e-5,
    max_steps: int = 10000,
    accumulate_grad_batches: int = 4,
    batch_size: int = 8,
    max_length: int = 512,
    mlm_probability: float = 0.15,
):
    logger = WandbLogger(
        log_model=False,
        project="XLM-RoBERTa-Continual-Pretraining",
        name="xlm-roberta-bilingual-mlm"
    )

     # This callback will monitor the validation loss and save the best model.
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",      # The metric to monitor
        mode="min",              # 'min' because lower loss is better
        save_top_k=1,            # Save only the best model
        dirpath="checkpoints/",  # Directory to save the model
        filename="best-model-{epoch:02d}-{val_loss:.2f}" # File name for the checkpoint
    )

    module = XLMRobertaMLMModule(
        learning_rate=learning_rate,
        max_steps=max_steps
    )

    # Initialize bilingual C4 data module
    datamodule = BilingualC4DataModule(
        tokenizer_name="xlm-roberta-base",
        max_length=max_length,
        batch_size=batch_size,
        mlm_probability=mlm_probability,
        streaming=True,  # Use streaming for large dataset
        train_size=None,  # Use full dataset, or set a number for testing
        val_size=10000,   # Limit validation size for faster validation
    )

    # Initialize trainer
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_steps=max_steps,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=1000,  # Validate every 1000 steps
        log_every_n_steps=100,    # Log every 100 steps
    )
    trainer.fit(module, datamodule=datamodule)

    print("Training finished. Saving best model...")
    
    # The path to the best model saved by the callback
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")

    # Load the LightningModule from the best checkpoint
    trained_model = XLMRobertaMLMModule.load_from_checkpoint(best_model_path)

    # Define a path to save the final Hugging Face model
    final_model_output_path = "./xlm-roberta-base-bilingual-specialized"

    # Save the underlying Hugging Face model and tokenizer for later use
    trained_model.encoder.save_pretrained(final_model_output_path)
    trained_model.tokenizer.save_pretrained(final_model_output_path)
    
    print(f"Best model saved to {final_model_output_path}")


if __name__ == "__main__":
    main(
        accelerator="gpu",
        devices=1,
        learning_rate=10e-4, # https://proceedings.neurips.cc/paper_files/paper/2019/file/c04c19c2c2474dbf5f7ac4372c5b9af1-Paper.pdf
        max_steps=20000,
        accumulate_grad_batches=8,
        batch_size=8,        # Adjust based on your GPU memory
        max_length=512,      # XLM-RoBERTa's max sequence length
        mlm_probability=0.15  # Standard MLM masking probability
    )
