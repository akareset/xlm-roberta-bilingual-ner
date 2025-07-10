from argparse import ArgumentParser
import lightning.pytorch as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModel
import datasets


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
        return torch.optim.AdamW(all_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)


def main(
    accelerator: str = "auto",
    devices: str | int = "auto",
    learning_rate: float = 5e-5,
    max_steps: int = 10000,
    accumulate_grad_batches: int = 4,
    use_wandb: bool = False,
):
    logger = WandbLogger(
            log_model=False,
            project="XLM-RoBERTa-Continual-Pretraining",
            name="xlm-roberta-bilingual-mlm"
        )

    module = XLMRobertaMLMModule(
        learning_rate=learning_rate,
        max_steps=max_steps
    )

    # TODO: Initialize data module, masking, etc.
    # datamodule = BilingualC4DataModule()

    # Initialize trainer
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_steps=max_steps,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    # TODO: Do we need to do a training loop by ourselves? 
    """autoencoder = LitAutoEncoder(Encoder(), Decoder())
optimizer = autoencoder.configure_optimizers()

for batch_idx, batch in enumerate(train_loader):
    loss = autoencoder.training_step(batch, batch_idx)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()"""
    # trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main(
        accelerator="gpu",
        devices=1,
        learning_rate=3e-5,
        max_steps=20000,
        accumulate_grad_batches=8,
        use_wandb=True
    )
