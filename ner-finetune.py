import lightning.pytorch as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModel, DataCollatorForTokenClassification, get_linear_schedule_with_warmup
from datasets import load_dataset
from typing import Optional, Dict, Any, List
import os

class XLMRobertaNER(L.LightningModule):
    def __init__(
        self,
        model_name, # or path to our pretrained model 
        num_classes,
        learning_rate,
        warmup_steps,
        weight_decay,
        dropout_prob,
        max_steps
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_labels = num_classes
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        # NER classification head
        self.encoder = AutoModel.from_pretrained(model_name)

        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_classes)

        self.loss_fn = torch.nn.CrossEntropyLoss()
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Get the last hidden state (the sequence of token embeddings)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        return logits

    def training_step(self, batch, batch_idx):  # why batch_idx?
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        
        # Flatten logits and labels for loss calculation
        # Logits shape: (batch_size, sequence_length, num_labels)
        # Labels shape: (batch_size, sequence_length)
        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        # For now, just log loss. NER metrics (SeqEval) can be added later as requested.
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    

    # TODO We probably need slightly different hyperparameters here, as the task is slightly different. I'll research this later.
    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler."""
        # Combine parameters from the encoder and the custom classifier head
        all_parameters = list(self.encoder.parameters()) + list(self.classifier.parameters())
        
        optimizer = torch.optim.AdamW(
            all_parameters, 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
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
    
class NERDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        model_name: str = "xlm-roberta-base",
        max_length: int = 128,
        batch_size: int = 16,
        language: Optional[str] = None,
        few_shot_k: Optional[int] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name
        self.language = language
        self.tokenizer_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.few_shot_k = few_shot_k
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Data collator specifically for token classification
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

    def prepare_data(self):
        load_dataset(self.dataset_name, self.language)

    def setup(self, stage: Optional[str] = None):
        raw_datasets = load_dataset(self.hparams.dataset_name, self.hparams.dataset_config_name)
        
        ner_feature = raw_datasets["train"].features["ner_tags"]
        
        # Get the human-readable names of the NER labels (e.g., 'O', 'B-PER', 'I-ORG').
        self.label_names = ner_feature.feature.names

        # Create a mapping from numerical ID to human-readable label name.
        self.label_map = {i: label for i, label in enumerate(self.label_names)}
        
        # output dimension of the classification head
        self.num_labels = len(self.label_names)

        if self.hparams.few_shot_k:
             # If few-shot, we create a small training set from the validation set.
            shuffled_val = raw_datasets["validation"].shuffle(seed=42)
            
            # Select the first 'k' instances from the shuffled validation set to form the few-shot training dataset.
            self.train_dataset = shuffled_val.select(range(self.hparams.few_shot_k))
            
            self.val_dataset = raw_datasets["test"]
            self.test_dataset = raw_datasets["test"]
        else:
            self.train_dataset = raw_datasets["train"]
            self.val_dataset = raw_datasets["validation"]
            self.test_dataset = raw_datasets["test"]

        self.train_dataset = self.train_dataset.map(self._tokenize_and_align_labels, batched=True)
        self.val_dataset = self.val_dataset.map(self._tokenize_and_align_labels, batched=True)
        self.test_dataset = self.test_dataset.map(self._tokenize_and_align_labels, batched=True)
    
    
    
    def _tokenize_and_align_labels(self, examples):
        """
        The core idea of this function is to solve the mismatch that happens when a word-based tokenizer splits a single word into multiple sub-word tokens.

        For example, the word "Washington" might be tokenized into ["Washing", "##ton"]. 
        The original dataset has only one label for "Washington" (e.g., B-LOC), but now we have two sub-word tokens.

        The function aligns the labels to these new sub-words using a simple strategy:

        Assign the Label to the First Sub-word: The original label (B-LOC) is assigned only to the first sub-word token ("Washing").
        Ignore Subsequent Sub-words: All other sub-word tokens from the same original word ("##ton") are given a special label of -100.
        Ignore Special Tokens: Special tokens added by the tokenizer, like [CLS] and [SEP], are also given the label -100.
            
            Example:
        - Original words:  ["Washington", "is", "a", "city"]
        - Original labels: [3 (B-LOC),  0 (O), 0 (O), 0 (O)]

        - Tokenized words: ["[CLS]", "Washing", "##ton", "is", "a", "city", "[SEP]"]
        - Aligned labels:  [-100,    3,         -100,    0,    0,   0,      -100]
        """

        # TODO 
        # Isn't it the problem that we ignore the subsequent subwords if split the word? 
        # I think Bendedict said that all subwords need to have labels, like ORG_Out, ORG_in etc. 
    
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=self.hparams.max_length
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens (like CLS, SEP) and padding tokens have None word_idx
                if word_idx is None:
                    label_ids.append(-100) # Ignore these tokens for loss calculation
                
                # If it's a new word or the first token of a word
                elif word_idx != previous_word_idx:
                    # Append the label of the current word
                    label_ids.append(label[word_idx])
                
                # If it's a subsequent token of the same word
                else:
                    label_ids.append(-100) # ignore it

                # update the previous_word_idx for the next iteration.
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, collate_fn=self.data_collator, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, collate_fn=self.data_collator, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, collate_fn=self.data_collator, num_workers=4)
    
#TODO  Carefully go through the code below, it's 1:25 and I've made too many variable name changes😅 And check the logic

def train_ner(
    model_name: str,
    dataset_name: str,
    language: Optional[str],
    run_name: str,
    seed: int,
    few_shot_k: Optional[int] = None,
):
    print(f"\n--- Starting Run: {run_name} (Seed: {seed}) ---")
    print(f"Model: {model_name}, Dataset: {dataset_name} ({language})")

    L.seed_everything(seed)

    datamodule = NERDataModule(model_name=model_name, dataset_name=dataset_name, language=language, batch_size=16, few_shot_k=few_shot_k)
    datamodule.setup()

    # TODO I used some temporar parameters, we need to establish the best one
    module = XLMRobertaNER(model_name=model_name,
                           num_labels=datamodule.num_labels, 
                           learning_rate=2e-5,
                           weight_decay=0.01,
                           warmup_steps = 1000,
                           dropout_prob=0.1,
                           max_steps=20000
                        )

    wandb_logger = WandbLogger(project="MNLP-NER-Finetuning", name=f"{run_name}-seed-{seed}")
    

    trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=10, logger=wandb_logger, log_every_n_steps=10)

    trainer.fit(module, datamodule=datamodule)
    
    wandb_logger.experiment.finish()
    print(f"--- Finished Run: {run_name} (Seed: {seed}) ---")


if __name__ == "__main__":
    SEEDS = [42, 1116, 22314]
    BASE_MODEL = "xlm-roberta-base"
    SPECIALIZED_MODEL = "./xlm-roberta-base-bilingual-specialized"

    if not os.path.exists(SPECIALIZED_MODEL):
        print(f"Warning: Specialized model not found at {SPECIALIZED_MODEL}")
        print("Skipping experiments for the specialized model.")
        run_specialized = False
    else:
        run_specialized = True

    print("--- Running Experiments for Base Model ---")
    for seed in SEEDS:
        train_ner(BASE_MODEL, "conll2003", None, "base-model-on-conll", seed)
        train_ner(BASE_MODEL, "wikiann", "en", "base-model-on-wikiann", seed)
        train_ner(BASE_MODEL, "masakhaner", "yo", "base-model-on-masakhaner-100", seed, few_shot_k=100)
    
    if run_specialized:
        print("\n--- Running Experiments for Bilingual Specialized Model ---")
        for seed in SEEDS:
            train_ner(SPECIALIZED_MODEL, "conll2003", None, "specialized-on-conll", seed)
            train_ner(SPECIALIZED_MODEL, "wikiann", "en", "specialized-on-wikiann", seed)
            train_ner(SPECIALIZED_MODEL, "masakhaner", "yo", "specialized-on-masakhaner-100", seed, few_shot_k=100)

    print("\n--- All experiments finished! ---")