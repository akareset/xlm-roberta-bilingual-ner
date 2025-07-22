import lightning.pytorch as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModel, DataCollatorForTokenClassification, get_linear_schedule_with_warmup
from datasets import load_dataset
from typing import Optional, Dict, Any, List
import os
import evaluate


class XLMRobertaNER(L.LightningModule):
    def __init__(
        self,
        model_name, # or path to our pretrained model 
        num_classes,
        learning_rate,
        weight_decay,
        dropout_prob,
        encoder: AutoModel = None # Allow passing a pre-loaded encoder
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_labels = num_classes

        # NER classification head
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = AutoModel.from_pretrained(model_name)

        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_classes)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.metric = evaluate.load("seqeval")



    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Get the last hidden state (the sequence of token embeddings)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        return logits

    def training_step(self, batch, batch_idx): 
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
        
        # Store predictions and labels for SeqEval
        predictions = torch.argmax(logits, dim=-1) # Get predicted label IDs
        # We need to filter out -100 labels and map IDs back to label names
        true_labels = [[self.trainer.datamodule.id_to_label[l.item()] for l in label_row if l.item() != -100] for label_row in labels]
        true_predictions = [[self.trainer.datamodule.id_to_label[p.item()] for p, l in zip(pred_row, label_row) if l.item() != -100] for pred_row, label_row in zip(predictions, labels)]

        self.metric.add_batch(predictions=true_predictions, references=true_labels)


        return loss

    def on_validation_epoch_end(self):
        # Calculate SeqEval metrics at the end of the validation epoch
        results = self.metric.compute(scheme="IOB2") # Use IOB2 scheme
        self.log_dict({
            "val_precision": results["overall_precision"],
            "val_recall": results["overall_recall"],
            "val_f1": results["overall_f1"],
            "val_accuracy": results["overall_accuracy"],
        }, logger=True)
        


    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        predictions = torch.argmax(logits, dim=-1)
        true_labels = [[self.trainer.datamodule.id_to_label[l.item()] for l in label_row if l.item() != -100] for label_row in labels]
        true_predictions = [[self.trainer.datamodule.id_to_label[p.item()] for p, l in zip(pred_row, label_row) if l.item() != -100] for pred_row, label_row in zip(predictions, labels)]

        self.metric.add_batch(predictions=true_predictions, references=true_labels)


        return loss
    
    def on_test_epoch_end(self):
        # Calculate SeqEval metrics at the end of the test epoch
        results = self.metric.compute(scheme="IOB2") # Use IOB2 scheme
        self.log_dict({
            "test_precision": results["overall_precision"],
            "test_recall": results["overall_recall"],
            "test_f1": results["overall_f1"],
            "test_accuracy": results["overall_accuracy"],
        }, logger=True)
        

    

    # TODO We probably need slightly different hyperparameters here, as the task is slightly different. I'll research this later.
    def configure_optimizers(self):
        # Combine parameters from the encoder and the custom classifier head
        all_parameters = list(self.encoder.parameters()) + list(self.classifier.parameters())
        
        optimizer = torch.optim.AdamW(
            all_parameters, 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )

        return optimizer
    
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
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.label_names = None
        self.label_map = None
        self.num_labels = None
        self.id_to_label = None


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Data collator specifically for token classification
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)



    def setup(self, stage: Optional[str] = None):

        # For CoNLL-2003, language is None. For WikiANN, it's "en". For MasakhaNER, it's "yo".
        raw_datasets = load_dataset(self.dataset_name, self.language)
        
        ner_feature = raw_datasets["train"].features["ner_tags"]
        
        # Get the human-readable names of the NER labels (e.g., 'O', 'B-PER', 'I-ORG').
        self.label_names = ner_feature.feature.names

        # Create a mapping from numerical ID to human-readable label name.
        self.id_to_label = {i: label for i, label in enumerate(self.label_names)} # Changed to id_to_label

        self.label_map = {i: label for i, label in enumerate(self.label_names)}
        
        # output dimension of the classification head
        self.num_labels = len(self.label_names)

        if self.few_shot_k:
             # If few-shot, we create a small training set from the validation set.
            shuffled_val = raw_datasets["validation"].shuffle(seed=2307)
            
            # Select the first k instances from the shuffled validation set to form the few-shot training dataset.
            self.train_dataset = shuffled_val.select(range(self.few_shot_k))
            
             # Use the next k instances for validation to avoid seeing the test set
            self.val_dataset = shuffled_val.select(range(self.few_shot_k, self.few_shot_k * 2))
            self.test_dataset = raw_datasets["test"]
        else:
            self.train_dataset = raw_datasets["train"]
            self.val_dataset = raw_datasets["validation"]
            self.test_dataset = raw_datasets["test"]

        column_names = raw_datasets["train"].column_names

        self.train_dataset = self.train_dataset.map(self._tokenize_and_align_labels, batched=True, remove_columns=column_names)
        self.val_dataset = self.val_dataset.map(self._tokenize_and_align_labels, batched=True, remove_columns=column_names)
        self.test_dataset = self.test_dataset.map(self._tokenize_and_align_labels, batched=True, remove_columns=column_names)
    
    
    
    def _tokenize_and_align_labels(self, examples):
        """
        Aligns labels with tokenized inputs, handling word splitting by the tokenizer.

        When a single word is split into multiple sub-word tokens (e.g., "Washington" -> ["Washing", "##ton"]),
        this function ensures that all sub-words receive a meaningful label.

        The strategy is as follows:
        1.  The first sub-word of a split token gets the original label (e.g., "B-LOC").
        2.  Subsequent sub-words from the same original word are assigned the corresponding "inside" label
            (e.g., "I-LOC"). This assumes a B-I labeling scheme where I-Tag = B-Tag + 1.
        3.  Special tokens like [CLS] and [SEP] are assigned -100 to be ignored by the loss function.

            Example (assuming B-LOC=3, I-LOC=4):
        - Original words:  ["Washington", "is", "a", "city"]
        - Original labels: [3,           0,    0,    0]

        - Tokenized words: ["[CLS]", "Washing", "##ton", "is", "a", "city", "[SEP]"]
        - Aligned labels:  [-100,    3,         4,       0,    0,   0,      -100]
        """
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
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to the corresponding "I-" tag.
                else:
                    current_label = label[word_idx]
                    # If the label is a B-tag (odd number), convert it to an I-tag (even number)
                    if current_label % 2 == 1:
                        label_ids.append(current_label + 1)
                    else:
                        label_ids.append(current_label)
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
    

def train_ner(
    model_name: str,
    dataset_name: str,
    language: Optional[str],
    run_name: str,
    seed: int,
    few_shot_k: Optional[int] = None,
    is_specialized: bool = False
):
    """
    Function to run a single NER training/evaluation experiment
    """

    print(f"\n--- Starting Run: {run_name} (Seed: {seed}) ---")
    print(f"Model: {model_name}, Dataset: {dataset_name} ({language})")

    L.seed_everything(seed)

    datamodule = NERDataModule(model_name="xlm-roberta-base", dataset_name=dataset_name, language=language, batch_size=16, few_shot_k=few_shot_k)
    datamodule.setup()

    if is_specialized:
        print(f"Loading specialized encoder from {model_name}")
        pretrain_module = L.LightningModule.load_from_checkpoint(model_name, map_location="cpu")
        
        # Extract the encoder from the loaded module
        specialized_encoder = pretrain_module.encoder
        module = XLMRobertaNER(
            model_name="xlm-roberta-base", # Pass name for tokenizer init inside module
            num_classes=datamodule.num_labels, 
            learning_rate=5e-5,
            weight_decay=0.01,
            dropout_prob=0.1,
            encoder=specialized_encoder # Pass the loaded specialized encoder
        )
    else:
        print(f"Loading base model {model_name}")
        module = XLMRobertaNER(
            model_name=model_name,
            num_classes=datamodule.num_labels, 
            learning_rate=5e-5,
            weight_decay=0.01,
            dropout_prob=0.1,
        )


    wandb_logger = WandbLogger(project="MNLP-NER-Finetuning", name=f"{run_name}-seed-{seed}")
    

    trainer = L.Trainer(
        accelerator="gpu", 
        devices=1, 
        max_epochs=10,
        max_steps=15000, 
        logger=wandb_logger, 
        log_every_n_steps=100, 
        enable_progress_bar=False,
        enable_checkpointing=False)

    trainer.fit(module, datamodule=datamodule)
    
    print(f"--- Running Final Evaluation on MasakhaNER for {run_name} (Seed: {seed}) ---")
    masakhaner_dm = NERDataModule(model_name="xlm-roberta-base", dataset_name="masakhaner", language="yo", batch_size=16)
    masakhaner_dm.setup()
    # module.id_to_label = masakhaner_dm.label_map # Ensure correct map for testing
    trainer.test(module, datamodule=masakhaner_dm)

    wandb_logger.experiment.finish()
    print(f"--- Finished Run: {run_name} (Seed: {seed}) ---")


if __name__ == "__main__":
    seeds = [23, 7, 2025]
    base_model = "xlm-roberta-base"
    specialized_model = os.path.join(os.path.expanduser("~"), "transfer_project", "checkpoints", "best-model-lr-1e-5-val_loss=3.67.ckpt")

    if not os.path.exists(specialized_model):
        print(f"Warning: Specialized model not found at {specialized_model}")
        print("Skipping experiments for the specialized model.")
        run_specialized = False
    else:
        run_specialized = True

    print("--- Running Experiments for Base Model ---")
    for seed in seeds:
        #train_ner(base_model, "conll2003", None, "base-model-on-conll", seed)
        train_ner(base_model, "wikiann", "en", "base-model-on-wikiann", seed)
        train_ner(base_model, "masakhaner", "yo", "base-model-on-masakhaner-100", seed, few_shot_k=100)
    
    if run_specialized:
        print("\n--- Running Experiments for Bilingual Specialized Model ---")
        for seed in seeds:
            #train_ner(specialized_model, "conll2003", None, "specialized-on-conll", seed, is_specialized=True)
            train_ner(specialized_model, "wikiann", "en", "specialized-on-wikiann", seed, is_specialized=True)
            train_ner(specialized_model, "masakhaner", "yo", "specialized-on-masakhaner-100", seed, is_specialized=True, few_shot_k=100)

    print("\n--- All experiments finished! ---")