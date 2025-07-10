#!/usr/bin/env python3
"""
Test script to verify that the bilingual C4 data loading works correctly.
This script tests the data module without running the full training.
"""

from main import BilingualC4DataModule
from transformers import AutoTokenizer
import torch

def test_data_loading():
    """Test the bilingual C4 data module"""
    print("Testing BilingualC4DataModule...")
    
    # Initialize data module with small batch size for testing
    datamodule = BilingualC4DataModule(
        tokenizer_name="xlm-roberta-base",
        max_length=128,  # Smaller for testing
        batch_size=4,    # Small batch for testing
        mlm_probability=0.15,
        streaming=True,
        train_size=100,  # Small size for testing
        val_size=50,     # Small size for testing
        num_workers=0,   # No multiprocessing for testing
    )
    
    print("Preparing data...")
    datamodule.prepare_data()
    
    print("Setting up datasets...")
    datamodule.setup("fit")
    
    print("Getting train dataloader...")
    train_loader = datamodule.train_dataloader()
    
    print("Getting validation dataloader...")
    val_loader = datamodule.val_dataloader()
    
    print("Testing one batch from train loader...")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        
        # Print some example tokens
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        print(f"  Example input (first sequence): {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)[:100]}...")
        
        # Check that masking occurred
        masked_positions = (batch['labels'][0] != -100).sum()
        print(f"  Number of masked tokens in first sequence: {masked_positions}")
        
        if i >= 2:  # Only test a few batches
            break
    
    print("Testing one batch from validation loader...")
    for i, batch in enumerate(val_loader):
        print(f"Validation Batch {i+1}:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        
        if i >= 1:  # Only test one validation batch
            break
    
    print("Data loading test completed successfully!")

if __name__ == "__main__":
    test_data_loading()
