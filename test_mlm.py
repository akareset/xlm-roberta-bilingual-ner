#!/usr/bin/env python3
"""
Quick test script to verify our custom MLM head works correctly
"""

import torch
from main import XLMRobertaMLMModule

def test_mlm_head():
    """Test the custom MLM head with sample inputs"""
    print("🧪 Testing custom MLM head...")
    
    # Initialize the module
    module = XLMRobertaMLMModule()
    
    # Create sample inputs
    batch_size = 2
    seq_length = 10
    
    # Sample token IDs and attention mask
    input_ids = torch.randint(0, module.tokenizer.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Sample labels for MLM (same as input_ids for this test)
    labels = input_ids.clone()
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Vocab size: {module.tokenizer.vocab_size}")
    
    # Forward pass (now returns only logits)
    with torch.no_grad():
        logits = module(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_length}, {module.tokenizer.vocab_size})")
    
    # Calculate loss manually (like in training_step)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, module.tokenizer.vocab_size), 
        labels.view(-1)
    )
    print(f"Loss: {loss}")
    
    # Check shapes
    assert logits.shape == (batch_size, seq_length, module.tokenizer.vocab_size)
    assert loss.item() > 0  # Should have some loss
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_mlm_head()
