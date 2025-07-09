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
    
    # Forward pass
    with torch.no_grad():
        outputs = module(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_length}, {module.tokenizer.vocab_size})")
    print(f"Loss: {outputs.loss}")
    
    # Check shapes
    assert outputs.logits.shape == (batch_size, seq_length, module.tokenizer.vocab_size)
    assert outputs.loss is not None
    assert outputs.loss.item() > 0  # Should have some loss
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_mlm_head()
