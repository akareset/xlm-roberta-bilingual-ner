#!/usr/bin/env python3
"""
Quick test script to verify our custom MLM head and loss calculation works correctly
"""

import torch
from torch.optim import AdamW
from transformers import DataCollatorForLanguageModeling
from main import XLMRobertaMLMModule

def test_mlm_head_and_loss():
    """Test the custom MLM head with proper MLM masking and loss calculation"""
    print("🧪 Testing custom MLM head and loss calculation...")
    
    # Initialize the module
    module = XLMRobertaMLMModule()
    
    # Initialize the data collator (same as in your main script)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=module.tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    
    # Create sample inputs - use actual text
    sample_texts = [
        "Hello world, this is a test sentence for masked language modeling.",
        "Another example sentence to verify our MLM implementation works correctly."
    ]
    
    # Tokenize the sample texts
    tokenized_inputs = module.tokenizer(
        sample_texts,
        truncation=True,
        padding=True,
        max_length=32,  # Shorter for testing
        return_tensors="pt"
    )
    
    print(f"Original input_ids shape: {tokenized_inputs['input_ids'].shape}")
    print(f"Sample input_ids: {tokenized_inputs['input_ids'][0][:10]}")  # First 10 tokens
    
    # Apply MLM masking using the data collator
    # Convert to list of dicts format that data collator expects
    batch_data = []
    for i in range(len(sample_texts)):
        batch_data.append({
            'input_ids': tokenized_inputs['input_ids'][i],
            'attention_mask': tokenized_inputs['attention_mask'][i]
        })
    
    # Apply masking
    masked_batch = data_collator(batch_data)
    
    print(f"Masked input_ids shape: {masked_batch['input_ids'].shape}")
    print(f"Labels shape: {masked_batch['labels'].shape}")
    
    # Show the masking effect
    print("\n🔍 Examining masking:")
    for i in range(min(2, len(sample_texts))):
        original_ids = tokenized_inputs['input_ids'][i][:15]  # First 15 tokens
        masked_ids = masked_batch['input_ids'][i][:15]
        labels = masked_batch['labels'][i][:15]
        
        print(f"\nSample {i+1}:")
        print(f"Original:  {original_ids}")
        print(f"Masked:    {masked_ids}")
        print(f"Labels:    {labels}")
        
        # Decode to see the actual tokens
        original_tokens = module.tokenizer.convert_ids_to_tokens(original_ids)
        masked_tokens = module.tokenizer.convert_ids_to_tokens(masked_ids)
        
        print(f"Original tokens: {original_tokens[:10]}")
        print(f"Masked tokens:   {masked_tokens[:10]}")
        
        # Show which positions are masked (labels != -100)
        masked_positions = (labels != -100).nonzero(as_tuple=True)[0]
        print(f"Masked positions: {masked_positions.tolist()}")
    
    # Forward pass
    with torch.no_grad():
        logits = module(
            input_ids=masked_batch['input_ids'], 
            attention_mask=masked_batch['attention_mask']
        )
    
    print(f"\n📊 Model output:")
    print(f"Logits shape: {logits.shape}")
    print(f"Expected shape: {masked_batch['input_ids'].shape + (module.tokenizer.vocab_size,)}")
    
    # Calculate loss exactly like in training_step
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, module.tokenizer.vocab_size), 
        masked_batch['labels'].view(-1)
    )
    
    print(f"\n💡 Loss calculation:")
    print(f"Loss: {loss.item():.4f}")
    
    # Additional verification: manual loss calculation
    print("\n🔬 Manual loss verification:")
    
    # Reshape for easier handling
    logits_flat = logits.view(-1, module.tokenizer.vocab_size)  # (batch_size * seq_len, vocab_size)
    labels_flat = masked_batch['labels'].view(-1)  # (batch_size * seq_len,)
    
    # Find positions that are actually masked (labels != -100)
    masked_positions = labels_flat != -100
    num_masked = masked_positions.sum().item()
    
    print(f"Total positions: {len(labels_flat)}")
    print(f"Masked positions: {num_masked}")
    print(f"Percentage masked: {num_masked/len(labels_flat)*100:.2f}%")
    
    if num_masked > 0:
        # Calculate loss only on masked positions
        masked_logits = logits_flat[masked_positions]
        masked_labels = labels_flat[masked_positions]
        
        manual_loss = torch.nn.functional.cross_entropy(masked_logits, masked_labels)
        print(f"Manual loss (masked only): {manual_loss.item():.4f}")
        print(f"Difference from auto loss: {abs(loss.item() - manual_loss.item()):.6f}")
        
        # Show some predictions vs targets
        print(f"\n🎯 Sample predictions:")
        with torch.no_grad():
            predictions = torch.argmax(masked_logits[:5], dim=-1)  # First 5 masked positions
            targets = masked_labels[:5]
            
            for j, (pred, target) in enumerate(zip(predictions, targets)):
                pred_token = module.tokenizer.decode([pred])
                target_token = module.tokenizer.decode([target])
                print(f"Position {j}: Predicted '{pred_token}' | Target '{target_token}'")
    
    # Verify loss is reasonable for MLM
    expected_random_loss = torch.log(torch.tensor(float(module.tokenizer.vocab_size)))
    print(f"\n📈 Loss analysis:")
    print(f"Current loss: {loss.item():.4f}")
    print(f"Random guess loss: {expected_random_loss.item():.4f}")
    print(f"Loss is {'reasonable' if loss.item() < expected_random_loss.item() else 'too high (possible issue)'}")
    
    # Check shapes
    assert logits.shape == (*masked_batch['input_ids'].shape, module.tokenizer.vocab_size)
    assert loss.item() > 0  # Should have some loss
    assert loss.item() < 20  # Should not be extremely high
    
    print("\n✅ All tests passed!")
    return loss.item()

def test_mlm_learning_capability():
    """
    Test that the model can learn by running a few optimization steps
    and verifying that the loss decreases.
    """
    print("🧪 Testing model's learning capability...")
    
    # Initialize the module
    module = XLMRobertaMLMModule()
    
    # Initialize the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=module.tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    
    # Create sample inputs
    sample_texts = [
        "Hello world, this is a test sentence for masked language modeling.",
        "Another example sentence to verify our MLM implementation works correctly."
    ]
    
    # Tokenize and create a batch
    tokenized_inputs = module.tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")
    batch_data = [{'input_ids': tokenized_inputs['input_ids'][i]} for i in range(len(sample_texts))]
    masked_batch = data_collator(batch_data)

    # --- 1. Check initial loss on the untrained model ---
    module.eval()  # Set to evaluation mode for a clean baseline
    with torch.no_grad():
        initial_logits = module(masked_batch['input_ids'], masked_batch['attention_mask'])
        initial_loss = torch.nn.functional.cross_entropy(
            initial_logits.view(-1, module.tokenizer.vocab_size),
            masked_batch['labels'].view(-1)
        )
    print(f"\nInitial Loss (untrained): {initial_loss.item():.4f}")

    # --- 2. Perform a few training steps ---
    module.train()  # IMPORTANT: Set model to training mode
    optimizer = AdamW(module.parameters(), lr=5e-5) # A standard optimizer

    print("\n🚀 Starting mini training loop (10 steps)...")
    for step in range(10):
        optimizer.zero_grad()
        
        # Forward pass with gradients enabled
        logits = module(masked_batch['input_ids'], masked_batch['attention_mask'])
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, module.tokenizer.vocab_size),
            masked_batch['labels'].view(-1)
        )
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 2 == 0:
            print(f"Step {step+1}/10, Loss: {loss.item():.4f}")

    # --- 3. Check final loss to see if it has decreased ---
    module.eval() # Set back to eval mode for final check
    with torch.no_grad():
        final_logits = module(masked_batch['input_ids'], masked_batch['attention_mask'])
        final_loss = torch.nn.functional.cross_entropy(
            final_logits.view(-1, module.tokenizer.vocab_size),
            masked_batch['labels'].view(-1)
        )
    print(f"\nFinal Loss (after 10 steps): {final_loss.item():.4f}")

    print("\n🎯 Sample predictions after training:")
    with torch.no_grad():
        # Reshape for easier handling
        logits_flat = final_logits.view(-1, module.tokenizer.vocab_size)
        labels_flat = masked_batch['labels'].view(-1)
        
        # Find positions that are actually masked
        masked_positions = labels_flat != -100
        
        if masked_positions.sum().item() > 0:
            masked_logits = logits_flat[masked_positions]
            masked_labels = labels_flat[masked_positions]
            
            predictions = torch.argmax(masked_logits[:5], dim=-1) # First 5 masked positions
            targets = masked_labels[:5]
            
            for j, (pred, target) in enumerate(zip(predictions, targets)):
                pred_token = module.tokenizer.decode([pred])
                target_token = module.tokenizer.decode([target])
                print(f"Position {j}: Predicted '{pred_token}' | Target '{target_token}'")

    # --- 4. Assert that learning has occurred ---
    assert final_loss.item() < initial_loss.item(), "Test failed: Loss did not decrease after training steps."
    
    print("\n✅✅✅ Success! Loss decreased, indicating the model is learning correctly.")

if __name__ == "__main__":
    test_mlm_learning_capability()