#!/usr/bin/env python3
"""
This script loads a trained MLM model from a checkpoint and evaluates its predictions
on sample English and Yoruba sentences.
"""

import torch
from main import XLMRobertaMLMModule

def fill_mask(sentence: str, model: XLMRobertaMLMModule):
    """
    Takes a sentence with a '<mask>' token, runs it through the model,
    and prints the top 5 predictions for the masked position.
    """
    print(f"--- Evaluating: \"{sentence}\" ---")

    # Ensure model is in evaluation mode
    model.eval()
    tokenizer = model.tokenizer

    # Tokenize the input
    inputs = tokenizer(sentence, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs)

    # Find the index of the masked token
    try:
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0][0]
    except IndexError:
        print("Error: '<mask>' token not found in the input sentence.")
        return

    # Get the logits for the masked token
    masked_token_logits = logits[0, mask_token_index, :]

    # Get the top 5 predicted tokens
    top_5_tokens = torch.topk(masked_token_logits, 5, dim=0).indices.tolist()

    print("Top 5 predictions:")
    for token_id in top_5_tokens:
        predicted_token = tokenizer.decode([token_id])
        print(f"- {predicted_token}")
    print("-" * 20)


def main():
    """Main function to load model and run evaluations."""
    # --- Configuration ---
    # IMPORTANT: Update this path to your checkpoint file
    checkpoint_path = "checkpoints/best-model-lr-1e-5-val_loss=12.57.ckpt"
    
    print(f"🚀 Loading model from checkpoint: {checkpoint_path}")
    
    try:
        # Load the Lightning Module from the checkpoint
        model = XLMRobertaMLMModule.load_from_checkpoint(checkpoint_path)
    except FileNotFoundError:
        print(f"❌ Error: Checkpoint file not found at '{checkpoint_path}'.")
        print("Please make sure the path is correct.")
        return

    # --- Sample Sentences ---
    # Feel free to change these or add your own
    english_sentence = "The capital of Germany is <mask>."
    yoruba_sentence = "Olu-ilu orile-ede Naijiria ni <mask>." # "The capital of Nigeria is <mask>."

    # --- Run Evaluation ---
    fill_mask(english_sentence, model)
    fill_mask(yoruba_sentence, model)


if __name__ == "__main__":
    main()
