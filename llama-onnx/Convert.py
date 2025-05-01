import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Create output directory
output_dir = "llama3-onnx"
os.makedirs(output_dir, exist_ok=True)

# Model ID
model_id = "meta-llama/Llama-3.2-1B"

# Load model and tokenizer
print(f"Loading model {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Crucial: disable cache to avoid DynamicCache issue
model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=False)
model.eval()

# Create sample input
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Explicitly create position_ids
seq_length = input_ids.shape[1]
position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)

# Export to ONNX
output_path = os.path.join(output_dir, "model.onnx")
print(f"Exporting model to {output_path}...")

with torch.no_grad():
    torch.onnx.export(
        model,
        # Important: include position_ids in the tuple
        (input_ids, attention_mask, position_ids),
        output_path,
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "position_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=15,
        do_constant_folding=True,
    )

print(f"Model successfully exported to {output_path}")