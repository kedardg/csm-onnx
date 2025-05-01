import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import time

# Load the tokenizer
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create ONNX Runtime session
session = ort.InferenceSession("llama3-onnx/model.onnx")
print("Model loaded successfully!")

# Get the actual input names from the ONNX model
input_names = [input.name for input in session.get_inputs()]
output_names = [output.name for output in session.get_outputs()]
print(f"Required inputs: {input_names}")
print(f"Model outputs: {output_names}")

# Initial input text
input_text = "What are you upto?"
print(f"Input prompt: {input_text}")
print("Generating response...")

# Tokenize the initial input
current_tokens = tokenizer(input_text, return_tensors="np").input_ids[0].tolist()

# Track the generated text
generated_text = ""
full_text = input_text

# Define generation parameters
max_new_tokens = 30
temperature = 0.4# Lower for more deterministic outputs, higher for more creative

# Generation loop
for _ in range(max_new_tokens):
    # Convert current tokens to model input format
    input_ids = np.array([current_tokens], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    
    # Generate position_ids
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1)
    
    # Create input feed dictionary
    input_feed = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    # Add position_ids to the feed
    if "position_ids" in input_names:
        input_feed["position_ids"] = position_ids
    
    # Handle any other required inputs (like onnx::Gather inputs)
    for name in input_names:
        if "onnx::Gather" in name and name not in input_feed:
            if name == "onnx::Gather_2" or name == "onnx::Gather_3":
                input_feed[name] = position_ids
    
    # Run inference
    start_time = time.time()
    outputs = session.run(None, input_feed)
    inference_time = time.time() - start_time
    
    # Get logits from the output
    logits = outputs[0]
    
    # Reshape the logits if necessary
    if len(logits.shape) > 3:
        logits = logits.reshape(-1, logits.shape[-1])
    
    # Get the last token logits (for the next token prediction)
    if len(logits.shape) == 3:  # [batch, seq_len, vocab_size]
        last_token_logits = logits[0, -1, :]
    else:  # Unexpected shape, adapt
        last_token_logits = logits[-1, :]
    
    # Apply temperature sampling
    if temperature > 0:
        # Apply temperature to logits
        last_token_logits = last_token_logits / temperature
        
        # Convert to probabilities with softmax
        probs = np.exp(last_token_logits) / np.sum(np.exp(last_token_logits))
        
        # Sample from the distribution
        next_token_id = np.random.choice(len(probs), p=probs)
    else:
        # Greedy sampling (just pick the most likely token)
        next_token_id = np.argmax(last_token_logits)
    
    # Decode the token
    next_token = tokenizer.decode(next_token_id)
    
    # Update tracking variables
    generated_text += next_token
    current_tokens.append(next_token_id)
    
    # Print the new token (streaming style)
    print(next_token, end="", flush=True)
    
    # Check for EOS token to stop generation early
    if next_token_id == tokenizer.eos_token_id:
        break

print("\n\nFull response:")
print(generated_text)
print(f"\nGenerated {len(generated_text)} characters in {max_new_tokens} inference steps")
print(f"Average inference time per token: {inference_time:.4f} seconds")