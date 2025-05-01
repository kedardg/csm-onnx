import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import time
import torch
import os
from transformers import AutoModelForCausalLM

def generate_text_full_context(model_path, prompt, max_new_tokens=20, temperature=0.7):
    """
    Generate text using the entire context for each prediction
    
    This approach processes the entire accumulated context for each prediction,
    rather than just the last token, which should maintain coherence.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(model_path)
    
    # Get input and output information
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    print(f"Model inputs: {input_names}")
    print(f"Model outputs: {output_names}")
    
    # Initial tokenization
    encoded_input = tokenizer(prompt, return_tensors="np")
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    
    # Track generation
    all_token_ids = input_ids[0].tolist()
    generated_text = ""
    
    # Statistics
    gen_times = []
    
    # Print the prompt
    print(f"Prompt: {prompt}")
    print("Generating: ", end="", flush=True)
    
    # Generate new tokens
    for i in range(max_new_tokens):
        start_time = time.time()
        
        # Create the inputs for full context
        current_input = np.array([all_token_ids], dtype=np.int64)
        current_mask = np.ones_like(current_input, dtype=np.int64)
        
        # Create position IDs (always starting from 0)
        position_ids = np.arange(len(all_token_ids), dtype=np.int64).reshape(1, -1)
        
        # Prepare the input feed
        feed = {
            "input_ids": current_input,
            "attention_mask": current_mask
        }
        
        # Add position_ids if required
        if "position_ids" in input_names:
            feed["position_ids"] = position_ids
            
        # Add any other required inputs
        for name in input_names:
            if "onnx::Gather" in name and name not in feed:
                feed[name] = position_ids
        
        # Run inference
        try:
            outputs = session.run(None, feed)
        except Exception as e:
            print(f"\nError during inference: {str(e)}")
            break
        
        # Get the logits
        logits = outputs[0]
        
        # Handle different output shapes
        if len(logits.shape) == 4:  # Shape like [1, 1, seq, vocab]
            logits = logits.squeeze(1)
            
        # Get the last token's logits
        last_token_logits = logits[0, -1, :]
        
        # Apply temperature and sample
        if temperature > 0:
            # Apply temperature
            last_token_logits = last_token_logits / temperature
            
            # Convert to probabilities with softmax (stable version)
            last_token_logits = last_token_logits - np.max(last_token_logits)
            probs = np.exp(last_token_logits)
            probs = probs / np.sum(probs)
            
            # Sample from the distribution
            next_token_id = np.random.choice(len(probs), p=probs)
        else:
            # Greedy sampling
            next_token_id = np.argmax(last_token_logits)
        
        # Decode the new token
        next_token = tokenizer.decode(next_token_id)
        
        # Print the token
        print(next_token, end="", flush=True)
        
        # Add to results
        generated_text += next_token
        all_token_ids.append(next_token_id)
        
        # Record timing
        end_time = time.time()
        gen_times.append(end_time - start_time)
        
        # Check for end of generation
        if next_token_id == tokenizer.eos_token_id:
            break
    
    print("\n")
    
    # Print statistics
    total_time = sum(gen_times)
    avg_time = total_time / len(gen_times) if gen_times else 0
    tokens_per_second = len(gen_times) / total_time if total_time > 0 else 0
    
    print(f"Generated {len(gen_times)} tokens in {total_time:.2f} seconds")
    print(f"Average time per token: {avg_time:.4f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    
    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "full_text": prompt + generated_text,
        "tokens_per_second": tokens_per_second
    }

def generate_with_reexport(prompt, max_new_tokens=20, temperature=0.7):
    """
    Alternative approach: Using the PyTorch model directly with timing
    """
    # Start timing
    total_start_time = time.time()
    
    # Model ID
    model_id = "meta-llama/Llama-3.2-1B"
    
    print(f"Prompt: {prompt}")
    print("Loading PyTorch model...")
    
    # Load tokenizer and model
    load_start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True)
    model.eval()
    load_time = time.time() - load_start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Print generation start
    print("Generating with PyTorch: ", end="", flush=True)
    
    # Track token generation times
    token_times = []
    generation_start_time = time.time()
    
    # Simple printing callback to show generation progress
    class GenerationMonitor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.start_time = time.time()
            self.times = []
            
        def __call__(self, beam_idx, input_ids, scores):
            current_time = time.time()
            # Only track times after the first token (which is part of initialization)
            if len(self.times) > 0:
                self.times.append(current_time - self.start_time)
            self.start_time = current_time
            
            # Print the generated token
            if len(input_ids) > 0 and len(input_ids[0]) > 0:
                token = self.tokenizer.decode(input_ids[0][-1])
                print(token, end="", flush=True)
            return True
    
    # Create the callback
    monitor = GenerationMonitor(tokenizer)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0,
            do_sample=(temperature > 0),
            callback=monitor,
            callback_interval=1
        )
    
    # Calculate timing
    generation_time = time.time() - generation_start_time
    total_time = time.time() - total_start_time
    
    # Get the generated text
    if len(outputs.shape) > 1 and outputs.shape[0] > 0:
        generated_tokens = outputs[0, inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        generated_text = "(Error: Unable to decode output)"
    
    # Print timing statistics
    print("\n")
    
    # Calculate statistics for callback times (excluding the first one which includes setup)
    callback_times = monitor.times[1:] if len(monitor.times) > 1 else []
    avg_token_time = sum(callback_times) / len(callback_times) if callback_times else 0
    tokens_per_second = len(callback_times) / sum(callback_times) if callback_times else 0
    
    num_new_tokens = len(outputs[0]) - len(inputs.input_ids[0])
    print(f"Generated {num_new_tokens} tokens in {generation_time:.2f} seconds")
    print(f"Average time per token: {avg_token_time:.4f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Total time (including model loading): {total_time:.2f} seconds")
    
    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "full_text": prompt + generated_text,
        "timing": {
            "load_time": load_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "avg_token_time": avg_token_time,
            "tokens_per_second": tokens_per_second
        }
    }

# Example usage
if __name__ == "__main__":
    model_path = "llama3-onnx/model.onnx"
    prompt = "What are you upto?"
    
    # Choose one of the two approaches
    use_full_context = True
    
    if use_full_context:
        print("===== ONNX FULL CONTEXT METHOD =====")
        result = generate_text_full_context(
            model_path=model_path,
            prompt=prompt,
            max_new_tokens=30,
            temperature=0.4
        )
    else:
        print("===== PYTORCH DIRECT METHOD =====")
        result = generate_with_reexport(
            prompt=prompt,
            max_new_tokens=30,
            temperature=0.4
        )
    
    print("\nGeneration complete!")
    print(f"Full text: {result['full_text']}")