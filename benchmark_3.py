import sys
import time
from tinygrad.llm.model import Transformer
from tinygrad.llm.cli import SimpleTokenizer

def main():
  prompt = "Explain the difference between locks, semaphores, and condition variables in concurrent programming, and provide a Python example showing how to use a condition variable to solve the Producer-Consumer problem."
  
  model_path = "/root/.cache/tinygrad/downloads/14b435b1656ec42daf9b503ed66245b4"
  max_context = 4096
  
  print(f"Loading model from {model_path}...")
  model, kv = Transformer.from_gguf(model_path, max_context)
  model_name = kv.get('general.name') or kv.get('general.basename') or "Qwen2.5 72B"
  print(f"Loaded model: {model_name}")
  
  tok = SimpleTokenizer.from_gguf_kv(kv)
  
  # Prepare prompt
  ids = tok.prefix() + tok.role("user") + tok.encode(prompt) + tok.end_turn() + tok.role("assistant")
  print(f"Prompt encoded into {len(ids)} tokens.")
  
  print("Generating response (temperature=0.0)...")
  
  tokens = []
  dec = tok.stream_decoder()
  
  start_time = time.perf_counter()
  last_time = start_time
  
  token_durations = []
  prefill_duration = 0.0
  
  for next_id in model.generate(ids, temperature=0.0):
    now = time.perf_counter()
    duration = now - last_time
    last_time = now
    
    if len(tokens) == 0:
      prefill_duration = duration
      print(f"\n--- Prefill phase completed in {prefill_duration:.4f}s ({len(ids)/prefill_duration:.2f} tok/s) ---")
    else:
      token_durations.append(duration)
      # Print progress indicator of speed for debugging
      print(f"\n[Token {len(tokens)}]: {duration:.4f}s ({1.0/duration:.2f} tok/s)")
      
    if tok.is_end(next_id) or len(tokens) >= 100:
      break
      
    tokens.append(next_id)
    token_str = dec(next_id)
    sys.stdout.write(token_str)
    sys.stdout.flush()
    
  total_time = time.perf_counter() - start_time
  print("\n\n--- Detailed Benchmark Results ---")
  print(f"Total time: {total_time:.4f}s")
  print(f"Prefill duration: {prefill_duration:.4f}s")
  
  if len(token_durations) > 0:
    first_decode_duration = token_durations[0]
    print(f"First decode step (including decode compile): {first_decode_duration:.4f}s")
    
    if len(token_durations) > 1:
      steady_durations = token_durations[1:]
      steady_time = sum(steady_durations)
      steady_count = len(steady_durations)
      steady_speed = steady_count / steady_time
      print(f"Steady-state decode (tokens 2-{len(tokens)}): {steady_count} tokens in {steady_time:.4f}s ({steady_speed:.2f} tokens/sec)")
    else:
      print("Not enough decode tokens for steady-state calculation.")
      
    total_decode_time = sum(token_durations)
    print(f"Overall decode speed: {len(token_durations)} tokens in {total_decode_time:.4f}s ({len(token_durations)/total_decode_time:.2f} tokens/sec)")

if __name__ == "__main__":
  main()
