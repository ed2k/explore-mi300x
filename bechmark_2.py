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
  st = time.perf_counter()
  
  tokens = []
  dec = tok.stream_decoder()
  pt = None
  
  for next_id in model.generate(ids, temperature=0.0):
    if len(tokens) == 0:
      pt = time.perf_counter()
      prefill_time = pt - st
      prefill_toks_sec = len(ids) / prefill_time
      print(f"\n--- Prefill phase completed ---")
      print(f"Prefill time: {prefill_time:.4f}s for {len(ids)} tokens ({prefill_toks_sec:.2f} tokens/sec)")
      print(f"--- Generation phase started ---")
    
    if tok.is_end(next_id) or len(tokens) >= 100:
      break
    tokens.append(next_id)
    token_str = dec(next_id)
    sys.stdout.write(token_str)
    sys.stdout.flush()
    
  et = time.perf_counter()
  print()
  print(f"--- Generation phase completed ---")
  if pt is not None:
    gen_time = et - pt
    gen_toks_sec = len(tokens) / gen_time
    print(f"Generated {len(tokens)} tokens in {gen_time:.4f}s ({gen_toks_sec:.2f} tokens/sec)")
  else:
    print("No tokens generated.")
    
  total_time = et - st
  print(f"Total time: {total_time:.4f}s")

if __name__ == "__main__":
  main()
