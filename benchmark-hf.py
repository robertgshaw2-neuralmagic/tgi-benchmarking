from transformers import LlamaForCausalLM, AutoTokenizer
import torch, tqdm, time

WARMUP_ITERATIONS = 1
ITERATIONS = 5
NUM_TOKENS = 512

model_id = "meta-llama/Llama-2-7b-hf"
model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token_id = tokenizer.eos_token_id
inputs = tokenizer(["Today"], return_tensors="pt").to("cuda")

print("Warming up...")

for _ in tqdm.tqdm(range(WARMUP_ITERATIONS)):
    with torch.no_grad():
        _ = model.generate(
            **inputs, 
            max_new_tokens=100, 
            min_new_tokens=100,
            use_cache=True,
        )

print("Running Benchmark...")

with torch.no_grad():
    
    start = time.perf_counter()

    for _ in tqdm.tqdm(range(ITERATIONS)):
        output = model.generate(
            **inputs, 
            max_new_tokens=NUM_TOKENS, 
            min_new_tokens=NUM_TOKENS,
            use_cache=True,
        )

    torch.cuda.synchronize()
    end = time.perf_counter()

    print(f"USE_CACHE = {NUM_TOKENS}")
    print(f"TOKENS = {NUM_TOKENS}")
    print(f"TIME = {end - start: 0.2f}")
    print(f"TOKENS/SEC = {NUM_TOKENS * ITERATIONS / (end - start): 0.2f}")

    print(output.shape)