import torch, time, argparse, gc
from text_generation_server.models.flash_llama import FlashLlama
from text_generation_server.models.flash_causal_lm import FlashCausalLMBatch
from text_generation_server.pb import generate_pb2

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument('--batch-sizes', type=int, nargs='+', required=True)
parser.add_argument('--max-new-tokens', type=int, default=100)

def warmup(model):
    max_input_length = 256
    max_batch_size = 32
    max_prefill_tokens = max_input_length * max_batch_size - 32

    warmup_requests = []
    n_tokens = 0
    while n_tokens < max_prefill_tokens:
        warmup_requests.append(
            generate_pb2.Request(
                id=0,
                inputs="_text" * max_input_length,
                truncate=min(max_input_length, max_prefill_tokens - n_tokens),
                parameters=generate_pb2.NextTokenChooserParameters(
                    do_sample=False
                ),
                stopping_parameters=generate_pb2.StoppingCriteriaParameters(
                    max_new_tokens=2
                )
            ),
        )
        
        n_tokens += max_input_length

    warmup_batch = generate_pb2.Batch(id=0, requests=warmup_requests, size=len(warmup_requests))

    fclm_warmup_batch = FlashCausalLMBatch.from_pb(
        pb=warmup_batch,
        tokenizer=model.tokenizer,
        dtype=model.dtype,
        device=model.device,
    )

    max_supported_total_tokens = model.warmup(batch=fclm_warmup_batch)

def make_clm_batch(model, batch_size=1, max_new_tokens=100):
    parameters = generate_pb2.NextTokenChooserParameters(
        watermark=False,
        temperature=1.0,
        repetition_penalty=1.0,
        top_k=0,
        top_p=1.0,
        typical_p=1.0,
        do_sample=False
    )

    stopping_parameters = generate_pb2.StoppingCriteriaParameters(
        max_new_tokens=max_new_tokens,
        ignore_eos_token=True
    )

    input_lst = [
        "In a galaxy far, far away"
    ]

    requests = [
        generate_pb2.Request(
            id=idx,
            inputs=inputs,
            truncate=256,
            parameters=parameters,    
            stopping_parameters=stopping_parameters
        )
        for idx, inputs in enumerate(input_lst * batch_size)
    ]

    return FlashCausalLMBatch.from_pb(
        pb=generate_pb2.Batch(id=0, requests=requests),
        tokenizer=model.tokenizer,
        dtype=model.dtype,
        device=model.device,
    )

def run_benchmark(model_id, batch_sizes=[1], max_new_tokens=100):
    model = FlashLlama(model_id=model_id, dtype=torch.bfloat16)
    warmup(model)

    for batch_size in batch_sizes:

        clm_batch = make_clm_batch(model, batch_size=batch_size, max_new_tokens=max_new_tokens)

        with torch.no_grad():

            start = time.perf_counter()
            for _ in range(max_new_tokens):
                generations, clm_batch = model.generate_token(clm_batch)

            torch.cuda.synchronize()
            end = time.perf_counter()

        total_time = end - start
        total_tokens = batch_size * max_new_tokens

        print(f"BATCH_SIZE = {batch_size} // THROUGHPUT: {total_tokens / total_time :0.2f} tokens/sec")

        del clm_batch
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parser.parse_args()
    
    run_benchmark(
        model_id=args.model_id,
        batch_sizes=args.batch_sizes,
        max_new_tokens=args.max_new_tokens
    )