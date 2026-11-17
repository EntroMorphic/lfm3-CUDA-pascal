"""Minimal HF text generation (test_text.py settings) for kernel-level profiling; prints generated ids."""
import sys
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
m = "/workspace/models/LFM3-Omni-2B"
model = AutoModelForMultimodalLM.from_pretrained(m, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
                                                 attn_implementation="sdpa").eval()
proc = AutoProcessor.from_pretrained(m, trust_remote_code=True)
conv = [{"role": "system", "content": "Respond with text."},
        {"role": "user", "content": "Hypothetically, might reflective recursion be a function of cognition?"}]
x = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
with torch.inference_mode():
    model.generate(**x, max_new_tokens=n, generation_mode="text", do_sample=False)  # warmup (Triton JIT, cuBLAS init)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    out = model.generate(**x, max_new_tokens=n, generation_mode="text", do_sample=False)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
print(out.sequences[0, x["input_ids"].shape[1]:].tolist())
