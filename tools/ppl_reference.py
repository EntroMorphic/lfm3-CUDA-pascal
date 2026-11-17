"""Reference perplexity for LFM3-Omni-2B (or, with --model-class causal, the text-only LFM3-1.7B) with the standard
Hugging Face stack (teacher forced).

Builds the shared evaluation windows (reference tokenizer, so every engine scores identical ids) and
scores them with Transformers in each requested attention configuration. The language model is
called exactly as Lfm3OmniForConditionalGeneration._generate prefills (empty DynamicCache, all-ones
attention mask, explicit positions), so SDPA picks the same kernels as real use, including the
flash -> mem-efficient switch on sliding layers past 256 tokens. Logits come from the tied lm_head
in 256-row tiles; NLL = -log_softmax(logits.float())[next token] over all 128000 logits.

Run in the reference container (lfm3-omni-dev) with Thor's ptxas for the Triton conv kernels:
  TRITON_PTXAS_PATH=/workspace/tools/ptxas TRITON_PTXAS_BLACKWELL_PATH=/workspace/tools/ptxas \\
  python /workspace/liquid-CUDA/tools/ppl_reference.py --text /workspace/data/wikitext-2-raw-v1/test.txt \\
      --output /workspace/liquid-CUDA/build/ppl-v1 --lengths 256 257 4096 32768

Long-context corpus (--corpus-dir, e.g. PG-19's test split): one window per book holding at least L - 1 tokens,
[BOS] + the book's first L - 1 tokens, books in file-name order; summary.json lists each window's book.
--chunk-rows N prefills each window in N-token chunks through the cache (HF's own continuation path), so lengths whose
one-pass attention mask does not fit in memory can be scored; such engines are named hf-CONFIG-chunkN.

Writes windows-L.i32 (int32, W x L ids), nll-hf-CONFIG-L.f32 (float32, W x (L-1)) with a
nll-hf-CONFIG-L.meta.json binding it to the windows it scored, and summary.json. ppl_compare.py
refuses any NLL file whose metadata does not match.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer
from transformers.cache_utils import DynamicCache


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_meta(path, **fields):
    path.write_text(json.dumps(fields, indent=2) + "\n")


def build_windows(tokens, bos, length, max_windows):
    step = length - 1
    count = len(tokens) // step
    if count < 1:
        raise ValueError(f"corpus has {len(tokens)} tokens, fewer than one window of length {length}")
    if max_windows:
        count = min(count, max_windows)
    rows = [[bos] + tokens[i * step:(i + 1) * step] for i in range(count)]
    return np.asarray(rows, dtype=np.int32)


def register_eager_blocked(block_rows):
    """Eager attention computed in blocks of query rows: the same math as the model's eager_attention_forward
    (scores, additive eager mask, FP32 softmax, P.V) without materializing all L x L scores at once. In FP32 at
    32,768 tokens, SDPA falls back to its unfused math kernel and needs over 96 GB (validation/g4-gates-v2/)."""
    if not isinstance(block_rows, int) or block_rows < 1:
        raise ValueError("--attention-block-rows must be at least 1")
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

    def eager_blocked(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        groups = module.num_key_value_groups
        key = key.repeat_interleave(groups, dim=1) if groups > 1 else key
        value = value.repeat_interleave(groups, dim=1) if groups > 1 else value
        out = torch.empty_like(query[..., :value.shape[-1]])
        for r0 in range(0, query.shape[2], block_rows):
            r1 = min(query.shape[2], r0 + block_rows)
            weights = torch.matmul(query[:, :, r0:r1], key.transpose(2, 3)) * scaling
            if attention_mask is not None:
                weights = weights + attention_mask[:, :, r0:r1, : key.shape[-2]]
            weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
            out[:, :, r0:r1] = torch.matmul(weights, value)
        return out.transpose(1, 2).contiguous(), None

    AttentionInterface.register("eager_blocked", eager_blocked)
    AttentionMaskInterface.register("eager_blocked", ALL_MASK_ATTENTION_FUNCTIONS["eager"])


def language_model(model):
    """The decoder stack and its config: Omni nests it under model.language_model; LFM3-1.7B is model.model."""
    if hasattr(model.model, "language_model"):
        return model.model.language_model, model.config.text_config
    return model.model, model.config


def book_windows(corpus_dir, tok, bos, length):
    """[BOS] + the first length - 1 tokens of every book long enough, in file-name order."""
    rows, books = [], []
    for path in sorted(Path(corpus_dir).glob("*.txt")):
        ids = tok(path.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
        if bos in ids:
            raise ValueError(f"{path.name} contains the BOS id")
        if len(ids) >= length - 1:
            rows.append([bos] + ids[:length - 1])
            books.append(path.stem)
    if not rows:
        raise ValueError(f"no book in {corpus_dir} has {length - 1} tokens")
    return np.asarray(rows, dtype=np.int32), books


def nll_rows(model, hidden, targets, out, tile=256):
    for r0 in range(0, len(targets), tile):
        r1 = min(len(targets), r0 + tile)
        logits = model.lm_head(hidden[r0:r1]).float()
        out[r0:r1] = (-torch.log_softmax(logits, dim=-1).gather(1, targets[r0:r1, None])[:, 0]).cpu().numpy()


@torch.inference_mode()
def score(model, windows, chunk_rows=0):
    lm, text_config = language_model(model)
    out = np.empty((windows.shape[0], windows.shape[1] - 1), dtype=np.float32)
    for w, row in enumerate(windows):
        ids = torch.from_numpy(row.astype(np.int64))[None].cuda()
        length = ids.shape[1]
        cache = DynamicCache(config=text_config)
        step = chunk_rows or length
        for c0 in range(0, length, step):
            c1 = min(length, c0 + step)
            # the mask covers the cached prefix and this chunk, as a cached prefill (prompt continuation) builds it
            hidden = lm(input_ids=ids[:, c0:c1], attention_mask=torch.ones_like(ids[:, :c1]),
                        position_ids=torch.arange(c0, c1, device="cuda")[None], past_key_values=cache,
                        use_cache=True).last_hidden_state[0]
            # rows c0..c1-1 predict tokens c0+1..c1 (the last row of the window predicts nothing)
            n = min(c1, length - 1) - c0
            nll_rows(model, hidden[:n], ids[0, c0 + 1:c0 + 1 + n], out[w, c0:c0 + n])
            del hidden
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", type=Path, help="one text file: non-overlapping windows over its tokens")
    p.add_argument("--corpus-dir", type=Path, help="a directory of documents (*.txt): one prefix window per long-enough document")
    p.add_argument("--chunk-rows", type=int, default=0, help="prefill in chunks of this many tokens (0 = one pass)")
    p.add_argument("--model", default="/workspace/models/LFM3-Omni-2B")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[256, 257, 4096, 32768])
    p.add_argument("--configs", nargs="+", default=["sdpa", "eager"], choices=["sdpa", "eager", "eager_blocked"])
    p.add_argument("--eager-max-length", type=int, default=4096, help="eager materializes L x L attention in FP32")
    p.add_argument("--max-windows", type=int, default=0, help="cap windows per length (0 = all)")
    p.add_argument("--attention-block-rows", type=int, default=2048,
                   help="query rows per block for --configs eager_blocked (exact eager attention, bounded memory)")
    p.add_argument("--model-class", choices=["omni", "causal"], default="omni",
                   help="causal: a text-only Lfm3ForCausalLM checkpoint such as LFM3-1.7B")
    p.add_argument("--memory-fraction", type=float, default=0.0,
                   help="cap PyTorch's allocator at this fraction of device memory (0 = no cap): an OOM then raises "
                        "instead of exhausting Thor's unified memory")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                   help="float32: a precision anchor (G4 tier 2b); engines are named hf-CONFIG-fp32, TF32 disabled")
    a = p.parse_args()
    if (a.text is None) == (a.corpus_dir is None):
        p.error("give exactly one of --text and --corpus-dir")
    a.output.mkdir(parents=True, exist_ok=False)
    if "eager_blocked" in a.configs:
        register_eager_blocked(a.attention_block_rows)
    if a.dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    suffix = "-fp32" if a.dtype == "float32" else ""

    if a.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    if a.model_class == "omni":
        tok = AutoProcessor.from_pretrained(a.model, trust_remote_code=True).tokenizer
    else:
        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    bos = tok.convert_tokens_to_ids("<|startoftext|>")
    if a.text is not None:
        tokens = tok(a.text.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
        if bos in tokens:
            raise ValueError("corpus contains the BOS id; windows would not be [BOS] + content")
        corpus = {"corpus": str(a.text), "corpus_sha256": sha256(a.text), "corpus_tokens": len(tokens),
                  "window_rule": "[BOS] + next L-1 corpus tokens, non-overlapping; score positions 1..L-1"}
    else:
        docs = sorted(a.corpus_dir.glob("*.txt"))
        corpus = {"corpus": str(a.corpus_dir), "corpus_files": len(docs),
                  "corpus_sha256": hashlib.sha256("".join(f"{sha256(d)}  {d.name}\n" for d in docs).encode()).hexdigest(),
                  "window_rule": "one window per document with >= L-1 tokens: [BOS] + its first L-1 tokens; score 1..L-1"}
    suffix_chunk = f"-chunk{a.chunk_rows}" if a.chunk_rows else ""
    summary = {"model": a.model, "model_class": a.model_class, **corpus, "bos_id": bos, "chunk_rows": a.chunk_rows,
               "nll_definition": "-log_softmax(lm_head(final hidden).float(), -1)[next token], all 128000 logits",
               "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
               "dtype": a.dtype,
               "precision_note": ("FP32 SDPA runs PyTorch's memory-efficient attention, whose GEMMs use 3xTF32 (FP32-level "
                                  "accuracy) regardless of the TF32 flags (mem_eff_attention/gemm_kernel_utils.h); "
                                  "eager_blocked uses plain FP32 matmuls with TF32 disabled")
                                 if a.dtype == "float32" else None,
               "attention_block_rows": a.attention_block_rows if "eager_blocked" in a.configs else None,
               "precision": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                              "float32_matmul_precision": torch.get_float32_matmul_precision()},
               "windows": {}, "results": []}
    windows = {}
    for length in a.lengths:
        books = None
        if a.text is not None:
            w = build_windows(tokens, bos, length, a.max_windows)
        else:
            w, books = book_windows(a.corpus_dir, tok, bos, length)
            if a.max_windows:
                w, books = w[:a.max_windows], books[:a.max_windows]
        path = a.output / f"windows-{length}.i32"
        w.tofile(path)
        windows[length] = w
        summary["windows"][str(length)] = {"count": int(w.shape[0]), "file": path.name, "sha256": sha256(path),
                                           **({"books": books} if books else {})}

    for config in a.configs:
        cls = AutoModelForMultimodalLM if a.model_class == "omni" else AutoModelForCausalLM
        model = cls.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="cuda",
                                    trust_remote_code=True, attn_implementation=config).eval()
        actual = language_model(model)[1]._attn_implementation
        if actual != config:
            raise RuntimeError(f"requested attn_implementation={config} but the language model runs {actual}")
        for length in a.lengths:
            if config == "eager" and length > a.eager_max_length:
                summary["results"].append({"engine": f"hf-{config}{suffix}{suffix_chunk}", "length": length, "skipped": "length above eager limit"})
                continue
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            nll = score(model, windows[length], a.chunk_rows)
            seconds = time.perf_counter() - t0
            peak_gb = torch.cuda.max_memory_allocated() / 2**30
            if not np.isfinite(nll).all():
                raise RuntimeError(f"non-finite NLL for hf-{config}{suffix} at length {length}")
            path = a.output / f"nll-hf-{config}{suffix}{suffix_chunk}-{length}.f32"
            nll.tofile(path)
            write_meta(a.output / f"nll-hf-{config}{suffix}{suffix_chunk}-{length}.meta.json", engine=f"hf-{config}{suffix}{suffix_chunk}", length=length,
                       windows_file=f"windows-{length}.i32", windows_sha256=summary["windows"][str(length)]["sha256"],
                       nll_sha256=sha256(path), attn_implementation=actual, dtype=a.dtype, precision=summary["precision"],
                       versions=summary["versions"])
            mean = float(nll.astype(np.float64).mean())
            summary["results"].append({"engine": f"hf-{config}{suffix}{suffix_chunk}", "length": length, "windows": int(nll.shape[0]),
                                       "scored_tokens": int(nll.size), "mean_nll": mean, "perplexity": math.exp(mean),
                                       "seconds": seconds, "peak_allocated_gib": peak_gb, "file": path.name,
                                       "sha256": sha256(path)})
            print(json.dumps(summary["results"][-1]), flush=True)
        del model
        torch.cuda.empty_cache()
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
