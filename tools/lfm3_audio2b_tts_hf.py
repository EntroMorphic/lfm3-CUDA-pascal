#!/usr/bin/env python3
"""LFM3-Audio-2B text-to-speech in HF, exactly as its model card shows it, for comparison with LFM3-Omni-2B
(validation/speech-audio2b-v1).

Why: does Omni's audio sibling, which has the same backbone (including the 256-key sliding window behind Omni's
long-form failure, validation/speech-long-form-v1), read long text unsplit? The card's TTS path: system prompt
"Perform TTS in <language> using a <gender> voice.", the assistant turn prefilled with <|audio_start|>
(continue_final_message), generate(), codec decode of audio_codes[:, :, :audio_lengths].
Arms (sampling):
  eval: the settings both cards report their evaluations with (audio_temperature=0.4, audio_top_k=0,
        audio_repetition_penalty=1.2, audio_repetition_window=64): the settings Omni's runs used.
  card: LFM3-Audio-2B's recommended speech settings (1.0, 128, 1.1, 4096).
torch.manual_seed(seed) before generate, as tools/lfm3_speech_hf.py does. BF16, SDPA.
Writes OUT_DIR/audio2b-ARM/CASE-sSEED/{summary.json, response.wav, response.txt} (tools/lfm3_asr.py's layout).
Usage (lfm3-omni-dev): lfm3_audio2b_tts_hf.py MODEL_DIR PROMPTS.json OUT_DIR eval|card
"""
import json
import sys
import time
from pathlib import Path

ARMS = {"eval": dict(audio_temperature=0.4, audio_top_k=0, audio_repetition_penalty=1.2, audio_repetition_window=64),
        "card": dict(audio_temperature=1.0, audio_top_k=128, audio_repetition_penalty=1.1, audio_repetition_window=4096)}


def main(model_dir, prompts, out_dir, arm):
    import soundfile as sf
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    torch.set_grad_enabled(False)
    sampling = ARMS[arm]
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda",
                                                     trust_remote_code=True, attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    codec = processor.audio_tokenizer.to(device="cuda").eval()
    spec = json.load(open(prompts))
    engine = f"audio2b-{arm}"
    for c in spec["cases"]:
        assert c["mode"] == "audio", "TTS cases only"
        system = spec["systems"][c["system"]]
        conversation = [{"role": "system", "content": system}, {"role": "user", "content": c["user"]},
                        {"role": "assistant", "content": "<|audio_start|>"}]
        inputs = processor.apply_chat_template(conversation, continue_final_message=True, tokenize=True,
                                               return_dict=True, return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]
        for seed in spec["seeds"]:
            name = f"{c['case']}-s{seed}"
            d = Path(out_dir) / engine / name
            d.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            o = model.generate(**inputs, max_new_tokens=c["max_new_tokens"], return_dict_in_generate=True, **sampling)
            n = int(o.audio_lengths[0])
            wav = codec.decode(o.audio_codes[:1, :, :n]).audio_values[0].squeeze(0).float().cpu().numpy() if n else []
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            slots = int(o.sequences.shape[1] - prompt_len)
            sf.write(d / "response.wav", wav, 24_000)
            (d / "response.txt").write_text("\n")
            (d / "summary.json").write_text(json.dumps(dict(
                case=name, mode="audio", seed=seed, engine=engine, sampling=sampling, prompt_tokens=int(prompt_len),
                conversation=[{"role": "system", "content": system}, {"role": "user", "content": c["user"]}], text="",
                audio_seconds=len(wav) / 24_000, frames=n, slots=slots, hit_cap=slots >= c["max_new_tokens"],
                generate_seconds=seconds), indent=1) + "\n")
            print(name, engine, f"{n} frames, {len(wav) / 24_000:.1f}s audio in {seconds:.1f}s",
                  "(hit max_new_tokens)" if slots >= c["max_new_tokens"] else "", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[4] not in ARMS:
        sys.exit(__doc__)
    main(*sys.argv[1:])
