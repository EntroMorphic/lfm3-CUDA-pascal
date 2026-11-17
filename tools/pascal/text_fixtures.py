"""Build the four forward fixtures of text-correct-v1 on the workstation (no model, no GPU).

The inputs follow tools/lfm3_reference.py: the chat template (system "Respond with text." + the prompt) for
`baseline` and `arithmetic`, rendered and tokenized by the engine itself from "messages"; the default prompt's content
tokens repeated to the length for `window257` and `single`, tokenized here with the checkpoint's tokenizer.json
(the `tokenizers` library, as Transformers uses it). Expected ids are the Thor engine's pins.

Usage: python3 tools/pascal/text_fixtures.py CHECKPOINT THOR_PINS_FORWARD_JSON OUTPUT_DIR
"""
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

PROMPT = "Hypothetically, might reflective recursion be a function of cognition?"
CASES = [  # name, prompt or None, length or None, tokens (tools/check_lfm3_forward.py)
    ("baseline", PROMPT, None, 16),
    ("arithmetic", "What is 17 plus 25? Answer briefly.", None, 16),
    ("window257", None, 257, 4),
    ("single", None, 1, 4),
]


def main():
    checkpoint, pins_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    pins = json.loads(pins_path.read_text())
    content = Tokenizer.from_file(str(checkpoint / "tokenizer.json")).encode(PROMPT, add_special_tokens=False).ids
    out.mkdir(parents=True, exist_ok=True)
    expected = {}
    for name, prompt, length, tokens in CASES:
        ids = pins[name]["generated_ids"]
        expected[name] = ids
        if length is None:
            request = dict(messages=[dict(role="system", content="Respond with text."), dict(role="user", content=prompt)])
        else:
            input_ids = (content * (length // len(content) + 1))[:length]
            request = dict(input_ids=input_ids, capacity=min(4096, max(512, length + tokens)))
        # As check_lfm3_forward.py replays a fixture without teacher forcing: as many tokens as the reference made.
        request.update(max_new_tokens=len(ids), check_reset=True)
        (out / f"{name}.request.json").write_text(json.dumps(request) + "\n")
    (out / "expected.json").write_text(json.dumps(dict(
        source="liquid-CUDA validation/g4-gates-v2/pins/forward.json (generated_ids)",
        thor_binary_sha256=pins["_provenance"]["binary_sha256"], generated_ids=expected), indent=1) + "\n")
    print(f"{len(CASES)} fixtures, {len(content)} content tokens -> {out}")


if __name__ == "__main__":
    main()
