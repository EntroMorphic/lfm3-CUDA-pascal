"""G4 tier 2b anchor: ppl_reference's eager_blocked attention computes the model's own eager attention.

The model's eager_attention_forward and repeat_kv are taken from its remote code by AST (no model weights), and
compared on CPU against eager_blocked across grouped heads, batch 2, causal and sliding-window eager masks, block
sizes that do and do not divide the length. One block covering the sequence must be bitwise equal; smaller blocks
change only matmul shapes, so they must agree to FP32 rounding.
"""
import ast
from pathlib import Path
import types

import pytest

torch = pytest.importorskip("torch")
masking = pytest.importorskip("transformers.masking_utils")
if not hasattr(masking, "eager_mask"):
    pytest.skip("this transformers has no masking_utils.eager_mask", allow_module_level=True)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT.parent / "models/LFM3-Omni-2B/modeling_lfm3.py"
if not MODEL.exists():
    pytest.skip("model remote code not present", allow_module_level=True)


def functions(path, names):
    source = path.read_text()
    tree = ast.parse(source)
    return "\n".join(ast.get_source_segment(source, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names)


@pytest.fixture(scope="module")
def ns():
    space = {"torch": torch, "nn": torch.nn}
    exec(functions(ROOT / "tools/ppl_reference.py", {"register_eager_blocked"}), space)
    exec("from typing import TypedDict\ntry:\n    from typing import Unpack\nexcept ImportError:\n    from typing_extensions import Unpack\n"
         "class TransformersKwargs(TypedDict): pass\n" + functions(MODEL, {"repeat_kv", "eager_attention_forward"}), space)
    return space


@pytest.mark.parametrize("block", [1, 7, 64, 4096])
@pytest.mark.parametrize("shape", [(1, 16, 4, 37, 8), (2, 16, 4, 50, 8), (2, 4, 4, 19, 16)])
@pytest.mark.parametrize("window", [None, 5])
def test_eager_blocked_equals_the_models_eager_attention(ns, block, shape, window):
    torch.manual_seed(0)
    B, H, KV, L, D = shape
    module = types.SimpleNamespace(num_key_value_groups=H // KV, training=False)
    q, k, v = torch.randn(B, H, L, D), torch.randn(B, KV, L, D), torch.randn(B, KV, L, D)
    mask_function = masking.causal_mask_function if window is None else masking.sliding_window_causal_mask_function(window)
    mask = masking.eager_mask(batch_size=B, q_length=L, kv_length=L, cache_position=torch.arange(L),
                              mask_function=mask_function, dtype=torch.float32)
    ns["register_eager_blocked"](block)
    reference, _ = ns["eager_attention_forward"](module, q, k, v, mask, D ** -0.5)
    blocked, _ = ALL_ATTENTION_FUNCTIONS["eager_blocked"](module, q, k, v, mask, D ** -0.5, sliding_window=window)
    if block >= L:
        assert torch.equal(reference, blocked)
    else:
        assert torch.allclose(reference, blocked, rtol=0, atol=1e-5), (reference - blocked).abs().max().item()


@pytest.mark.parametrize("rows", [0, -3])
def test_block_rows_must_be_positive(ns, rows):
    with pytest.raises(ValueError, match="at least 1"):
        ns["register_eager_blocked"](rows)
