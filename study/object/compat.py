"""Scoped, reversible compatibility adapters for LeRobot 0.5.1 / TF 5.3.0.

No site-packages edits, model weight file edits, strict=False, or suppressed
tokenizer warnings. Existing two manually applied initialization fixes are OK.
"""
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path


def check_versions():
    actual = {k: version(k) for k in ("lerobot", "torch", "transformers", "flash-attn", "hf-libero")}
    if actual["lerobot"] != "0.5.1" or actual["transformers"] != "5.3.0":
        raise RuntimeError(f"This adapter targets lerobot=0.5.1 / transformers=5.3.0, got {actual}")
    return actual


@contextmanager
def installed(checkpoint):
    import torch
    from safetensors import safe_open
    from lerobot.policies.groot import processor_groot as pg
    from lerobot.policies.groot.action_head import flow_matching_action_head as ah
    from lerobot.policies.groot.groot_n1 import GR00TN15
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    check_versions()
    path = Path(checkpoint) / "model.safetensors"
    prefix = "_groot_model.backbone.eagle_model.language_model."
    embedding_key, head_key = prefix+"model.embed_tokens.weight", prefix+"lm_head.weight"
    with safe_open(str(path), framework="pt") as f:
        if (f.metadata() or {}).get(embedding_key) != head_key:
            raise RuntimeError("Checkpoint differs from the verified shared-weight checkpoint. Stop and inspect.")
    old_beta, old_init = ah.Beta, GR00TN15.__init__
    old_create, old_collate = GrootPolicy._create_groot_model, pg.collate

    def cpu_beta(*args, **kwargs):
        with torch.device("cpu"):
            return old_beta(*args, **kwargs)

    def init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            self.post_init()

    def create(self):
        model = old_create(self)
        lm = model.backbone.eagle_model.language_model
        if lm.model.embed_tokens.weight.shape != lm.lm_head.weight.shape:
            raise RuntimeError("Embedding/head shape mismatch")
        lm.model.embed_tokens.weight = lm.lm_head.weight
        return model

    def collate(features, eagle_processor):
        def call(*args, **kwargs):
            tensor_type = kwargs.pop("return_tensors", None)
            if tensor_type is not None:
                common = dict(kwargs.get("common_kwargs") or {})
                common["return_tensors"] = tensor_type
                kwargs["common_kwargs"] = common
            return eagle_processor(*args, **kwargs)
        return old_collate(features, call)

    ah.Beta, GR00TN15.__init__ = cpu_beta, init
    GrootPolicy._create_groot_model, pg.collate = create, collate
    try:
        yield
    finally:
        ah.Beta, GR00TN15.__init__ = old_beta, old_init
        GrootPolicy._create_groot_model, pg.collate = old_create, old_collate


def tokenizer_audit(output_path):
    """Inventory only: does NOT certify training-time tokenization equivalence."""
    import hashlib
    import json
    from PIL import Image
    from lerobot.policies.groot.processor_groot import _build_eagle_processor, DEFAULT_TOKENIZER_ASSETS_REPO
    proc = _build_eagle_processor(DEFAULT_TOKENIZER_ASSETS_REPO)
    tokenizer = proc.tokenizer
    samples = ["put the black bowl between the plate and the ramekin",
               "['pick up the object']", "<image 1><IMG_CONTEXT></image>", "Move 12.5 cm."]
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_text = backend.to_str() if backend is not None else "unavailable"
    result = {"class": type(tokenizer).__name__, "backend_sha256": hashlib.sha256(backend_text.encode()).hexdigest(),
              "fix_mistral_regex": tokenizer.init_kwargs.get("fix_mistral_regex"),
              "examples": [{"text": s, "ids": tokenizer.encode(s, add_special_tokens=False)} for s in samples],
              "certified_equivalent_to_training": False,
              "review": "Compare with the checkpoint training tokenizer; do not blindly enable the Mistral regex patch."}
    image = Image.new("RGB", (224, 224), (128, 128, 128))
    conv = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": samples[0]}]}]
    text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    out = proc(text=[text], images=[image], images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1,
               "use_thumbnail": False}, common_kwargs={"return_tensors": "pt"}, padding=True)
    result["tensor_shapes"] = {k: list(out[k].shape) for k in ("pixel_values", "image_sizes", "input_ids", "attention_mask")}
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
