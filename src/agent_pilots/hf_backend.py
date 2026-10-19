"""Local Hugging Face decoding with an independent RNG stream per request."""

from __future__ import annotations

import math
from importlib.metadata import version
from pathlib import Path

import torch

BACKEND_VERSION = "hf-autoregressive-v1"


def sample_next(logits, generators, temperature, top_p):
    """Sample one token per row without coupling request RNG streams.

    Ties at the top-p boundary are broken by original token id (ascending),
    making the retained prefix stable. A generator is consumed only for the
    corresponding row, including rows that have not yet emitted EOS.
    """
    if logits.ndim != 2 or logits.shape[0] != len(generators) or logits.shape[1] < 1:
        raise ValueError("logits must be [batch, vocab] and match generator count")
    if not math.isfinite(float(temperature)) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if not math.isfinite(float(top_p)) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not torch.is_floating_point(logits) or not torch.isfinite(logits).any(dim=-1).all():
        raise ValueError("each logits row must contain at least one finite value")
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("logits may contain -inf masks but not NaN or +inf")
    if any(not isinstance(generator, torch.Generator) or generator.device != logits.device
           for generator in generators):
        raise ValueError("every generator must be on the same device as logits")
    if temperature == 0:
        return logits.argmax(dim=-1)
    if not len(generators):
        return torch.empty(0, dtype=torch.long, device=logits.device)
    scaled = logits.float() / float(temperature)
    sorted_logits, sorted_ids = torch.sort(scaled, dim=-1, descending=True, stable=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    keep = torch.ones_like(probs, dtype=torch.bool)
    keep[:, 1:] = probs.cumsum(dim=-1)[:, :-1] < float(top_p)
    filtered = torch.zeros_like(probs).scatter(1, sorted_ids, probs.masked_fill(~keep, 0))
    filtered /= filtered.sum(dim=-1, keepdim=True)
    # Multinomial alone is per row; sorting/filtering remains batched on device.
    return torch.stack([
        torch.multinomial(row, 1, generator=generator).squeeze(0)
        for row, generator in zip(filtered, generators, strict=True)
    ])


def _collect_eos_ids(model, tokenizer):
    values = [getattr(tokenizer, "eos_token_id", None)]
    for config in (getattr(model, "config", None), getattr(model, "generation_config", None)):
        values.append(getattr(config, "eos_token_id", None) if config is not None else None)
    eos = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (tuple, list, set)):
            eos.update(int(item) for item in value if item is not None)
        else:
            eos.add(int(value))
    if not eos:
        raise ValueError("Model/tokenizer must define at least one EOS token id")
    return frozenset(eos)


class HFBackend:
    def __init__(self, model_path, config):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("HFBackend requires a CUDA device")
        self.model_path = str(Path(model_path).resolve())
        self.temperature = float(config["temperature"])
        self.top_p = float(config["top_p"])
        self.max_tokens = int(config["max_tokens"])
        self.batch_size = int(config.get("request_batch_size", 24))
        self.max_model_len = int(config.get("max_model_len", 8192))
        if not 1 <= self.batch_size <= 24 or self.max_tokens < 1 or not 1 <= self.max_model_len <= 8192:
            raise ValueError("batch size, generation length and context length must be positive")
        if not math.isfinite(self.temperature) or self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("invalid temperature or top_p")
        self.device = torch.device("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer needs a pad_token_id or eos_token_id")
        self.pad_id = int(self.tokenizer.pad_token_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
        ).eval()
        self.max_model_len = min(self.max_model_len, int(getattr(self.model.config, "max_position_embeddings", 8192)))
        self.eos_ids = _collect_eos_ids(self.model, self.tokenizer)
        self.metadata = {
            "backend_version": BACKEND_VERSION,
            "torch_version": str(torch.__version__),
            "transformers_version": version("transformers"),
            "model_path": self.model_path,
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "max_model_len": self.max_model_len,
            "dtype": "bfloat16",
            "device": "cuda:0",
            "independent_per_request_rng": True,
            "output_tokens_include_eos": True,
            "vllm_output_equivalence": False,
            "attention_implementation": "sdpa",
            "position_ids": "attention_mask.cumsum(-1)-1; current token position for cached decode",
            "top_p_boundary_tie_break": "descending logit then ascending token id",
        }

    @torch.inference_mode()
    def generate(self, prompts, seeds):
        if len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must have equal length")
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError("all prompts must be strings")
        if any(not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64 for seed in seeds):
            raise ValueError("each seed must be an integer")
        results = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start:start + self.batch_size]
            batch_seeds = seeds[start:start + self.batch_size]
            encoded = self.tokenizer(
                batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False,
            )
            prompt_lengths = encoded["attention_mask"].sum(dim=-1).tolist()
            if any(length < 1 for length in prompt_lengths):
                raise ValueError("each prompt must contain at least one token")
            if any(length + self.max_tokens > self.max_model_len for length in prompt_lengths):
                raise ValueError("prompt plus max_tokens exceeds model context length")
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            batch_size = input_ids.shape[0]
            generators = [torch.Generator(device=self.device).manual_seed(seed) for seed in batch_seeds]
            generated = [[] for _ in range(batch_size)]
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            finish_reasons = ["length"] * batch_size
            past_key_values = None
            current_ids = input_ids
            current_mask = attention_mask

            for step in range(self.max_tokens):
                position_ids = current_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(current_mask == 0, 0)
                position_ids = position_ids[:, -current_ids.shape[1]:]
                output = self.model(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                next_tokens = torch.full(
                    (batch_size,), self.pad_id, dtype=torch.long, device=self.device,
                )
                active_before = ~finished
                active_rows = torch.nonzero(active_before, as_tuple=False).flatten()
                if active_rows.numel():
                    active_logits = output.logits[active_rows, -1, :]
                    active_generators = [generators[index] for index in active_rows.tolist()]
                    sampled = sample_next(active_logits, active_generators, self.temperature, self.top_p)
                    next_tokens[active_rows] = sampled
                    for index, token in zip(active_rows.tolist(), sampled.tolist(), strict=True):
                        generated[index].append(token)
                        if token in self.eos_ids:
                            finished[index] = True
                            finish_reasons[index] = "stop"
                if finished.all() or step + 1 == self.max_tokens:
                    break
                current_ids = next_tokens[:, None]
                current_mask = torch.cat(
                    [current_mask, active_before.long()[:, None]], dim=1,
                )
                past_key_values = output.past_key_values
                if past_key_values is None:
                    raise RuntimeError("Model did not return the required KV cache")

            for i, token_ids in enumerate(generated):
                visible = token_ids[:-1] if token_ids and token_ids[-1] in self.eos_ids else token_ids
                results.append({
                    "response": self.tokenizer.decode(
                        visible, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                    ),
                    "finish_reason": finish_reasons[i],
                    "input_tokens": int(prompt_lengths[i]),
                    "output_tokens": len(token_ids),
                })
        return results
