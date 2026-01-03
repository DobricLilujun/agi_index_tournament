"""
Back-translation based text rewriter.
Mimics the DictRewriter interface but uses translation pivots (NLLB or ChatGPT)
to produce paraphrases while preserving document structure.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Optional: Hugging Face transformers for NLLB
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    _HF_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _HF_AVAILABLE = False
    AutoModelForSeq2SeqLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore


@dataclass
class BackTranslationConfig:
    backend: str = "nllb"  # "nllb" or "chatgpt"
    pivot_languages: List[str] = field(default_factory=lambda: ["fra_Latn", "deu_Latn", "spa_Latn"])
    model_name: str = "facebook/nllb-200-distilled-600M"
    src_lang: str = "eng_Latn"
    hf_auth_token: Optional[str] = None  # HF token if model is gated (passed as `token`)
    device: Optional[str] = None  # e.g., "cuda" or "cpu"
    max_length: int = 512
    temperature: float = 1.0  # for ChatGPT-style backends
    top_p: float = 0.9  # for ChatGPT-style backends
    chatgpt_fn: Optional[Callable[[str], str]] = None  # user-supplied chat completion function
    seed: Optional[int] = None


def _load_prompt_template(filename: str) -> str:
    """Load a Jinja-style prompt template from the templates folder.

    Falls back to an inline default if the template file is missing.
    """
    base = Path(__file__).resolve().parents[2]  # project root (…/agi_index_tournament)
    template_path = base / "templates" / filename
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")

    # Fallback defaults
    if "forward" in filename:
        return (
            "You are a translation engine. Translate from {src_lang} to {pivot_lang}. "
            "Only return the translated text.\n\n{text}\n"
        )
    return (
        "You are a translation engine. Translate the following from {pivot_lang} back to {src_lang}. "
        "Only return the translated text.\n\n{text}\n"
    )


class BackTranslationRewriter:
    """
    Back-translation rewriter with a DictRewriter-like interface.

    Supports two backends:
    - NLLB (huggingface transformers): offline multilingual translation
    - ChatGPT (or any chat LLM): user provides `chatgpt_fn(prompt) -> str`

    Key features:
    - Multiple pivot languages with configurable selection strategies
    - Line-by-line processing to preserve formatting
    - Deterministic runs via optional seed
    """

    def __init__(self, config: Optional[BackTranslationConfig] = None):
        self.config = config or BackTranslationConfig()
        self._rng = random.Random(self.config.seed)

        self._tokenizer = None
        self._model = None

        if self.config.backend not in {"nllb", "chatgpt"}:
            raise ValueError("backend must be 'nllb' or 'chatgpt'")

        if self.config.backend == "nllb" and not _HF_AVAILABLE:
            raise ImportError("transformers is required for NLLB backend")

        if self.config.backend == "chatgpt" and self.config.chatgpt_fn is None:
            raise ValueError("chatgpt backend requires a chatgpt_fn callback")

    # ------------------------------------------------------------------
    # Backend loaders
    # ------------------------------------------------------------------
    def _load_nllb(self):
        if self._model is not None and self._tokenizer is not None:
            return
        # Use slow tokenizer and set src_lang so language-specific BOS is available.
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=False,
            src_lang=self.config.src_lang,
            token=self.config.hf_auth_token,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            token=self.config.hf_auth_token,
        )
        device = self.config.device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self._model.to(device)
        self._device = device

    # ------------------------------------------------------------------
    # Pivot management
    # ------------------------------------------------------------------
    def set_pivot_languages(self, pivot_languages: List[str]):
        if not pivot_languages:
            raise ValueError("pivot_languages cannot be empty")
        self.config.pivot_languages = pivot_languages

    def get_pivot_languages(self) -> List[str]:
        return list(self.config.pivot_languages)

    # ------------------------------------------------------------------
    # Translation helpers
    # ------------------------------------------------------------------
    def _translate_nllb_one(self, text: str, src_lang: str, tgt_lang: str) -> str:
        self._load_nllb()
        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=self.config.max_length)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        # Prefer convert_tokens_to_ids; fall back to lang_code_to_id if available.
        if hasattr(self._tokenizer, "convert_tokens_to_ids"):
            bos_id = self._tokenizer.convert_tokens_to_ids(tgt_lang)
        else:  # pragma: no cover - compatibility fallback
            bos_id = self._tokenizer.lang_code_to_id[tgt_lang]  # type: ignore
        outputs = self._model.generate(
            **inputs,
            forced_bos_token_id=bos_id,
            max_length=self.config.max_length,
        )
        decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded[0] if decoded else text

    def _translate_chatgpt_one(self, text: str, src_lang: str, tgt_lang: str) -> str:
        # The user supplies chatgpt_fn; prompt is loaded from template for easy editing.
        prompt_tpl = _load_prompt_template("back_translation_forward_prompt.jinja")
        prompt = prompt_tpl.format(src_lang=src_lang, pivot_lang=tgt_lang, text=text)
        result = self.config.chatgpt_fn(prompt)
        return result.strip() if isinstance(result, str) else text

    def _back_translate_once(self, text: str, pivot_lang: str) -> Tuple[str, Dict[str, str]]:
        """
        Perform a single round of back-translation through a pivot language.
        Returns paraphrased text and metadata.
        """
        if self.config.backend == "nllb":
            forward = self._translate_nllb_one(text, src_lang=self.config.src_lang, tgt_lang=pivot_lang)
            back = self._translate_nllb_one(forward, src_lang=pivot_lang, tgt_lang=self.config.src_lang)
        else:
            forward_prompt = _load_prompt_template("back_translation_forward_prompt.jinja").format(
                src_lang=self.config.src_lang,
                pivot_lang=pivot_lang,
                text=text,
            )
            # Use provided fn twice for forward/backward steps
            forward = self.config.chatgpt_fn(forward_prompt)  # type: ignore
            forward = forward.strip() if isinstance(forward, str) else text
            back_prompt = _load_prompt_template("back_translation_back_prompt.jinja").format(
                pivot_lang=pivot_lang,
                src_lang=self.config.src_lang,
                text=forward,
            )
            back = self.config.chatgpt_fn(back_prompt)  # type: ignore
            back = back.strip() if isinstance(back, str) else text

        metadata = {
            "pivot": pivot_lang,
            "forward": forward,
            "back": back,
        }
        return back, metadata

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def rewrite(
        self,
        text: str,
        pivot_strategy: str = "random",  # "random" or "round_robin"
        rounds: int = 1,
        pivot_override: Optional[List[str]] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Rewrite text using back-translation.

        Args:
            text: Input text (line breaks preserved).
            pivot_strategy: How to choose pivot language per round (random | round_robin).
            rounds: Number of back-translation rounds to apply.
            pivot_override: Optional list of pivot languages to use for this call.

        Returns:
            (rewritten_text, metadata_per_line)
            metadata_per_line is a list with one entry per input line containing:
              - pivot: pivot language code used
              - forward: forward translation
              - back: back-translated text
        """
        if rounds < 1:
            raise ValueError("rounds must be >= 1")

        pivot_languages = pivot_override or self.config.pivot_languages
        if not pivot_languages:
            raise ValueError("No pivot languages configured")

        # If using ChatGPT-style backend, process the full text in one go per round.
        if self.config.backend == "chatgpt":
            current_text = text
            metadata_all: List[Dict[str, str]] = []
            rr_idx = 0

            for _ in range(rounds):
                if pivot_strategy == "round_robin":
                    pivot = pivot_languages[rr_idx % len(pivot_languages)]
                    rr_idx += 1
                else:
                    pivot = self._rng.choice(pivot_languages)

                current_text, meta = self._back_translate_once(current_text, pivot)
                metadata_all.append(meta)

            return current_text, metadata_all

        # NLLB path preserves line structure.
        lines = text.split("\n")
        rewritten_lines: List[str] = []
        metadata_all: List[Dict[str, str]] = []

        # Round-robin tracker
        rr_idx = 0
        for line in lines:
            if not line.strip():
                rewritten_lines.append("")
                metadata_all.append({"pivot": None, "forward": "", "back": ""})
                continue

            current_text = line

            for _ in range(rounds):
                if pivot_strategy == "round_robin":
                    pivot = pivot_languages[rr_idx % len(pivot_languages)]
                    rr_idx += 1
                else:
                    pivot = self._rng.choice(pivot_languages)

                current_text, meta = self._back_translate_once(current_text, pivot)
                metadata_all.append(meta)  # store each round's metadata

            rewritten_lines.append(current_text)

        rewritten_text = "\n".join(rewritten_lines)
        return rewritten_text, metadata_all


__all__ = ["BackTranslationRewriter", "BackTranslationConfig"]
