# translator.py
# -*- coding: utf-8 -*-

from typing import List, Dict, Optional, Callable, Union, Tuple
import warnings

# Hugging Face transformers backends
try:
    from transformers import (
        MarianMTModel,
        MarianTokenizer,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False


class Translator:

    def __init__(
        self,
        backend: str = "marian",  # "marian" | "nllb" | "llm",
        src_lang: Optional[str] = None,  # default source language (if any)
        tgt_lang: Optional[str] = None,  # default target language (if any)
        is_local_model: bool = False,  # if using local LLM model (for future use)

    ):
        if backend not in {"marian", "nllb", "llm"}:
            raise ValueError("backend must be one of 'marian', 'nllb', or 'llm'")
        self.backend = backend
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # MarianMT settings
        if "marian" in backend:
            self.marian_model_name: Optional[str] = None  # e.g., "Helsinki-NLP/opus-mt-en-de"
            self.marian_multilingual_prefix: bool = False  # use >>xx<< prefix style for multilingual models
        elif "nllb" in backend:
            if not _HF_AVAILABLE:
                raise ImportError("transformers library is required for NLLB backend")
            self.nllb_language_map: Dict[str, str] = {
                # short code to BCP-47 mapping for common languages
                "eng": "eng_Latn",
                "fra": "fra_Latn",
                "deu": "deu_Latn",
                "spa": "spa_Latn",
                "ita": "ita_Latn",
                "por": "por_Latn",
                "rus": "rus_Cyrl",
                "zho": "zho_Hans",
                "jpn": "jpn_Jpan",
                # add more as needed
            }
        else:
            # LLM Local Loading and Infernce
            if is_local_model:
                self.llm_chat_fn: Optional[Callable[[str, Optional[str]], str]] = None  # user-supplied chat function
                self.llm_system_prompt: Optional[str] = None  # optional system prompt for LLM
            # LLM VLLM API
            else:
                self.llm_chat_fn: Optional[Callable[[str, Optional[str]], str]] = None  # user-supplied chat function
                self.llm_system_prompt: Optional[str] = None  # optional system prompt for LLM


    def _load_marian(self):
        if self._marian_model is not None:
            return
        if not self.marian_model_name:
            raise ValueError("For Marian backend, marian_model_name must be provided.")
        self._marian_tok = MarianTokenizer.from_pretrained(self.marian_model_name)
        self._marian_model = MarianMTModel.from_pretrained(self.marian_model_name)
        if self.device:
            self._marian_model.to(self.device)

    def _load_nllb(self):
        if self._nllb_model is not None:
            return
        self._nllb_tok = AutoTokenizer.from_pretrained(self.nllb_model_name)
        self._nllb_model = AutoModelForSeq2SeqLM.from_pretrained(self.nllb_model_name)
        if self.device:
            self._nllb_model.to(self.device)


    def translate_text(
        self,
        text: str,
    ) -> str:
        
        if self.backend == "marian":
            return self._translate_marian_one(text, src_lang=src_lang, tgt_lang=tgt_lang, **gen_kwargs)
        elif self.backend == "nllb":
            return self._translate_nllb_one(text, src_lang=src_lang, tgt_lang=tgt_lang, **gen_kwargs)
        elif self.backend == "llm":
            return self._translate_llm_one(text, src_lang=src_lang, tgt_lang=tgt_lang)
        else:
            raise ValueError("Unsupported backend")


