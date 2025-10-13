# -*- coding: utf-8 -*-
# Hybrite (Hybrid Rewrite) dataset reconstruction pipeline
# Modules: dictionary-guided rewriting via constituency/dependency cues, back-translation,
# masked LM token substitution, and stylistic paraphrasing
#
# Dependencies:
#   pip install spacy transformers sentencepiece torch nltk sacremoses
#   python -m spacy download en_core_web_sm
# Optional:
#   pip install deep-translator   # if using online translation fallback

import os
import re
import random
from typing import List, Dict, Any, Tuple
import warnings

warnings.filterwarnings("ignore")

# ============ Imports ============
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception:
    pass

import numpy as np
import spacy

from transformers import (
    pipeline, AutoModelForMaskedLM, AutoTokenizer,
    MarianMTModel, MarianTokenizer
)

# ============ Helpers: Back-translation (local MarianMT preferred) ============
class LocalBackTranslator:
    """
    Local back-translation using MarianMT models.
    Provide language pairs that exist on Hugging Face (Helsinki-NLP/opus-mt-XX-YY).
    Default uses English <-> German/French/Spanish as examples.
    """
    def __init__(self, target_langs: List[str] = ("de", "fr", "es")):
        self.langs = target_langs
        self.models = {}
        # Map to Marian model names
        # en->xx and xx->en models
        for lang in self.langs:
            en2xx = f"Helsinki-NLP/opus-mt-en-{lang}"
            xx2en = f"Helsinki-NLP/opus-mt-{lang}-en"
            try:
                self.models[(f"en->{lang}")]=(
                    MarianMTModel.from_pretrained(en2xx),
                    MarianTokenizer.from_pretrained(en2xx)
                )
                self.models[(f"{lang}->en")]=(
                    MarianMTModel.from_pretrained(xx2en),
                    MarianTokenizer.from_pretrained(xx2en)
                )
            except Exception as e:
                # If model not available locally, mark missing
                self.models[(f"en->{lang}")] = None
                self.models[(f"{lang}->en")] = None

    def _translate(self, text: str, direction: str) -> str:
        pair = self.models.get(direction)
        if not pair:
            return ""
        model, tokenizer = pair
        batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        gen = model.generate(**batch, max_new_tokens=200)
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)
        return out[0] if out else ""

    def back_translate(self, text: str) -> List[str]:
        results = []
        for lang in self.langs:
            try:
                to_lang = self._translate(text, f"en->{lang}")
                if not to_lang:
                    continue
                back = self._translate(to_lang, f"{lang}->en")
                if back and back != text:
                    results.append(back)
            except Exception:
                continue
        return results
    

# ============ Core Hybrite Pipeline ============
class HybritePipeline:
    def __init__(
        self,
        dict_map: Dict[str, List[str]] = None,
        backtrans_langs: List[str] = ("de", "fr", "es"),
        paraphrase_model_name: str = "t5-small",
        mlm_model_name: str = "bert-base-uncased",
        device: int = -1,
        seed: int = 42
    ):
        random.seed(seed)
        np.random.seed(seed)

        # spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("spaCy model en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")

        # Dictionary for dictionary-based rewriting
        self.word_dict = dict_map or {
            "good": ["excellent", "great", "wonderful", "fantastic"],
            "bad": ["terrible", "awful", "poor", "horrible"],
            "big": ["large", "huge", "massive", "enormous"],
            "small": ["tiny", "little", "minute", "compact"],
            "fast": ["quick", "rapid", "swift", "speedy"],
            "slow": ["sluggish", "leisurely", "unhurried", "gradual"],
            "today": ["nowadays", "at present", "currently"],
            "very": ["highly", "extremely", "remarkably", "particularly"]
        }

        # Paraphrase model (style paraphrasing)
        try:
            self.paraphraser = pipeline("text2text-generation", model=paraphrase_model_name, device=device)
        except Exception:
            self.paraphraser = None

        # Masked LM
        try:
            tok = AutoTokenizer.from_pretrained(mlm_model_name)
            mdl = AutoModelForMaskedLM.from_pretrained(mlm_model_name)
            self.mlm = pipeline("fill-mask", model=mdl, tokenizer=tok, device=device)
            self.mask_token = tok.mask_token
        except Exception:
            self.mlm = None
            self.mask_token = "[MASK]"

        # Back-translation (local MarianMT)
        self.bt = LocalBackTranslator(target_langs=list(backtrans_langs))

    # -------- Linguistic structure extraction --------
    def get_dependency_info(self, sent: str) -> List[Dict[str, Any]]:
        doc = self.nlp(sent)
        info = []
        for t in doc:
            info.append({
                "text": t.text,
                "lemma": t.lemma_,
                "pos": t.pos_,
                "tag": t.tag_,
                "dep": t.dep_,
                "head": t.head.text if t.head != t else "ROOT",
                "children": [c.text for c in t.children]
            })
        return info

    def get_constituency_cues(self, sent: str) -> Dict[str, Any]:
        doc = self.nlp(sent)
        return {
            "noun_phrases": [np.text for np in doc.noun_chunks],
            "entities": [(ent.text, ent.label_) for ent in doc.ents]
        }

    # -------- Dictionary-based rewriting (structure guided) --------
    def dictionary_rewrite(self, sent: str, max_variants: int = 5) -> List[str]:
        words = sent.split()
        out = []
        # Use dependency/NP cues mainly to avoid replacing named entities
        doc = self.nlp(sent)
        ents = {(e.start, e.end) for e in doc.ents}

        def in_ent(idx: int) -> bool:
            for s, e in ents:
                if s <= idx < e:
                    return True
            return False

        for _ in range(max_variants):
            cand = words[:]
            for i, tok in enumerate(doc):
                if in_ent(i):
                    continue
                src = re.sub(r"[^\w']", "", tok.text.lower())
                if src in self.word_dict and random.random() < 0.5:
                    replacement = random.choice(self.word_dict[src])
                    # preserve capitalization
                    if tok.text[:1].isupper():
                        replacement = replacement.capitalize()
                    # preserve trailing punctuation
                    trailing = ""
                    if re.match(r".*[.,!?;:]$", tok.text):
                        trailing = tok.text[-1]
                    cand[i] = replacement + trailing
            cand_sent = self._tidy_spaces(" ".join(cand))
            if cand_sent != sent and cand_sent not in out:
                out.append(cand_sent)
        return out

    # -------- Back-translation --------
    def back_translate(self, sent: str) -> List[str]:
        if not self.bt:
            return []
        return self.bt.back_translate(sent)

    # -------- Masked LM substitution --------
    def mlm_substitute(self, sent: str, mask_ratio: float = 0.15, rounds: int = 3) -> List[str]:
        if not self.mlm:
            return []
        words = sent.split()
        n_mask = max(1, int(len(words) * mask_ratio))
        outs = []

        for _ in range(rounds):
            idxs = random.sample(range(len(words)), min(n_mask, len(words)))
            cand = words[:]
            for idx in idxs:
                original = cand[idx]
                # keep punctuation attached
                core = re.sub(r"([.,!?;:]+)$", "", original)
                punct = "" if core == original else original[len(core):]
                token = core if core else original
                cand[idx] = self.mask_token + punct

                masked = self._tidy_spaces(" ".join(cand))
                try:
                    preds = self.mlm(masked)
                except Exception:
                    cand[idx] = original
                    continue

                # transformers returns list of candidates when single mask present.
                # If multiple masks, pipeline returns list-per-mask; so handle both.
                best = None
                if isinstance(preds, list) and preds and isinstance(preds[0], dict):
                    # single [MASK]
                    best = preds[0]["token_str"].strip()
                elif isinstance(preds, list) and preds and isinstance(preds[0], list):
                    # multiple masks: take the first list's top
                    best = preds[0][0]["token_str"].strip()
                if best:
                    cand[idx] = best + punct
                else:
                    cand[idx] = original

            new_sent = self._tidy_spaces(" ".join(cand))
            if new_sent != sent and new_sent not in outs:
                outs.append(new_sent)
        return outs

    # -------- Stylistic paraphrasing --------
    def paraphrase(self, sent: str, n: int = 3, max_len: int = 128, temperature: float = 0.8) -> List[str]:
        if not self.paraphraser:
            return []
        prompt = f"paraphrase: {sent}"
        try:
            outs = self.paraphraser(
                prompt,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=n,
                max_length=max_len,
                no_repeat_ngram_size=2,
                clean_up_tokenization_spaces=True
            )
        except Exception:
            return []
        results = []
        for o in outs:
            txt = o.get("generated_text", "").strip()
            if txt and txt != sent and txt not in results:
                results.append(self._tidy_spaces(txt))
        return results

    # -------- Orchestrator --------
    def reconstruct_dataset(self, sents: List[str]) -> Dict[str, List[str]]:
        res = {
            "original": sents[:],
            "dictionary_based": [],
            "back_translated": [],
            "masked_lm": [],
            "stylistic": [],
            "all_variants": []
        }
        for s in sents:
            dict_vars = self.dictionary_rewrite(s)
            bt_vars = self.back_translate(s)
            mlm_vars = self.mlm_substitute(s)
            para_vars = self.paraphrase(s)

            res["dictionary_based"].extend(dict_vars)
            res["back_translated"].extend(bt_vars)
            res["masked_lm"].extend(mlm_vars)
            res["stylistic"].extend(para_vars)
            res["all_variants"].extend(dict_vars + bt_vars + mlm_vars + para_vars)
        return res

    # -------- Evaluation (generalizability proxies) --------
    def evaluate_generalizability(self, originals: List[str], variants: List[str]) -> Dict[str, float]:
        if not originals or not variants:
            return {
                "vocabulary_diversity": 0.0,
                "sentence_length_variance": 0.0,
                "syntactic_complexity": 0.0
            }
        orig_vocab = set()
        for s in originals:
            orig_vocab.update(self._tok(s))
        var_vocab = set()
        for s in variants:
            var_vocab.update(self._tok(s))

        vocab_div = len(var_vocab) / max(1, len(orig_vocab))

        olens = [len(self._tok(s)) for s in originals]
        vlens = [len(self._tok(s)) for s in variants]
        o_var = np.var(olens) if len(olens) > 1 else 0.0
        v_var = np.var(vlens) if len(vlens) > 1 else 0.0
        len_var_ratio = (v_var / o_var) if o_var > 0 else 0.0

        o_depth = self._avg_dep_depth(originals)
        v_depth = self._avg_dep_depth(variants)
        syn_complex = (v_depth / o_depth) if o_depth > 0 else 0.0

        return {
            "vocabulary_diversity": float(vocab_div),
            "sentence_length_variance": float(len_var_ratio),
            "syntactic_complexity": float(syn_complex)
        }

    # -------- Internals --------
    def _tok(self, s: str) -> List[str]:
        return [w for w in re.findall(r"\w+|[^\w\s]", s)]

    def _avg_dep_depth(self, sents: List[str]) -> float:
        total, cnt = 0, 0
        for s in sents:
            doc = self.nlp(s)
            for t in doc:
                total += self._depth(t)
                cnt += 1
        return total / cnt if cnt else 0.0

    def _depth(self, token) -> int:
        d = 0
        cur = token
        while cur.head != cur:
            d += 1
            cur = cur.head
        return d

    def _tidy_spaces(self, s: str) -> str:
        # Fix spaces before punctuation
        s = re.sub(r"\s+([.,!?;:])", r"\1", s)
        # Collapse multi-spaces
        s = re.sub(r"\s{2,}", " ", s)
        # Fix " ( " or " ) " style
        s = re.sub(r"\(\s+", "(", s)
        s = re.sub(r"\s+\)", ")", s)
        return s.strip()

