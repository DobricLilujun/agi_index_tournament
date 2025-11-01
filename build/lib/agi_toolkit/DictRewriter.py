import random
import nltk
from nltk import word_tokenize, pos_tag
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

# 下载必要资源
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

class DictRewriter:
    def __init__(self):
        pass

    @staticmethod
    def ptb_to_wordnet_pos(ptb_tag: str):
        if not ptb_tag:
            return None

        tag = ptb_tag.upper()

        if tag in ('NN', 'NNS', 'NNP', 'NNPS'):
            return wn.NOUN

        elif tag in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'):
            return wn.VERB

        elif tag in ('JJ', 'JJR', 'JJS'):
            return wn.ADJ

        elif tag in ('RB', 'RBR', 'RBS', 'WRB'):
            return wn.ADV

        else:
            return None

    @staticmethod
    def synset_id_str(ss):
        """生成唯一 WordNet synset ID"""
        return f"{ss.offset():08d}-{ss.pos()}"

    def wsd_sentence(self, sentence: str):
        """对句子进行词义消歧 (Word Sense Disambiguation)"""
        tokens = word_tokenize(sentence)
        tags = pos_tag(tokens)
        tag_map = dict(tags)

        results = []
        for w in tokens:
            ptb = tag_map.get(w)
            wn_pos = self.ptb_to_wordnet_pos(ptb)
            ss = lesk(tokens, w, wn_pos) if wn_pos else lesk(tokens, w)
            if ss:
                item = {
                    "word": w,
                    "pos": ptb,
                    "wnet_pos": wn_pos,
                    "wnet_number": self.synset_id_str(ss),
                    "gloss": ss.definition(),
                    "synonyms": [lemma.name() for lemma in ss.lemmas()]
                }
            else:
                item = {
                    "word": w,
                    "pos": ptb,
                    "wnet_pos": wn_pos,
                    "wnet_number": None,
                    "gloss": None,
                    "synonyms": []
                }
            results.append(item)
        return results

    def rewrite(
        self,
        text: str,
        ratio: float = 0.3,
        exclude_word_class: list = None,
    ):
        """
        按整体比例改写文本（保留换行）
        exclude_word_class: 例如 ["NN", "NNP", "VB"]
        """
        assert 0 <= ratio <= 1, "ratio must be from 0 to 1"
        exclude_word_class = exclude_word_class or []

        # === Step 1. 保留换行结构 ===
        lines = text.split("\n")
        line_tokens = []
        all_words = []
        wsd_info_all = []

        # === Step 2. 对所有行进行 WSD 并汇总 ===
        for line in lines:
            if not line.strip():
                line_tokens.append([])  # 空行
                continue

            wsd_info = self.wsd_sentence(line)
            tokens = [w["word"] for w in wsd_info]
            all_words.extend(tokens)
            wsd_info_all.extend(wsd_info)
            line_tokens.append(tokens)

        # === Step 3. 获取所有可替换的词索引（基于全局） ===
        candidate_indices = [
            i for i, w in enumerate(wsd_info_all)
            if (w["pos"] not in exclude_word_class and w["synonyms"])
        ]

        if not candidate_indices:
            return text  # 没有可替换词则直接返回

        # === Step 4. 全局抽样替换 ===
        num_to_replace = int(len(candidate_indices) * ratio)
        replace_indices = set(random.sample(candidate_indices, num_to_replace))

        for i, w in enumerate(wsd_info_all):
            if i in replace_indices:
                synonyms = w["synonyms"]
                original = w["word"]
                if synonyms:
                    candidates = [s for s in synonyms if s.lower() != original.lower()]
                    if candidates:
                        all_words[i] = random.choice(candidates).replace('_', ' ')

        # === Step 5. 根据行结构重建文本 ===
        new_text_lines = []
        idx = 0
        for tokens in line_tokens:
            if not tokens:
                new_text_lines.append("")
                continue
            line_len = len(tokens)
            new_text_lines.append(" ".join(all_words[idx: idx + line_len]))
            idx += line_len

        return "\n".join(new_text_lines), wsd_info_all
