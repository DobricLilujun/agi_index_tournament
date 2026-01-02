import random
import nltk
from nltk import word_tokenize, pos_tag
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from typing import List, Dict, Optional
from .SynonymSources import MultiSourceSynonymRetriever

# ============================================================================
# NLTK Resources Initialization
# ============================================================================
# Download necessary NLTK resources for POS tagging, WSD, and synonym lookup
# These resources are cached locally (~/.nltk_data/) after first download

nltk.download('punkt_tab')  # Punkt tokenizer - splits text into sentences and words
nltk.download('averaged_perceptron_tagger')  # POS tagger (general version)
nltk.download('averaged_perceptron_tagger_eng')  # POS tagger (English-specific)
nltk.download('wordnet')  # WordNet synonyms/antonyms database
nltk.download('omw-1.4')  # Open Multilingual Wordnet - cross-lingual mappings

class DictRewriter:
    """
    Multi-source text rewriting class supporting multiple synonym dictionaries.
    Performs word sense disambiguation (WSD) and synonym replacement with configurable sources.
    Supports: WordNet, FastText, Sense2Vec, FASS (Fast-Sense Dictionary)
    Preserves the original document structure (line breaks, formatting).
    """
    def __init__(self, sources: List[str] = None, source_weights: Dict[str, float] = None,
                 aggregation_method: str = 'ranked'):
        """
        Initialize the multi-source rewriter.
        
        Args:
            sources: List of synonym sources to use. 
                    Options: ['wordnet', 'fasttext', 'sense2vec', 'fass']
                    Default: ['wordnet', 'fasttext']
            source_weights: Custom weights for source ranking
            aggregation_method: How to aggregate synonyms from multiple sources
                               Options: 'ranked', 'all', 'union', 'intersection'
        """
        self.sources = sources or ['wordnet', 'fasttext']
        self.source_weights = source_weights or {}
        self.aggregation_method = aggregation_method
        
        # Initialize multi-source retriever
        self.synonym_retriever = MultiSourceSynonymRetriever(
            sources=self.sources,
            source_weights=self.source_weights
        )
        
        # Log available sources
        print(f"✓ DictRewriter initialized with sources: {list(self.synonym_retriever.sources.keys())}")
        print(f"  Aggregation method: {aggregation_method}")

    def set_sources(self, sources: List[str], aggregation_method: str = None):
        """
        Change the synonym sources and aggregation method at runtime.
        
        Args:
            sources: List of sources to use
            aggregation_method: Aggregation strategy ('ranked', 'all', 'union', 'intersection')
        """
        self.sources = sources
        if aggregation_method:
            self.aggregation_method = aggregation_method
        
        self.synonym_retriever = MultiSourceSynonymRetriever(
            sources=self.sources,
            source_weights=self.source_weights
        )
        print(f"✓ Sources updated to: {self.sources}")
        print(f"  Aggregation method: {self.aggregation_method}")
    
    def get_available_sources(self) -> List[str]:
        """Get list of available synonym sources."""
        return ['wordnet', 'fasttext', 'sense2vec', 'fass']
    
    def get_active_sources(self) -> List[str]:
        """Get currently active synonym sources."""
        return list(self.synonym_retriever.sources.keys())
    
    def add_fass_dictionary(self, filepath: str):
        """
        Add custom FASS (Fast-Sense Dictionary) from JSON file.
        
        Args:
            filepath: Path to JSON file with format {'word': ['syn1', 'syn2', ...]}
        """
        if 'fass' in self.synonym_retriever.sources:
            self.synonym_retriever.sources['fass'].load_from_file(filepath)
            print(f"✓ FASS dictionary loaded from {filepath}")
        else:
            print("⚠ FASS source not active. Add 'fass' to sources first.")
    
    @staticmethod
    def ptb_to_wordnet_pos(ptb_tag: str):
        """
        Convert PTB (Penn Treebank) POS tags to WordNet POS constants.
        
        Args:
            ptb_tag (str): Penn Treebank POS tag (e.g., 'NN', 'VB', 'JJ', 'RB')
        
        Returns:
            int or None: WordNet POS constant (NOUN, VERB, ADJ, ADV) or None if unmapped
        """
        if not ptb_tag:
            return None

        tag = ptb_tag.upper()

        # Noun tags (singular, plural, proper noun)
        if tag in ('NN', 'NNS', 'NNP', 'NNPS'):
            return wn.NOUN
        # Verb tags (base, past, gerund, past participle, etc.)
        elif tag in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'):
            return wn.VERB
        # Adjective tags (positive, comparative, superlative)
        elif tag in ('JJ', 'JJR', 'JJS'):
            return wn.ADJ
        # Adverb tags (regular, comparative, superlative, wh-adverb)
        elif tag in ('RB', 'RBR', 'RBS', 'WRB'):
            return wn.ADV

        else:
            return None

    @staticmethod
    def synset_id_str(ss):
        """
        Generate a unique WordNet synset ID string.
        
        Args:
            ss (Synset): WordNet synset object
        
        Returns:
            str: Unique identifier in format 'offset-pos' (e.g., '02084442-n')
        """
        return f"{ss.offset():08d}-{ss.pos()}"

    def wsd_sentence(self, sentence: str):
        """
        Perform Word Sense Disambiguation (WSD) on a sentence.
        Uses the Lesk algorithm to find the most likely meaning of each word based on context.
        Integrates multi-source synonym retrieval for enhanced vocabulary coverage.
        
        Args:
            sentence (str): Input sentence to disambiguate
        
        Returns:
            list: List of dictionaries containing word sense information for each token
        """
        tokens = word_tokenize(sentence)
        tags = pos_tag(tokens)
        tag_map = dict(tags)

        results = []
        for w in tokens:
            # Get the PTB POS tag for the word
            ptb = tag_map.get(w)
            # Convert PTB tag to WordNet POS format
            wn_pos = self.ptb_to_wordnet_pos(ptb)
            # Apply Lesk algorithm for word sense disambiguation
            ss = lesk(tokens, w, wn_pos) if wn_pos else lesk(tokens, w)
            
            # Get synonyms from all configured sources
            source_synonyms = self.synonym_retriever.get_synonyms(w, ptb, 
                                                                  aggregation=self.aggregation_method,
                                                                  top_k=3)
            
            # Extract ranked/primary synonyms
            primary_synonyms = source_synonyms.get('ranked', [])
            if not primary_synonyms:
                primary_synonyms = source_synonyms.get('union', [])
            if not primary_synonyms and ss:
                primary_synonyms = [lemma.name() for lemma in ss.lemmas()]
            
            if ss:
                # Successfully found the word sense - record synonyms and definition
                item = {
                    "word": w,
                    "pos": ptb,
                    "wnet_pos": wn_pos,
                    "wnet_number": self.synset_id_str(ss),
                    "gloss": ss.definition(),
                    "synonyms": primary_synonyms,
                    "source_synonyms": source_synonyms  # Keep all sources for reference
                }
            else:
                # Word sense not found - record empty values but still try multi-source retrieval
                item = {
                    "word": w,
                    "pos": ptb,
                    "wnet_pos": wn_pos,
                    "wnet_number": None,
                    "gloss": None,
                    "synonyms": primary_synonyms,
                    "source_synonyms": source_synonyms
                }
            results.append(item)
        return results

    def rewrite(
        self,  
        text: str,  # Input text to be rewritten
        ratio: float = 0.3,  # Proportion of words to replace (0.0-1.0)
        is_replace_by_description: bool = False,  # Control noun replacement by description
        exclude_word_class: list = None,  # e.g., ["NN", "NNP"]
        sources: List[str] = None,  # Override sources for this rewrite
        aggregation: str = None,  # Override aggregation for this rewrite
    ):
        """
        Rewrite text by replacing words with synonyms at a specified ratio.
        Supports multiple synonym sources with configurable aggregation.
        Preserves document structure (line breaks) while augmenting vocabulary.
        
        Args:
            text (str): Input text to be rewritten
            ratio (float): Proportion of words to replace (0.0-1.0, default 0.3 = 30%)
            is_replace_by_description (bool): Whether to replace nouns with definitions
            exclude_word_class (list): List of POS tags to exclude from replacement
            sources (list): Override active sources for this rewrite
            aggregation (str): Override aggregation method for this rewrite
        
        Returns:
            tuple: (rewritten_text, wsd_info_list) - augmented text and disambiguation data
        """
        assert 0 <= ratio <= 1, "ratio must be from 0 to 1"
        exclude_word_class = exclude_word_class or []
        
        # Temporarily override sources if specified
        original_sources = None
        original_aggregation = None
        if sources or aggregation:
            original_sources = self.sources
            original_aggregation = self.aggregation_method
            if sources:
                self.set_sources(sources, aggregation or self.aggregation_method)
            if aggregation:
                self.aggregation_method = aggregation

        # Step 1: Split text into lines, preserving line breaks
        lines = text.split("\n")
        line_tokens = []
        all_words = []
        wsd_info_all = []

        # Step 2: Perform WSD on each line and collect all word information
        for line in lines:
            if not line.strip():
                line_tokens.append([])  # Empty line
                continue

            wsd_info = self.wsd_sentence(line)
            tokens = [w["word"] for w in wsd_info]
            all_words.extend(tokens)
            wsd_info_all.extend(wsd_info)
            line_tokens.append(tokens)

        # Step 3: Identify all replaceable words (based on global statistics)
        # Filter out: single-character words, prepositions (IN), articles/determiners (DT), 
        # conjunctions (CC), and other function words that cannot be meaningfully replaced
        exclude_pos_tags = {'IN', 'DT', 'CC', 'TO', 'PRP', 'PRP$', 'WDT', 'WP', 'WP$'}
        
        candidate_indices = [
            i for i, w in enumerate(wsd_info_all)
            if (len(w["word"]) > 1 and 
                w["pos"] not in exclude_pos_tags and 
                w["pos"] not in exclude_word_class and 
                w["synonyms"])
        ]

        if not candidate_indices:
            return text  # No replaceable words - return original text

        # Step 4-5 (Merged): Replace words using both description and synonym methods
        # Calculate total number of words to replace based on ratio
        num_to_replace = int(len(candidate_indices) * ratio)
        
        if num_to_replace == 0:
            return text, wsd_info_all
        
        # Randomly select indices to replace
        replace_indices = set(random.sample(candidate_indices, num_to_replace))
        
        # If description replacement is enabled, identify replaceable nouns
        if is_replace_by_description:
            # Find nouns with definitions among the selected replacement indices
            noun_indices_with_gloss = [
                i for i in replace_indices
                if wsd_info_all[i]["pos"] in ('NN', 'NNS', 'NNP', 'NNPS') and wsd_info_all[i]["gloss"]
            ]
            
            # Randomly split replacements between description and synonym methods
            # Use roughly 50-50 split if both methods are applicable
            if noun_indices_with_gloss:
                num_description_replace = len(noun_indices_with_gloss) // 2
                description_indices = set(random.sample(noun_indices_with_gloss, num_description_replace))
                synonym_indices = replace_indices - description_indices
            else:
                # No nouns with definitions, use synonym replacement only
                description_indices = set()
                synonym_indices = replace_indices
            
            # Replace selected nouns with their definitions
            for i in description_indices:
                w = wsd_info_all[i]
                all_words[i] = f'"{w["gloss"]}"'
            
            # Replace remaining words with synonyms
            for i in synonym_indices:
                w = wsd_info_all[i]
                synonyms = w["synonyms"]
                original = w["word"]
                if synonyms:
                    # Exclude synonyms identical to original word, randomly select replacement
                    candidates = [s for s in synonyms if s.lower() != original.lower()]# Exclude identical synonyms
                    if candidates:
                        all_words[i] = random.choice(candidates).replace('_', ' ')
        else:
            # Only use synonym replacement
            for i in replace_indices:
                w = wsd_info_all[i]
                synonyms = w["synonyms"]
                original = w["word"]
                if synonyms:
                    # Exclude synonyms identical to original word, randomly select replacement
                    candidates = [s for s in synonyms if s.lower() != original.lower()]
                    if candidates:
                        all_words[i] = random.choice(candidates).replace('_', ' ')


        # Step 6: Reconstruct text according to original line structure
        new_text_lines = []
        idx = 0
        for tokens in line_tokens:
            if not tokens:
                new_text_lines.append("")
                continue
            line_len = len(tokens)
            new_text_lines.append(" ".join([w.strip() for w in all_words[idx: idx + line_len]]))
            idx += line_len

        result = "\n".join(new_text_lines), wsd_info_all
        
        # Restore original sources if they were overridden
        if original_sources:
            self.set_sources(original_sources, original_aggregation)
        
        return result
