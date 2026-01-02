"""
Multi-source Synonym Dictionary Manager.
Integrates WordNet, FastText, Sense2Vec, and FASS for high-quality synonym retrieval.
Supports source selection and ranked aggregation.
"""

import random
import nltk
from nltk.corpus import wordnet as wn
from typing import List, Dict, Tuple, Optional
import warnings

# Download required resources
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class SynonymSource:
    """Base class for synonym sources."""
    
    def get_synonyms(self, word: str, pos: str = None) -> List[str]:
        """
        Get synonyms for a word.
        
        Args:
            word: Target word
            pos: Part-of-speech tag (PTB format: NN, VB, JJ, RB, etc.)
        
        Returns:
            List of synonyms
        """
        raise NotImplementedError


class WordNetSource(SynonymSource):
    """WordNet-based synonym retrieval."""
    
    def __init__(self):
        """Initialize WordNet source."""
        self.name = "WordNet"
        self.priority = 1  # Highest priority - most reliable
    
    def get_synonyms(self, word: str, pos: str = None, synset=None) -> List[str]:
        """
        Get synonyms from WordNet.
        
        Args:
            word: Target word
            pos: PTB POS tag
            synset: Specific synset to use
        
        Returns:
            List of unique synonyms
        """
        if not synset:
            # Get all synsets for the word
            synsets = wn.synsets(word)
        else:
            synsets = [synset]
        
        synonyms = set()
        for ss in synsets:
            for lemma in ss.lemmas():
                syn = lemma.name().replace('_', ' ')
                if syn.lower() != word.lower():
                    synonyms.add(syn)
        
        return list(synonyms)


class FastTextSource(SynonymSource):
    """FastText-based semantic similarity for synonym retrieval."""
    
    def __init__(self):
        """Initialize FastText source."""
        self.name = "FastText"
        self.priority = 2  # Medium priority
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load FastText model."""
        try:
            import fasttext.util
            # Try to load pre-trained model (requires gensim)
            try:
                import gensim.downloader as api
                self.model = api.load("fasttext-wiki-news-subwords-300")
            except:
                warnings.warn("FastText model not available. Falling back to NLTK embeddings.")
                self.model = None
        except ImportError:
            warnings.warn("fasttext package not installed. Install with: pip install fasttext")
            self.model = None
    
    def get_synonyms(self, word: str, pos: str = None, top_k: int = 5) -> List[str]:
        """
        Get semantically similar words using FastText.
        
        Args:
            word: Target word
            pos: POS tag (unused for FastText)
            top_k: Number of similar words to return
        
        Returns:
            List of similar words ranked by similarity
        """
        if not self.model:
            return []
        
        try:
            # Get similar words from model
            similar_words = self.model.most_similar(word.lower(), topn=top_k)
            return [w for w, score in similar_words]
        except:
            return []


class Sense2VecSource(SynonymSource):
    """Sense2Vec-based multi-sense word embeddings."""
    
    def __init__(self):
        """Initialize Sense2Vec source."""
        self.name = "Sense2Vec"
        self.priority = 2  # Medium priority
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load Sense2Vec model."""
        try:
            from sense2vec import Sense2Vec
            # Download and load pre-trained model
            try:
                self.model = Sense2Vec().from_disk("s2v_reddit_2015_md")
            except:
                warnings.warn("Sense2Vec model not available. Download with: python -m sense2vec download s2v_reddit_2015_md")
                self.model = None
        except ImportError:
            warnings.warn("sense2vec package not installed. Install with: pip install sense2vec")
            self.model = None
    
    def get_synonyms(self, word: str, pos: str = None, top_k: int = 5) -> List[str]:
        """
        Get multi-sense similar words using Sense2Vec.
        
        Args:
            word: Target word
            pos: PTB POS tag for sense selection
            top_k: Number of similar words to return
        
        Returns:
            List of similar words with senses
        """
        if not self.model:
            return []
        
        try:
            # Map POS tag to Sense2Vec format
            sense_tag = self._map_pos_to_sense(pos)
            query = f"{word.lower()}|{sense_tag}" if sense_tag else word.lower()
            
            # Get similar words
            similar = self.model.most_similar(query, n_similar=top_k)
            return [word.replace(f"|{sense_tag}", "").replace("|", " ") for word, score in similar]
        except:
            return []
    
    @staticmethod
    def _map_pos_to_sense(pos: str) -> str:
        """Map PTB POS tags to Sense2Vec sense tags."""
        if not pos:
            return ""
        
        pos_map = {
            'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'NOUN', 'NNPS': 'NOUN',
            'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',
            'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
            'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV', 'WRB': 'ADV'
        }
        return pos_map.get(pos.upper(), "")


class FASSSource(SynonymSource):
    """Fast-Sense Dictionary (FASS) - domain-optimized synonyms."""
    
    def __init__(self):
        """Initialize FASS source."""
        self.name = "FASS"
        self.priority = 3  # Lower priority for specialized use
        # Custom synonym dictionary for FASS
        self.fass_dict = self._load_fass_dictionary()
    
    def _load_fass_dictionary(self) -> Dict[str, List[str]]:
        """
        Load or create FASS (Fast-Sense Dictionary).
        Returns a basic FASS dictionary structure.
        Users can extend this with their own domain-specific synonyms.
        """
        # Basic FASS structure - can be extended with domain-specific synonyms
        fass_dict = {
            # Example entries (minimal)
            "good": ["excellent", "fine", "great", "positive"],
            "bad": ["poor", "terrible", "negative", "awful"],
            "big": ["large", "huge", "great", "enormous"],
            "small": ["tiny", "little", "minute", "compact"],
        }
        return fass_dict
    
    def load_from_file(self, filepath: str):
        """
        Load FASS dictionary from file.
        File format: JSON with structure {'word': ['synonym1', 'synonym2', ...]}
        """
        import json
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.fass_dict.update(json.load(f))
        except Exception as e:
            warnings.warn(f"Failed to load FASS dictionary from {filepath}: {e}")
    
    def get_synonyms(self, word: str, pos: str = None) -> List[str]:
        """
        Get synonyms from FASS dictionary.
        
        Args:
            word: Target word
            pos: POS tag (unused for FASS)
        
        Returns:
            List of FASS synonyms
        """
        return self.fass_dict.get(word.lower(), [])


class MultiSourceSynonymRetriever:
    """
    Multi-source synonym retriever with ranking and aggregation.
    Combines multiple synonym sources for highest quality results.
    """
    
    def __init__(self, sources: List[str] = None, source_weights: Dict[str, float] = None):
        """
        Initialize multi-source retriever.
        
        Args:
            sources: List of source names to use. 
                    Options: ['wordnet', 'fasttext', 'sense2vec', 'fass']
                    Default: ['wordnet', 'fasttext']
            source_weights: Dictionary mapping source names to weight values.
                           Used for ranking aggregation.
        """
        self.sources = {}
        self.source_weights = source_weights or {}
        
        # Initialize available sources
        all_sources = {
            'wordnet': WordNetSource(),
            'fasttext': FastTextSource(),
            'sense2vec': Sense2VecSource(),
            'fass': FASSSource(),
        }
        
        # Select requested sources (default: WordNet + FastText)
        if sources is None:
            sources = ['wordnet', 'fasttext']
        
        for source_name in sources:
            if source_name.lower() in all_sources:
                self.sources[source_name.lower()] = all_sources[source_name.lower()]
                # Set default weights if not provided
                if source_name.lower() not in self.source_weights:
                    self.source_weights[source_name.lower()] = all_sources[source_name.lower()].priority
    
    def get_synonyms(self, word: str, pos: str = None, aggregation: str = 'ranked', 
                     top_k: int = 10) -> Dict[str, List[str]]:
        """
        Get synonyms from all active sources.
        
        Args:
            word: Target word
            pos: POS tag
            aggregation: Aggregation strategy
                        'ranked': Sort by source priority and remove duplicates
                        'all': Return all synonyms from all sources
                        'union': Return unique synonyms from any source
                        'intersection': Return only synonyms common to all sources
            top_k: Maximum number of synonyms to return
        
        Returns:
            Dictionary with keys being source names and values being synonym lists
        """
        all_synonyms = {}
        
        # Collect synonyms from all active sources
        for source_name, source in self.sources.items():
            try:
                syns = source.get_synonyms(word, pos)
                all_synonyms[source_name] = syns[:top_k]
            except Exception as e:
                warnings.warn(f"Error retrieving from {source_name}: {e}")
                all_synonyms[source_name] = []
        
        # Apply aggregation strategy
        if aggregation == 'ranked':
            return self._aggregate_ranked(all_synonyms, top_k)
        elif aggregation == 'all':
            return all_synonyms
        elif aggregation == 'union':
            return self._aggregate_union(all_synonyms, top_k)
        elif aggregation == 'intersection':
            return self._aggregate_intersection(all_synonyms, top_k)
        else:
            return all_synonyms
    
    @staticmethod
    def _aggregate_ranked(all_synonyms: Dict[str, List[str]], top_k: int) -> Dict[str, List[str]]:
        """
        Aggregate synonyms ranked by source priority.
        Removes duplicates, preserves order by source priority.
        """
        ranked_list = []
        seen = set()
        
        # Sort by source priority (WordNet first, then FastText, etc.)
        priority_order = {'wordnet': 1, 'fasttext': 2, 'sense2vec': 2, 'fass': 3}
        sorted_sources = sorted(all_synonyms.items(), 
                               key=lambda x: priority_order.get(x[0], 999))
        
        for source_name, syns in sorted_sources:
            for syn in syns:
                if syn.lower() not in seen:
                    ranked_list.append(syn)
                    seen.add(syn.lower())
                    if len(ranked_list) >= top_k:
                        break
            if len(ranked_list) >= top_k:
                break
        
        return {'ranked': ranked_list[:top_k], **all_synonyms}
    
    @staticmethod
    def _aggregate_union(all_synonyms: Dict[str, List[str]], top_k: int) -> Dict[str, List[str]]:
        """
        Union aggregation: combine all synonyms, remove duplicates.
        """
        union_set = set()
        for syns in all_synonyms.values():
            union_set.update(s.lower() for s in syns)
        
        union_list = list(union_set)
        random.shuffle(union_list)
        return {'union': union_list[:top_k], **all_synonyms}
    
    @staticmethod
    def _aggregate_intersection(all_synonyms: Dict[str, List[str]], top_k: int) -> Dict[str, List[str]]:
        """
        Intersection aggregation: return only synonyms common to all sources.
        """
        if not all_synonyms:
            return {'intersection': []}
        
        # Convert to lowercase sets for comparison
        synonym_sets = [set(s.lower() for s in syns) for syns in all_synonyms.values()]
        intersection = synonym_sets[0]
        for s_set in synonym_sets[1:]:
            intersection &= s_set
        
        return {'intersection': list(intersection)[:top_k], **all_synonyms}

