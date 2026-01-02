# AGI Index Tournament - AI Coding Guidelines

## Project Overview
**Purpose**: Evaluate AGI capabilities across 6 reasoning benchmarks at three difficulty levels (Easy/Medium/High) by measuring model generalization on unseen complex problems.

**Core Definition of AGI in This Project**: A model's ability to draw inferences across instances—successfully solving complex problems never encountered before. This is operationalized as **generalization capability across benchmark diversity**.

## Architecture: Three-Layer Framework

### 1. **Benchmark Layer** (`src/agi_toolkit/datasets/`)
- Each dataset is loaded via HuggingFace `datasets` library
- Currently implements: MMLU (Math), GSM8K, ThinkBench, MME-Reasoning, GPQA Diamond, EQUATOR
- Pattern: Dataset-specific loader classes (e.g., `MMLUMathTester`, `test_gpqa_diamond.py`)
- Key principle: Load raw data, apply dataset-specific preprocessing, return standardized format

### 2. **Evaluator Layer** (`src/agi_toolkit/evaluators/`)
- **Critical pattern**: Each dataset has a parallel evaluator (e.g., `MMLUEvaluator`, `GSM8KEvaluator`)
- **Unified interface** via `UnifiedEvaluator`: Maps dataset name → evaluator class
- Evaluators compute metrics (accuracy, domain-specific scores) with protocol-based thresholds ("quick", "standard", "comprehensive")
- Example: `MMLUEvaluator.evaluate()` normalizes to uppercase, computes accuracy, optionally provides per-subject breakdown

### 3. **Data Transformation Layer** (`src/agi_toolkit/`)
- **Hybrid Rewrite Pipeline** (`hybritePipe.py`): Multi-module data augmentation system
  - Modules: dictionary-guided rewriting (via spaCy + WordNet), back-translation (MarianMT), masked LM token substitution (BERT), stylistic paraphrasing (T5)
  - Preserves linguistic structure using dependency/constituency parsing
- **Dictionary Rewriter** (`DictRewriter.py`): Word-sense disambiguation via NLTK, synonym substitution with POS filtering
- **Translator** (`translator.py`): Backend abstraction (MarianMT, NLLB, LLM) for multi-language support

## Critical Patterns & Conventions

### Import Path Inconsistency (Known Issue)
- Mixed absolute/relative imports exist in codebase (e.g., `src.agi_toolkit.datasets.*` vs `agi_toolkit.evaluators.*`)
- When adding new evaluators: Use `from agi_toolkit.evaluators.X import Y` pattern to match existing structure
- Standardization effort encouraged but not yet enforced

### Evaluator Interface Contract
```python
class *Evaluator:
    def __init__(self, protocol: str = "standard"):
        self.protocol = protocol
        self.thresholds = {"quick": X, "standard": Y, "comprehensive": Z}
    
    def evaluate(self, predictions: List[str], ground_truths: List[str], **kwargs) -> Dict:
        # Returns: {"accuracy": float, "correct": int, "total": int, "passed": bool, "metric": str}
```
- All evaluators must support protocol parameter for flexible evaluation rigor
- Additional kwargs (e.g., `subjects`, `domains`) enable per-category analysis

### Dataset Loader Pattern
- Load via HuggingFace: `from datasets import load_dataset`
- Random sampling for testing (e.g., `random.sample()` for subset selection)
- Return standardized dicts with "query"/"question", "answer"/"ground_truth" keys
- Docstrings should document size, sample questions, and expected answer format

## External Dependencies & Critical Setup
- **NLP Core**: NLTK (must download punkt, wordnet, omw-1.4), spaCy (must load `en_core_web_sm`)
- **ML Models**: transformers (HuggingFace), torch, numpy
- **Translation**: MarianMT models lazy-loaded via `transformers` (Helsinki-NLP/opus-mt-XX-YY series)
- **Data**: datasets library (HuggingFace Hub access required)

**Setup steps not in README**: 
1. `python -m spacy download en_core_web_sm`
2. NLTK downloads occur on first import in `DictRewriter.py`

## Workflow Commands
- **No standard build/test setup yet**: Project is in development phase
- **Manual testing**: Import evaluators directly, call `.evaluate()` with sample data
- **Development iteration**: Modify datasets in `src/agi_toolkit/datasets/`, evaluators auto-picked via `UnifiedEvaluator.EVALUATORS` dict

## Adding New Benchmarks
1. Create `test_yourname.py` in `datasets/` folder (follow MMLUMathTester pattern)
2. Create `YournameEvaluator.py` in `evaluators/` folder (follow MMLUEvaluator contract)
3. Register in `UnifiedEvaluator.EVALUATORS` dict
4. Update README dataset table and `unified_test.py` DATASETS catalog

## Project Goals & Non-Negotiables
- Maintain **benchmark diversity** (reasoning, creativity, planning, image understanding, decision-making, agent)
- Ensure **difficulty progression** (Easy → Medium → High tests generalization depth)
- Support **reproducibility**: Standardize random seeds, document preprocessing steps
- **No cherry-picking datasets**: Evaluate across full spectrum—partial evaluation defeats AGI measurement goal
