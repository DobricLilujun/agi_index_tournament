# AGI_INDEX_TOURNAMENT

This project was developed to evaluate and showcase AGI index. Large language models, even multimodal large language models, and the reasoning models that emerged in 2025, have already come very close to and in some aspects even surpassed all current graduate students, whether in evaluation or problem-solving approaches. Sometimes they even have **"aha moments"** that produce thought processes beyond imagination. We believe that AGI is approaching very closely, but there are few benchmarks in the market that can systematically evaluate a model's AGI capabilities, because everyone has different definitions of AGI. Therefore, in this project, we define AGI as the comprehensive ability of a model or agent to draw inferences from one instance to another—that is, the model's ability to successfully solve and provide correct answers to complex problems it has never encountered before, which is what we call **"general."** To this end, we propose a tournament that systematically evaluates how far models are from achieving AGI across different capability focuses and difficulty levels.

<div align=center><img src="agi_tournament_logo.png" width="80%"></div>


## Methods


## Datasets

| Capability          | Difficulty Level  | Option1                                                      | Option2                                           | Option3                              |
| ------------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------- | ------------------------------------ |
| Reasoning           | Easy              | MMLU (Massive Multitask Language Understanding) (Math) (15k) | GSM8K (8.5k)                                      |                                      |
| Reasoning           | Medium difficulty | ThinkBench (2.9k)                                            | MME-Reasoning (1.1k)                              |                                      |
| Reasoning           | High difficulity  | GPQA Diamond (0.45k)                                         | EQUATOR (1k)                                      |                                      |
| Creativity          | Easy              | Alternative Uses Task (AUT) (0.1k)                           | Divergent Association Task (DAT) (0.2k)           | Creativity_eval (0.07k) (Deja Teste) |
| Creativity          | Medium difficulty | LiveIdeaBench (0.5k)                                         | Japanese Creativity Questions (JCQ) (0.3k)        |                                      |
| Creativity          | High difficulity  | DeepMath-Creative (0.18k)                                    | Creation-MMBench (0.7k)                           |                                      |
| Planning            | Easy              | Blocksworld (0.1k)                                           | OSU-NLP-Group/ TravelPlanner (40k)                |                                      |
| Planning            | Medium difficulty | FlowBench (0.4k)                                             | Plancraft (1k)                                    |                                      |
| Planning            | High difficulity  | Travel-Sim (0.3k)                                            | SafeAgentBench (0.75k)                            |                                      |
| Image Understanding | Easy              | VQA (Visual Question Answering) (250k)                       | COCO Captions (120k)                              |                                      |
| Image Understanding | Medium difficulty | MME-RealWorld (29k)                                          | MMIU (Multimodal Multi-image Understanding) (77K) |                                      |
| Image Understanding | High difficulity  | VHELM (Holistic Evaluation of Vision Language Models) (5K)   | Document Haystack (2K)                            |                                      |
| Decision Making     | Easy              | Basic two-person game problem                                | Basic Risk Preference Test                        |                                      |
| Decision Making     | Medium difficulty | Decision-Making Behavior Evaluation Framework (0.8K)         | GAMA-Bench (0.5K)                                 |                                      |
| Decision Making     | High difficulity  | INVESTORBENCH (2K)                                           | MarketSenseAI 2.0 (0.3K)                          |                                      |
| Agent               | Easy              | HumanEval (0.01K)                                            | Tool Calling                                      |                                      |
| Agent               | Medium difficulty | MultiAgentBench (1K)                                         | AgentBench (0.6K)                                 |                                      |
| Agent               | High difficulity  | WebArena (0.8K)                                              | Agent-SafetyBench (2k)                            |                                      |

---

## Changelog

### [2026-01-02] Multi-Source Dictionary Integration

#### Features Added

**Multi-Source Dictionary System**
- Integrated 4 synonym sources:
  - `WordNet` - High-quality academic synonym database (Priority 1)
  - `FastText` - Semantic similarity based on word vectors (Priority 2)
  - `Sense2Vec` - Multi-sense word embeddings (Priority 2)
  - `FASS` - Custom domain-specific synonym dictionary (Priority 3)

- 4 aggregation strategies:
  - `ranked` (default) - Priority-based ordering with deduplication
  - `union` - Maximum coverage
  - `intersection` - Common synonyms only
  - `all` - All sources information for debugging

**DictRewriter Enhancements**
- Multi-source initialization with `sources`, `source_weights`, `aggregation_method` parameters
- Runtime source switching via `set_sources()` method
- Dynamic FASS dictionary loading with `add_fass_dictionary()` method
- Per-call source override with `rewrite(..., sources=..., aggregation=...)`
- Multi-source synonym retrieval in WSD results

**New Methods**
- `get_active_sources()` - Returns currently active sources
- `get_available_sources()` - Lists all 4 available sources

---

### [2026-01-03] Back-Translation Rewriter System

#### Features Added

**BackTranslationRewriter**
- Dual-backend architecture supporting both NLLB and ChatGPT-based back-translation
- Line-by-line processing for NLLB backend to preserve document structure
- Full-text processing for ChatGPT backend for better semantic coherence
- Multi-pivot language support with configurable selection strategies



