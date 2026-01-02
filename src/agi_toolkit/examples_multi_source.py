"""
Quick Start Guide for Multi-Source DictRewriter

This module demonstrates how to use the enhanced DictRewriter with multiple synonym sources.
"""

from agi_toolkit.DictRewriter import DictRewriter

# ============================================================================
# Example 1: Basic Usage (Default - WordNet + FastText)
# ============================================================================
print("=" * 80)
print("Example 1: Basic Usage (Default Sources)")
print("=" * 80)

rewriter = DictRewriter()
text = "The beautiful garden has many colorful flowers."
rewritten, wsd_info = rewriter.rewrite(text, ratio=0.4)

print(f"Original:  {text}")
print(f"Rewritten: {rewritten}")
print(f"Active sources: {rewriter.get_active_sources()}\n")


# ============================================================================
# Example 2: WordNet Only (Fast)
# ============================================================================
print("=" * 80)
print("Example 2: WordNet Only (Fastest)")
print("=" * 80)

rewriter_fast = DictRewriter(sources=['wordnet'])
text = "This is a good and fast project."
rewritten_fast, _ = rewriter_fast.rewrite(text, ratio=0.5)

print(f"Original:  {text}")
print(f"Rewritten: {rewritten_fast}\n")


# ============================================================================
# Example 3: Multiple Sources with Aggregation
# ============================================================================
print("=" * 80)
print("Example 3: Multiple Sources (Better Coverage)")
print("=" * 80)

rewriter_multi = DictRewriter(
    sources=['wordnet', 'fass'],
    aggregation_method='ranked'
)
rewritten_multi, _ = rewriter_multi.rewrite(text, ratio=0.5)

print(f"Original:  {text}")
print(f"Rewritten: {rewritten_multi}")
print(f"Active sources: {rewriter_multi.get_active_sources()}")
print(f"Aggregation method: {rewriter_multi.aggregation_method}\n")


# ============================================================================
# Example 4: View Multi-Source Synonyms
# ============================================================================
print("=" * 80)
print("Example 4: Viewing Multi-Source Synonyms")
print("=" * 80)

test_sentence = "The project is good."
wsd_results = rewriter.wsd_sentence(test_sentence)

for item in wsd_results:
    if item['synonyms']:
        print(f"\nWord: '{item['word']}' (POS: {item['pos']})")
        print(f"  Ranked synonyms: {item['synonyms'][:5]}")
        if 'source_synonyms' in item:
            print(f"  Source breakdown:")
            for source_name, syns in item['source_synonyms'].items():
                if syns and source_name != 'ranked':
                    print(f"    - {source_name:10s}: {syns[:3]}")


# ============================================================================
# Example 5: Runtime Source Switching
# ============================================================================
print("\n" + "=" * 80)
print("Example 5: Runtime Source Switching")
print("=" * 80)

rewriter = DictRewriter(sources=['wordnet'])
print(f"Initial sources: {rewriter.get_active_sources()}")

# Change sources dynamically
rewriter.set_sources(['wordnet', 'fass'], aggregation_method='union')
print(f"Updated sources: {rewriter.get_active_sources()}")
print(f"Updated aggregation: {rewriter.aggregation_method}\n")


# ============================================================================
# Example 6: Per-Call Source Override
# ============================================================================
print("=" * 80)
print("Example 6: Per-Call Source Override")
print("=" * 80)

rewriter = DictRewriter(sources=['wordnet'])
text = "The beautiful garden has many colorful flowers."

# Rewrite with default sources
result1, _ = rewriter.rewrite(text, ratio=0.5)
print(f"Using WordNet: {result1}")

# Override just for this call - use only synonyms from additional sources
result2, _ = rewriter.rewrite(text, ratio=0.5, sources=['fass'])
print(f"Override with FASS: {result2}")

# Next call uses original sources again
result3, _ = rewriter.rewrite(text, ratio=0.5)
print(f"Back to WordNet: {result3}\n")


# ============================================================================
# Example 7: Custom FASS Dictionary
# ============================================================================
print("=" * 80)
print("Example 7: Using Custom FASS Dictionary")
print("=" * 80)

import json
import tempfile

# Create custom domain-specific synonyms
custom_fass = {
    "good": ["excellent", "outstanding", "fantastic"],
    "bad": ["poor", "terrible", "awful"],
    "fast": ["quick", "rapid", "swift"],
    "slow": ["sluggish", "gradual", "leisurely"]
}

# Save to temporary file
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(custom_fass, f)
    fass_path = f.name

# Load and use custom dictionary
rewriter = DictRewriter(sources=['wordnet', 'fass'])
rewriter.add_fass_dictionary(fass_path)

text = "This is a good and fast project."
rewritten, _ = rewriter.rewrite(text, ratio=1.0)

print(f"Original:  {text}")
print(f"Rewritten (with custom FASS): {rewritten}\n")


# ============================================================================
# Example 8: Aggregation Strategy Comparison
# ============================================================================
print("=" * 80)
print("Example 8: Aggregation Strategy Comparison")
print("=" * 80)

test_sentence = "This is good."
text_to_rewrite = "This is a good project."

strategies = ['ranked', 'union', 'intersection']

for strategy in strategies:
    rewriter = DictRewriter(
        sources=['wordnet', 'fass'],
        aggregation_method=strategy
    )
    
    wsd = rewriter.wsd_sentence(test_sentence)
    rewritten, _ = rewriter.rewrite(text_to_rewrite, ratio=0.5)
    
    print(f"\nStrategy: {strategy}")
    for item in wsd:
        if item['word'] == 'good':
            print(f"  Synonyms: {item['synonyms'][:5]}")
    print(f"  Rewritten: {rewritten}")


print("\n" + "=" * 80)
print("All examples completed!")
print("=" * 80)
