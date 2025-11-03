import numpy as np
from typing import Dict, List


class GPQADiamondEvaluator:
    """GPQA Diamond标准评估器"""

    def __init__(self, protocol="standard"):
        self.protocol = protocol

        self.nontriviality_threshold = 55

    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str],
        domains: List[str] = None,
    ) -> Dict:

        predictions = [str(p).upper() for p in predictions]
        ground_truths = [str(g).upper() for g in ground_truths]

        correct = sum(p == g for p, g in zip(predictions, ground_truths))
        total = len(predictions)
        accuracy = correct / total * 100

        results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "metric": "Accuracy (4-choice)",
            "is_nontrivial": accuracy > self.nontriviality_threshold,
            "nontriviality_threshold": self.nontriviality_threshold,
        }

        if domains:
            results["by_domain"] = self._evaluate_by_domain(
                predictions, ground_truths, domains
            )

        results["baselines"] = {
            "random": 25,
            "non_expert_with_google": 37,
            "phd_expert": 69.7,
        }
        results["exceeds_expert"] = accuracy > 69.7

        return results

    def _evaluate_by_domain(self, predictions, ground_truths, domains):

        domain_stats = {}
        domain_names = set(domains)

        for domain in domain_names:
            indices = [i for i, d in enumerate(domains) if d == domain]
            if not indices:
                continue

            domain_preds = [predictions[i] for i in indices]
            domain_truth = [ground_truths[i] for i in indices]

            correct = sum(p == t for p, t in zip(domain_preds, domain_truth))
            total = len(indices)

            domain_stats[domain] = {
                "accuracy": correct / total * 100,
                "correct": correct,
                "total": total,
            }

        return domain_stats
