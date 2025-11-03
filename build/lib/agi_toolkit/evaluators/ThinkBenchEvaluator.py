import numpy as np
from typing import Dict, List


class ThinkBenchEvaluator:

    def __init__(self, protocol="standard"):
        self.protocol = protocol
        self.thresholds = {"quick": 65, "standard": 65, "comprehensive": 65}

    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str],
        reasoning_types: List[str] = None,
    ) -> Dict:

        correct = sum(p == g for p, g in zip(predictions, ground_truths))
        total = len(predictions)
        overall_accuracy = correct / total * 100

        results = {
            "overall_accuracy": overall_accuracy,
            "correct": correct,
            "total": total,
            "passed": overall_accuracy >= self.thresholds[self.protocol],
            "metric": "Accuracy",
        }

        if reasoning_types:
            results["by_reasoning_type"] = self._evaluate_by_type(
                predictions, ground_truths, reasoning_types
            )

            if len(set(reasoning_types)) > 1:
                results["robustness"] = self._calculate_robustness(
                    results["by_reasoning_type"]
                )

        return results

    def _evaluate_by_type(self, predictions, ground_truths, reasoning_types):

        type_stats = {}
        for rtype in set(reasoning_types):
            indices = [i for i, rt in enumerate(reasoning_types) if rt == rtype]
            if not indices:
                continue

            type_preds = [predictions[i] for i in indices]
            type_truth = [ground_truths[i] for i in indices]

            correct = sum(p == t for p, t in zip(type_preds, type_truth))
            total = len(indices)

            type_stats[rtype] = {
                "accuracy": correct / total * 100,
                "correct": correct,
                "total": total,
            }

        return type_stats

    def _calculate_robustness(self, type_stats):

        if len(type_stats) < 2:
            return None

        accuracies = [s["accuracy"] for s in type_stats.values()]
        robustness_score = (
            min(accuracies) / max(accuracies) if max(accuracies) > 0 else 0
        )

        return {
            "robustness_score": robustness_score,
            "performance_variation": max(accuracies) - min(accuracies),
        }
