import numpy as np
from typing import Dict, List


class MMEReasoningEvaluator:

    def __init__(self, protocol="standard"):
        self.protocol = protocol
        self.thresholds = {"quick": 60, "standard": 60, "comprehensive": 60}

    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str],
        reasoning_types: List[str] = None,
    ) -> Dict:

        reasoning_types = reasoning_types or []

        results = {
            "inductive": self._evaluate_type(
                predictions, ground_truths, reasoning_types, "inductive"
            ),
            "deductive": self._evaluate_type(
                predictions, ground_truths, reasoning_types, "deductive"
            ),
            "abductive": self._evaluate_type(
                predictions, ground_truths, reasoning_types, "abductive"
            ),
        }

        scores = [r["accuracy"] for r in results.values() if r["total"] > 0]
        comprehensive_score = np.mean(scores) if scores else 0

        results["comprehensive_score"] = comprehensive_score
        results["passed"] = comprehensive_score >= self.thresholds[self.protocol]

        results["balance_analysis"] = self._analyze_balance(results)

        return results

    def _evaluate_type(self, predictions, ground_truths, reasoning_types, target_type):
        indices = [i for i, rt in enumerate(reasoning_types) if rt == target_type]

        if not indices:
            return {"accuracy": 0, "correct": 0, "total": 0}

        type_preds = [predictions[i] for i in indices]
        type_truth = [ground_truths[i] for i in indices]

        correct = sum(p == t for p, t in zip(type_preds, type_truth))
        total = len(indices)

        return {"accuracy": correct / total * 100, "correct": correct, "total": total}

    def _analyze_balance(self, results):
        accuracies = [
            results["inductive"]["accuracy"],
            results["deductive"]["accuracy"],
            results["abductive"]["accuracy"],
        ]

        max_acc = max(accuracies)
        min_acc = min(accuracies)
        imbalance = max_acc - min_acc

        return {
            "imbalance_score": imbalance,
            "strongest": (
                "inductive"
                if max_acc == accuracies[0]
                else "deductive" if max_acc == accuracies[1] else "abductive"
            ),
            "weakest": (
                "inductive"
                if min_acc == accuracies[0]
                else "deductive" if min_acc == accuracies[1] else "abductive"
            ),
            "balance_rating": (
                "BALANCED"
                if imbalance < 10
                else "PARTIALLY BALANCED" if imbalance < 20 else "UNBALANCED"
            ),
        }
