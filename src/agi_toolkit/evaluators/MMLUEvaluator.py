import numpy as np
from typing import Dict, List


class MMLUEvaluator:
    def __init__(self, protocol="standard"):
        self.protocol = protocol
        self.thresholds = {"quick": 70, "standard": 70, "comprehensive": 70}

    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str],
        subjects: List[str] = None,
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
            "passed": accuracy >= self.thresholds[self.protocol],
            "metric": "Accuracy",
        }

        if subjects:
            results["by_subject"] = self._evaluate_by_subject(
                predictions, ground_truths, subjects
            )

        return results

    def _evaluate_by_subject(self, predictions, ground_truths, subjects):

        subject_stats = {}
        for subject in set(subjects):
            indices = [i for i, s in enumerate(subjects) if s == subject]
            if not indices:
                continue

            subj_preds = [predictions[i] for i in indices]
            subj_truth = [ground_truths[i] for i in indices]

            correct = sum(p == t for p, t in zip(subj_preds, subj_truth))
            total = len(indices)

            subject_stats[subject] = {
                "accuracy": correct / total * 100,
                "correct": correct,
                "total": total,
            }

        return subject_stats
