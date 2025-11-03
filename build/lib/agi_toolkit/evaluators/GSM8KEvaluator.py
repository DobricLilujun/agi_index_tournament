import re
from typing import Dict, List, Union


class GSM8KEvaluator:

    def __init__(self, protocol="standard", tolerance=1e-5):
        self.protocol = protocol
        self.tolerance = tolerance
        self.thresholds = {"quick": 80, "standard": 80, "comprehensive": 80}

    def evaluate(self, predictions: List[str], ground_truths: List[str]) -> Dict:

        correct = 0
        detailed_results = []

        for pred, truth in zip(predictions, ground_truths):
            pred_num = self._extract_numeric(str(pred))
            truth_num = self._extract_numeric(str(truth))

            is_correct = False
            if pred_num is not None and truth_num is not None:
                is_correct = abs(pred_num - truth_num) < self.tolerance
            elif str(pred).strip() == str(truth).strip():
                is_correct = True

            if is_correct:
                correct += 1

            detailed_results.append(
                {"prediction": pred, "ground_truth": truth, "is_correct": is_correct}
            )

        accuracy = correct / len(predictions) * 100

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(predictions),
            "passed": accuracy >= self.thresholds[self.protocol],
            "metric": "Exact Match",
            "tolerance": self.tolerance,
            "detailed_results": detailed_results,
        }

    def _extract_numeric(self, text: str) -> Union[float, None]:

        text = str(text).replace(",", "")
        patterns = [
            r"####\s*([-+]?[0-9]*\.?[0-9]+)",
            r"(?:answer|answer is|the answer is)\s*:?\s*([-+]?[0-9]*\.?[0-9]+)",
            r"([-+]?[0-9]*\.?[0-9]+)\s*(?:dollars?|yuan)?$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        try:
            return float(re.sub(r"[^\d.-]", "", text))
        except:
            return None
