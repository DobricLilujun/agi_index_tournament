import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics.pairwise import cosine_similarity


class EQUATORTester:
    def __init__(self, protocol="standard", embedding_model=None):
        self.protocol = protocol
        self.embedding_model = embedding_model
        self.thresholds = {"quick": 70, "standard": 70, "comprehensive": 70}

    def evaluate(
        self,
        model_answers: List[str],
        reference_answers: List[str],
        key_facts_list: List[List[str]],
    ) -> Dict:
        scores = []
        detailed_results = []

        for model_ans, ref_ans, key_facts in zip(
            model_answers, reference_answers, key_facts_list
        ):

            fact_coverage = self._calculate_fact_coverage(model_ans, key_facts)

            semantic_sim = self._calculate_semantic_similarity(model_ans, ref_ans)

            equator_score = 0.5 * fact_coverage + 0.5 * semantic_sim
            scores.append(equator_score)

            detailed_results.append(
                {
                    "fact_coverage": fact_coverage,
                    "semantic_similarity": semantic_sim,
                    "equator_score": equator_score,
                    "passed": equator_score >= 0.7,
                }
            )

        mean_score = np.mean(scores) * 100
        pass_rate = sum(1 for s in scores if s >= 0.7) / len(scores) * 100

        return {
            "mean_equator_score": mean_score,
            "std_equator_score": np.std(scores) * 100,
            "pass_rate": pass_rate,
            "passed": mean_score >= self.thresholds[self.protocol],
            "metric": "EQUATOR Score",
            "detailed_results": detailed_results,
        }

    def _calculate_fact_coverage(self, answer: str, key_facts: List[str]) -> float:

        if not key_facts:
            return 1.0

        answer_lower = answer.lower()
        facts_covered = sum(1 for fact in key_facts if fact.lower() in answer_lower)

        return facts_covered / len(key_facts)

    def _calculate_semantic_similarity(self, answer1: str, answer2: str) -> float:

        if self.embedding_model is None:

            words1 = set(answer1.lower().split())
            words2 = set(answer2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0

        try:
            emb1 = self.embedding_model.encode(answer1)
            emb2 = self.embedding_model.encode(answer2)
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            return (similarity + 1) / 2
        except:
            return 0.5
