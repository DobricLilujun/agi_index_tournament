from typing import Dict, List
from src.agi_toolkit.datasets.MMLUEvaluator import MMLUEvaluator
from src.agi_toolkit.datasets.GSM8KEvaluator import GSM8KEvaluator
from agi_toolkit.evaluators.ThinkBenchEvaluator import ThinkBenchEvaluator
from src.agi_toolkit.datasets.MMEReasoningEvaluator import MMEReasoningEvaluator
from src.agi_toolkit.datasets.GPQADiamondEvaluator import GPQADiamondEvaluator
from src.agi_toolkit.datasets.EQUATOREvaluator import EQUATORTester


class UnifiedEvaluator:

    EVALUATORS = {
        "mmlu": MMLUEvaluator,
        "gsm8k": GSM8KEvaluator,
        "thinkbench": ThinkBenchEvaluator,
        "mme": MMEReasoningEvaluator,
        "gpqa": GPQADiamondEvaluator,
        "equator": EQUATORTester,
    }

    def __init__(self, dataset: str, protocol: str = "standard"):
        if dataset not in self.EVALUATORS:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.dataset = dataset
        self.evaluator = self.EVALUATORS[dataset](protocol)

    def evaluate(self, **kwargs) -> Dict:

        return self.evaluator.evaluate(**kwargs)

    @staticmethod
    def compare_results(results_dict: Dict[str, Dict]) -> Dict:

        comparison = {}

        for dataset, results in results_dict.items():
            comparison[dataset] = {
                "accuracy": results.get("accuracy", results.get("mean_equator_score")),
                "status": "Passed" if results.get("passed") else "Failed",
            }

        return comparison


# evaluator = UnifiedEvaluator('mmlu', protocol='standard')
# mmlu_results = evaluator.evaluate(
#     predictions=['A', 'B', 'C', 'A'],
#     ground_truths=['A', 'B', 'D', 'A'],
#     subjects=['algebra', 'geometry', 'calculus', 'algebra']
# )
# print(f"MMLU Accuracy: {mmlu_results['accuracy']:.2f}%")

# # 示例2: GSM8K
# evaluator = UnifiedEvaluator('gsm8k')
# gsm8k_results = evaluator.evaluate(
#     predictions=['42', '100', '5'],
#     ground_truths=['42', '99', '5']
# )
# print(f"GSM8K Accuracy: {gsm8k_results['accuracy']:.2f}%")

# # 示例3: GPQA Diamond
# evaluator = UnifiedEvaluator('gpqa')
# gpqa_results = evaluator.evaluate(
#     predictions=['A', 'B', 'C'],
#     ground_truths=['A', 'B', 'C'],
#     domains=['Biology', 'Physics', 'Chemistry']
# )
# print(f"GPQA Accuracy: {gpqa_results['accuracy']:.2f}%")
# print(f"Non-trivial: {gpqa_results['is_nontrivial']}")

# # 示例4: 批量评估
# all_results = {
#     'mmlu': mmlu_results,
#     'gsm8k': gsm8k_results,
#     'gpqa': gpqa_results
# }

# comparison = UnifiedEvaluator.compare_results(all_results)
# print("\n比较结果:")
# for dataset, comparison_data in comparison.items():
#     print(f"  {dataset}: {comparison_data['accuracy']:.2f}% ({comparison_data['status']})")
