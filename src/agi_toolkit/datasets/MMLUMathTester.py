# MMLU (Math) Dataset Testing Code
# Easy Difficulty - 15k samples
# Multiple-choice questions across mathematical domains

from datasets import load_dataset
import random
from typing import Dict, List
import json


class MMLUMathTester:
    def __init__(self):
        self.dataset = None
        self.results = []

    def load_data(self, split="test", limit=None):
        print("Loading MMLU dataset...")
        # MMLU has multiple math-related subjects
        math_subjects = [
            "abstract_algebra",
            "college_mathematics",
            "elementary_mathematics",
            "high_school_mathematics",
            "high_school_statistics",
        ]

        self.dataset = []
        for subject in math_subjects:
            try:
                ds = load_dataset("cais/mmlu", subject, split=split)
                for item in ds:
                    self.dataset.append(
                        {
                            "subject": subject,
                            "question": item["question"],
                            "choices": item["choices"],
                            "answer": item["answer"],
                        }
                    )
            except Exception as e:
                print(f"Error loading {subject}: {e}")

        if limit:
            self.dataset = self.dataset[:limit]

        print(f"Loaded {len(self.dataset)} questions from MMLU Math")


    def test_single_question(self, question_data: Dict, model_answer: str) -> Dict:

        correct = model_answer.upper() == question_data["answer"]

        return {
            "question": question_data["question"],
            "choices": question_data["choices"],
            "correct_answer": question_data["answer"],
            "model_answer": model_answer,
            "is_correct": correct,
            "subject": question_data["subject"],
        }

    def run_evaluation(self, model_fn, num_samples=100):

        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        samples = random.sample(self.dataset, min(num_samples, len(self.dataset)))

        correct_count = 0
        for idx, sample in enumerate(samples):

            model_answer = model_fn(sample["question"], sample["choices"])

            result = self.test_single_question(sample, model_answer)
            self.results.append(result)

            if result["is_correct"]:
                correct_count += 1

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{num_samples} questions...")

        accuracy = correct_count / len(samples) * 100
        print(f"\nAccuracy: {accuracy:.2f}% ({correct_count}/{len(samples)})")

        return {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(samples),
            "results": self.results,
        }

    # Save results to a target files
    def save_results(self, filename="mmlu_math_results.json"):

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")


if __name__ == "__main__":

    tester = MMLUMathTester()

    tester.load_data(split="test", limit=100)

    def dummy_model(question: str, choices: List[str]) -> str:

        return random.choice(["A", "B", "C", "D"])

    print("\n" + "=" * 50)
    print("Running MMLU Math Evaluation")
    print("=" * 50)
    results = tester.run_evaluation(dummy_model, num_samples=20)

    tester.save_results()

    print("\n" + "=" * 50)
    print("Example Question:")
    print("=" * 50)
    if tester.dataset:
        example = tester.dataset[0]
        print(f"Subject: {example['subject']}")
        print(f"Question: {example['question']}")
        print("Choices:")
        for i, choice in enumerate(example["choices"]):
            print(f"  {chr(65+i)}. {choice}")
        print(f"Correct Answer: {example['answer']}")
