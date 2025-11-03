# MMLU (Math) Dataset Testing Code
# Easy Difficulty - 15k samples
# Multiple-choice questions across mathematical domains

"""
MMLU (Massive Multitask Language Understanding) Math Dataset
- 包含数学相关的多项选择题
- 涵盖从基础数学到高等数学的各个领域
- 标准的4选1格式
"""

from datasets import load_dataset
import random
from typing import Dict, List
import json

class MMLUMathTester:
    def __init__(self):
        """初始化MMLU Math测试器"""
        self.dataset = None
        self.results = []
        
    def load_data(self, split='test', limit=None):
        """
        加载MMLU数据集
        
        Args:
            split: 数据集分割 ('test', 'validation', 'dev')
            limit: 限制加载的样本数量
        """
        print("Loading MMLU dataset...")
        # MMLU has multiple math-related subjects
        math_subjects = [
            'abstract_algebra',
            'college_mathematics', 
            'elementary_mathematics',
            'high_school_mathematics',
            'high_school_statistics'
        ]
        
        self.dataset = []
        for subject in math_subjects:
            try:
                ds = load_dataset('cais/mmlu', subject, split=split)
                for item in ds:
                    self.dataset.append({
                        'subject': subject,
                        'question': item['question'],
                        'choices': item['choices'],
                        'answer': item['answer']
                    })
            except Exception as e:
                print(f"Error loading {subject}: {e}")
        
        if limit:
            self.dataset = self.dataset[:limit]
        
        print(f"Loaded {len(self.dataset)} questions from MMLU Math")
        
    def test_single_question(self, question_data: Dict, model_answer: str) -> Dict:
        """
        测试单个问题
        
        Args:
            question_data: 问题数据字典
            model_answer: 模型的答案 (A/B/C/D)
            
        Returns:
            测试结果字典
        """
        correct = model_answer.upper() == question_data['answer']
        
        return {
            'question': question_data['question'],
            'choices': question_data['choices'],
            'correct_answer': question_data['answer'],
            'model_answer': model_answer,
            'is_correct': correct,
            'subject': question_data['subject']
        }
    
    def run_evaluation(self, model_fn, num_samples=100):
        """
        运行评估
        
        Args:
            model_fn: 模型函数，接受问题和选项，返回答案
            num_samples: 评估的样本数量
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        samples = random.sample(self.dataset, min(num_samples, len(self.dataset)))
        
        correct_count = 0
        for idx, sample in enumerate(samples):
            # 调用模型
            model_answer = model_fn(sample['question'], sample['choices'])
            
            # 测试结果
            result = self.test_single_question(sample, model_answer)
            self.results.append(result)
            
            if result['is_correct']:
                correct_count += 1
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{num_samples} questions...")
        
        accuracy = correct_count / len(samples) * 100
        print(f"\nAccuracy: {accuracy:.2f}% ({correct_count}/{len(samples)})")
        
        return {
            'accuracy': accuracy,
            'correct': correct_count,
            'total': len(samples),
            'results': self.results
        }
    
    def save_results(self, filename='mmlu_math_results.json'):
        """保存结果到JSON文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")


# 示例使用
if __name__ == "__main__":
    # 初始化测试器
    tester = MMLUMathTester()
    
    # 加载数据（限制为100个样本用于演示）
    tester.load_data(split='test', limit=100)
    
    # 示例模型函数（随机选择）
    def dummy_model(question: str, choices: List[str]) -> str:
        """示例模型：随机选择答案"""
        return random.choice(['A', 'B', 'C', 'D'])
    
    # 运行评估
    print("\n" + "="*50)
    print("Running MMLU Math Evaluation")
    print("="*50)
    results = tester.run_evaluation(dummy_model, num_samples=20)
    
    # 保存结果
    tester.save_results()
    
    # 打印示例问题
    print("\n" + "="*50)
    print("Example Question:")
    print("="*50)
    if tester.dataset:
        example = tester.dataset[0]
        print(f"Subject: {example['subject']}")
        print(f"Question: {example['question']}")
        print("Choices:")
        for i, choice in enumerate(example['choices']):
            print(f"  {chr(65+i)}. {choice}")
        print(f"Correct Answer: {example['answer']}")
