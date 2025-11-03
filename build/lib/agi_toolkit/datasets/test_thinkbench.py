# ThinkBench Dataset Testing Code
# Medium Difficulty - 2.9k samples
# Dynamic Out-of-Distribution reasoning evaluation

"""
ThinkBench Dataset
- 动态OOD（Out-of-Distribution）数据生成
- 测试模型的鲁棒性和泛化能力
- 包含多种推理任务
"""

from datasets import load_dataset
import random
from typing import Dict, List
import json

class ThinkBenchTester:
    def __init__(self):
        """初始化ThinkBench测试器"""
        self.dataset = None
        self.results = []
        
    def load_data(self, split='test', limit=None):
        """
        加载ThinkBench数据集
        
        Args:
            split: 数据集分割
            limit: 限制加载的样本数量
        """
        print("Loading ThinkBench dataset...")
        try:
            # 从Hugging Face加载数据集
            ds = load_dataset('jiuyinjiu/ThinkBench', split=split)
            
            self.dataset = []
            for item in ds:
                self.dataset.append({
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'reasoning_type': item.get('type', 'unknown'),
                    'difficulty': item.get('difficulty', 'medium')
                })
            
            if limit:
                self.dataset = self.dataset[:limit]
            
            print(f"Loaded {len(self.dataset)} questions from ThinkBench")
            
        except Exception as e:
            print(f"Error loading ThinkBench: {e}")
            print("Note: You may need to request access to this dataset")
    
    def test_single_question(self, question_data: Dict, model_answer: str) -> Dict:
        """
        测试单个问题
        
        Args:
            question_data: 问题数据字典
            model_answer: 模型的答案
            
        Returns:
            测试结果字典
        """
        # 简单的字符串匹配（实际应用中可能需要更复杂的比较）
        correct_answer = str(question_data['answer']).strip().lower()
        predicted_answer = str(model_answer).strip().lower()
        
        is_correct = correct_answer == predicted_answer
        
        return {
            'question': question_data['question'],
            'correct_answer': question_data['answer'],
            'model_answer': model_answer,
            'is_correct': is_correct,
            'reasoning_type': question_data['reasoning_type'],
            'difficulty': question_data['difficulty']
        }
    
    def run_evaluation(self, model_fn, num_samples=100):
        """
        运行评估
        
        Args:
            model_fn: 模型函数，接受问题，返回答案
            num_samples: 评估的样本数量
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        samples = random.sample(self.dataset, min(num_samples, len(self.dataset)))
        
        correct_count = 0
        reasoning_stats = {}
        
        for idx, sample in enumerate(samples):
            # 调用模型
            model_answer = model_fn(sample['question'])
            
            # 测试结果
            result = self.test_single_question(sample, model_answer)
            self.results.append(result)
            
            if result['is_correct']:
                correct_count += 1
            
            # 统计不同推理类型的表现
            reasoning_type = result['reasoning_type']
            if reasoning_type not in reasoning_stats:
                reasoning_stats[reasoning_type] = {'correct': 0, 'total': 0}
            reasoning_stats[reasoning_type]['total'] += 1
            if result['is_correct']:
                reasoning_stats[reasoning_type]['correct'] += 1
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{num_samples} questions...")
        
        accuracy = correct_count / len(samples) * 100
        print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct_count}/{len(samples)})")
        
        print("\nAccuracy by Reasoning Type:")
        for reasoning_type, stats in reasoning_stats.items():
            type_acc = stats['correct'] / stats['total'] * 100
            print(f"  {reasoning_type}: {type_acc:.2f}% ({stats['correct']}/{stats['total']})")
        
        return {
            'accuracy': accuracy,
            'correct': correct_count,
            'total': len(samples),
            'reasoning_stats': reasoning_stats,
            'results': self.results
        }
    
    def save_results(self, filename='thinkbench_results.json'):
        """保存结果到JSON文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")


# 示例使用
if __name__ == "__main__":
    # 初始化测试器
    tester = ThinkBenchTester()
    
    # 加载数据
    tester.load_data(split='test', limit=100)
    
    # 示例模型函数
    def dummy_model(question: str) -> str:
        """示例模型：返回简单答案"""
        return "Answer"
    
    # 运行评估
    print("\n" + "="*50)
    print("Running ThinkBench Evaluation")
    print("="*50)
    
    if tester.dataset:
        results = tester.run_evaluation(dummy_model, num_samples=20)
        tester.save_results()
        
        # 打印示例问题
        print("\n" + "="*50)
        print("Example Question:")
        print("="*50)
        example = tester.dataset[0]
        print(f"Question: {example['question']}")
        print(f"Answer: {example['answer']}")
        print(f"Reasoning Type: {example['reasoning_type']}")
        print(f"Difficulty: {example['difficulty']}")
    else:
        print("Dataset could not be loaded. Please check access permissions.")
