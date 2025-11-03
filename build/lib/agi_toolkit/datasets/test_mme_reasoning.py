# MME-Reasoning Dataset Testing Code
# Medium Difficulty - 1.1k samples
# Comprehensive logical reasoning benchmark for MLLMs

"""
MME-Reasoning Dataset
- 全面的逻辑推理基准测试
- 包含归纳、演绎和溯因推理
- 1,188个精心策划的多模态问题
"""

from datasets import load_dataset
import random
from typing import Dict, List
import json

class MMEReasoningTester:
    def __init__(self):
        """初始化MME-Reasoning测试器"""
        self.dataset = None
        self.results = []
        
    def load_data(self, split='test', limit=None):
        """
        加载MME-Reasoning数据集
        
        Args:
            split: 数据集分割
            limit: 限制加载的样本数量
        """
        print("Loading MME-Reasoning dataset...")
        try:
            # 从Hugging Face加载数据集
            ds = load_dataset('U4R/MME-Reasoning', split=split)
            
            self.dataset = []
            for item in ds:
                self.dataset.append({
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'reasoning_type': item.get('reasoning_type', 'unknown'),  # inductive, deductive, abductive
                    'difficulty_level': item.get('difficulty', 'medium'),
                    'image': item.get('image', None)  # MME-Reasoning is multimodal
                })
            
            if limit:
                self.dataset = self.dataset[:limit]
            
            print(f"Loaded {len(self.dataset)} questions from MME-Reasoning")
            
        except Exception as e:
            print(f"Error loading MME-Reasoning: {e}")
            print("Note: This dataset may require special access or different loading method")
    
    def test_single_question(self, question_data: Dict, model_answer: str) -> Dict:
        """
        测试单个问题
        
        Args:
            question_data: 问题数据字典
            model_answer: 模型的答案
            
        Returns:
            测试结果字典
        """
        correct_answer = str(question_data['answer']).strip().lower()
        predicted_answer = str(model_answer).strip().lower()
        
        is_correct = correct_answer == predicted_answer
        
        return {
            'question': question_data['question'],
            'correct_answer': question_data['answer'],
            'model_answer': model_answer,
            'is_correct': is_correct,
            'reasoning_type': question_data['reasoning_type'],
            'difficulty': question_data['difficulty_level']
        }
    
    def run_evaluation(self, model_fn, num_samples=100):
        """
        运行评估
        
        Args:
            model_fn: 模型函数，接受问题和图像（如果有），返回答案
            num_samples: 评估的样本数量
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        samples = random.sample(self.dataset, min(num_samples, len(self.dataset)))
        
        correct_count = 0
        reasoning_type_stats = {}
        
        for idx, sample in enumerate(samples):
            # 调用模型（如果是多模态模型，传入图像）
            model_answer = model_fn(sample['question'], sample.get('image'))
            
            # 测试结果
            result = self.test_single_question(sample, model_answer)
            self.results.append(result)
            
            if result['is_correct']:
                correct_count += 1
            
            # 统计不同推理类型的表现
            reasoning_type = result['reasoning_type']
            if reasoning_type not in reasoning_type_stats:
                reasoning_type_stats[reasoning_type] = {'correct': 0, 'total': 0}
            reasoning_type_stats[reasoning_type]['total'] += 1
            if result['is_correct']:
                reasoning_type_stats[reasoning_type]['correct'] += 1
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{num_samples} questions...")
        
        accuracy = correct_count / len(samples) * 100
        print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct_count}/{len(samples)})")
        
        print("\nAccuracy by Reasoning Type:")
        for reasoning_type, stats in reasoning_type_stats.items():
            type_acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {reasoning_type}: {type_acc:.2f}% ({stats['correct']}/{stats['total']})")
        
        return {
            'accuracy': accuracy,
            'correct': correct_count,
            'total': len(samples),
            'reasoning_type_stats': reasoning_type_stats,
            'results': self.results
        }
    
    def save_results(self, filename='mme_reasoning_results.json'):
        """保存结果到JSON文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")


# 示例使用
if __name__ == "__main__":
    # 初始化测试器
    tester = MMEReasoningTester()
    
    # 加载数据
    tester.load_data(split='test', limit=100)
    
    # 示例模型函数
    def dummy_model(question: str, image=None) -> str:
        """
        示例模型：返回简单答案
        在实际应用中，多模态模型应处理图像输入
        """
        return "Answer"
    
    # 运行评估
    print("\n" + "="*50)
    print("Running MME-Reasoning Evaluation")
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
        print(f"Difficulty: {example['difficulty_level']}")
        print(f"Has Image: {example['image'] is not None}")
    else:
        print("Dataset could not be loaded. Please check dataset availability.")
