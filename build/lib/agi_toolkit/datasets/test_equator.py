# EQUATOR Dataset Testing Code
# High Difficulty - ~1k samples
# Deterministic framework for evaluating LLM reasoning with open-ended questions

"""
EQUATOR (Evaluation of Question Answering Thoroughness in Open-ended Reasoning)
- 确定性评分框架
- 开放式问题评估
- 结合向量数据库进行语义相似度匹配
- 注重事实准确性和推理能力
"""

import random
from typing import Dict, List, Optional
import json
import numpy as np

class EQUATORTester:
    def __init__(self):
        """初始化EQUATOR测试器"""
        self.dataset = None
        self.results = []
        
    def load_data(self, dataset_path: Optional[str] = None, limit=None):
        """
        加载EQUATOR评估数据
        
        Args:
            dataset_path: 数据集路径（如果有）
            limit: 限制加载的样本数量
            
        注意：EQUATOR是一个框架，需要配合具体的问答数据集使用
        """
        print("Loading EQUATOR evaluation dataset...")
        
        # EQUATOR通常使用现有的开放式QA数据集
        # 这里我们创建一个示例数据结构
        if dataset_path:
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    self.dataset = json.load(f)
            except Exception as e:
                print(f"Error loading dataset: {e}")
        else:
            # 示例数据结构
            print("Note: EQUATOR requires a custom dataset with open-ended questions")
            print("Creating example dataset structure...")
            self.dataset = self._create_example_dataset()
        
        if limit and self.dataset:
            self.dataset = self.dataset[:limit]
        
        if self.dataset:
            print(f"Loaded {len(self.dataset)} questions for EQUATOR evaluation")
        
    def _create_example_dataset(self) -> List[Dict]:
        """创建示例数据集结构"""
        return [
            {
                'question': 'What is the capital of France?',
                'reference_answer': 'Paris is the capital of France.',
                'key_facts': ['Paris', 'capital', 'France'],
                'reasoning_required': False
            },
            {
                'question': 'Explain the process of photosynthesis.',
                'reference_answer': 'Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar.',
                'key_facts': ['plants', 'sunlight', 'water', 'carbon dioxide', 'oxygen', 'sugar'],
                'reasoning_required': True
            },
            # 添加更多示例...
        ]
    
    def calculate_semantic_similarity(self, answer1: str, answer2: str) -> float:
        """
        计算两个答案之间的语义相似度
        
        Args:
            answer1: 第一个答案
            answer2: 第二个答案
            
        Returns:
            相似度分数 (0-1)
            
        注意：实际应用中应使用向量嵌入和余弦相似度
        这里使用简单的词汇重叠作为示例
        """
        # 简单的词汇重叠相似度（实际应使用嵌入模型）
        words1 = set(answer1.lower().split())
        words2 = set(answer2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def evaluate_factual_accuracy(self, model_answer: str, reference_answer: str, 
                                  key_facts: List[str]) -> Dict:
        """
        评估事实准确性
        
        Args:
            model_answer: 模型的答案
            reference_answer: 参考答案
            key_facts: 关键事实列表
            
        Returns:
            评估结果字典
        """
        # 检查关键事实是否包含在答案中
        model_answer_lower = model_answer.lower()
        facts_covered = sum(1 for fact in key_facts if fact.lower() in model_answer_lower)
        fact_coverage = facts_covered / len(key_facts) if key_facts else 0.0
        
        # 计算语义相似度
        similarity = self.calculate_semantic_similarity(model_answer, reference_answer)
        
        # EQUATOR评分：结合事实覆盖率和语义相似度
        equator_score = (fact_coverage * 0.5 + similarity * 0.5)
        
        return {
            'fact_coverage': fact_coverage,
            'semantic_similarity': similarity,
            'equator_score': equator_score,
            'facts_covered': facts_covered,
            'total_facts': len(key_facts)
        }
    
    def test_single_question(self, question_data: Dict, model_answer: str) -> Dict:
        """
        测试单个问题
        
        Args:
            question_data: 问题数据字典
            model_answer: 模型的答案
            
        Returns:
            测试结果字典
        """
        evaluation = self.evaluate_factual_accuracy(
            model_answer,
            question_data['reference_answer'],
            question_data['key_facts']
        )
        
        # 根据EQUATOR评分判断是否"正确"（阈值可调整）
        is_correct = evaluation['equator_score'] >= 0.7
        
        return {
            'question': question_data['question'],
            'reference_answer': question_data['reference_answer'],
            'model_answer': model_answer,
            'is_correct': is_correct,
            'reasoning_required': question_data['reasoning_required'],
            **evaluation
        }
    
    def run_evaluation(self, model_fn, num_samples=100):
        """
        运行EQUATOR评估
        
        Args:
            model_fn: 模型函数，接受问题，返回开放式答案
            num_samples: 评估的样本数量
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        samples = random.sample(self.dataset, min(num_samples, len(self.dataset)))
        
        total_equator_score = 0
        correct_count = 0
        reasoning_scores = {'reasoning': [], 'non_reasoning': []}
        
        for idx, sample in enumerate(samples):
            # 调用模型
            model_answer = model_fn(sample['question'])
            
            # 测试结果
            result = self.test_single_question(sample, model_answer)
            self.results.append(result)
            
            total_equator_score += result['equator_score']
            if result['is_correct']:
                correct_count += 1
            
            # 分类统计推理型和非推理型问题
            if sample['reasoning_required']:
                reasoning_scores['reasoning'].append(result['equator_score'])
            else:
                reasoning_scores['non_reasoning'].append(result['equator_score'])
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{num_samples} questions...")
        
        avg_equator_score = total_equator_score / len(samples) * 100
        accuracy = correct_count / len(samples) * 100
        
        print(f"\nOverall EQUATOR Score: {avg_equator_score:.2f}%")
        print(f"Binary Accuracy (≥70% threshold): {accuracy:.2f}% ({correct_count}/{len(samples)})")
        
        if reasoning_scores['reasoning']:
            avg_reasoning = np.mean(reasoning_scores['reasoning']) * 100
            print(f"Reasoning Questions Score: {avg_reasoning:.2f}%")
        
        if reasoning_scores['non_reasoning']:
            avg_non_reasoning = np.mean(reasoning_scores['non_reasoning']) * 100
            print(f"Non-Reasoning Questions Score: {avg_non_reasoning:.2f}%")
        
        return {
            'equator_score': avg_equator_score,
            'accuracy': accuracy,
            'correct': correct_count,
            'total': len(samples),
            'reasoning_scores': reasoning_scores,
            'results': self.results
        }
    
    def save_results(self, filename='equator_results.json'):
        """保存结果到JSON文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")


# 示例使用
if __name__ == "__main__":
    # 初始化测试器
    tester = EQUATORTester()
    
    # 加载数据（使用示例数据）
    tester.load_data()
    
    # 示例模型函数
    def dummy_model(question: str) -> str:
        """示例模型：返回简短答案"""
        if "capital" in question.lower():
            return "Paris"
        elif "photosynthesis" in question.lower():
            return "Plants convert light into energy using chlorophyll"
        return "This is a sample answer."
    
    # 运行评估
    print("\n" + "="*50)
    print("Running EQUATOR Evaluation")
    print("="*50)
    
    if tester.dataset:
        results = tester.run_evaluation(dummy_model, num_samples=len(tester.dataset))
        tester.save_results()
        
        # 打印示例问题
        print("\n" + "="*50)
        print("Example Question:")
        print("="*50)
        example = tester.dataset[0]
        print(f"Question: {example['question']}")
        print(f"Reference Answer: {example['reference_answer']}")
        print(f"Key Facts: {example['key_facts']}")
        print(f"Requires Reasoning: {example['reasoning_required']}")
    else:
        print("Dataset could not be loaded.")
