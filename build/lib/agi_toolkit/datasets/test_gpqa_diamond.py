# GPQA Diamond Dataset Testing Code
# High Difficulty - 198 questions (0.45k)
# Graduate-Level Google-Proof Q&A in biology, physics, and chemistry

"""
GPQA Diamond Dataset
- 研究生级别的高难度问题
- 涵盖生物、物理和化学
- 即使专家使用Google也难以回答
- 包含198个精心筛选的高质量问题
"""

from datasets import load_dataset
import random
from typing import Dict, List
import json

class GPQADiamondTester:
    def __init__(self):
        """初始化GPQA Diamond测试器"""
        self.dataset = None
        self.results = []
        
    def load_data(self, limit=None):
        """
        加载GPQA Diamond数据集
        
        Args:
            limit: 限制加载的样本数量（默认全部198题）
        """
        print("Loading GPQA Diamond dataset...")
        try:
            # 从Hugging Face加载GPQA数据集的Diamond子集
            ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond')
            
            self.dataset = []
            for item in ds['train']:  # GPQA Diamond在train split中
                self.dataset.append({
                    'question': item['Question'],
                    'choices': [
                        item['Correct Answer'],
                        item['Incorrect Answer 1'],
                        item['Incorrect Answer 2'],
                        item['Incorrect Answer 3']
                    ],
                    'correct_answer_index': 0,  # 正确答案总是第一个
                    'subject': item.get('Subdomain', 'unknown'),
                    'high_level_domain': item.get('High-level domain', 'unknown')
                })
            
            # 打乱选项顺序
            for item in self.dataset:
                combined = list(zip(item['choices'], [0, 1, 2, 3]))
                random.shuffle(combined)
                item['choices'], indices = zip(*combined)
                item['correct_answer_index'] = indices.index(0)
                item['correct_answer_letter'] = chr(65 + item['correct_answer_index'])  # A, B, C, D
            
            if limit:
                self.dataset = self.dataset[:limit]
            
            print(f"Loaded {len(self.dataset)} questions from GPQA Diamond")
            
        except Exception as e:
            print(f"Error loading GPQA Diamond: {e}")
    
    def test_single_question(self, question_data: Dict, model_answer: str) -> Dict:
        """
        测试单个问题
        
        Args:
            question_data: 问题数据字典
            model_answer: 模型的答案 (A/B/C/D或0/1/2/3)
            
        Returns:
            测试结果字典
        """
        # 转换答案格式
        if model_answer.upper() in ['A', 'B', 'C', 'D']:
            model_answer_index = ord(model_answer.upper()) - 65
        else:
            try:
                model_answer_index = int(model_answer)
            except:
                model_answer_index = -1
        
        correct_answer_index = question_data['correct_answer_index']
        is_correct = model_answer_index == correct_answer_index
        
        return {
            'question': question_data['question'],
            'choices': question_data['choices'],
            'correct_answer': question_data['correct_answer_letter'],
            'model_answer': model_answer,
            'is_correct': is_correct,
            'subject': question_data['subject'],
            'domain': question_data['high_level_domain']
        }
    
    def run_evaluation(self, model_fn, num_samples=None):
        """
        运行评估
        
        Args:
            model_fn: 模型函数，接受问题和选项列表，返回答案
            num_samples: 评估的样本数量（None表示全部）
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        if num_samples:
            samples = random.sample(self.dataset, min(num_samples, len(self.dataset)))
        else:
            samples = self.dataset
        
        correct_count = 0
        domain_stats = {}
        
        for idx, sample in enumerate(samples):
            # 调用模型
            model_answer = model_fn(sample['question'], sample['choices'])
            
            # 测试结果
            result = self.test_single_question(sample, model_answer)
            self.results.append(result)
            
            if result['is_correct']:
                correct_count += 1
            
            # 统计不同领域的表现
            domain = result['domain']
            if domain not in domain_stats:
                domain_stats[domain] = {'correct': 0, 'total': 0}
            domain_stats[domain]['total'] += 1
            if result['is_correct']:
                domain_stats[domain]['correct'] += 1
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(samples)} questions...")
        
        accuracy = correct_count / len(samples) * 100
        print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct_count}/{len(samples)})")
        print(f"Note: Random guessing baseline = 25.0%")
        print(f"Note: Human expert performance ≈ 69.7%")
        
        print("\nAccuracy by Domain:")
        for domain, stats in domain_stats.items():
            domain_acc = stats['correct'] / stats['total'] * 100
            print(f"  {domain}: {domain_acc:.2f}% ({stats['correct']}/{stats['total']})")
        
        return {
            'accuracy': accuracy,
            'correct': correct_count,
            'total': len(samples),
            'domain_stats': domain_stats,
            'results': self.results
        }
    
    def save_results(self, filename='gpqa_diamond_results.json'):
        """保存结果到JSON文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")


# 示例使用
if __name__ == "__main__":
    # 初始化测试器
    tester = GPQADiamondTester()
    
    # 加载数据（全部198题）
    tester.load_data()
    
    # 示例模型函数（随机选择）
    def dummy_model(question: str, choices: List[str]) -> str:
        """示例模型：随机选择答案"""
        return random.choice(['A', 'B', 'C', 'D'])
    
    # 运行评估
    print("\n" + "="*50)
    print("Running GPQA Diamond Evaluation")
    print("="*50)
    
    if tester.dataset:
        results = tester.run_evaluation(dummy_model, num_samples=50)
        tester.save_results()
        
        # 打印示例问题
        print("\n" + "="*50)
        print("Example Question:")
        print("="*50)
        example = tester.dataset[0]
        print(f"Domain: {example['high_level_domain']}")
        print(f"Subject: {example['subject']}")
        print(f"\nQuestion: {example['question'][:200]}...")  # 只显示前200字符
        print("\nChoices:")
        for i, choice in enumerate(example['choices']):
            print(f"  {chr(65+i)}. {choice[:100]}...")
        print(f"\nCorrect Answer: {example['correct_answer_letter']}")
    else:
        print("Dataset could not be loaded.")
