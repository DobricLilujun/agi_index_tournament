# GSM8K Dataset Testing Code
# Easy Difficulty - 8.5k samples
# Grade school math word problems with step-by-step solutions

"""
GSM8K (Grade School Math 8K) Dataset
- 小学数学应用题数据集
- 包含自然语言形式的解题步骤
- 答案为数值形式
"""

from datasets import load_dataset
import random
import re
from typing import Dict, Optional
import json

class GSM8KTester:
    def __init__(self):
        """初始化GSM8K测试器"""
        self.dataset = None
        self.results = []
        
    def load_data(self, split='test', limit=None):
        """
        加载GSM8K数据集
        
        Args:
            split: 数据集分割 ('train', 'test')
            limit: 限制加载的样本数量
        """
        print("Loading GSM8K dataset...")
        try:
            ds = load_dataset('openai/gsm8k', 'main', split=split)
            
            self.dataset = []
            for item in ds:
                # 解析答案（格式: "#### 答案"）
                answer_text = item['answer']
                # 提取最终答案
                final_answer = self._extract_answer(answer_text)
                
                self.dataset.append({
                    'question': item['question'],
                    'full_answer': answer_text,
                    'final_answer': final_answer
                })
            
            if limit:
                self.dataset = self.dataset[:limit]
            
            print(f"Loaded {len(self.dataset)} questions from GSM8K")
            
        except Exception as e:
            print(f"Error loading GSM8K: {e}")
    
    def _extract_answer(self, answer_text: str) -> str:
        """
        从完整答案中提取最终数值答案
        
        Args:
            answer_text: 完整答案文本
            
        Returns:
            最终数值答案
        """
        # GSM8K答案格式: "step1\nstep2\n#### final_answer"
        if '####' in answer_text:
            parts = answer_text.split('####')
            final_answer = parts[-1].strip()
            # 移除逗号（例如 1,000 -> 1000）
            final_answer = final_answer.replace(',', '')
            return final_answer
        return answer_text.strip()
    
    def _normalize_answer(self, answer: str) -> float:
        """
        标准化答案为数值
        
        Args:
            answer: 答案字符串
            
        Returns:
            标准化后的数值
        """
        try:
            # 移除所有非数字字符（除了小数点和负号）
            answer = re.sub(r'[^\d.-]', '', str(answer))
            return float(answer)
        except:
            return None
    
    def test_single_question(self, question_data: Dict, model_answer: str) -> Dict:
        """
        测试单个问题
        
        Args:
            question_data: 问题数据字典
            model_answer: 模型的答案
            
        Returns:
            测试结果字典
        """
        correct_answer = self._normalize_answer(question_data['final_answer'])
        predicted_answer = self._normalize_answer(model_answer)
        
        # 检查是否正确（允许小的浮点误差）
        is_correct = False
        if correct_answer is not None and predicted_answer is not None:
            is_correct = abs(correct_answer - predicted_answer) < 1e-5
        
        return {
            'question': question_data['question'],
            'correct_answer': question_data['final_answer'],
            'model_answer': model_answer,
            'is_correct': is_correct,
            'full_solution': question_data['full_answer']
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
        for idx, sample in enumerate(samples):
            # 调用模型
            model_answer = model_fn(sample['question'])
            
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
    
    def save_results(self, filename='gsm8k_results.json'):
        """保存结果到JSON文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")


# 示例使用
if __name__ == "__main__":
    # 初始化测试器
    tester = GSM8KTester()
    
    # 加载数据
    tester.load_data(split='test', limit=100)
    
    # 示例模型函数（返回随机答案）
    def dummy_model(question: str) -> str:
        """示例模型：返回随机数字"""
        return str(random.randint(1, 1000))
    
    # 运行评估
    print("\n" + "="*50)
    print("Running GSM8K Evaluation")
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
        print(f"Question: {example['question']}")
        print(f"\nFull Solution:")
        print(example['full_answer'])
        print(f"\nFinal Answer: {example['final_answer']}")
