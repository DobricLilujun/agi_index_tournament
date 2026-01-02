# Multi-Source Dictionary Integration Guide

## Overview

DictRewriter 现在支持多个同义词源的集成和聚合，包括：

- **WordNet** - 经过验证的高质量同义词库（默认）
- **FastText** - 基于词向量的语义相似度
- **Sense2Vec** - 多义词感知的词向量
- **FASS** - Fast-Sense Dictionary（自定义领域词典）

## Installation

### 基础依赖
```bash
pip install nltk spacy gensim
python -m spacy download en_core_web_md
```

### 可选依赖
```bash
# FastText support
pip install fasttext

# Sense2Vec support
pip install sense2vec
python -m sense2vec download s2v_reddit_2015_md
```

## Quick Start

### 1. 默认配置（推荐）
```python
from agi_toolkit.DictRewriter import DictRewriter

# 使用 WordNet + FastText（默认）
rewriter = DictRewriter()
# Active sources: ['wordnet', 'fasttext']

text = "The project develops powerful algorithms."
rewritten, wsd_info = rewriter.rewrite(text, ratio=0.3)
print(rewritten)
```

### 2. 仅使用 WordNet
```python
rewriter = DictRewriter(sources=['wordnet'])
rewritten, _ = rewriter.rewrite(text, ratio=0.3)
```

### 3. 多源聚合
```python
rewriter = DictRewriter(
    sources=['wordnet', 'fasttext', 'fass'],
    aggregation_method='ranked'  # Options: 'ranked', 'union', 'intersection', 'all'
)
rewritten, _ = rewriter.rewrite(text, ratio=0.3)
```

## 聚合策略（Aggregation Methods）

### ranked (默认)
按优先级排序，移除重复
- WordNet (优先级 1) → FastText (优先级 2) → Sense2Vec (优先级 2) → FASS (优先级 3)
- 返回质量最高的同义词

```python
rewriter = DictRewriter(
    sources=['wordnet', 'fasttext', 'fass'],
    aggregation_method='ranked'
)
```

### union
合并所有源的同义词，移除重复
```python
rewriter = DictRewriter(
    sources=['wordnet', 'fasttext'],
    aggregation_method='union'
)
```

### intersection
仅返回所有源都有的同义词（最严格）
```python
rewriter = DictRewriter(
    sources=['wordnet', 'fasttext', 'sense2vec'],
    aggregation_method='intersection'
)
```

### all
保留所有源的同义词（用于调试和分析）
```python
rewriter = DictRewriter(
    sources=['wordnet', 'fasttext'],
    aggregation_method='all'
)
```

## API Reference

### 初始化

```python
DictRewriter(
    sources: List[str] = None,           # ['wordnet', 'fasttext']
    source_weights: Dict[str, float] = None,  # 自定义权重
    aggregation_method: str = 'ranked'   # 聚合策略
)
```

### 主要方法

#### 1. rewrite()
重写文本，支持多源选择
```python
rewriter.rewrite(
    text: str,                      # 输入文本
    ratio: float = 0.3,            # 替换比例 (0-1)
    is_replace_by_description: bool = False,  # 是否用定义替换名词
    exclude_word_class: list = None,  # 排除的词性 ['NN', 'NNP']
    sources: List[str] = None,     # 覆盖当前源（仅此次调用）
    aggregation: str = None        # 覆盖聚合方法（仅此次调用）
)
```

**返回:** (rewritten_text, wsd_info_list)

#### 2. wsd_sentence()
对句子进行词义消歧，返回多源同义词
```python
wsd_info = rewriter.wsd_sentence("The project is good.")
# 返回: [
#   {
#       "word": "The",
#       "pos": "DT",
#       "synonyms": [...],
#       "source_synonyms": {
#           "wordnet": [...],
#           "fasttext": [...],
#           "ranked": [...]  # 聚合后的同义词
#       },
#       ...
#   },
#   ...
# ]
```

#### 3. set_sources()
运行时切换源
```python
rewriter.set_sources(
    sources=['wordnet', 'fasttext'],
    aggregation_method='union'
)
```

#### 4. get_active_sources()
获取当前活跃的源
```python
sources = rewriter.get_active_sources()
# ['wordnet', 'fasttext']
```

#### 5. get_available_sources()
获取所有可用的源
```python
available = rewriter.get_available_sources()
# ['wordnet', 'fasttext', 'sense2vec', 'fass']
```

#### 6. add_fass_dictionary()
加载自定义 FASS 字典
```python
rewriter.add_fass_dictionary('path/to/custom_fass.json')
```

## 使用示例

### 示例 1: 基础重写
```python
from agi_toolkit.DictRewriter import DictRewriter

rewriter = DictRewriter()
text = "The beautiful garden has many colorful flowers."
rewritten, _ = rewriter.rewrite(text, ratio=0.5)
print(rewritten)
```

### 示例 2: 对比不同源
```python
text = "This is a good and fast project."

# WordNet only
r1 = DictRewriter(sources=['wordnet'])
result1, _ = r1.rewrite(text, ratio=1.0)
print("WordNet:", result1)

# WordNet + FastText
r2 = DictRewriter(sources=['wordnet', 'fasttext'])
result2, _ = r2.rewrite(text, ratio=1.0)
print("Multi-source:", result2)
```

### 示例 3: 自定义 FASS 字典
```python
import json

# 创建自定义词典
custom_dict = {
    "good": ["excellent", "wonderful", "outstanding"],
    "bad": ["poor", "terrible", "awful"],
    "fast": ["quick", "rapid", "swift"]
}

# 保存为 JSON
with open('custom_fass.json', 'w') as f:
    json.dump(custom_dict, f)

# 使用自定义字典
rewriter = DictRewriter(sources=['wordnet', 'fass'])
rewriter.add_fass_dictionary('custom_fass.json')

text = "This is a good and fast project."
rewritten, _ = rewriter.rewrite(text, ratio=1.0)
print(rewritten)  # 使用自定义同义词
```

### 示例 4: 单次调用覆盖
```python
# 创建基础重写器
rewriter = DictRewriter(sources=['wordnet'])

# 大部分调用使用 WordNet
result1, _ = rewriter.rewrite("Good text.", ratio=0.5)

# 这次调用临时使用 FastText
result2, _ = rewriter.rewrite("Good text.", ratio=0.5, sources=['fasttext'])

# 后续调用恢复到 WordNet
result3, _ = rewriter.rewrite("Good text.", ratio=0.5)
```

### 示例 5: 查看多源同义词
```python
rewriter = DictRewriter(sources=['wordnet', 'fasttext', 'fass'])

sentence = "The project is good."
wsd_results = rewriter.wsd_sentence(sentence)

for item in wsd_results:
    if item['word'].lower() == 'good':
        print(f"Word: {item['word']}")
        print(f"Ranked synonyms: {item['synonyms'][:5]}")
        
        # 查看各源的同义词
        for source_name, syns in item['source_synonyms'].items():
            if syns:
                print(f"  {source_name}: {syns[:3]}")
```

## 性能考虑

### 源加载时间
```
WordNet:  快速（本地）
FastText: 中速（首次下载较慢）
Sense2Vec: 中速（需要下载模型）
FASS:     快速（内存中字典）
```

### 聚合策略性能
```
ranked > all > union > intersection
```

### 优化建议
1. **开发阶段**: 使用 `sources=['wordnet']`（快速反馈）
2. **生产环境**: 使用 `sources=['wordnet', 'fasttext']` + `aggregation='ranked'`
3. **特定领域**: 添加自定义 FASS 字典

## FAQ

### Q: FastText 和 Sense2Vec 哪个更好？
**A**: 
- FastText: 更广泛的词汇覆盖，更快
- Sense2Vec: 更好的多义词处理，更准确
- 推荐: 组合使用 (WordNet + FastText)

### Q: 如何处理不在字典中的词？
**A**: 系统会自动跳过无法找到同义词的词，保留原文。

### Q: 可以混合多个自定义字典吗？
**A**: 可以。加载多个 FASS JSON 文件：
```python
rewriter.add_fass_dictionary('dict1.json')
rewriter.add_fass_dictionary('dict2.json')  # 会合并
```

### Q: 如何评估重写质量？
**A**: 查看 `test_multi_source_dict.ipynb` 中的对比测试

## 故障排除

### FastText 加载失败
```python
# 解决方案 1: 安装依赖
pip install fasttext gensim

# 解决方案 2: 移除 FastText 源
rewriter = DictRewriter(sources=['wordnet'])
```

### Sense2Vec 模型未找到
```bash
python -m sense2vec download s2v_reddit_2015_md
```

### NLTK 数据缺失
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

## 参考文献

- **WordNet**: https://wordnet.princeton.edu/
- **FastText**: https://fasttext.cc/
- **Sense2Vec**: https://github.com/explosion/sense2vec
- **FASS**: Custom domain-specific dictionaries

