# DictRewriter Multi-Source - Quick Reference Card

## 一行启动

```python
from agi_toolkit.DictRewriter import DictRewriter
rewriter = DictRewriter()  # 默认: WordNet + FastText
rewritten, _ = rewriter.rewrite("Text here", ratio=0.3)
```

## 源选择

```python
# 仅 WordNet（快）
DictRewriter(sources=['wordnet'])

# WordNet + FastText（平衡）- 默认
DictRewriter(sources=['wordnet', 'fasttext'])

# 全部源（最全面）
DictRewriter(sources=['wordnet', 'fasttext', 'sense2vec', 'fass'])

# 自定义权重
DictRewriter(sources=['wordnet', 'fasttext'], 
             source_weights={'wordnet': 1.5, 'fasttext': 1.0})
```

## 聚合策略

```python
# Ranked (推荐) - 按优先级排序，去重
DictRewriter(aggregation_method='ranked')

# Union - 最大覆盖
DictRewriter(aggregation_method='union')

# Intersection - 最严格
DictRewriter(aggregation_method='intersection')

# All - 保留所有信息（调试用）
DictRewriter(aggregation_method='all')
```

## 主要方法

| 方法 | 功能 |
|------|------|
| `rewrite(text, ratio, sources=None)` | 重写文本（可单次覆盖源） |
| `wsd_sentence(sent)` | 词义消歧 + 多源同义词 |
| `set_sources(sources, agg)` | 运行时切换源 |
| `get_active_sources()` | 获取活跃源列表 |
| `get_available_sources()` | 获取所有可用源 |
| `add_fass_dictionary(path)` | 加载自定义词典 |

## 常见场景

### 场景 1: 快速原型
```python
r = DictRewriter(sources=['wordnet'])
rewritten, _ = r.rewrite(text, 0.5)
```

### 场景 2: 生产环境（推荐）
```python
r = DictRewriter()  # 默认配置
rewritten, _ = r.rewrite(text, 0.5)
```

### 场景 3: 高质量要求
```python
r = DictRewriter(sources=['wordnet', 'fasttext'], 
                 aggregation_method='ranked')
rewritten, _ = r.rewrite(text, 0.5)
```

### 场景 4: 创意写作
```python
r = DictRewriter(sources=['wordnet', 'fasttext'], 
                 aggregation_method='union')
rewritten, _ = r.rewrite(text, 0.5)
```

### 场景 5: 领域特定
```python
r = DictRewriter(sources=['wordnet', 'fass'])
r.add_fass_dictionary('my_domain.json')
rewritten, _ = r.rewrite(text, 0.5)
```

### 场景 6: 单次覆盖
```python
r = DictRewriter(sources=['wordnet'])
# 这次用 FastText
rewritten, _ = r.rewrite(text, 0.5, sources=['fasttext'])
# 下次回到 WordNet
rewritten, _ = r.rewrite(text, 0.5)
```

## WSD 多源结果

```python
wsd_results = rewriter.wsd_sentence("Good text")

for item in wsd_results:
    if item['synonyms']:
        print(f"Word: {item['word']}")
        print(f"  Ranked: {item['synonyms'][:5]}")
        print(f"  All sources: {item['source_synonyms']}")
```

## 自定义 FASS 词典

```python
import json

# 创建词典
fass_dict = {
    "good": ["excellent", "wonderful"],
    "bad": ["poor", "terrible"]
}

# 保存
with open('custom.json', 'w') as f:
    json.dump(fass_dict, f)

# 使用
r = DictRewriter(sources=['fass'])
r.add_fass_dictionary('custom.json')
```

## 参数速查

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sources` | List[str] | ['wordnet', 'fasttext'] | 使用的源 |
| `source_weights` | Dict | None | 源权重 |
| `aggregation_method` | str | 'ranked' | 聚合策略 |
| `ratio` | float | 0.3 | 替换比例 |
| `is_replace_by_description` | bool | False | 名词用定义替换 |

## 源性能指标

| 源 | 初始化 | 执行时间 | 覆盖范围 | 准确度 |
|----|--------|---------|---------|--------|
| WordNet | 快 | 0.5s | 中 | 高 |
| FastText | 中 | 0.6s | 大 | 中高 |
| Sense2Vec | 慢 | 0.8s | 很大 | 很高 |
| FASS | 快 | 0.5s | 自定义 | 自定义 |

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| FastText 警告 | `pip install fasttext gensim` |
| Sense2Vec 警告 | `python -m sense2vec download s2v_reddit_2015_md` |
| 没有同义词 | 检查 NLTK 资源是否完整 |
| 源没有激活 | 使用 `get_active_sources()` 检查 |

## 环境变量

```python
# 查看当前配置
r = DictRewriter()
print(r.get_active_sources())      # ['wordnet', 'fasttext']
print(r.aggregation_method)         # 'ranked'
print(r.get_available_sources())   # 所有可用源
```

## 高级用法

```python
# 不同的源用于不同的调用
r = DictRewriter(sources=['wordnet'])

# 大部分使用 WordNet
for text in texts:
    rewritten, _ = r.rewrite(text, 0.5)

# 特殊情况切换源
special_rewritten, _ = r.rewrite(special_text, 0.5, 
                                 sources=['fasttext'],
                                 aggregation='union')

# 恢复原状态
rewritten, _ = r.rewrite(normal_text, 0.5)
```

## 完整示例

```python
from agi_toolkit.DictRewriter import DictRewriter
import json

# 1. 创建自定义词典
fass = {"good": ["excellent"], "bad": ["poor"]}
with open('dict.json', 'w') as f:
    json.dump(fass, f)

# 2. 初始化多源重写器
rewriter = DictRewriter(
    sources=['wordnet', 'fasttext', 'fass'],
    aggregation_method='ranked'
)

# 3. 加载自定义字典
rewriter.add_fass_dictionary('dict.json')

# 4. 执行重写
text = "This is a good project."
rewritten, wsd_info = rewriter.rewrite(text, ratio=0.5)

# 5. 查看结果
print(f"Original:  {text}")
print(f"Rewritten: {rewritten}")

# 6. 查看详细信息
for item in wsd_info:
    if item['synonyms']:
        print(f"{item['word']}: {item['synonyms'][:3]}")
```

## 相关文件

- 完整指南: `MULTISOURCE_GUIDE.md`
- 实现细节: `MULTISOURCE_IMPLEMENTATION.md`
- 示例代码: `examples_multi_source.py`
- 测试笔记本: `test_multi_source_dict.ipynb`
- 源代码: `SynonymSources.py`, `DictRewriter.py`

