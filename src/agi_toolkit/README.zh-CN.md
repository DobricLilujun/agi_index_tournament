# AGI Tournament Toolkit Design

> 面向“问题重写（Rewrite）+ 系统评估（Evaluation）”的一套工具与实验框架设计草案。

---

## 🎯 项目目标计划

### 1) Rewrite：4R 框架
围绕同一问题/任务，构造可控、可组合的重写策略，用于提升鲁棒性测试、泛化评估与对抗性分析。

- **R1 — Reword（改写措辞）**
  - Dictionary Rewriting（词典/同义替换）
  - Back Translation（回译）
  - LLM Rewriting（大模型改写）

- **R2 — Random Rearrange（随机重排）**
  - 句子随机打乱
  - 单词（word）随机打乱

- **R3 — Realize（理解与重述）**
  - 总结（Summarization）
  - Agentic Debating（代理式辩论/多轮自辩）
  - When to Stop?（停止条件/收敛判据）

- **R4 — Recheck（复核与校验）**
  - LLM as a judge（LLM 作为裁判）
  - 比较相似度与正确性
  - 评估“现模型 vs 新模型”生成问题的相似度（以及语义一致性）

---

### 2) 评估正确性（Correctness）
对上述策略进行系统性评估与测试，覆盖：长问题 / 短问题 / 不同 Benchmark 等。

- **工程与框架**
  - 构造统一的 `evaluation` 框架（代码层面）
  - 兼容接入多种 benchmark（可插拔）

- **指标与流水线**
  - 对 **AGI Index** 进行测试与实现
  - 将“单个问题”的**完整 pipeline** 纳入代码（从重写→推理→判分→汇总）

- **实验设计**
  - 测试不同 Rewrite 策略的**组合效果**，进行模式识别与对比
  - 加入 **ablation test（消融实验）**：
    - 选用最新的人工 benchmark
    - 同样跑我们的 pipeline
    - 观察“重写是否带来收益/收益来自哪一环”
  - 研究如何用“最强组合”成功 **confuse** 最强模型（对抗性/脆弱点探索）

---

### 3) 评估多样性（Diversity）
从多个维度衡量 rewrite 策略是否带来更强覆盖与更丰富的变体空间。

- 评估不同 rewrite 策略对**多样性**的提升
- 评估不同 rewrite 策略对**正确性**的提升
- 评估不同 rewrite 策略对**效率**的提升（计算开销/时间/成本）
- 评估不同 rewrite 策略对**复杂问题**的提升
- 评估不同 rewrite 策略对**长问题**的提升
- 评估不同 rewrite 策略对**短问题**的提升
- 评估不同 rewrite 策略对**不同领域问题**的提升

---

### 4) 评估泛化性（Generalization）
评估不同主体（模型/人类）在不同能力维度上的泛化表现，并用统一指标呈现。

- 评估不同模型在不同能力下的泛化性：使用我们的 **AGI Index** 做测试与体现
- 评估人类在不同能力下的泛化性：同样使用 **AGI Index** 做测试与体现
- 使用 **SPoSE modelling** 对比研究与测试：
  - 对比人脑与 LLMs 的能力分布
  - 绘制“人脑 vs 模型”的**分布/能力对比图**
