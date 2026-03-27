#import "../../config.typ": *

#set page(height: auto, width: 24cm)
#set text(16pt, font: "IBM Plex Sans SC", lang: "zh")
#show raw: text.with(font: ("Zed Plex Mono", "IBM Plex Sans SC"))
#show math.equation: set text(16pt)
#set table(inset: 9pt)

#show: template-post.with(
  title: "论文阅读：MSA — 将 LLM 记忆扩展到 1 亿 Token 的稀疏注意力框架",
  description: "Memory Sparse Attention (MSA) 论文的深度阅读笔记，包含方法拆解、推理流程分析和批判性思考。",
  tags: ("论文阅读", "LLM", "注意力机制"),
  category: "论文阅读笔记",
  date: datetime(year: 2026, month: 3, day: 27, hour: 10, minute: 0, second: 0),
)

#quote[
  *论文*：MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens \
  *作者*：Yu Chen, Runkai Chen, Sheng Yi, Xinda Zhao 等 \
  *机构*：Evermind, Shanda Group, Peking University \
  *会议*：NeurIPS 2026 \
  *链接*：arXiv:2603.23516v1
]

= 一句话总结

MSA 的目标是让 LLM 拥有"终身记忆"：端到端地从多达 1 亿 token 的文档库中检索并生成答案，仅需 2 张 A800 GPU，且从 16K 扩展到 100M token 时性能仅下降不到 9%。

= 问题背景

当前 LLM 的有效上下文长度通常限制在 128K\~1M token，而认知科学估计人类终身记忆约为 2\~3 亿 token。现有长期记忆方案各有硬伤：

#table(
  columns: (1fr, 2fr, 2fr),
  table.header([类别], [代表方法], [核心缺陷]),
  [基于参数的记忆], [LoRA / CPT / Titans], [容量有限，易灾难性遗忘，训练开销大],
  [基于外部存储的记忆], [RAG / MemAgent], [检索与生成解耦，精度受限于浅层语义匹配],
  [基于潜在状态的记忆], [稀疏注意力 / 线性注意力], [KV 方法算力爆炸，线性注意力固定状态导致遗忘],
)

= 方法拆解

== 路由投影器 + Top-k 稀疏注意力

MSA 在标准注意力的 $W_Q$、$W_K$、$W_V$ 之外，额外引入两个线性投影矩阵 $W_(Q R)^h$ 和 $W_(K R)^h$（本质上就是两个 `nn.Linear`），用于生成专用的路由表示：

$
  K_(i,h) = H_i W_K^h, quad V_(i,h) = H_i W_V^h, quad K_(i,h)^R = H_i W_(K R)^h
$

每个文档被分割为固定长度块（block size = 64），对 $K$、$V$、$K^R$ 分别做块级均值池化（mean pooling），64 个 token 压缩为 1 个，得到 $overline(K)$、$overline(V)$、$overline(K)^R$。

推理时，查询经路由投影器生成 $Q^R$，与所有文档的 $overline(K)^R$ 计算余弦相似度，选出 Top-16 最相关的文档：

$
  S_(i j) = max_(op("token") t) (op("mean")_(op("head") h) (cos((Q_(q,h)^R)_t, overline(K)_(i j,h)^R)))
$

仅选中文档的压缩 KV 参与后续注意力计算。

#note(title: "关键设计")[路由仅应用于模型后半部分的层（后 18 层）。低层隐藏状态缺乏高层语义抽象，路由效果差。]

路由投影器的维度与标准 KV 投影一致：每头 128 维，8 个 KV 头，跨 18 层。可以从论文给出的内存数据反推验证：100M token 的路由键 $overline(K)^R$ 约占 56GB，与 $100M / 64 times 8 times 128 times 18 times 2$ 字节的计算结果一致。

== 文档级 RoPE

MSA 为每个记忆文档分配独立的位置 ID（都从 0 开始），将位置语义与文档总数解耦：

#table(
  columns: (1fr, 1fr, 2fr),
  table.header([策略], [适用范围], [作用]),
  [并行 RoPE], [每个记忆文档], [每个文档独立编号，不受文档总数影响],
  [全局 RoPE], [用户查询 + 生成], [位置 ID 从 $k$ 偏移（Top-k 检索数），保持因果依赖],
)

#warning(title: "100M 的真实含义")[
  Document-wise RoPE 能 work 的前提是*每个文档都足够短*。MSA 的 100M token 不是一个文档 100M，而是十万个短文档加起来 100M。如果你的场景是一本 100M token 的超长小说，位置 ID 仍会远超训练范围，Document-wise RoPE 救不了你。MSA 本质上解决的是"文档数量超多"的问题，不是"单文档超长"的问题。
]

== KV 缓存压缩 + Memory Parallel

100M token 的全部 KV 缓存（含路由键）约需 169GB，超过 2×A800 的 160GB 总显存。MSA 的解法是分层存储：

#table(
  columns: (1fr, 1fr, 1fr, 2fr),
  table.header([数据], [存储位置], [大小], [原因]),
  [$overline(K)^R$ 路由键], [GPU 显存], [\~56GB], [每次查询都要全扫描，必须低延迟],
  [$overline(K)$, $overline(V)$ 内容 KV], [CPU 内存], [\~113GB], [仅 Top-16 文档按需加载，每次 \~19MB],
  [模型权重], [GPU 显存（每卡复制）], [\~8GB×2], [4B 模型，BF16 格式],
)

每张 GPU 实际占用：模型权重 \~8GB + 路由键分片 \~28GB = \~36GB，剩余 \~44GB 足够完成推理。选出 Top-16 文档后，从 CPU 加载的内容 KV 仅约 19MB，PCIe 传输不到 1ms。

== Memory Interleave

针对多跳推理，MSA 将检索和生成交替进行。从 PDF 的 Figure 3 可以看到具体的 token 格式：

```
Query: When was Erik Watts' father born?

Model output: [4]<|object_ref_end|>
  → System fetches doc[4]: "Erik Watts ... is the son of Bill Watts."

Model output: [3]<|object_ref_end|>
  → System fetches doc[3]: "Bill Watts (born May 5, 1939) ..."

Model output: <End-of-Retrieve>
  <|im_start|>The answer is: May 5, 1939<|im_end|>
```

每轮模型生成文档 ID 后，系统取出原文追加到上下文，重新路由选出新的 Top-16 文档，继续生成。模型通过生成 `<End-of-Retrieve>` token 自主决定何时停止检索。

= 推理全流程

假设记忆库有 10 万篇文档（共 100M token），用户提问"刘备是怎么死的？"：

+ *离线阶段（仅一次）*：全部文档过一遍 4B 模型 forward pass，生成并缓存压缩 KV。
+ *路由*：查询经模型生成 $Q^R$，与 GPU 上的 156 万个路由键条目计算余弦相似度，选出 Top-16 文档。
+ *加载*：从 CPU 搬运 Top-16 文档的 $overline(K)$, $overline(V)$（\~19MB）到 GPU。
+ *生成*：注意力上下文 = [Top-16 压缩 KV ; 查询本地 KV]，逐 token 自回归生成答案。
+ *（如多跳）*：模型输出文档 ID → 系统追加原文 → 重新路由 → 继续生成，循环直到 `<End-of-Retrieve>`。

每个 token 生成时，本地 KV 正常增长，内存 KV 保持不变，不需要逐 token 重新拼接。

= 实验结果

== QA 任务

MSA（4B 参数）以 3.760 平均分超越了使用 Qwen3-235B 和 Llama-3.3-70B 等前沿生成器的 SOTA RAG 系统：

#table(
  columns: (2fr, 1fr),
  table.header([系统], [平均分]),
  [标准 RAG（最佳 R\@k）], [3.242],
  [RAG + 重排序], [3.372],
  [HippoRAG2], [3.275],
  [*MSA（\@adaptive）*], [*3.760*],
)

== NIAH 任务（32K\~1M token）

#table(
  columns: (2fr, 1fr, 1fr, 1fr),
  table.header([模型], [32K], [1M], [衰减]),
  [Qwen3-4B-Instruct], [0.95], [0.25], [-73.7%],
  [Qwen3-Next-80B-A3B], [1.00], [0.81], [-19.0%],
  [RL-MemoryAgent-14B], [0.98], [0.93], [-5.8%],
  [*MSA（4B）*], [*0.99*], [*0.95*], [*-3.9%*],
)

== 消融实验

#table(
  columns: (2fr, 1fr, 1fr),
  table.header([变体], [平均分], [相对下降]),
  [MSA-S2（完整）], [3.976], [---],
  [移除 Memory Interleave], [3.497], [-5.3%],
  [移除持续预训练], [2.537], [-31.3%],
  [移除原始文档文本注入], [2.325], [-37.1%],
)

持续预训练和原始文本注入是两个最关键的组件，移除后性能分别崩溃 31.3% 和 37.1%。

= 批判性分析

== MSA 本质上是 RAG 的变体

从最高层抽象看，MSA 仍然是 *存文档 → 查相关 → 拼起来生成答案* 的 retrieve-then-generate 范式。可以把它放在一个光谱上：

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  table.header([BM25+LLM], [Dense RAG], [RETRO], [REALM], [MSA]),
  [独立检索器], [独立 embedding], [跨注意力], [联合训练], [注意力内部路由],
  [独立生成器], [独立生成器], [集成解码], [端到端], [共享参数],
  [文本拼接], [文本拼接], [隐状态], [梯度贯通], [KV 直接计算],
)

MSA 更靠"融合"端，但没有跳出这个光谱。真正有意义的区别是：（1）检索结果是压缩 KV 而非文本，省掉重编码；（2）路由器和生成器共享同一个模型，语义对齐更好。

== 离线与在线成本的权衡

#table(
  columns: (1fr, 1fr, 1fr),
  table.header([], [RAG], [MSA]),
  [离线成本], [低（轻量 embedding 模型）], [高（完整 4B forward pass）],
  [离线存储], [小（向量索引）], [大（169GB KV cache）],
  [在线检索], [快（ANN 近似搜索）], [慢（暴力扫描 156 万条目）],
  [在线生成], [慢（重新编码检索文本）], [快（直接用压缩 KV）],
)

MSA 在生成阶段省掉了重编码，但离线编码和在线路由的成本都显著高于 RAG。

== Memory Interleave 的本质

论文将 Memory Interleave 描述为"自适应记忆交织机制"，但实际上它的推理循环等价于 LLM 的 tool calling / function calling：

```
loop:
    model generates tokens
    if output contains [doc_id]<|object_ref_end|>:
        system fetches original text, appends to context
        re-route, continue loop
    if output contains <End-of-Retrieve>:
        return answer
```

"模型自主判断证据充足"只是模型在该位置生成了 `<End-of-Retrieve>` 而非 `[doc_id]`。单跳和多跳走的是同一个 pipeline，区别仅在于循环执行了几次。

== 命名问题

MSA 叫"Memory Sparse Attention"，但并没有修改注意力机制本身的稀疏性（对比 Longformer、BigBird、NSA 等真正的 sparse attention）。MSA 的"稀疏"是指从外部文档库中选择少数文档参与注意力，粒度是*文档级*而非*token 级*，选择机制是*独立的路由投影器*而非*注意力分数*本身。

更准确的定位可能是 *Sparse Retrieval with Attention-based Fusion*。

= 个人总结

MSA 的真正贡献不在于范式创新，而在于系统工程：通过 KV 缓存压缩、Document-wise RoPE、分层存储和 Memory Parallel 这套组合拳，让一个 4B 的小模型在 2×A800 上就能跑 1 亿 token 级别的文档库检索。

它证明了一件事：*把 RAG 的检索器深度集成进注意力机制、让检索和生成共享同一个模型的表示空间*，比用更大的生成器 + 独立检索器更有效。在实际应用中，如果你的场景是"海量短文档的问答检索"，MSA 是比传统 RAG 更优的选择；但如果你需要理解一本完整的超长文档，仍然需要 Ring Attention 等真正扩展单序列长度的技术。
