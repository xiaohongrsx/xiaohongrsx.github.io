#import "../../config.typ": *

#set page(height: auto, width: 24cm)
#set text(16pt, font: "IBM Plex Sans SC", lang: "zh")
#show raw: text.with(font: ("Zed Plex Mono", "IBM Plex Sans SC"))
#show math.equation: set text(16pt)
#set table(inset: 9pt)

#show: template-post.with(
  title: "论文阅读：InfLLM-V2 — 把可训练稀疏注意力做成全注意力的“受限版本”",
  description: "InfLLM-V2 论文阅读笔记：零额外参数、Dense-Sparse Switchable Attention、三阶段块选择与高效 kernel，如何把短序列预训练和平滑的长上下文适配真正打通。",
  tags: ("论文阅读", "LLM", "注意力机制"),
  category: "论文阅读笔记",
  date: datetime(year: 2026, month: 3, day: 30, hour: 11, minute: 30, second: 0),
)

#quote[
  *论文*：InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation \
  *作者*：Weilin Zhao, Zihan Zhou, Zhou Su, Chaojun Xiao, Yuxuan Li, Yanghao Li, Yudi Zhang, Weilun Zhao, Zhen Li, Yuxiang Huang, Ao Sun, Xu Han, Zhiyuan Liu \
  *机构*：清华大学, OpenBMB, 哈尔滨工业大学 \
  *链接*：arXiv:2509.24663v1
]

= 一句话总结

InfLLM-V2 的真正创新，不是“再发明一种稀疏注意力”，而是把*可训练稀疏注意力*做成了*与标准全注意力高度同构的受限版本*：短序列继续跑 dense attention，长序列再切到 sparse attention；参数不重来、输出形式不突变、训练流程仍然遵循“短序列预训练 -> 长序列微调”。

= 问题背景：为什么现有长上下文方案不够顺手

Transformer 的全注意力在长序列上的计算和显存开销是平方级的。要把上下文长度从 4K 拉到 32K、128K，光靠硬算基本不现实，于是大家会考虑稀疏注意力。但现有方法通常卡在两个方向：

#table(
  columns: (1fr, 2fr, 2fr),
  table.header([路线], [代表方法], [核心问题]),
  [免训练稀疏注意力], [InfLLM, MInference], [不需要重新训练，但为了避免性能明显下降，稀疏程度往往不敢开太大，提速有限],
  [可训练稀疏注意力], [NSA], [可以学出更强的稀疏模式，但会引入三套 KV 投影、多个注意力分支和门控，和标准 LLM 的训练流程不对齐],
  [继续用全注意力], [Full Attention], [效果上限高，但长上下文成本太高，预填充和解码都很贵],
)

论文的核心观察是：*可训练稀疏注意力之所以难用，往往不是因为它不够强，而是因为它和原始 Transformer 太不像了。*  
如果长上下文微调阶段把 attention 模块整个换成另一种复杂结构，模型就相当于突然被迫学一套新网络，训练容易不稳，短上下文效率也会变差。

= 先看一个具体例子

假设现在有一个 *32K token* 的长文档，模型正在处理第 *20000* 个 token。论文把序列按块切分，每块大小为 $B = 64$，因此这个 token 位于第 *313* 个块。

如果使用全注意力，这个 token 需要看前面全部历史，大约是两万个 token。  
而在 InfLLM-V2 中，它只看三类块：

+ *初始块*：保留最前面的少量块，提供全局锚点
+ *局部块*：保留当前位置最近的一批块，负责短程依赖
+ *远程 top-k 块*：从更远的历史里动态挑选最相关的少量块

块集合可以写成：

$
  I(i) = I_("init") union I_("local")(i) union I_("topk")(i)
$

训练设置中总共保留 *96* 个块，每块 *64* 个 token，也就是只看：

$
  96 times 64 = 6144
$

个 token。也就是说，原本可能要看前面约 *20000* 个 token，现在只看约 *6K* 个 token。

```
全部历史:  [1 ...................................................... 313]
真正保留:  [首块] [最近 32 个局部块 ...................... 313] [若干远程 top-k 块]
```

#note(
  title: "直觉",
)[这有点像在读一本很长的书：你不会每次都从第一页重读到当前页，而是保留开头的目录、手边最近几页，再加上少量真正相关的早期章节。]

= 创新一：让 sparse 尽量长得像 dense

== 共享 KV 投影

InfLLM-V2 最关键的一刀，是*不再像 NSA 那样为不同分支各自维护一套 KV 投影*。它直接复用原始 dense attention 的 $W_K$ 和 $W_V$。

这件事听起来像工程细节，实际上很重要：

+ 短序列预训练学到的表示可以直接继承到长序列阶段
+ 切到 sparse 时不需要重新发明一套 KV 空间
+ 历史缓存不需要因为“换了投影参数”而重编码

== 对齐计算图

NSA 的计算图更像是三路注意力并行跑完，再通过门控融合：

$
  O = g_("cmp") O_("cmp") + g_("slc") O_("slc") + g_("win") O_("win")
$

而 InfLLM-V2 的做法是：

+ *把 Selected Attention 和 Sliding Attention 合并成统一的 sparse attention*
+ *压缩分支只负责选块，不再参与最终输出*
+ *最终只保留一个主输出分支*

于是它更像是“标准注意力只不过少看了一部分 token”，而不是“突然换成多分支网络”。

#table(
  columns: (1fr, 1.4fr, 1.4fr),
  table.header([设计维度], [NSA], [InfLLM-V2]),
  [KV 投影参数], [三组独立参数], [单组共享参数，直接复用预训练权重],
  [注意力模块], [三个独立模块 + 门控聚合], [统一 sparse attention 单分支输出],
  [压缩模块], [MLP 压缩且参与输出], [无参数池化，只负责块选择],
  [短序列效率], [短序列也得算三模块], [短序列直接走 dense attention],
)

== Dense-Sparse Switchable Attention

InfLLM-V2 不是让所有输入都硬走 sparse，而是根据序列长度动态切换路径：

```python
if seq_len <= threshold:
    y = dense_attention(x)
else:
    scores = compressed_block_scores(Q, K)
    blocks = init_blocks | local_blocks | topk_blocks(scores)
    y = sparse_attention_over_selected_blocks(Q, K, V, blocks)
```

这里最重要的一点是：*不是 dense 和 sparse 两条路都跑一遍再选*，而是按输入长度选择一条更划算的执行路径。  
所以“切换”本身不是主要瓶颈，真正有成本的是 sparse 路径里的*块选择*。

= 创新二：三阶段块表示压缩

难点在于：如果你想先选块再做 sparse attention，就必须先估计“哪些块值得看”。  
但如果这一步本身太贵，稀疏注意力省下来的算力又会被选块过程吃回去。

InfLLM-V2 的做法不是直接把一个 64-token 大块粗暴压成一个向量，而是设计了一个*三阶段从细到粗的压缩流程*：

#table(
  columns: (1fr, 2fr, 2fr),
  table.header([阶段], [做什么], [作用]),
  [第一阶段], [对 $K$ 做滑动平均池化，得到更细粒度的中间表示], [先保留局部细节，不急着把整块“一锅端”],
  [第二阶段], [在 GQA 的头组内对注意力分数求和], [强制同组头共享块选择模式，利于块稀疏实现],
  [第三阶段], [对聚合分数做最大池化，得到最终块分数], [保留最显著的相关性信号，用于 top-k 选块],
)

其中第二阶段的核心式子是：

$
  S^("shared") = sum_(h=1)^G S^("C1")(h)
$

直觉上，这三步分别对应：

+ *先粗筛局部相关区域*
+ *再在头组内把信息合并，减少块选择的离散度*
+ *最后只保留每个块最强的证据*

这比“一次大平均池化直接选块”更稳，因为它不会过早把细粒度信息抹平。

= 训练是怎么做的

这篇论文在训练流程上其实非常朴素，甚至可以说它的一个重要优点就是：*几乎不额外折腾训练范式*。

#table(
  columns: (1fr, 2fr, 2fr),
  table.header([阶段], [做什么], [为什么这样做]),
  [第一阶段], [用标准 dense attention 在短序列上预训练], [先把模型按普通 LLM 的方式训会],
  [第二阶段], [把注意力模块切成 InfLLM-V2，并复用原始 $W_K$、$W_V$], [保持参数与表示空间连续],
  [第三阶段], [在长序列数据上继续微调，短样本走 dense，长样本走 sparse], [同时保住短文本能力并适配长上下文],
)

论文里的 recipe 大致是：

+ 短上下文预训练：*4K* 长度，*8T tokens*
+ 长上下文微调：*5B tokens*
+ 微调时混合四段长度：*0\~4K*、*4\~12K*、*12\~24K*、*24\~32K*
+ 稀疏块大小：$B = 64$
+ 总可见块数：*96*，也就是约 *6K visible tokens*

训练目标本身没有变化，仍然是普通的下一 token 预测。  
从这个角度看，InfLLM-V2 不是“从零训练一个全新的 sparse 模型”，而是：

*先用 dense 把模型训会，再用一个和 dense 很像的 sparse 机制做长上下文适配。*

```python
# stage 1: short-context pretraining
for x in short_sequences:
    y = dense_attention_model(x)
    loss = next_token_loss(y)
    update(all_params)

# stage 2: long-context adaptation
load_pretrained_weights()
replace_attention_with_infllm_v2()

for x in mixed_length_sequences:
    if len(x) <= threshold:
        y = dense_attention(x)
    else:
        scores = compressed_scores(Q, K)
        blocks = init_blocks | local_blocks | topk_blocks(scores)
        y = sparse_attention_over(blocks)

    loss = next_token_loss(y)
    update(shared_model_params)
```

#warning(
  title: "这里最值得记住的一点",
)[InfLLM-V2 的训练优势不在于用了什么花哨的新 loss，而在于*长上下文阶段没有发生架构突变*：参数基本对齐，输出形式基本对齐，只有“能看到的上下文范围”发生了变化。]

= 切换会不会慢

我们在讨论这篇论文时，一个很自然的问题是：既然它要在 dense 和 sparse 之间切换，那切换本身会不会拖慢？

我的理解是：*切换本身不是主要瓶颈，真正昂贵的是 sparse 路径里的块选择过程。*

#table(
  columns: (1fr, 1fr, 2fr),
  table.header([成本项], [是不是主要瓶颈], [原因]),
  [dense/sparse 路径切换], [否], [通常按序列长度选择一条路径，而不是两条都跑],
  [重新编码历史缓存], [否], [共享同一套 $W_K$、$W_V$，不需要因为换模式而重做 KV],
  [块选择], [是], [要先计算粗粒度分数，再做 top-k 选块，这部分如果实现不好会很慢],
)

论文也正是针对这第三项做了 kernel 级优化：

+ *融合头组求和*：把 GQA 头组内求和直接融合到 SRAM 计算循环里，减少写回 HBM 的数据量
+ *LSE 近似*：避免为了精确求 softmax 归一化而额外付出接近 2 倍开销

论文报告的结果也说明了这一点：如果“切换”本身很贵，后面不可能拿到这么高的速度收益。它最终在内核层面拿到了相对 FlashAttention *7.4x*（A100）到 *9.3x*（4090）的加速，端到端推理也拿到了 *2.13x* 的预填充加速和 *2.32x* 的解码加速。

= 实验结果说明了什么

我觉得实验里最值得关注的不是某一个绝对数字，而是下面这三件事：

#table(
  columns: (1.8fr, 1fr, 1fr, 1fr),
  table.header([任务], [FullAttn], [InfLLM-V2], [NSA]),
  [RULER 32K 平均分], [84.26], [Dense: *88.32* / Sparse: 82.62], [59.92],
  [LongBench / LongPPL], [42.30 / 2.06], [*42.54* / 2.12], [37.10 / 4.24],
  [长推理平均分], [42.79], [42.66], [37.28],
)

+ *第一*，InfLLM-V2 的 sparse 模式已经足够接近 Full Attention，说明“只看一部分块”并没有把长程信息砍坏
+ *第二*，它在短到长适配场景下明显强于 NSA，证明“架构对齐”这件事不是小修小补，而是决定训练能否稳定工作的关键
+ *第三*，它甚至保留了 dense 模式，在某些长上下文理解评测上 dense 结果还高于 FullAttn 基线，这说明长上下文微调后的模型并没有丢掉短文本或稠密路径能力

= 我对这篇文章的理解

如果只记一句话，我会记成：

*InfLLM-V2 的真正创新，是把长上下文稀疏注意力从“外挂的一套复杂结构”，变成了“标准全注意力的可切换稀疏版本”。*

更具体一点，我觉得它有三层价值：

+ *算法层*：共享 KV 投影、统一 sparse 输出、三阶段块压缩，让稀疏注意力更接近 dense attention
+ *训练层*：自然适配“短序列预训练 -> 长序列微调”这一主流范式，不需要架构大改
+ *系统层*：没有停留在“理论上更省算力”，而是继续把块选择做成真正能跑快的 kernel

所以这篇文章最打动我的地方，并不是某个单点技巧，而是它非常明确地抓住了一个原则：

*长上下文 attention 不仅要稀疏、要快，还必须和原始 Transformer 足够像。*  
只有这样，它才真的适合拿来训练、拿来部署，而不只是论文里的一个特殊模块。
