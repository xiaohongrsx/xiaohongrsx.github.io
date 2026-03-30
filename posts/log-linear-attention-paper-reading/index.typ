#import "../../config.typ": *

#set page(height: auto, width: 24cm)
#set text(16pt, font: "IBM Plex Sans SC", lang: "zh")
#show raw: text.with(font: ("Zed Plex Mono", "IBM Plex Sans SC"))
#show math.equation: set text(16pt)
#set table(inset: 9pt)

#show: template-post.with(
  title: "论文阅读：Log-Linear Attention — 基于 Fenwick 树与层次矩阵的对数线性复杂度注意力机制",
  description: "Log-Linear Attention 论文深度阅读笔记：结构化掩码矩阵统一视角、Fenwick 树分区、HODLR 层次矩阵、分块并行扫描训练算法，如何用 O(log T) 个隐藏状态在效率和表达能力之间取得新平衡。",
  tags: ("论文阅读", "LLM", "注意力机制"),
  category: "论文阅读笔记",
  date: datetime(year: 2026, month: 3, day: 30, hour: 20, minute: 0, second: 0),
)

#quote[
  *论文*：Log-Linear Attention \
  *作者*：Han Guo\*, Songlin Yang\*, Tarushii Goel, Eric P. Xing, Tri Dao, Yoon Kim \
  *机构*：MIT, Princeton University / Together AI, CMU / MBZUAI / GenBio AI \
  *链接*：arXiv:2506.04761v3, ICLR 2026
]

= 核心贡献

本文提出 Log-Linear Attention，将线性注意力中的单一固定大小隐藏状态替换为 $O(log T)$ 个随序列长度对数增长的层次化隐藏状态。通过 Fenwick 树分区和层次矩阵（HODLR）理论，实现了 $O(T log T)$ 训练复杂度和 $O(log T)$ 解码内存的新复杂度折中。

= 研究动机

== 高效注意力的结构化矩阵统一框架

论文首先建立了一个关键观察：广泛的高效注意力机制可以统一表示为

$
  P = A dot.circle M, quad O = P V
$

其中 $A in bb(R)^(T times T)$ 是类注意力矩阵（如 $Q K^top$），$M in bb(R)^(T times T)$ 是下三角因果掩码矩阵。不同模型通过对 $M$ 施加不同结构约束来实现不同的效率-表达能力折中：

#table(
  columns: (1.8fr, 1.8fr, 1fr, 1fr, 1fr),
  table.header([$M$ 的结构], [模型], [训练], [解码时间], [解码内存]),
  [全 1 下三角（半可分）], [Linear Attention / Mamba-2], [$O(T)$], [$O(1)$], [$O(1)$],
  [1-半可分], [RetNet / Mamba-2（含门控）], [$O(T)$], [$O(1)$], [$O(1)$],
  [Toeplitz], [长卷积模型（Multi-Hyena）], [$O(T log T)$], [$O(log^2 T)$], [$O(T)$],
  [*准层次矩阵（Quasi-H）*], [*Log-Linear Attention*], [$O(T log T)$], [$O(log T)$], [$O(log T)$],
  [无结构], [Softmax Attention], [$O(T^2)$], [$O(T)$], [$O(T)$],
)

核心洞察：*决定计算效率的不是 $A$ 的形式（是否包含 softmax），而是 $M$ 的结构。* 使用无结构的 $M$（如随机下三角矩阵）即使去掉 softmax 也无法降低复杂度。

== 线性注意力的根本瓶颈

线性注意力的循环形式 $S_t = S_(t-1) + v_t k_t^top$, $o_t = S_t q_t$ 将整个前缀 $[0, t)$ 压缩为单一 $d times d$ 隐藏状态 $S_t$。即使加入数据依赖门控（Mamba-2）或 delta 规则（Gated DeltaNet），固定大小的隐藏状态仍是关联回忆等需要精确检索上下文信息的任务的根本瓶颈。

本文的核心问题：*能否在保持次二次训练复杂度和亚线性解码内存的前提下，突破固定大小隐藏状态的表达瓶颈？*

= Fenwick 树分区与多尺度隐藏状态

== 前缀的层次化分解

从解码视角看，注意力机制可被视为对前缀 $[0, t)$ 的*分区-聚合*过程。Softmax attention 将每个 token 独立存储（$t$ 个大小为 1 的桶），线性注意力将所有 token 合并为一个桶。Log-Linear Attention 采用 Fenwick 树分解（Ryabko, 1992; Fenwick, 1994），将前缀划分为至多 $L = O(log T)$ 个不相交的桶 $B_t^((ell))$，桶大小 $|B_t^((ell))| = 2^(ell-1)$（$ell gt.eq 1$），加上一个大小为 1 的哨兵桶 $B_t^((0)) = {t}$。

分区方法：从 $t$ 开始，反复减去 $"lowbit"(t) = t and (-t)$，每步切出一个桶。

以 $t = 7$（二进制 `111`）为例：

- $"lowbit"(7) = 1$：桶 $B_7^((1)) = {6}$，大小 $2^0 = 1$
- $"lowbit"(6) = 2$：桶 $B_7^((2)) = {4, 5}$，大小 $2^1 = 2$
- $"lowbit"(4) = 4$：桶 $B_7^((3)) = {0, 1, 2, 3}$，大小 $2^2 = 4$

```
时间轴:  0  1  2  3  4  5  6  [7]
        |--- B⁽³⁾ ----|--B⁽²⁾--|B⁽¹⁾| B⁽⁰⁾
        |  大小 4      | 大小 2 |  1 |  1
```

以 $t = 8$（二进制 `1000`）为例，$"lowbit"(8) = 8$，所有 8 个历史 token 合并为单一桶 $B_8^((4)) = {0, dots, 7}$，退化为线性注意力。$t$ 的二进制中有多少个 1，就有多少个非空桶。

== 层级判定算法

给定 $s < t$，可通过 Fenwick 树遍历确定 $s$ 所属的桶层级 $ell(t, s)$：

```python
def level(t, s):
    if s == t: return 0
    cursor = t
    while cursor > 0:
        lb = cursor & (-cursor)       # lowbit
        if cursor - lb <= s < cursor:
            return lb.bit_length()    # ℓ = log₂(lb) + 1
        cursor -= lb
```

$t = 7$ 时各 $s$ 的层级：

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  table.header([$s$], [0], [1], [2], [3], [4], [5], [6], [7]),
  [$ell(7, s)$], [3], [3], [3], [3], [2], [2], [1], [0],
)

== 循环形式与输出计算

每个桶维护独立的 $d times d$ 隐藏状态 $S_t^((ell)) = sum_(s in B_t^((ell))) v_s k_s^top$。输出由*数据依赖的非负层级权重* $lambda_t^((ell))$ 加权求和：

$
  o_t = sum_(ell=0)^(L-1) lambda_t^((ell)) dot q_t^top S_t^((ell))
$

$lambda_t^((ell))$ 通过对当前 token 隐藏表示 $x_t$ 施加线性投影得到，为 per-head 参数。额外参数量：Mamba-2 增加 < 3%，Gated DeltaNet 增加 < 0.4 %。

*退化条件*：当 $lambda_t^((ell))$ 在各层级间相同（或线性相关）时，Log-Linear Attention 退化为标准线性注意力。

= 层次矩阵 $M^H$ 与并行训练形式

== 并行形式

将循环形式重写为矩阵形式以支持并行训练：

$
  O = (Q K^top dot.circle M^H) V, quad M^H_(t s) = cases(lambda_t^(ell(t, s)) & "if" s lt.eq t, 0 & "otherwise")
$

其中 $ell(t, s)$ 由 Fenwick 树分区确定。$T = 8$ 时 $M^H$ 的结构（$lambda^((k))$ 简记为 $lambda^k$）：

```
         s=0   s=1   s=2   s=3   s=4   s=5   s=6   s=7
  t=0: [ λ⁰                                              ]
  t=1: [ λ¹    λ⁰                                        ]
  t=2: [ λ²    λ²    λ⁰                                  ]
  t=3: [ λ²    λ²    λ¹    λ⁰                            ]
  t=4: [ λ³    λ³    λ³    λ³    λ⁰                      ]
  t=5: [ λ³    λ³    λ³    λ³    λ¹    λ⁰                ]
  t=6: [ λ³    λ³    λ³    λ³    λ²    λ²    λ⁰          ]
  t=7: [ λ³    λ³    λ³    λ³    λ²    λ²    λ¹    λ⁰    ]
```

== HODLR 结构与低秩性

$M^H$ 属于 *HODLR（Hierarchically Off-Diagonal Low-Rank）* 矩阵。其核心性质可通过递归二分验证：

将 $8 times 8$ 矩阵沿中点二分为四个 $4 times 4$ 子块。*左下非对角块*为：

$
  mat(
    lambda_4^((3)), lambda_4^((3)), lambda_4^((3)), lambda_4^((3));
    lambda_5^((3)), lambda_5^((3)), lambda_5^((3)), lambda_5^((3));
    lambda_6^((3)), lambda_6^((3)), lambda_6^((3)), lambda_6^((3));
    lambda_7^((3)), lambda_7^((3)), lambda_7^((3)), lambda_7^((3));
  ) = vec(lambda_4^((3)), lambda_5^((3)), lambda_6^((3)), lambda_7^((3))) dot mat(1, 1, 1, 1)
$

这是一个*秩 1* 矩阵。两个对角子块的结构与原矩阵自相似——递归二分后左下角仍为秩 1。

这种"递归划分后每层非对角块均为低秩"的性质正是 H-matrix 的定义特征。

== 复杂度推导

秩 $r$ 矩阵 $M in bb(R)^(n times n)$ 与向量的乘法复杂度为 $O(r n)$ 而非 $O(n^2)$。$M^H$ 含 $O(log T)$ 层低秩块（每层秩 1），因此与 $M^H$ 的结构化矩阵乘法复杂度为 $O(T log T)$。

= 解码：基于 lssb 的状态更新

定义 $"lssb"(t)$ 为 $t$ 的二进制中最低有效设置位的索引（即 $max{ell in bb(N) | 2^ell "divides" t}$）。隐藏状态 $\{S_t^((ell))\}_ell$ 按如下规则更新：

$
  S_t^((ell)) = cases(
    v_t k_t^top & "if" ell = 0,
    0 & "if" 0 < ell lt.eq "lssb"(t),
    sum_(ell'=0)^(ell-1) S_(t-1)^((ell')) & "if" ell = "lssb"(t) + 1,
    S_(t-1)^((ell)) & "if" ell > "lssb"(t) + 1
  )
$

该规则等价于 Fenwick 树的更新操作：$ell = 0$ 层接收当前 token 的外积；$"lssb"(t)$ 以下的层级合并后提升一级；更高层保持不变。

以 $T = 8$ 为例，各时间步的状态演化：

#table(
  columns: (0.6fr, 0.6fr, 1fr, 1fr, 1.5fr, 1.5fr),
  table.header([$t$], [二进制], [lssb], [$ell = 0$], [$ell = 1$], [$ell = 2$]),
  [0], [`0`], [—], [$v_0 k_0^top$], [$emptyset$], [$emptyset$],
  [1], [`01`], [0], [$v_1 k_1^top$], [$v_0 k_0^top$], [$emptyset$],
  [2], [`10`], [1], [$v_2 k_2^top$], [$emptyset$], [$sum_(i=0)^1 v_i k_i^top$],
  [3], [`11`], [0], [$v_3 k_3^top$], [$v_2 k_2^top$], [$sum_(i=0)^1 v_i k_i^top$],
  [4], [`100`], [2], [$v_4 k_4^top$], [$emptyset$], [$emptyset$],
)

$t = 4$ 时，$"lssb"(4) = 2$，$ell = 1, 2$ 层清零，原有内容合并至 $ell = 3$ 层：$S_4^((3)) = sum_(i=0)^3 v_i k_i^top$。

每步更新涉及至多 $O(log T)$ 个状态，解码时间和空间均为 $O(log T)$。

= 训练：分块并行扫描算法

== 矩阵分解

给定块大小 $C$，$M^H$ 可分解为块对角矩阵 $D$（块内交互）与各层级块间矩阵 $M^((ell))$ 之和：

$
  M^H = underbrace(D, "块内") + sum_(ell = ell_C)^(L-1) underbrace(M^((ell)), "块间层级" ell)
$

其中 $ell_C = log_2 C + 1$ 为与块大小对齐的最低层级。$ell < ell_C$ 的层级折叠进 $D$。

== 两阶段计算

*块内阶段*（$ell < ell_C$）：$D$ 是 $T\/C$ 个大小为 $C times C$ 的因果对角块，各块独立做稠密注意力并行处理。每块 $O(C^2)$，总计 $O(T C)$。

*块间阶段*（$ell gt.eq ell_C$）：每个 $M^((ell))$ 为缩放的半可分结构，可复用 Mamba-2 / Gated DeltaNet 的分块并行原语。以 $T = 8, C = 2$ 为例：

*层级 2*（相邻块间依赖）：块 0 的状态 $S_0 = sum_(s in "chunk"_0) v_s k_s^top$ 传递至块 1；块 2 的状态传递至块 3。两对独立，可并行执行。对应一次 Mamba-2 分块扫描原语，$O(T)$。

$
  o_t^((2)) = lambda_t^((2)) dot q_t^top dot S_("prev-chunk"), quad t in "receiving chunk"
$

*层级 3*（更远块间依赖）：块 0+块 1 的合并状态 $S_(0 1)$ 传递至块 2 和块 3。同样一次扫描原语，$O(T)$。

*输出聚合*：$o_t = o_t^("intra") + sum_(ell gt.eq ell_C) o_t^((ell))$

== 复杂度分析

#table(
  columns: (1fr, 1.5fr, 1.5fr),
  table.header([], [Mamba-2], [Log-Linear Mamba-2]),
  [块内], [$O(T C)$], [$O(T C)$],
  [块间], [1 次扫描 $O(T)$], [$O(log(T \/ C))$ 次扫描],
  [*总计*], [$O(T)$], [$O(T log T)$],
)

额外开销仅为对数因子。$T = 16384, C = 64$ 时块间扫描次数为 $log_2(256) = 8$，且每次调用的均为已有的高效 Triton 内核。

= 与已有模型的组合

Log-Linear Attention 保留原模型中 $A$ 的参数化形式，仅将掩码组合为 $M = M^S dot.circle M^H$：

$
  O = (Q K^top dot.circle M^S dot.circle M^H) V quad "(Log-Linear Mamba-2)"
$

$
  O = ((Q K^top dot.circle L) dot I + K K^top dot.circle (L - I)^(-1) dot.circle M^S dot.circle M^H) V quad "(Log-Linear Gated DeltaNet)"
$

其中 $M^S$ 为原模型的半可分掩码（$M^S_(i j) = product_(k=j+1)^i alpha_k$）。半可分矩阵与 H-矩阵的 Hadamard 积仍为 H-矩阵。

任何具有结构化记忆和高效分块原语的线性注意力均可通过此方式提升为对数线性变体。

= 实验结果

== 合成任务：MQAR（多查询关联回忆）

序列长度 256，4-64 个键值对，5 个种子的平均准确率（标准差）：

#table(
  columns: (2fr, 1fr, 1fr, 1fr),
  table.header([模型], [维度 16], [维度 32], [维度 64]),
  [Transformer], [$gt.eq 99$], [$gt.eq 99$], [$gt.eq 99$],
  [Mamba-2], [46.9 (2.3)], [75.1 (4.9)], [89.6 (6.1)],
  [+ Log-Linear], [*55.9 (9.1)*], [*76.5 (4.8)*], [*92.9 (2.7)*],
  [Gated DeltaNet], [38.4 (1.0)], [79.0 (2.1)], [$gt.eq 99$],
  [+ Log-Linear], [*40.0 (1.4)*], [*84.4 (1.2)*], [$gt.eq 99$],
)

对数线性变体在所有设置下均优于对应的线性基线。

== 语言建模

500 亿 token 预训练，序列长度 16K，21 层：

#table(
  columns: (2fr, 1fr, 1fr, 1fr),
  table.header([模型], [Wiki. ppl ↓], [LMB. ppl ↓], [LMEval avg ↑]),
  [Transformer (693M)], [21.56], [22.14], [44.0],
  [Transformer (778M, 24L)], [21.13], [21.17], [45.6],
  [Mamba-2 (802M)], [22.44], [24.14], [44.8],
  [*LL Mamba-2* (825M)], [*22.11*], [*21.86*], [*44.9*],
  [Gated DeltaNet (793M)], [21.73], [19.71], [45.0],
  [*LL Gated DeltaNet* (796M)], [*21.45*], [*18.09*], [*45.5*],
)

Log-Linear Gated DeltaNet 在所有指标上优于层数匹配的 Transformer（21 层），在半数指标上优于参数匹配的 Transformer（24 层）。

== 长上下文评估

*NIAH（Needle-In-A-Haystack）*：多针检索任务中 Log-Linear Gated DeltaNet 全部 9 个指标改进。

*逐位置损失分析*：对数线性变体在各位置上持续降低损失，表明长程上下文利用能力得到改善。

*训练效率*：自定义 Triton 内核在序列长度 >8K 时前后向速度超过 FlashAttention-2；完整模型在 32K 时吞吐量超过 Transformer。

= 总结与讨论

#table(
  columns: (1.2fr, 1fr, 1fr, 1fr),
  table.header([], [Softmax Attention], [Linear Attention], [*Log-Linear Attention*]),
  [隐藏状态数], [$T$], [1], [$O(log T)$],
  [解码内存], [$O(T)$], [$O(1)$], [$O(log T)$],
  [训练复杂度], [$O(T^2)$], [$O(T)$], [$O(T log T)$],
  [$M$ 的结构], [无结构], [半可分], [准层次矩阵（Quasi-H）],
)

Log-Linear Attention 的理论贡献在于建立了对数线性注意力与层次矩阵理论（HODLR）之间的精确联系：注意力算子等价于与准 H-矩阵的结构化矩阵乘法。Fenwick 树分区作为 H-矩阵的具体构造方案，同时满足训练端的 $O(T log T)$ 并行复杂度和推理端的 $O(log T)$ 在线更新复杂度。

局限性方面，Fenwick 树分区引入了"近处高分辨率、远处低分辨率"的固定归纳偏置，这在 $t$ 为 2 的幂时退化为单桶（等价于线性注意力）。此外，与 Transformer 相比仍存在性能差距，最优的 $lambda$ 参数化策略有待进一步探索。
