#import "../../config.typ": *

#set page(height: auto, width: 24cm)
#set text(16pt, font: "IBM Plex Sans SC", lang: "zh")
#show raw: text.with(font: ("Zed Plex Mono", "IBM Plex Sans SC"))
#show math.equation: set text(16pt)
#set table(inset: 9pt)

#show: template-post.with(
  title: "论文阅读：TurboQuant — 近最优失真率的在线向量量化",
  description: "TurboQuant 论文阅读笔记：如何通过随机旋转 + 逐坐标最优标量量化，在 data-oblivious 的在线设置下达到信息论下界 2.7 倍以内的 MSE 和内积失真。",
  tags: ("论文阅读", "LLM", "量化"),
  category: "论文阅读笔记",
  date: datetime(year: 2026, month: 3, day: 28, hour: 16, minute: 0, second: 0),
)

#quote[
  *论文*：TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate \
  *作者*：Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni \
  *机构*：Google Research, New York University, Google DeepMind \
  *链接*：arXiv:2504.19874v1
]

= 一句话总结

TurboQuant 对输入向量做*随机旋转*使每个坐标服从已知的 Beta 分布，然后*逐坐标应用预计算的最优标量量化器*（Lloyd-Max），无需看数据、无需预处理，在任意位宽下 MSE 和内积失真均在信息论下界的 $approx 2.7$ 倍以内。

= 背景：为什么需要在线向量量化

向量量化（VQ）的核心目标是将高维浮点向量压缩为低位宽整数表示，同时尽量保持几何结构（距离和内积）。它在两个场景中至关重要：

- *KV cache 压缩*：LLM 推理时必须缓存所有已生成 token 的键/值向量，内存随上下文长度线性增长。量化 KV cache 是降低内存和通信开销的最直接手段。
- *近邻搜索*：向量数据库需要在数十亿嵌入中做内积/余弦相似度搜索，量化压缩索引大小并加速距离计算。

现有方法存在明显的权衡：

#table(
  columns: (1fr, 2fr, 2fr),
  table.header([类别], [代表方法], [核心缺陷]),
  [离线 / 数据依赖],
  [GPTQ, AWQ, 乘积量化（PQ）],
  [需要校准数据和大量预处理（k-means、Hessian），不适合 KV cache 等动态场景],

  [在线 / 数据无关], [RTN, KIVI], [简单均匀量化，失真率随位宽的改善是*多项式*级别，远非最优],
  [在线 + 理论保证], [QJL, PolarQuant], [QJL 仅支持 1-bit；PolarQuant 仅优化内积，MSE 非最优],
)

TurboQuant 的目标：设计一个*在线、数据无关、加速器友好*的量化器，在*所有位宽*下同时达到 MSE 和内积的*近最优*失真率。

= 问题定义

设计量化映射 $Q: bb(R)^d -> {0, 1}^B$（$B = b dot d$，位宽 $b$）和反量化映射 $Q^(-1): {0, 1}^B -> bb(R)^d$，最小化以下两种失真：

*MSE 失真*（量化前后向量的 L2 距离）：
$
  D_("mse") := bb(E)_Q [norm(x - Q^(-1)(Q(x)))_2^2]
$

*内积失真*（量化前后内积的误差）：
$
  D_("prod") := bb(E)_Q [(angle.l y, x angle.r - angle.l y, Q^(-1)(Q(x)) angle.r)^2]
$

对于内积量化器，还要求*无偏性*：
$
  bb(E)_Q [angle.l y, Q^(-1)(Q(x)) angle.r] = angle.l y, x angle.r
$

#note(
  title: "关于单位范数假设",
)[论文假设 $norm(x)_2 = 1$，这不具限制性：对一般向量先存储 L2 范数（一个浮点数），量化归一化后的向量，反量化时乘回范数即可。]

= 核心方法一：MSE 最优 TurboQuant

== 随机旋转与 Beta 分布

TurboQuant 的第一步是对输入向量 $x in S^(d-1)$ 乘以一个随机旋转矩阵 $Pi in bb(R)^(d times d)$（通过对随机高斯矩阵做 QR 分解生成）。旋转后 $y = Pi dot x$ 均匀分布在单位超球面 $S^(d-1)$ 上。

关键数学事实（引理 1）：$y$ 的每个坐标 $y_j$ 服从 Beta 分布：

$
  f_X (x) = frac(Gamma(d\/2), sqrt(pi) dot Gamma((d-1)\/2)) (1 - x^2)^((d-3)\/2), quad x in [-1, 1]
$

高维时该分布收敛到正态分布 $N(0, 1\/d)$。更重要的是，不同坐标 $y_i$ 和 $y_j$（$i != j$）在高维中*近似独立*。

这两个性质的组合意味着：*可以把 d 维向量量化问题拆解为 d 个独立的一维标量量化问题*，每个坐标的分布已知且相同，大幅简化了算法设计。

== 最优标量量化（Lloyd-Max）

既然每个坐标都服从已知分布 $f_X$，最优标量量化就是一个*连续一维 k-means 问题*：将区间 $[-1, 1]$ 划分为 $2^b$ 个区域，每个区域用一个质心 $c_i$ 代表，使加权 MSE 最小：

$
  C(f_X, b) := min_(c_1 <= c_2 <= dots.c <= c_(2^b)) sum_(i=1)^(2^b) integral_((c_(i-1)+c_i)\/2)^((c_i+c_(i+1))\/2) |x - c_i|^2 dot f_X (x) dif x
$

最优解满足 *Voronoi 划分*：区间边界是相邻质心的中点。这个优化问题通过 Lloyd-Max 迭代算法数值求解，对每个实际位宽 $b$ 只需求解一次，结果预存为码本。

例如在中等维度 $d$ 下，$b = 1$ 的最优码本为 $\{plus.minus sqrt(2 \/ (pi d))\}$，$b = 2$ 的最优码本为 $\{plus.minus 0.453 \/ sqrt(d), plus.minus 1.51 \/ sqrt(d)\}$。

== 算法流程

*量化（Quant_mse）*：
+ 计算 $y = Pi dot x$（随机旋转）
+ 对每个坐标 $j in [d]$，找到最近质心：$"idx"_j = op("argmin")_(k in [2^b]) |y_j - c_k|$
+ 输出索引向量 $"idx" in [2^b]^d$（共 $b dot d$ bits）

*反量化（DeQuant_mse）*：
+ 查表：$tilde(y)_j = c_("idx"_j)$
+ 逆旋转：$tilde(x) = Pi^top dot tilde(y)$
+ 输出重建向量 $tilde(x)$

== 定理 1：MSE 性能保证

对任意位宽 $b >= 1$ 和任意 $x in S^(d-1)$，TurboQuant_mse 的 MSE 失真满足：

$
  D_("mse") <= frac(sqrt(3) pi, 2) dot frac(1, 4^b) approx frac(2.7, 4^b)
$

*推导过程*：

由于 $Pi$ 是正交矩阵，旋转保持 L2 范数不变：
$
  norm(x - tilde(x))_2^2 = norm(Pi x - tilde(y))_2^2 = sum_(j=1)^d |y_j - tilde(y)_j|^2
$

每个坐标 $y_j$ 独立地服从同一分布 $f_X$，且被同一最优标量量化器量化，因此：
$
  D_("mse") = bb(E)[sum_(j=1)^d |y_j - tilde(y)_j|^2] = d dot C(f_X, b)
$

对于 $b <= 4$，数值求解 $C(f_X, b)$ 得到精确值：

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  table.header([位宽 $b$], [$b = 1$], [$b = 2$], [$b = 3$], [$b = 4$]),
  [$D_("mse")$], [$approx 0.36$], [$approx 0.117$], [$approx 0.03$], [$approx 0.009$],
)

对于更大的位宽 $b > 4$，使用 Panter-Dite 高分辨率近似公式：
$
  C(f_X, b) <= frac(1, 12) dot (integral f_X (x)^(1\/3) dif x)^3 dot frac(1, 4^b) = frac(sqrt(3) pi, 2 d) dot frac(1, 4^b)
$

乘以 $d$ 即得整体上界 $D_("mse") <= frac(sqrt(3) pi, 2) dot frac(1, 4^b)$。

= 核心方法二：内积最优 TurboQuant

== 为什么 MSE 量化器对内积有偏

MSE 量化器最小化了 $norm(x - tilde(x))_2^2$，但*不保证内积的无偏性*。以 $b = 1$ 为例，此时量化器将每个坐标映射到 $plus.minus sqrt(2 \/ (pi d))$（即符号函数），可以证明：

$
  bb(E)[angle.l y, Q_("mse")^(-1)(Q_("mse")(x)) angle.r] = frac(2, pi) dot angle.l y, x angle.r
$

内积被系统性地*缩小了 $2 \/ pi approx 0.637$ 倍*。偏差随位宽增加而减小，但在低位宽时不可忽略。

#warning(
  title: "实际影响",
)[如果直接用 MSE 量化器做近邻搜索或注意力计算，所有内积都会被低估，导致 softmax 分布变平、检索召回率下降。]

== QJL：1-bit 无偏内积量化

量化 Johnson-Lindenstrauss（QJL）变换是一种 1-bit 量化方案，*天然无偏*：

$
  Q_("qjl")(x) = "sign"(S dot x) in {-1, +1}^d
$

其中 $S in bb(R)^(d times d)$，$S_(i j) tilde.op N(0, 1)$。反量化：

$
  Q_("qjl")^(-1)(z) = frac(sqrt(pi\/2), d) dot S^top dot z
$

QJL 的性能保证：对任意 $x in S^(d-1)$ 和任意 $y in bb(R)^d$：
- *无偏*：$bb(E)[angle.l y, Q_("qjl")^(-1)(Q_("qjl")(x)) angle.r] = angle.l y, x angle.r$
- *方差*：$"Var"(angle.l y, Q_("qjl")^(-1)(Q_("qjl")(x)) angle.r) <= frac(pi, 2d) dot norm(y)_2^2$

== 两阶段算法

TurboQuant_prod 将 MSE 量化和 QJL 组合为两阶段方案：

+ *阶段 1*：用位宽 $b - 1$ 的 TurboQuant_mse 量化 $x$，最小化残差的 L2 范数。残差 $r = x - Q_("mse")^(-1)(Q_("mse")(x))$ 满足 $norm(r)_2^2 approx D_("mse")(b-1)$。
+ *阶段 2*：对残差 $r$ 应用 1-bit QJL 量化，得到 $"qjl" = "sign"(S dot r)$。

总位宽 = $(b - 1) dot d + 1 dot d = b dot d$ bits。

*量化（Quant_prod）*：
+ $"idx" = "Quant"_("mse")(x)$
+ $r = x - "DeQuant"_("mse")("idx")$
+ $"qjl" = "sign"(S dot r)$
+ 输出 $("idx", "qjl", gamma = norm(r)_2)$

*反量化（DeQuant_prod）*：
+ $tilde(x)_("mse") = "DeQuant"_("mse")("idx")$
+ $tilde(x)_("qjl") = frac(sqrt(pi\/2), d) dot gamma dot S^top dot "qjl"$
+ 输出 $tilde(x) = tilde(x)_("mse") + tilde(x)_("qjl")$

#note(
  title: "直觉理解",
)[MSE 量化器把残差的 L2 范数压得很小，QJL 对这个小残差做 1-bit 无偏估计。残差越小，QJL 的方差越低，最终内积误差就越小。两者各取所长：MSE 量化器负责"压小残差"，QJL 负责"消除偏差"。]

== 定理 2：内积性能保证

对任意位宽 $b >= 1$，任意 $x in S^(d-1)$ 和 $y in bb(R)^d$：

*无偏性*：
$
  bb(E)[angle.l y, tilde(x) angle.r] = angle.l y, x angle.r
$

*失真上界*：
$
  D_("prod") <= frac(sqrt(3) pi^2 dot norm(y)_2^2, d) dot frac(1, 4^b)
$

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  table.header([位宽 $b$], [$b = 1$], [$b = 2$], [$b = 3$], [$b = 4$]),
  [$D_("prod")$ ($norm(y) = 1$)],
  [$approx 1.57 \/ d$],
  [$approx 0.56 \/ d$],
  [$approx 0.18 \/ d$],
  [$approx 0.047 \/ d$],
)

*推导核心*：反量化输出 $tilde(x) = tilde(x)_("mse") + tilde(x)_("qjl")$，内积误差为：
$
  angle.l y, x angle.r - angle.l y, tilde(x) angle.r = angle.l y, r - tilde(x)_("qjl") angle.r
$

由 QJL 的无偏性，$bb(E)[tilde(x)_("qjl")] = r$（条件期望），因此整体无偏。方差项由 QJL 的方差界给出：
$
  D_("prod") = "Var"(angle.l y, tilde(x)_("qjl") angle.r) <= frac(pi, 2d) dot norm(y)_2^2 dot norm(r)_2^2
$

将 $norm(r)_2^2 <= D_("mse")(b-1)$ 代入，结合 MSE 上界即得最终结果。

= 信息论下界

TurboQuant 不仅给出了上界，还证明了任何量化算法都*无法做得更好*的下界，从而说明 TurboQuant 是近最优的。

== Shannon 失真率下界

对于均匀分布在 $S^(d-1)$ 上的随机向量 $x$，Shannon 下界给出：

$
  D(B) >= frac(d, 2 pi e) dot 2^((2\/d)(h(x) - B))
$

其中 $h(x)$ 是微分熵，$B$ 是总 bit 复杂度。代入超球面均匀分布的熵并化简，得到：

$
  D(B) >= 2^(-2B\/d) = frac(1, 4^b) quad (B = b dot d)
$

== 定理 3：下界

利用 Yao 的极大极小原理（将"最坏情况输入上的随机算法"转化为"随机输入上的确定性算法"），证明对*任何*量化算法 $Q$：

$
  D_("mse")(Q) >= frac(1, 4^b), quad D_("prod")(Q) >= frac(norm(y)_2^2, d) dot frac(1, 4^b)
$

== 与上界的对比

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  table.header([], [TurboQuant 上界], [信息论下界], [比值]),
  [$D_("mse")$], [$frac(sqrt(3) pi, 2) dot frac(1, 4^b)$], [$frac(1, 4^b)$], [$frac(sqrt(3) pi, 2) approx 2.72$],
  [$D_("prod")$],
  [$frac(sqrt(3) pi^2, d) dot frac(1, 4^b)$],
  [$frac(1, d) dot frac(1, 4^b)$],
  [$sqrt(3) pi^2 approx 17.1$],
)

对 MSE 而言，TurboQuant 与最优值仅差 $approx 2.7$ 倍。在低位宽时差距更小：$b = 1$ 时实际比值仅约 $1.45$。

#note(
  title: "位宽依赖性的意义",
)[上界和下界都以 $1 \/ 4^b$ 的速率随位宽指数衰减。这意味着每增加 1 bit，失真降低 4 倍。相比之下，简单的均匀量化（RTN）失真仅以 $1 \/ 2^b$ 衰减——TurboQuant 在位宽效率上实现了*指数级改进*。]

= 实验结果

== KV Cache 量化：大海捞针测试

在 Llama-3.1-8B-Instruct 上，将 KV cache 压缩至 25%（4× 压缩），评估 4K\~104K token 长度下的信息检索能力：

#table(
  columns: (1fr, 1fr, 1fr),
  table.header([方法], [KV 大小], [检索性能]),
  [Full Cache], [16 bit], [完美],
  [SnapKV / PyramidKV], [25%], [长序列显著衰减],
  [KIVI], [25%], [部分位置遗漏],
  [PolarQuant], [25%], [接近完美],
  [*TurboQuant*], [*25%*], [*与全精度完全一致*],
)

== KV Cache 量化：LongBench 端到端生成

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  table.header([方法], [KV 大小], [单文档QA], [多文档QA], [摘要], [平均]),
  [Full Cache], [16], [45.29], [45.16], [26.55], [50.06],
  [KIVI], [3], [43.38], [37.99], [27.16], [48.50],
  [PolarQuant], [3.9], [45.18], [44.48], [26.23], [49.78],
  [*TurboQuant*], [*2.5*], [*44.16*], [*44.96*], [*24.80*], [*49.44*],
  [*TurboQuant*], [*3.5*], [*45.01*], [*45.31*], [*26.00*], [*50.06*],
)

3.5-bit TurboQuant 的平均分与 16-bit Full Cache *完全持平*，同时实现了 *4.5×* 压缩。2.5-bit 时仅有轻微下降。

== 近邻搜索

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  table.header([方法], [$d = 200$], [$d = 1536$], [$d = 3072$]),
  [乘积量化（PQ）], [37.04s], [239.75s], [494.42s],
  [RabitQ], [597.25s], [2267.59s], [3957.19s],
  [*TurboQuant*], [*0.0007s*], [*0.0013s*], [*0.0021s*],
)

TurboQuant 的量化时间比 PQ 快 $approx 10^5$ 倍，因为不需要运行 k-means 构建码本。在召回率上，TurboQuant 也持续优于 PQ 和 RabitQ。

= 总结

TurboQuant 的核心洞察极为简洁：*随机旋转消除最坏情况输入，将向量量化问题归约为已知分布上的标量量化*。这一设计同时满足了三个通常互相矛盾的目标：

+ *近最优失真率*：MSE 和内积失真均在信息论下界的常数倍以内，位宽效率以 $1 \/ 4^b$ 指数衰减。
+ *在线 / Data-oblivious*：无需校准数据、无需预训练码本，对 KV cache 等动态生成的向量可以即时应用。
+ *加速器友好*：核心操作仅有矩阵乘法（旋转）和逐元素查表（量化），天然支持 GPU 向量化。

与 PolarQuant 等同期工作相比，TurboQuant 的独特优势在于*同时优化了 MSE 和内积两个失真度量*，并通过两阶段设计（MSE 量化 + 残差 QJL）巧妙地解决了 MSE 量化器的内积偏差问题。其理论贡献——信息论下界的严格证明——也为未来量化算法的设计提供了明确的性能基准。
