#import "../../config.typ": *

#set page(height: auto, width: 24cm)
#set text(16pt, font: "IBM Plex Sans SC", lang: "zh")
#show raw: text.with(font: ("Zed Plex Mono", "IBM Plex Sans SC"))
#show math.equation: set text(16pt)
#set table(inset: 9pt)

#show: template-post.with(
  title: "论文阅读：Mamba-3 — 从 SSM 原理出发同时改善质量、能力与效率",
  description: "Mamba-3 论文深度阅读笔记：指数梯形离散化、复值 SSM（数据依赖 RoPE）、MIMO 三大创新如何同时解决次二次方模型的质量、状态跟踪能力和推理效率问题。",
  tags: ("论文阅读", "LLM", "SSM"),
  category: "论文阅读笔记",
  date: datetime(year: 2026, month: 3, day: 29, hour: 19, minute: 0, second: 0),
)

#quote[
  *论文*：Mamba-3: Improved Sequence Modeling using State Space Principles \
  *作者*：Aakash Lahoti\*, Kevin Y. Li\*, Berlin Chen\*, Caitlin Wang\* 等 \
  *机构*：Carnegie Mellon University, Princeton University, Together AI, Cartesia AI \
  *链接*：arXiv:2603.15569v1
]

= 一句话总结

Mamba-3 从 SSM 理论视角出发，通过三项创新——*指数梯形离散化*（更精确的递推）、*复值 SSM*（解锁旋转动态 = 数据依赖 RoPE）、*MIMO*（免费提升推理算力利用率）——同时解决了次二次方模型在质量、能力和效率上的三大瓶颈。1.5B 规模下比 Transformer 高 2.2 个百分点，以一半状态大小匹配 Mamba-2 的困惑度。

= 前置知识：SSM 与 Mamba-1/2 回顾

== SSM 的核心思想

状态空间模型用*固定大小的隐藏状态*替代 Transformer 的 KV cache 来压缩历史信息。离散化后的递推关系：

$
  bold(h)_t = alpha_t bold(h)_(t-1) + gamma_t bold(B)_t x_t, quad y_t = bold(C)_t^top bold(h)_t
$

推理时每步只需更新固定大小的状态 $bold(h)_t in bb(R)^(N times P)$，内存 $O(1)$，计算 $O(1)$——不像 Transformer 要 attend 所有历史 token。

== Mamba-1：选择性机制

Mamba-1 让 $bold(B)_t$, $bold(C)_t$, $Delta_t$ 都由当前 token 投影得到（数据依赖），使模型能根据输入内容动态决定记忆策略。$Delta_t$ 大 → 遗忘旧状态、强写入当前；$Delta_t$ 小 → 保留历史、忽略当前。

== Mamba-2：状态空间对偶（SSD）

Mamba-2 发现 SSM 和线性注意力是同一件事：$bold(Y) = (bold(L) dot.o bold(C) bold(B)^top) bold(X)$，其中 $bold(B)$ = Key，$bold(C)$ = Query，$bold(X)$ = Value，$bold(L)$ = 带衰减的结构化掩码。为了用 GPU 矩阵乘法加速训练，$bold(A)_t$ 被简化为标量乘单位矩阵。

== Mamba-2 留下的三个问题

#table(
  columns: (1fr, 2fr),
  table.header([问题], [原因]),
  [质量退化], [为训练效率牺牲了表达力；Mamba-2 的离散化是一阶近似，缺乏理论证明],
  [状态跟踪能力缺失], [标量转移 $alpha_t in (0, 1)$ 只能衰减，无法表达旋转动态 → 奇偶性任务准确率 0.9%],
  [推理 GPU 利用率极低], [解码算术强度仅 ~2.5 ops/byte，H100 的矩阵乘法峰值为 ~295 ops/byte，99% 算力闲置],
)

Mamba-3 的三项创新分别对应解决这三个问题。

= 创新一：指数梯形离散化

== 问题：Mamba-1/2 的离散化缺乏理论依据

SSM 需要将连续 ODE 离散化。Mamba-1 声称用了 ZOH，但实际实现用了一个额外近似——Mamba-3 论文首次为其提供了理论证明，并将其命名为*指数-欧拉*（Exponential-Euler）。

核心推导思路：先用指数调整精确处理状态转移 $e^(Delta_t A_t)$，再用不同数值方法近似*状态输入积分*：

$
  bold(h)_t approx e^(Delta_t A_t) bold(h)_(t-1) + integral_(tau_(t-1))^(tau_t) e^((tau_t - tau)A_t) bold(B)(tau) x(tau) d tau
$

== 指数-欧拉（一阶，Mamba-1/2）

用欧拉法（矩形法则）近似积分，取右端点值乘区间宽度：

$
  bold(h)_t = e^(Delta_t A_t) bold(h)_(t-1) + Delta_t bold(B)_t x_t
$

一阶近似，误差 $O(Delta_t^2)$。这正是 Mamba-1 和 Mamba-2 实际使用的公式。

== 指数梯形（二阶，Mamba-3）

用广义梯形法则近似积分——取两个端点的*数据依赖加权平均*：

$
  bold(h)_t = underbrace(e^(Delta_t A_t), alpha_t) bold(h)_(t-1) + underbrace((1 - lambda_t) Delta_t e^(Delta_t A_t), beta_t) bold(B)_(t-1) x_(t-1) + underbrace(lambda_t Delta_t, gamma_t) bold(B)_t x_t
$

其中 $lambda_t in [0, 1]$ 是数据依赖参数。二阶近似，误差 $O(Delta_t^3)$。

对比 Mamba-2 的两项递推，Mamba-3 变成了*三项递推*——多了 $beta_t bold(B)_(t-1) x_(t-1)$，上一步的输入也参与当前状态更新。

== 特殊情况

#table(
  columns: (1fr, 2fr),
  table.header([$lambda_t$], [结果]),
  [$lambda_t = 1$], [$beta_t = 0$，退化为 Mamba-2 的欧拉方法],
  [$lambda_t = 1\/2$], [经典梯形法则（两端点等权平均）],
  [$lambda_t$ 自由学习], [Mamba-3 默认设置，最大表达力],
)

== 隐式宽度-2 卷积

三项递推等价于先对状态输入 $bold(v)_t = bold(B)_t x_t$ 做*宽度-2 的数据依赖卷积*，再送入标准线性递推：

$
  tilde(bold(v))_t = beta_t bold(v)_(t-1) + gamma_t bold(v)_t, quad bold(h)_t = alpha_t bold(h)_(t-1) + tilde(bold(v))_t
$

这与 Mamba-1 中在递推*外部*对原始输入做的短因果卷积不同——Mamba-3 的卷积在递推*内部*对状态输入做。

== 淘汰短卷积

Mamba-3 还在 $bold(B)$, $bold(C)$ 上添加了可学习的头特定偏置（初始化全 1），引入数据无关分量。*指数梯形的隐式卷积 + BC 偏置*的协同效应使短因果卷积变得多余：

#table(
  columns: (2fr, 1fr),
  table.header([配置], [困惑度 ↓]),
  [Mamba-3 无偏置、无梯形], [16.68],
  [Mamba-3 无偏置], [16.49],
  [*Mamba-3（完整）*], [*15.72*],
  [Mamba-3 + 短卷积], [15.85（加了反而更差）],
)

== SSD 并行形式

Mamba-3 仍是 SSD 的实例。掩码 $bold(L)$ 变为 1-半可分矩阵与 *2-带矩阵*（而非 Mamba-2 的对角矩阵）的乘积：

$
  bold(L) = underbrace(mat(1; alpha_1, 1; alpha_2 alpha_1, alpha_2, 1; dots.v, , dots.down), "半可分（衰减）") dot underbrace(mat(gamma_0; beta_1, gamma_1; 0, beta_2, gamma_2; , , dots.down), "2-带（梯形）")
$

= 创新二：复值 SSM 与数据依赖 RoPE

== 问题：实值 SSM 无法做旋转

Mamba-2 的 $alpha_t in (0, 1)$ 只能让状态*单调衰减*，无法表达*旋转*。而奇偶性任务需要"遇到 1 就翻转状态"：

$
  bold(h)_t = R(pi x_t) bold(h)_(t-1), quad R(theta) = mat(cos theta, -sin theta; sin theta, cos theta)
$

旋转矩阵 $R(pi)$ 的特征值为 $\{-1, -1\}$——*负数*，Mamba-2 的 $(0, 1)$ 范围无法表达。

== 解法：复值 SSM

Mamba-3 从复值 SSM 出发：$bold(A)(t) + i bold(theta)(t)$ 的虚部 $bold(theta)$ 编码旋转角度。

*命题 2*：离散化后，复值 SSM 等价于实值 SSM + 块对角旋转矩阵：

$
  bold(h)_t = e^(Delta_t A_t) dot bold(R)_t dot bold(h)_(t-1) + Delta_t bold(B)_t x_t
$

其中 $bold(R)_t$ 由 $2 times 2$ 旋转矩阵 $R(Delta_t theta_t^((i)))$ 组成的块对角矩阵。

== RoPE 技巧

直接在递推中乘 $bold(R)_t$ 需要 $O(N^2)$ 的矩阵-向量乘法。*命题 3* 证明旋转可以"挪"到 $bold(B)$, $bold(C)$ 上：

$
  bold(h)_t = e^(Delta_t A_t) bold(h)_(t-1) + (product_(i=0)^t bold(R)_i^top) Delta_t bold(B)_t x_t
$

$
  y_t = [(product_(i=0)^t bold(R)_i^top) bold(C)_t]^top bold(h)_t
$

因为 $bold(R)_t$ 是块对角的 $2 times 2$ 旋转，累积旋转只需*角度相加* $theta_("累积") += theta_t$，然后对 $bold(B)$, $bold(C)$ 的每对维度做 cos/sin 旋转——整个操作 $O(N)$，而非 $O(N^2)$。

#table(
  columns: (1fr, 1fr, 1fr),
  table.header([], [标准 RoPE], [Mamba-3 RoPE]),
  [旋转角度], [固定频率 $theta[i] = 10000^(-2i\/N)$], [数据依赖的 $Delta_t theta_t^((i))$],
  [作用对象], [Q 和 K], [C 和 B（= Q 和 K）],
  [依赖什么], [只跟位置有关], [跟输入内容和位置都有关],
)

#note(
  title: "首个有理论动机的数据依赖 RoPE",
)[标准 RoPE 的旋转角度是固定的，无法实现"遇到 1 旋转 180°、遇到 0 不旋转"这样的条件逻辑。Mamba-3 的 $theta_t$ 由当前 token 投影产生，旋转角度根据输入动态调整。]

== 状态跟踪实验

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  table.header([任务], [Mamba-3], [Mamba-3（标准 RoPE）], [Mamba-3（无 RoPE）], [Mamba-2]),
  [奇偶性], [*100.0%*], [1.6%], [2.3%], [0.9%],
  [模运算（无括号）], [*98.5%*], [20.7%], [1.5%], [47.8%],
  [模运算（有括号）], [87.8%], [2.6%], [0.7%], [0.9%],
)

数据依赖 RoPE 是关键：标准 RoPE 几乎不起作用，因为旋转角度必须依赖输入内容才能实现条件翻转。

= 创新三：MIMO

== 问题：GPU 算力严重闲置

SSM 解码的核心操作是状态更新 $bold(h)_t = alpha_t bold(h)_(t-1) + Delta_t bold(B)_t bold(x)_t^top$。瓶颈在于读写状态 $bold(h)_t in bb(R)^(N times P)$ 的*内存带宽*，而外积 $bold(B)_t bold(x)_t^top$ 的计算量太小，GPU 的矩阵乘法单元几乎全部闲置。

算术强度仅 ~2.5 ops/byte，而 H100 峰值为 ~295 ops/byte。

== 解法：外积升级为矩阵乘法

把 $bold(B)_t in bb(R)^N$ 升级为 $bold(B)_t in bb(R)^(N times R)$，$bold(x)_t in bb(R)^P$ 升级为 $bold(X)_t in bb(R)^(P times R)$：

$
  "SISO:" quad bold(B)_t bold(x)_t^top in bb(R)^(N times P) quad "(rank 1 外积)" \
  "MIMO:" quad bold(B)_t bold(X)_t^top in bb(R)^(N times P) quad "(rank " R " 矩阵乘法)"
$

#table(
  columns: (1fr, 1fr, 1fr),
  table.header([特性], [SISO], [MIMO（秩 $R$）]),
  [算术强度], [$Theta(1) approx 2.5$ ops/byte], [$Theta(R)$（提升 $R$ 倍）],
  [解码 FLOPs], [$5 N P$], [$4 N P R + N P$],
  [状态大小], [$N times P$], [$N times P$（*不变*）],
  [实际延迟], [基线], [*几乎不变*],
)

== 为什么延迟几乎不变

操作是*内存受限*的：瓶颈在于读写状态 $bold(h)_t$（$N times P$ 个数），这个量不变。增加的 $R$ 倍计算可以叠加在内存操作的等待时间上，GPU 计算单元原本就是空闲的。

类比：开卡车运货，路程（内存带宽）是 10 分钟，SISO 搬 1 箱（1 分钟），MIMO 搬 4 箱（4 分钟）。总时间从 21 分钟到 24 分钟，多了 14%，但运了 4 倍的货。

== 为什么能提升模型质量

SISO 每步向状态写入 rank-1 的更新（"一维信息"），MIMO 写入 rank-$R$ 的更新（"$R$ 维信息"）。状态大小不变但每步写入的信息更丰富，压缩质量更高。

== 训练策略

MIMO 可分解为 $R^2$ 个 SISO 并行计算。通过设块大小 $C_("MIMO") = C_("SISO") \/ R$，训练开销从 $R^2$ 倍降到 $R$ 倍。实测 $R = 4$ 时仅 *2 倍训练减速*。

MLP 宽度略减以保持参数匹配（1.5B 模型仅减 6.6%）。

= 完整 Mamba-3 架构

整体遵循 Llama 风格，交替 Mamba-3 块 + SwiGLU 块 + 预归一化。

关键架构决策：

- *BC/QK 归一化*：$bold(B)$, $bold(C)$ 投影后加 RMSNorm，稳定训练并移除 Mamba-2 的门控后归一化
- *BC 偏置*：头特定通道偏置引入数据无关分量，与指数梯形协同替代短卷积
- *无短卷积、无激活函数*：Mamba-1/2 中被认为必需的短因果卷积和 SiLU 完全被移除
- *两个变体*：SISO（默认，公平比较用）和 MIMO（$R = 4$，更强但训练略慢）

= 实验结果

== 1.5B 下游准确率

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  table.header([模型], [困惑度 ↓], [HellaSwag], [PIQA], [Arc-C], [*平均*]),
  [Transformer], [10.51], [60.6], [73.8], [40.4], [55.4],
  [GDN], [10.45], [61.3], [74.3], [41.2], [55.8],
  [Mamba-2], [10.47], [61.4], [73.6], [41.8], [55.7],
  [*Mamba-3 SISO*], [10.35], [61.9], [73.6], [42.7], [*56.4*],
  [*Mamba-3 MIMO*], [*10.24*], [*62.3*], [*75.3*], [*44.5*], [*57.6*],
)

Mamba-3 MIMO 比 Transformer 高 *+2.2*，比 Mamba-2 高 *+1.9* 个百分点。

== 推理内核延迟

#table(
  columns: (1fr, 1fr, 1fr),
  table.header([模型], [BF16, $d_("state")=64$], [BF16, $d_("state")=128$]),
  [Mamba-2], [0.127 ms], [0.203 ms],
  [GDN], [0.176 ms], [0.257 ms],
  [*Mamba-3 SISO*], [*0.110 ms*], [*0.156 ms*],
  [Mamba-3 MIMO ($R=4$)], [0.137 ms], [0.179 ms],
)

SISO 是所有模型中最快的。MIMO 增加了 $R$ 倍 FLOPs 但延迟只增约 15%。

== 性能-效率权衡

状态大小 64 的 Mamba-3 MIMO 匹配状态大小 128 的 Mamba-2 的困惑度——即*以一半延迟达到相同性能*。

= 总结

Mamba-3 的三项创新都来自 SSM 的理论视角，且不是从线性注意力或测试时回归等其他视角能自然想到的：

+ *指数梯形离散化*：首次为 Mamba-1/2 的启发式公式提供理论证明，并推广为二阶精确方法，引入隐式状态输入卷积，配合 BC 偏置淘汰了短因果卷积
+ *复值 SSM*：通过复值状态转移实现旋转动态，等价于数据依赖 RoPE，解锁了 SSM 的状态跟踪能力（奇偶性从 0.9% → 100%）
+ *MIMO*：将 SISO 外积升级为矩阵乘法，在不增加状态大小和解码延迟的前提下提升 $R$ 倍 FLOPs 和建模能力

三者的协同使 Mamba-3 在质量（+2.2 vs Transformer）、能力（解锁状态跟踪）和效率（一半状态大小匹配 Mamba-2）上全面推进了性能-效率帕累托前沿。
