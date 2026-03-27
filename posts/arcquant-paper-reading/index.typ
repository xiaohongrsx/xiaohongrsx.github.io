#import "../../config.typ": *

#set page(height: auto, width: 24cm)
#set text(16pt, font: "IBM Plex Sans SC", lang: "zh")
#show raw: text.with(font: ("Zed Plex Mono", "IBM Plex Sans SC"))
#show math.equation: set text(16pt)
#set table(inset: 9pt)

#show: template-post.with(
  title: "论文阅读：ARCQuant — 用增强残差通道提升 NVFP4 量化精度",
  description: "ARCQuant 论文阅读笔记：如何在统一 NVFP4 精度下，通过两次量化达到 FP8 级别的精度，同时保持硬件兼容性。",
  tags: ("论文阅读", "LLM", "量化"),
  category: "论文阅读笔记",
  date: datetime(year: 2026, month: 3, day: 27, hour: 14, minute: 0, second: 0),
)

#quote[
  *论文*：ARCQuant: Boosting NVFP4 Quantization with Augmented Residual Channels for LLMs \
  *作者*：Haoqian Meng, Yilun Luo, Yafei Zhao, Wenyuan Liu, Peng Zhang, Xindian Ma \
  *机构*：天津大学计算机科学与技术学院 \
  *链接*：arXiv:2601.07475v1 \
  *代码*：https://github.com/actypedef/ARCQuant
]

= 一句话总结

ARCQuant 发现*两次 FP4 量化的精度平方等于一次 FP8 量化的精度*（$epsilon_4^2 = epsilon_8$），据此设计了一种将量化残差沿规约维度拼接到矩阵乘法中的方案，在统一 NVFP4 精度下以约 3%\~9% 的额外计算换取接近 W4A8 级别的精度。

= 背景：为什么 W4A4 在 NVFP4 上很难

LLM 推理的主要瓶颈是内存带宽。将权重和激活都量化到 4 比特（W4A4）可以最大化吞吐量，但激活中的*异常值*（少数通道数值特别大）使 4 比特表示精度崩溃。

NVIDIA Blackwell 架构引入了 *NVFP4 格式*：每 16 个 E2M1 元素共享一个 E4M3 缩放因子。其核心优势是*分块隔离*——异常值只影响自己所在的 16 元素块，不会拉垮整个张量。

然而现有 PTQ 方法在 NVFP4 上全部失效：

#table(
  columns: (1fr, 2fr, 2fr),
  table.header([方法], [代表工作], [为什么不行]),
  [旋转], [QuaRot], [Hadamard 变换将异常值的大数值扩散到所有维度，破坏分块隔离，反而增大局部动态范围],
  [平滑], [SmoothQuant], [将激活难度转嫁给权重，但 4 比特权重容量太小，补偿微弱],
  [混合精度], [Atom], [异常值用 FP8 保护，但 NVFP4 块大小为 16、FP8 块大小为 32，Tensor Core 不支持混用],
)

#note(
  title: "实验验证",
)[论文实验显示，QuaRot 在 Llama 3.1-8B 的 NVFP4 量化上甚至*低于不做任何优化的 RTN*，直接证实了旋转方法破坏分块隔离的论断。]

= 核心方法：增强残差通道

== 整体流程

ARCQuant 的核心思想是将误差补偿嵌入矩阵乘法的规约维度，实现"一次 GEMM 完成主计算 + 残差校正"：

+ *通道重排序*：按绝对最大值排序激活通道，将异常值集中到前 $S$ 个位置。
+ *主量化*：对重排序后的 $X$ 进行 NVFP4 分块量化，得到 $Q_X$ 和 $s_X$。
+ *残差计算*：仅对前 $S$ 个异常值通道计算残差 $R_o = X_o - s_(X_o) dot Q_(X_o)$。
+ *残差量化*：对 $R_o$ 再次 NVFP4 量化，得到 $Q_(R_o)$ 和 $s_(R_o)$。
+ *增强拼接*：沿 $K$ 维拼接 $Q_(X_("aug")) = [Q_X | Q_(R_o)]$，维度从 $K_("in")$ 扩展到 $K_("in") + S$。
+ *统一 GEMM*：单次 NVFP4 GEMM 完成全部计算。

权重侧对应操作：重排序后*直接复制*异常值权重列（不计算残差），构造 $Q_(W_("aug")) = [Q_W | Q_(W_o)]$。

== 自适应异常值选择

阈值设为 $tau = 2^(-3) M$（$M$ 为逐层动态范围最大值），即 $M \/ 8$。这个 $8 = 2^3$ 来自 FP8（E5M2，5 位指数）和 FP4（E2M1，2 位指数）的指数位宽差距：$5 - 2 = 3$ 位。

- 通道最大值 $< tau$：FP4 精度已足够，不需要补偿
- 通道最大值 $>= tau$：FP4 精度显著不足，需要残差补偿

$S$ 逐层自适应确定，典型范围 $S <= 512$，不同模型和层的差异显著。

== 数学等价性：为什么一次 GEMM 就够

原始矩阵乘法 $Y = X W^top$，ARCQuant 将其转换为增强运算 $(N, K_("in") + S, M)$：

$
  Y approx Q(X) Q(W)^top + Q(R_o) Q(W_o)^top = s_(X_("aug")) dot Q_(X_("aug")) (s_(W_("aug")) dot Q_(W_("aug")))^top
$

由于矩阵乘法的累加是线性的，GEMM 在扩展的规约维度上运算时，前 $K_("in")$ 个通道贡献主计算，后 $S$ 个通道贡献残差校正，累加器自动将两部分求和。

关键：增大的是 GEMM 的*规约维度*（$K$ 维），不是输出维度。GEMM 输出的形状与原始完全相同，后续的 Attention、LayerNorm、残差连接等操作不需要任何修改。

以 Qwen2.5-7B 的一个线性层为例：

```
原始：X [batch, 4096] × W [hidden, 4096]   → K = 4096
增强：X [batch, 4224] × W [hidden, 4224]   → K = 4096 + 128 = 4224
输出：Y [batch, hidden]                     → 形状不变
额外计算量 = 128 / 4096 ≈ 3%
```

== 为什么 Attention 不受影响

ARCQuant 作用在每个线性层（QKV 投影、输出投影、FFN）的 GEMM 内部。以 QKV 投影为例：

```
Q = X_aug @ W_Q_aug^T  →  输出维度不变
K = X_aug @ W_K_aug^T  →  输出维度不变
V = X_aug @ W_V_aug^T  →  输出维度不变
```

Q、K、V 的形状与原来完全一致，后续的 $Q K^top arrow.r "softmax" arrow.r dot V$ 注意力计算无需改动。增强维度在 GEMM 的求和过程中被"消化"掉了。

= 理论保证：两次 FP4 $approx$ 一次 FP8

核心等式：$epsilon_4^2 = epsilon_8$（NVFP4 精度限制 $epsilon_4 = 2^(-2)$，MXFP8 精度限制 $epsilon_8 = 2^(-4)$）。

对于 MXFP8 单阶段量化（E8M0 缩放因子，仅指数）：
$
  B_("mx") = alpha_("mx") M epsilon_8 < 2 M epsilon_8
$

对于 ARCQuant 双阶段 NVFP4 量化（E4M3 缩放因子，含尾数）：
$
  |e_("arc")| <= s_2 epsilon_4 <= (alpha_2 alpha_1 M epsilon_4) epsilon_4 = (alpha_1 alpha_2) M epsilon_8 = B_("arc")
$

其中 $sup alpha_1 alpha_2 = 1.125^2 approx 1.266$。由于 $1.266 < 2$，ARCQuant 的误差界*严格优于* MXFP8。

#note(
  title: "直觉理解",
)[两次 FP4 量化（第二次量化第一次的残差）的总精度与一次 FP8 量化可比。ARCQuant 本质上是在 W4A4 的硬件约束下达到了 W4A8 级别的精度。]

= 融合核函数设计

为避免在线残差计算的延迟开销，ARCQuant 实现了融合量化核函数，将以下操作整合为单一 CUDA kernel：

```
通道重排序 → RMSNorm → 主量化 → 残差量化 → 交错通道布局输出
```

关键设计：
- *交错通道布局*：主量化块与对应残差块在物理内存中交错排列（每 16 通道一对），避免跨距访问
- *输出严格 NVFP4 格式*：后续直接调用标准 CUTLASS GEMM 核函数，无需定制矩阵乘

= 实验结果

== 精度对比

ARCQuant 在所有 W4A4 方法中取得最佳精度，且超过 W4A8+RTN 基线：

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  table.header([模型], [方法], [0-shot 平均], [WikiText2 PPL], [MMLU 5-shot]),
  [Llama 3.1-8B], [FP16], [72.56], [6.24], [65.15],
  [], [Atom], [67.74], [7.52], [59.27],
  [], [FlatQuant], [70.51], [6.95], [61.33],
  [], [*ARCQuant*], [*70.90*], [*6.87*], [*62.61*],
  [Qwen2.5-7B], [FP16], [70.97], [6.85], [74.16],
  [], [Atom], [67.57], [8.96], [68.17],
  [], [*ARCQuant*], [*70.28*], [*7.28*], [*72.84*],
  [Qwen2.5-32B], [FP16], [74.82], [5.02], [83.26],
  [], [*ARCQuant*], [*74.80*], [*5.38*], [*82.61*],
)

Qwen2.5-32B 上实现近无损压缩（0-shot 平均仅差 0.02）。

== 效率

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  table.header([平台], [模型], [相比 FP16 加速], [内存节省], [相比裸 NVFP4 额外延迟]),
  [RTX 5090], [Llama 3.1-8B], [3.5x], [2.8x], [3%\~9%],
  [PRO 6000], [Qwen2.5-7B], [2.0x\~2.5x], [1.5x], [\~4.9%],
)

GEMM 延迟与增强通道数 $S$ 呈严格线性关系，在典型范围（$S <= 512$）内开销可忽略。

== 鲁棒性

- *跨格式泛化*：ARCQuant 在 INT4 和 MXFP4 上同样优于 RTN
- *校准鲁棒性*：更换校准集（WikiText2 / C4 / HumanEval）后，PPL 和 0-shot 准确率波动均 $< 0.03$
- *跨域迁移*：使用文本域校准的模型在编程任务上保留 $> 99%$ FP16 精度

= 总结

ARCQuant 提出了一种优雅的思路解决细粒度量化中的异常值问题：不改变数据格式、不做全局变换，而是在统一精度下"以极小的维度扩展换取精度恢复"。其核心洞察——$epsilon_4^2 = epsilon_8$——使得两次 NVFP4 量化在理论上可达到一次 MXFP8 的分辨率。通过将残差通道嵌入 GEMM 的规约维度，整个补偿过程在一次标准矩阵乘法中完成，无需混合精度、无需定制核函数、无需后处理。这一原则具有良好的可扩展性——随着新的细粒度格式出现，只需更新融合量化核函数即可复用整体框架。
