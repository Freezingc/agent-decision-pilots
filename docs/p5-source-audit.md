# P5 来源核查：工具写入超时后的主动状态诊断

## 核查护照

- 日期：2026-10-20（Asia/Shanghai；检索开始时为 2026-10-19 UTC），资料截止本次核查，不使用之后资料。
- 状态：**ANALYZED / 有界来源核查**。按 ARS runtime policy 与 academic-search 执行；原生子代理承担独立任务，但共享主线程背景，属于 same-context native audit，不是跨模型盲审、系统综述或实验复现。
- 范围：阅读下列一手论文的方法段和官方仓库；未运行其代码，未使用 GPU、付费接口或外部消息服务。引用量未查询；不能由“本次未找到”推出不存在。
- 本文件不报告 P5 实验结果，不修改原有实验协议。

## 结论与最直接的重叠

**重叠风险高。不能把“融合多信号形成置信度，再选择检查、等待、重试”作为新贡献。** [Verified Tool Calls（VTC）§8](https://arxiv.org/html/2608.02645v1#S8) 已明确提出这一未来工作；它不是本次推断出的潜在延伸。其已实现部分还有三值只读验证、退避后复核与幂等重试。

本次已读集合中，尚未核实一个同时实现“有真实终局审计的超时日志 → 区分执行/可见性延迟 → 学习相关多通道分布 → 选择检查/等待/重试”的完整系统。这个较窄的交集仍可研究，但**各个组成部分已有直接近邻**；把它写成 POMDP 或加上“右删失”不自动构成算法创新。更先决的问题是日志是否足以识别这些状态，以及优势能否超过廉价状态接口与历史分位数轮询。

## 已阅读的一手近邻

| 来源、元数据及阅读位置 | 已覆盖的机制 | 尚未由本次阅读核实的部分、代码状态 |
|---|---|---|
| Mansoor、Phadke、Rana，*Verified Tool Calls Improve LLM Agent Reliability Under Non-Atomic Failures*，2026，[arXiv:2608.02645v1 全文](https://arxiv.org/html/2608.02645v1)。读 §3.3、§4.4–4.7、§5、§7–8；HTML §8 第 369–370 行是最直接段落。 | 已有超时后先验证、True/False/Unknown、退避、同逻辑操作幂等键。§8 明确提出融合后置条件、工具响应、幂等信息和历史，按置信度选立即重试或等待再验证。 | §8 是方向声明，未见删失似然、联合延迟学习或主动选择检查通道。§4.6 基线省略幂等键，不能直接当新工作的唯一强对照。未从正文定位作者代码。页面 arXiv 版本头为 7 月 31 日，文内又标 8 月 24 日；日期呈现不一致，不据此推断优先权。 |
| Yifei She 等，*Learning to Wait: Synchronizing Agents with the Physical World*，2025-12-18，[arXiv:2512.16262v1 全文](https://arxiv.org/html/2512.16262v1)。读 §2–3、局限、附录 A/D。 | 已启动任务只有 PENDING/DONE；用历史等待时长、检查次数调整下次等待，权衡检查成本与等待延迟。是“学习何时再查”的直接近邻，不可只比较固定轮询。 | 实验是受控 Gamma 完成时间、短程 ICL；未见永久失败/已提交但不可见的区分、多检查渠道联合学习或正式删失估计。未从论文核实官方代码。 |
| Wenzhi Chen 等，*ReliHarness: A Self-Learning Reliability Harness for LLM Agent Tool Execution*，Electronics 15(18):4166，2026-09-14，[出版方正文](https://www.mdpi.com/2079-9292/15/18/4166)，[DOI](https://doi.org/10.3390/electronics15184166)。读出版方可返回的 §3.3.2、Algorithm 1、结论；日期由版本页及 DOI 元数据交叉核对。 | 心跳和 OS 线程存活两个渠道共同区别运行、僵死、死亡；检测循环按固定时序同时查两者，已有多信号进程诊断。 | 进程活着不证明业务写入已提交；文中 self-learning 主要是命令匹配/纠错，未核实删失后验或成本驱动的渠道选择。后续正文/PDF 请求遇 429，未读全篇、未核实代码，不将检索摘要代替其余方法。 |
| Alexander Tornede、Viktor Bengs、Eyke Hüllermeier，*Machine Learning for Online Algorithm Selection under Censored Feedback*，AAAI 2022，36(9):10370–10380，[论文页](https://ojs.aaai.org/index.php/AAAI/article/view/21279)，[原始 PDF](https://cdn.aaai.org/ojs/21279/21279-13-25292-1-2-20220628.pdf)，[DOI](https://doi.org/10.1609/aaai.v36i9.21279)。读 §1–2、§4–5.3、§7 代码脚注。 | 已把 timeout-aborted runtime 作为右删失反馈，以运行时间概率模型和在线决策处理，讨论直接拿截止值代替真实运行时间的偏差。 | 对象是选择运行算法，不是一次持续外部写入的多通道可见性；其 §4 的独立算法模型没有解决这里的渠道依赖。作者脚注链接的 [online_as](https://github.com/alexandertornede/online_as) 已核实仓库和 GPL-3.0，未执行。所查 HEAD：`c5ebf8d2a7493efd5ecfd0fddbae002090249ace`。 |
| Alexander Tornede 等，*Run2Survive: A Decision-theoretic Approach to Algorithm Selection based on Survival Analysis*，ACML 2020，PMLR 129:737–752，[论文页及 PDF](https://proceedings.mlr.press/v129/tornede20a.html)。读摘要与 PDF §2.2 的删失/生存分析说明，未精读全部实验。 | 生存分析结合决策效用选择算法，已覆盖“删失运行时间分布服务于风险敏感决策”的组合。 | 不能因此认定已解决未知提交状态、相关异步读或重试造成的状态变化；本次未另核验其独立代码。 |
| Sirin Nitinawarat、Venugopal V. Veeravalli，*Controlled Sensing for Sequential Multihypothesis Testing with Controlled Markovian Observations and Non-Uniform Control Cost*，2013 预印本／Sequential Analysis 34(1):1–24，2015，[原始 PDF](https://arxiv.org/pdf/1310.1844)，[DOI](https://doi.org/10.1080/07474946.2014.961864)。读 PDF §3、§5（尤其 pp.5–6、10–11）。 | 不同成本的传感动作、受控 Markov 相关观测、顺序辨识和停止已有理论；“有相关性且检查成本不同”本身不是空白。 | 模型假定给定各假设下的转移概率；其 self-tuning 指自适应未知假设，不等于从删失日志学习未知传感器规律。未核实官方代码。 |
| Bardia Mohammadi 等，*Atomix: Timely, Transactional Tool Use for Reliable Agentic Workflows*，2026-02-16，[arXiv:2602.14849v1](https://arxiv.org/html/2602.14849v1)。读 §3–5，重点 §3.3、§4.2、§5 limitations。 | 已有工具副作用事务、进度 frontier、补偿、幂等去重与故障重放。需要正面对照更强的执行语义，而非只和盲目 retry 比较。 | 依靠适配器报告进度，不是从不完整观察估计提交概率；单进程内存去重不保证 crash-safe。论文明确链接的 [mpi-dsg/atomix](https://github.com/mpi-dsg/atomix) 已读 README/目录，Apache-2.0；未执行，不能声称可直接复现实验。 |

补充边界：[Dynamic Observation Policies in Observation Cost-Sensitive Reinforcement Learning](https://arxiv.org/html/2307.02620)（Bellinger、Crowley、Tamblyn；已读 §1–4）已研究何时付费观察及动作重复；其所述 AC-NOMDP 的测量是完整无噪声状态，不能直接替代本问题的有噪声、滞后渠道模型。这里没有重做一般 robust POMDP／active diagnosis 的完整综述。

## 必须保留的工程基线

1. **客户端稳定 operation ID + 状态端点 + 服务端幂等。** [Microsoft Azure API Guidelines，固定 commit](https://github.com/microsoft/api-guidelines/blob/a7022a299442a8352431874e63ec4dff548a1b81/azure/Guidelines.md#long-running-operations--jobs) 的 LRO initiation、Status Monitor、Obtaining status 三节：客户端可指定 ID；同 ID 相同请求视作重试、不同请求冲突；GET 状态包含运行/成功/失败/取消；服务端用 Retry-After 指导轮询并保留终局记录。该指南是合同，不证明所有 API 已支持。实验必须明确哪些保证确实存在、为何不可用。只有响应里才获得的服务端 ID 可能随响应一起丢失；不能把这种设计缺陷当算法收益。
2. **有 deadline/预算的指数退避 + full jitter + 同键重试 + VTC 三值复核。** [FreeRTOS/backoffAlgorithm 官方 README](https://github.com/FreeRTOS/backoffAlgorithm) 已明确实现指数退避和 full jitter；查询 HEAD 为 `13e49c5fcf233890a7b8d975a3a12c7083aa7d8a`。比较时服务端支持的 Retry-After 应优先纳入。退避控制负载，不单独证明副作用未发生；必须同幂等协议配套。
3. **非 LLM 的历史适配基线。** 按命令/后端/负载分层，用历史完成或可见时间的经验/删失估计分位数设等待，配最便宜有效渠道；另比固定双渠道、最优开发集固定调度、基于同一已学模型的一步信息价值策略。全部方法拿到相同历史和相同接口。不要只比“固定 1 秒查一次”。
4. **学习心跳间隔的故障怀疑度。** [Akka 官方 PhiAccrualFailureDetector 固定源码](https://github.com/akka/akka-core/blob/838fe7279217fb185a098abb84808c134f299161/akka-remote/src/main/scala/akka/remote/PhiAccrualFailureDetector.scala)，类注释及 `phi` 方法，使用历史心跳间隔分布，计算 `-log10(尾概率)`。源码已读，未运行。该值本身不是包含故障先验的“已死亡后验”，更不是“业务未提交后验”；应作成熟延迟自适应对照。

## 先验证删失定义，再谈后验学习

以下是本次方法推断，不是上述论文已经证明的本问题结论。

令 `T_ack` 为响应到达时间，`T_commit` 为写入提交时间，`T_j = T_commit + D_j` 为第 j 个只读渠道首次能确认正确后置条件的时间。客户端在 c 超时只直接给出 `T_ack > c`；**不能推出 `T_commit > c`**。例如 c=5 秒时未收到响应，写入可能在 2 秒已完成。把这条记录当“执行时间超过 5 秒”会错标似然。

如果渠道的可见性单调且无误报/漏报，t 时的否定只能给出 `T_j > t`；先否定后肯定给区间删失，跟踪结束仍否定才是该可见时间的右删失。对同一操作在 1、2、3 秒连续读到否定，证据是 `T_j > 3`，不能按独立样本相乘 `S(1)S(2)S(3)`。非单调读副本、部分写入或错误 verifier 时，必须改为显式观测模型。

还存在可识别性硬点：如果 q 是永不成功/永不可见的概率，在简化混合模型里，未见成功概率为 `q + (1-q) S_vis(t)`。有限跟踪内，永久失败质量 q 与任意长延迟尾部通常不能无参数地区分。两个渠道还可能共享提交时间、后端拥塞或同一副本；将边缘概率直接相乘会重复计证。需要有终局审计、足够长跟踪、明确延迟上界或可辩护结构假设，不能凭一次有限日志拟合得到“可靠”的重试许可。

策略也影响日志：选择何时检查、何时取消/重试，会改变删失和潜在状态。至少保存操作身份、请求和检查时钟、渠道、停止原因、所有尝试及终局审计；按原始操作/后端负载簇划分开发测试。仅把观测到的成功作标签会产生选择偏差；离线比较新策略还需动作支持/探索或前瞻性评测，不能只重放旧策略看见的部分。

## 可反驳条件与推进门槛

| 假设 | 能推翻它的结果／止损条件 |
|---|---|
| 超时后确有影响决策的状态歧义 | 廉价、可信的 operation-ID 状态端点已充分给出终局，或服务端稳定幂等重放已经消除重复副作用风险。此时只能讨论检测延迟/负载优化，不声称额外执行安全收益。 |
| 多渠道主动选择有必要 | 单一便宜渠道在全部相关状态均支配其余渠道，或开发集冻结的简单调度达到相同风险下同等成本。此时不需要复杂规划。 |
| 日志能识别并校准执行后验 | 在终局可审计测试集，删失感知模型对未执行/已提交/仍在执行的校准、风险分层没有优于经验规则；或永久失败与长尾不可区分。先修数据和建模目标，不扩大策略。 |
| 渠道依赖建模有实际价值 | 优势只在条件独立仿真成立，换成共享延迟/共享故障就过度自信；或相关模型不优于同一预算下的单渠道。不得把独立假设模拟收益当真实可靠性。 |
| 动态规划超越“学到延迟” | 用同一个学习模型，一步策略/分位数等待已解释收益；完整策略只在知道真实参数的 oracle 下获胜。则贡献限于延迟估计，不能宣称新的序贯决策优势。 |
| 有真实任务必要性 | 系统只需要结构化 API 轮询，完全不依赖语言/语义判断。可以作为可靠性控制研究，但“Agent”包装不能替代与分布式系统、主动诊断基线的比较。 |

建议先做两个门：**门 A：拿到可审计的完成/可见性日志，检查识别和校准；门 B：在相同成功/误报/重复副作用约束下，比成本和完成检测延迟。** 阈值只用开发集，按操作/场景簇给置信区间。若 A 不过则不进入规划；若 B 不优于强规则则停止“新算法”方向。独立报告假完成、错误重试、真正重复副作用、最终成功、检查费用和检测延迟，避免任意加权总分掩盖风险。

## 检索记录与未知点

本轮使用学术/网络检索发现线索，再打开 arXiv、AAAI、PMLR、出版方原文和作者/维护组织 GitHub。以下为实际使用的主要查询；不是预注册、穷尽或可证明全覆盖的检索式：

```text
"Verified Tool Calls"
"tool" "timeout" "censored" "agent" retry
"timeout" "POMDP" "retry"
"right censored" "polling" optimal
"polling" "censored" "learning"
"interval-censored" "polling" "learning"
"retry" "censored feedback"
"censored feedback" "Machine Learning for Online Algorithm Selection"
"Bayesian" "RPC" "timeout" retry paper
"multi-channel" "timeout" "posterior" retry
"censored observations" "active diagnosis"
"active diagnosis" "delayed observations"
"active sensing" "unknown delays"
"controlled sensing" "controlled Markovian observations" cost arxiv
"Learning to Wait: Synchronizing Agents with the Physical World"
"Learning to Wait" "github" "2512.16262"
"ReliHarness" arxiv
ReliHarness dual channel execution reliability harness
"Verified Tool Calls" "github" Mansoor
"operation" "status" "idempotency" "Retries" github azure guidelines
"Exponential Backoff" "Jitter" aws github
"phi accrual failure detector" pdf Hayashibara
"Atomix" "Timely" "Transactional" tool arxiv
"censored" "tool execution" "retry"
"adaptive polling" "right censored"
```

- 未核实 VTC、Learning to Wait、ReliHarness 的完整官方实验代码；不以“没找到”当成“无代码”。online_as、Atomix 和工程库只做静态阅读，未验证可运行性。
- GitHub commit 值经 API 查询；Azure 固定版本已从 GitHub 页面阅读。部分 raw 文件请求超时，未将没有返回的源码算作已读；FreeRTOS 机制依据已读官方 README。
- ReliHarness 后续获取受限；只据已返回的出版方方法段作有限结论，不能排除未读部分还有更直接方法。
- 一个声称 hazard/censoring controller 的候选仓库直接访问返回 404，未作为证据。若干 2026 主动诊断/超时线索只获得第三方摘要，未作支持性引用。
- 没有找到“完全同题已实现”并不确认剩余交集新颖；当前最稳妥定位是 **VTC 明确未来工作下的一个窄、可证伪的统计建模问题**，需先过工程基线和数据可识别性门槛。
