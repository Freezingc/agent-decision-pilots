## Material Passport

- Origin Skill: ARS academic-research-suite / experiment-agent
- Origin Mode: run，附确定性重跑与储存轨迹核验
- Origin Date: 2026-10-19
- Verification Status: VERIFIED — 仅限本结构模拟、轨迹记账和固定种子重现；不含 LLM 或真实用户效果
- Version Label: pilot-p4-v1-diagnostic-supplement
- Model: 无推理模型；标准库 CPU 模拟
- Review provenance: 设计、实现、判读及核验由同一执行 agent 完成，没有独立模型或盲审验证
- Protocol: docs/protocol-v0.md，P4

## 判读与处置

**建议停止把此两步规划机制作为当前算法贡献，暂不推进真实用户或 LLM 交互实验。** 信息取得确实能胜过强制「总是问」或「先探查再一律问」，但这主要可由廉价、单步的信息价值规则解释。这里没有学到新策略，也没有建立需要复杂规划的实用增量。

已知模型下，单步规则与两步规划在 45 个网格点中的 43 个有相同精确期望效用；去掉不可用工具的无效成本变化后，是 33 个不同条件中的 31 个。另两个带噪条件的最大精确差为 **0.00077037 效用单位**。此处 reward 最大为 1，但它不是成功率，差值不能写成准确率的百分点。

有限样本上的 dev 调参规则与 planner 差值介于 -0.01354 和 +0.01707。这些差值不能直接归因于两步规划：例如 clean 的中成本条件中，未调参的单步规则已与 planner 完全一致，dev 网格却受抽样噪声影响选了较差规则。不能选用较弱的 dev 胜出者，忽略更强的不调参规则，来声称算法收益。

## 冻结的环境与可见信息

每个 episode 有两个相互独立的隐藏二元变量：E 是环境属性，P 是用户偏好。公开 context 包含各自先验 p(E=1)、p(P=1) 及环境权重 w。每个先验从 {0.2, 0.5, 0.8} 均匀抽取，w 同样从 {0.2, 0.5, 0.8} 抽取；所以有 27 个等概率 context。策略作最后的二元决定 a，得到：

`reward = w × 1[a=E] + (1-w) × 1[a=P]`

`net utility = reward − 实际执行的 probe/ask 成本`

probe 仅产生 E 的带噪观测；ask 仅产生 P 的带噪回答。两个来源使用不同的随机流。环境 probe 不读取 P；user answer 不读取 E。策略边界只接收 context、Bayes belief、已用过的工具及公开条件，不接收 episode 或隐藏真值。真值只用于产生观测和终态判分。

每个信息动作最多一次。回答缺失仍收取该动作的完整成本，且不更新 belief。最后决策不收费；所有策略共用相同动作与观测模型。这是抽象二元决策，没有语言理解、工具执行失败语义或多轮偏好澄清。

## 对照、数据分割与敏感性

- `no_information`：不探查、不询问，依公开先验决定。
- `never_ask`：环境 probe 一次（可用时），不问用户。
- `always_ask`：问用户一次（可用时），不 probe。
- `probe_then_ask`：可用时依序执行两个工具，不依成本早停。
- `fixed_dev`：在上述四个固定策略中，仅依 dev 的平均净效用选出一个。
- `simple_dev`：四个固定策略，加上 75 个简单规则，在 dev 上选择。规则只做单步 gross VOI > λ × 成本检查，两个工具的 λ 从 {0, 0.5, 1, 2, disabled} 选取，顺序为 probe 优先、ask 优先或最大单步净价值；观测后可停止。共 79 个候选，平局先选较少调用，再依固定网格次序。
- `simple_frozen_transfer`：只在 noisy、probe cost=0.08、ask cost=0.12 的 dev 上选一次规则，其他条件不重选 λ。选中 probe λ=0.5、ask λ=2、probe 优先。
- `myopic_known`：不调参，λ=1，选最大正单步净价值，观测后重新决策。
- `planner`：精确有限树 Bayes 规划，最多剩余两个信息动作，可先 ask 或先 probe。

**planner 与两种单步规则都知道真实的噪声、缺失率和先验。** 因而没有把参数知识的优势只给 planner；同时，这些结果只是已知模型的理想化诊断，没有验证从数据估参或处理错误参数。

dev 是 seed={17,23}，每 seed 128 个 episode，共 256；test 是 seed={101,211,307}，每 seed 256 个，共 768。没有 episode 跨 dev/test。所有策略使用相同 test cases，且对同一工具共用相同潜在观测；条件间也共用随机数。

成本固定为 probe∈{0.01,0.08,0.25}，ask∈{0.02,0.12,0.35}，和下列五种信息条件交叉，共 45 个网格点：

| 条件 | probe accuracy | user accuracy | 缺失或不可用 |
|---|---:|---:|---|
| clean | 1.00 | 1.00 | 无 |
| noisy | 0.85 | 0.80 | 无 |
| missing | 0.90 | 0.90 | 两来源各独立缺失 40% |
| no_probe | 不可用 | 0.90 | 没有环境信息工具 |
| no_user | 0.90 | 不可用 | 没有用户回答 |

不可用工具的成本变化不产生新任务；这些重复网格点用作条件检查，不能算独立证据。条件变化后仍把真实 likelihood 公开给策略；这不是未知噪声下的迁移学习结果。

## 实测结果与模型期望分开报告

以下是 probe cost=0.08、ask cost=0.12 的配对 test 净效用，N=768。所有 45 条件、9 策略的成本、reward、环境匹配、偏好匹配和调用率都在 `artifacts/p4/sensitivity.csv`；逐 seed 和配对差值在 `run-v1-supplement/summary.json`。

| 条件 | never ask | always ask | probe→ask | dev 简单规则 | 不调参单步 | 两步规划 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.69591 | 0.66581 | 0.64284 | 0.74128 | 0.75367 | 0.75367 |
| noisy | 0.64383 | 0.58013 | 0.52227 | 0.68581 | 0.69544 | 0.69544 |
| missing | 0.64331 | 0.59419 | 0.55117 | 0.69003 | 0.69367 | 0.69367 |
| no_probe | 0.67617 | 0.61659 | 0.61659 | 0.68628 | 0.69206 | 0.69206 |
| no_user | 0.66466 | 0.67617 | 0.66466 | 0.68971 | 0.68971 | 0.68971 |

同一中成本 noisy 条件中，planner 的 reward=0.72643、成本=0.03099、平均 probe=0.22135 次、ask=0.11068 次。环境匹配率 0.71484、偏好匹配率 0.62240；对照 no_information 分别为 0.64063、0.64453。故净效用增加不代表两种匹配率都增加；不应只展示一个加权分数掩盖偏好匹配下降。

精确已知模型期望在 27 个 context 上平均，独立于抽样出的 test 隐藏真值。真正与单步规则有差异的两个条件是：

| 条件 | planner 精确期望 | 单步精确期望 | 精确差 | test 配对观察差 |
|---|---:|---:|---:|---:|
| noisy p=0.01,a=0.02 | 0.73343 | 0.73325 | 0.00018 | -0.00592 |
| noisy p=0.08,a=0.02 | 0.71036 | 0.70959 | 0.00077 | -0.00167 |

精确期望的微小正差与 test 上的负差可以同时成立，因为后者有抽样噪声。保留这些结果，不加样或改环境追求正的 test 差值。`summary.json` 也保留未校正多重比较的配对常态近似 95% 描述区间；不以其作确认性显著性结论。

单元测试另含一个手算的互补信息例子：先验皆 0.8、w=0.5、两来源 accuracy=0.9、成本皆 0.01。单步规则停在 0.8；两步规划值为 0.8134。它证明实现能处理互补性，**不是自然任务或本轮平均算法收益的证据**，也未把它过度抽样进实验。

## 验证与可重现文件

先写的 12 个行为测试均因缺失 `agent_pilots.clarification` 而失败，之后实现通过。诊断补充的精确策略值测试也先确认缺失功能失败；最后 13 个标准库 unittest 全数通过。指定的 `.venv` 没有 pytest，因此改用 unittest，未安装依赖。

首轮 `run-v1` 保留原设计、原结果和哈希匹配的源代码快照。观察首轮后加入的单步对照／精确期望明确记为诊断补充，输出到 `run-v1-supplement`；原环境、种子、样本量与 dev 网格不变，没有覆盖首轮结果。

两轮主要运行分别约 19.5 秒和 21.3 秒。补充轮再次从头运行到 `replay-v1-supplement`，六份确定性文件（含完整 gzip 轨迹、summary、dev 网格结果）SHA-256 完全一致。运行秒数属环境测量，不要求逐位一致。

`artifacts/p4/validate.py` 另外核对了全部 **311,040** 条补充 test 记录：工具可用性与单次限制、同工具的配对观测、Bayes 更新、实际收费、隐藏终态 reward、net utility，以及与 summary 的均值一致性。结果在 `artifacts/p4/verification.json`。这是同作者的实现核验，并非独立验证。

从项目根目录重现（输出子目录须尚不存在）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/zhangyudong/FDU/paper/Eval/.venv/bin/python -m unittest discover -s tests -p test_clarification.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src timeout 90 /home/zhangyudong/FDU/paper/Eval/.venv/bin/python -m agent_pilots.clarification --output artifacts/p4/reproduce-new
timeout 60 /home/zhangyudong/FDU/paper/Eval/.venv/bin/python artifacts/p4/validate.py --replay artifacts/p4/reproduce-new
```

程序与所有依赖均为标准库。没有 GPU、付费 API、网络模型调用或真实参与者。没有进行参数学习或方法新颖性检索；本次止损依据是此已知模型结构中，强简单规则已解释几乎全部观察空间，不是对所有 Agent 澄清问题的否定。
