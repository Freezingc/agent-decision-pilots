# P2：不确定记忆更新的最小探针

## Material Passport

- Origin Skill：ARS experiment-agent / test-driven-development
- Origin Mode：run（用户已授权的小型可行性验证）
- Origin Date：2026-10-19
- Verification Status：ANALYZED（本地模型探针与 CPU 判分验证）
- Version Label：p2-frozen-v1
- Audit Scope：同上下文原生数据适配审查；非独立盲审、非完整 MemConflict 重跑。

**决定：当前切片不足以验证算法，停止据此扩展实验。** 零模型住所词法对照在测试集 52/52 正确，模型直接判断仅 43/52 正确；保守置信阈值没有带来额外收益。这说明本次适配存在足以解题的题型可分性，不能把模型零误覆盖解读为不确定记忆更新方法的证据。

## 来源、许可与适配

一手来源为 [MemConflict 官方仓库](https://github.com/TaoZhen1110/MemConflict)，固定 commit `ec51d5d36e87f7665d1337f3a88cbde95fc2a964` 的 [Data/Step4_4.jsonl](https://github.com/TaoZhen1110/MemConflict/blob/ec51d5d36e87f7665d1337f3a88cbde95fc2a964/Data/Step4_4.jsonl)，完整文件 SHA256：`8ef9ec8589eccb86f63ab3a819a9180217405351a8d5846866721ea74babe092`。其官方数据为 LLM 辅助构造的多会话用户对话，并非自然用户数据。2026-10-19 读取官方 README、GitHub 元数据和完整树；API license 为 null，树中未发现 LICENSE/COPYING。没有推断再分发许可。含对话原文的源数据、抽样、请求和响应全部留在被忽略的 `raw-downloads/`，公开文件仅包含 ID、标签、哈希与汇总。查询与许可检查见 [来源记录](../artifacts/p2/source-provenance.json)。

原文件含 30 个 profile。适配每个合格 profile 的前两个可锚定住所更新、前两个可锚定条件偏好对：旧／新住所都须在用户第一人称居住／搬迁谓词中出现；条件偏好须由用户第一人称描述，包含物品及至少两个条件内容词（条件仅一个内容词时须包含该词）。每个 profile 两类皆满足才进入样本。按 profile 的固定 SHA256 排序，前 8 个为 dev，其余为 test。最终 21 个 profile、84 对：dev 32 对、test 52 对，各集合两个类别均衡。每对原文已在同上下文逐条阅读，旧／新会话和 turn 索引、原文哈希见 [标签元数据](../artifacts/p2/frozen-v1/labels.jsonl)。

此处 `supersede` / `coexist` 是从官方事件元数据及对话锚点派生的关系标签，不是官方原生三分类标签，也不是官方问答／检索指标。没有强行给静态真假矛盾分配可从文本客观知道的真值。`uncertain` 是模型可选的暂缓动作，当前 gold 没有该类别。因此不能衡量“不可判定事实”识别。不同条件的偏好允许并存，不要求把偏好字段压成唯一值。

首版只要求数值字符串出现，误将出生地作为住所、第三人偏好作为本人偏好。该 104 条批次及模型输出完整保留，已用 [隔离记录](../artifacts/p2/quarantine.json) 标为适配失败；analyze 在读取其响应前直接拒绝该目录。修正后另建 `frozen-v1`，未覆盖旧批，也未把旧批性能当作结果。新批保留了同一系统提示，只修正输入适配和冻结样本。

## 冻结执行与计分

每条请求严格包含 `id/system/user/max_tokens`，只有两段带日期的用户原文，不含 gold、冲突类型、profile ID 或隐藏问答。模型输出 JSON `{label, confidence}`。一对一次调用，直接判断和保守策略共用响应；不额外生成模板或请求反例。

主线程统一运行本地 Qwen2.5-Coder-7B-Instruct，revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`，temperature 0、top_p 0.95、每请求最多 96 新 token、batch 16（末批 4），无付费 API。84 条全部 stop、JSON 有效 84/84。84 个 request_hash 均与冻结请求、temperature、top_p、max_tokens 对应。输入 19,678 token，输出 2,051 token（后端计数包含 EOS）；旧失败批费用单列，不能与此批去重或遗漏真实总成本。冻结 manifest、请求、标签、响应和模型元数据的哈希绑定在 [summary](../artifacts/p2/frozen-v1/summary.json)。后端元数据的 max_tokens=384 是 runner 初始默认，逐条收据与实际请求上限均为 96。

基线是最新优先、始终保留、模型直接判断，以及固定网格的保守置信阈值。阈值仅用 dev：误覆盖率不高于 5% 时，先最大化正确更新数、再覆盖数、再选择较低阈值；网格含高于 1 的全暂缓选项。选择结果为 **0**，故保守策略逐项等于直接模型。没有把自报置信度当作校准概率。

源适配审查后额外加入零模型词法诊断；三个固定候选是“新语句有第一人称居住／搬迁谓词”“旧新皆有该谓词”“始终保留”，仅凭 dev 正确计数选规则，平局按该顺序。选中第一项；只读用户可见文本，不读取冲突元数据或标签。该诊断用于揭示本切片可分性，不是预注册的新算法比较，也没有用测试结果调规则。

误覆盖分母为全部 gold coexist；漏更新分母为全部 gold supersede，保留和暂缓都计作漏更新。覆盖率为非暂缓数／全部对数。正确率保留全部对作分母，避免弃权虚增表现。按 profile 聚类重采样 1,000 次（seed 1701）给出组内探索性 95% 区间；零事件 bootstrap 得到 [0,0] 不能解释为总体风险为零。

## 冻结测试结果

测试集 13 个 profile，26 个真实更新、26 个应并存对。

| 方法 | 正确／52 | 误覆盖／26 | 漏更新／26 | 覆盖／52 |
| --- | ---: | ---: | ---: | ---: |
| 最新优先 | 26 | 26 | 0 | 52 |
| 始终保留 | 26 | 0 | 26 | 52 |
| 模型直接判断 | 43 | 0 | 3 | 45 |
| dev 选保守阈值（0） | 43 | 0 | 3 | 45 |
| 零模型词法对照 | 52 | 0 | 0 | 52 |

模型更新召回 23/26=88.46%，聚类 bootstrap 95% 区间约 [73.08%, 96.25%]；覆盖率 86.54%，区间约 [76.92%, 94.23%]。暂缓 7 对，其中一个真实更新、六个并存对；另两个真实更新被判为并存。dev 模型正确 24/32、覆盖 27/32、误覆盖 0/16、漏更新 4/16；词法对照 32/32。保守阈值没有新增可利用的风险—覆盖空间。

所有动态正例最终都是 Residence，负例都是条件偏好，常见搬迁词与偏好／条件词足以区分；这不是一般开放域冲突更新的代表性采样。没有实际记忆库、下游任务或更新造成的累积损害，也没有同属性下真实更新与错误矛盾的匹配难例。故当前结果既不证明通用算法成立，也不否定更难场景值得研究。若继续，应先获得许可清晰、同属性且词汇匹配的真更新／应保留数据，并保留强规则和同覆盖率基线；在此之前不增加模型调用、训练或声称创新。

## 复现与验证

```bash
PYTHONPATH=src /home/zhangyudong/FDU/paper/Eval/.local/eval-venv/bin/python \
  -m agent_pilots.memory analyze \
  --run-dir artifacts/p2/frozen-v1 \
  --responses artifacts/p2/frozen-v1/raw-downloads/responses.jsonl \
  --out artifacts/p2/frozen-v1/summary.json

/home/zhangyudong/FDU/paper/Eval/.local/eval-venv/bin/python \
  -m pytest tests/test_memory.py -q
```

测试先确认缺失功能失败，再实现；适配缺陷、隔离标志、词法对照和响应哈希也有定向红绿回归。最终 19 项通过，覆盖原文锚定、出生地／第三人排除、profile 隔离、标签不入 prompt、严格解析、弃权分母、dev-only 选择、完整 ID、token 核算、冻结文件篡改与 request/config 绑定。未提交 Git；本子任务未加载模型或使用 GPU。
