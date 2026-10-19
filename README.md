# Agent 决策算法的可行性实验

这是一组用于筛选科研选题的探索性实验。模拟结果、真实本地模型结果与理想上限分别记录；不把简单基线可解释的收益当作新算法贡献。

- 总体判断与候选算法：[中文实验报告](docs/feasibility-report-zh.md)
- 执行前协议：[protocol-v0.md](docs/protocol-v0.md)
- P1 恢复：[首轮记录](docs/p1-notes.md)
- P2 记忆：[记录](docs/p2-notes.md)
- P3 跳步：[记录](docs/p3-notes.md)
- P4 澄清：[记录](docs/p4-notes.md)

首轮 P1/P4 是标准库 CPU 模拟；P2 是官方合成记忆基准的关系切片；P3 是 Qwen2.5-Coder-7B 的 768 次真实生成和 MBPP+ 隐藏判分。未使用付费 API。

在 Python 3.11 环境安装分析/测试依赖：

```bash
python -m pip install -e '.[dev,evaluation]'
python -m pytest -q
```

本地推理另需 CUDA 及 `generation` 依赖。可复用已有模型缓存；本机路径与运行命令见各项记录。P3 沙箱需要 Linux bubblewrap。第三方原始数据与请求保存在被忽略的本地目录；源码、协议、合成轨迹、模型生成的代码分支、来源哈希及汇总保存在仓库。

结果用于决定继续、改题或止损，不是论文级有效性或新颖性声明。
