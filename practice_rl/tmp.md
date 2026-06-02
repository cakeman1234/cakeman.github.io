

- GRPO 的 KL 散度和 PPO 的熵奖励是什么关系？
- GRPO 和 PPO 在公式上的区别，除了 KL 和熵的区别，还有什么？
- 生成轨迹是什么，轨迹是什么，它和回答的关系是什么？
- 这里的 KL 散度用 TRL 的实现呢？

## 张量与公式对应
- 这里的 `stack` 方法的作用是什么？
- 为什么在 `token_loss` 计算一次 `mean`，在最后 `forward` 还要计算一次 `mean`？
- 这两次 `mean` 在公式上是怎么对应的？

## 2026-06-02

### 站点与笔记展示
- Notes 页正文为什么看起来没有真正变宽，为什么代码块边界和正文边界不一致？
- 为什么本地直接打开 `note.html?path=rl/ppo.md` 会 `Failed to fetch`？
- 如何简化个人站点的小改动发布流程，避免每次都手动 `add / commit / push`？

### GRPO 设计与实现
- 按照 `GRPOLoss` 的职责，它内部是否需要计算 advantage？
- 如果面试官让我手撕 GRPO，我现在这版写法是否合格？
- 在 `GRPOLoss` 之外，如果面试官让我继续写，应该继续写什么，是 trainer 吗？
- 如果继续写 trainer，`__init__` 里应该放哪些参数？
- `ref_model` 如果使用的是旧模型拷贝，也需要单独保留吗？
- `GRPOTrainer` 是否应该继承 `nn.Module`？
- GRPO 的 advantage 公式是不是“减平均值再除方差”，还是应该除标准差？
- `compute_advantages` 和 `forward` 都写完后，下一步最该补的是哪一层？
- `compute_logprobs` 这个函数的职责、输入输出和整体位置应该怎么设计？
- 抽取 generated token 的 log_probs 这一段代码到底在做什么？
- 为什么 `log_probs[pos - 1]` 还不够，还需要再用 `token_id` 去索引？
- 每个 token 的 `lp` 在 GRPO/PPO 公式里对应哪一步，有什么作用？
- 逐 token 计算 ratio 时，如果不同模型生成的 token 序列有偏差，会不会导致整段 token 全部错位？
- 为什么 model 可以对一个给定 token 序列计算 logprobs？
- 对固定前缀求“下一个 token 的整张 logprob 分布”，再取真实 token 对应值，这个理解是否正确？
