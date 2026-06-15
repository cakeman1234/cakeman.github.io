# Agent-R1 学习与复现路线

## 总目标

- 参考项目：`Agent-R1`
- 我的目标：为了算法实习，读懂参考项目的核心思想，并手搓一个最小可讲清楚的复现版本
- 最终产出：
  - 能讲清楚 Agent-R1 在解决什么问题
  - 能区分哪些是框架，哪些是算法核心
  - 能自己实现一个最小的 step-level agent RL 闭环
  - 能整理出一套用于面试展示的材料

## 总原则

- 不追求完整复刻原仓库
- 不把时间浪费在分布式工程细节上
- 重点放在：
  - step-level MDP
  - agent-env loop
  - trajectory 表示
  - reward -> return -> advantage
  - 最小 PPO / GRPO 风格更新
  - 一个最小任务

## 双线推进方式

本路线分为两条线，同时推进：

- **A 线：参考项目学习**
  - 目标是读懂 `Agent-R1`
- **B 线：我的复现实现**
  - 目标是同步手搓一个最小版本

原则：

- 第 1-2 部分先学习，不急着写
- 从第 3 部分开始，必须边学边写
- 后面每一部分都要有对应的“我的实现”

---

## 第一部分：问题定义与核心思想

### A 线：参考项目学习

- [ ] 阅读 `Agent-R1/README.md`
- [ ] 回答以下问题：
  - 多轮 agent 训练为什么不能简单看成单轮长文本生成？
  - 什么是 step-level MDP？
  - Agent-R1 的最小训练闭环是什么？
- [ ] 用自己的话写出一句话总结：
  - Agent-R1 到底在解决什么问题

### B 线：我的复现实现

- [ ] 暂时不写代码
- [ ] 只做一件事：
  - 画出最小闭环图：`obs -> model -> action -> env/tool feedback -> reward + next_obs`

### 本部分产出

- [ ] 一段自己的文字总结
- [ ] 一张最小闭环草图

---

## 第二部分：最小任务入口 GSM8K

### A 线：参考项目学习

- [ ] 阅读 `recipes/gsm8k/README.md`
- [ ] 阅读：
  - `recipes/gsm8k/base.yaml`
  - `recipes/gsm8k/reward_fn.py`
  - `recipes/gsm8k/tool.py`
- [ ] 回答以下问题：
  - 为什么 GSM8K 是最适合入门的任务？
  - plain GSM8K 和 GSM8K + Tool 的区别是什么？
  - prompt、tool、reward、env kwargs 分别来自哪里？

### B 线：我的复现实现

- [ ] 暂时不写训练代码
- [ ] 先定义自己的复现范围：
  - 任务是否直接选 `GSM8K + Tool`
  - 或者先做一个更简单的 toy task
- [ ] 写下自己的最小复现目标：
  - 单进程
  - 单任务
  - 单环境
  - 单种 advantage

### 本部分产出

- [ ] 一页任务数据流总结
- [ ] 一份自己的最小复现 scope

---

## 第三部分：Agent-Environment Loop

### A 线：参考项目学习

- [ ] 阅读：
  - `agent_r1/agent_flow/agent_env_loop.py`
  - `agent_r1/env/base.py`
  - `agent_r1/env/envs/tool.py`
  - `agent_r1/tool/base.py`
- [ ] 回答以下问题：
  - 什么是 `Observation`
  - 什么是 `Action`
  - 一次循环如何变成一个 step
  - 环境如何决定下一轮 prompt

### B 线：我的复现实现

- [ ] 开始写代码
- [x] 自己实现最小版本：
  - `Observation`
  - `Action`
  - `Env.reset()`
  - `Env.step()`
  - `AgentEnvLoop`
- [ ] 先不要接模型，哪怕用假数据也行
- [x] 设计多轮 tool agent 的输出协议：
  - 中间轮允许分析 + `tool_call`
  - 最终轮单独输出 `<answer>...</answer>`
  - 不把 `<answer>` 强行加到每一轮
  - 明确 tool parser 和 final answer parser 的职责边界
- [x] 完成本地最小 tool 链路验证：
  - `process_lc.py -> env_kwargs -> run_tests`
  - 验证正确代码 / 错误答案 / 语法错误三种分支
- [x] 完成最小多轮 agent 主体骨架：
  - `agent_core/types.py`
  - `agent_core/env.py`
  - `agent_core/loop.py`
- [x] 完成最小多轮 loop 调试：
  - 第 1 轮输出 `tool_call`
  - 第 2 轮输出 `<answer>`
- [x] 细化 `env.py` 的职责分层：
  - 解析层：`tool_call` / `<answer>` 提取
  - 状态层：`messages` / `finished` / `turn_count` / `last_tool_result`
  - 转移层：tool 分支 / final answer 分支 / invalid action 分支
- [x] 补全终局闭环：
  - 最终轮 `<answer>` 提交后自动评测
  - 返回最终 reward / done / info
  - 明确中间 step reward 和最终 reward 的关系
- [x] 增强环境健壮性：
  - 空 action 处理
  - 非法 tool_call 处理
  - 最大轮数控制
- [x] 接入本地真实 LeetCode 数据进行多轮联调：
  - 下载 `newfacade/LeetCodeDataset`
  - 用第一条真实样本跑通 `debug_agent_loop.py`
- [x] 新增最小 policy 层骨架：
  - `agent_core/policy.py`
  - `BasePolicy`
  - `DebugRulePolicy`
- [x] 让 `debug_agent_loop.py` 从“内嵌 fake_policy”切换为“调用 policy 对象”

### 本部分产出

- [x] 我自己的最小 loop 伪代码
- [x] 一个可以跑通 1 个 step 的 toy env
- [x] 一个可以跑通 2-step 的多轮 toy agent
- [x] 一个带终局评测的完整 toy env

---

## 第四部分：Step 与 Trajectory 表示

### A 线：参考项目学习

- [ ] 阅读：
  - `agent_r1/agent_flow/agent_flow.py`
- [ ] 搞清楚：
  - `AgentFlowStep` 里存了什么
  - 为什么 step 边界重要
  - 多个 step 如何组成 trajectory

### B 线：我的复现实现

- [ ] 自己定义最小数据结构：
  - `Step`
  - `Trajectory`
- [ ] 至少保存：
  - observation
  - action
  - reward
  - done
  - next_observation
  - step_index
  - trajectory_id

### 本部分产出

- [x] 我自己的 trajectory 数据结构
- [x] 2-step 真实样本 trajectory 调试结果
- [ ] 3-step toy trajectory 示例

---

## 第五部分：Reward、Return、Advantage

### A 线：参考项目学习

- [ ] 阅读：
  - `agent_r1/reward_loop/reward_loop.py`
  - `agent_r1/trainer/ppo/core_algos.py`
- [ ] 重点关注：
  - `compute_gae_advantage_return`
  - `compute_token_gae_advantage_return`
  - `compute_grpo_outcome_advantage`
- [ ] 回答以下问题：
  - step-level reward 和 token-level reward 的区别是什么？
  - 多步 trajectory 如何变成训练目标？
  - 这里相对普通单轮 RLHF 的核心变化是什么？

### B 线：我的复现实现

- [ ] 自己实现一个最小 advantage 版本
- [ ] 建议顺序：
  - 先做 step-level return
  - 再做 step-level GAE
  - 最后再看是否需要 token-level 版本
- [ ] 用手算 toy example 验证结果

### 本部分产出

- [x] 一篇 `Trajectory / Return / GAE` 学习笔记
- [ ] 一个手算例子
- [ ] 一个自己写的 `return/advantage` 实现

---

## 第六部分：Trainer 调用链

### A 线：参考项目学习

- [ ] 选择性阅读：
  - `agent_r1/trainer/main_agent_ppo.py`
  - `agent_r1/trainer/ppo/ray_trainer.py`
- [ ] 只看主调用链，不深挖分布式细节
- [ ] 回答以下问题：
  - rollout 在哪里触发
  - reward 在哪里计算
  - advantage 在哪里写回 batch
  - PPO update 在哪里发生

### B 线：我的复现实现

- [ ] 自己实现最小 trainer 主流程：
  - sample rollout
  - compute reward
  - compute return / advantage
  - policy update
- [ ] 先单进程、单 batch、单任务跑通

### 本部分产出

- [ ] 10 行以内的训练调用链总结
- [ ] 一个最小 trainer 主循环

---

## 第七部分：拼出最小复现项目

### A 线：参考项目学习

- [ ] 回头检查参考项目里哪些内容我故意不复现
- [ ] 明确哪些属于：
  - 算法核心
  - 工程框架
  - 任务适配

### B 线：我的复现实现

- [ ] 把前面模块拼起来：
  - env
  - loop
  - trajectory
  - reward
  - advantage
  - policy update
- [ ] 跑出一个最小可运行版本
- [ ] 最好能有：
  - 简单日志
  - 一次训练前后对比

### 本部分产出

- [ ] 一个最小 runnable repo
- [ ] 一份简短说明文档

---

## 第八部分：面试包装

### A 线：参考项目学习

- [ ] 总结参考项目的亮点：
  - step-level MDP
  - 多步 credit assignment
  - agent-env 解耦

### B 线：我的复现实现

- [ ] 准备下面这些材料：
  - 一张整体架构图
  - 一张算法流程图
  - 一个 toy example
  - 一个失败案例与分析
- [ ] 准备清楚回答：
  - 我复现了什么
  - 我没有复现什么
  - 为什么这样裁剪
  - 哪一部分最有算法价值

### 本部分产出

- [ ] 面试讲稿初版
- [ ] 项目介绍文档初版

---

## 推荐阅读顺序

1. `README.md`
2. `recipes/gsm8k/README.md`
3. `recipes/gsm8k/base.yaml`
4. `recipes/gsm8k/reward_fn.py`
5. `recipes/gsm8k/tool.py`
6. `agent_r1/agent_flow/agent_env_loop.py`
7. `agent_r1/env/base.py`
8. `agent_r1/env/envs/tool.py`
9. `agent_r1/agent_flow/agent_flow.py`
10. `agent_r1/reward_loop/reward_loop.py`
11. `agent_r1/trainer/ppo/core_algos.py`
12. `agent_r1/trainer/ppo/ray_trainer.py`

## 什么时候开始写复现

- 第 1 部分：不写代码，只理解问题
- 第 2 部分：不写训练代码，只确定最小复现范围
- 第 3 部分：正式开始写
- 第 4-7 部分：边学边写，学习和实现同步推进
- 第 8 部分：整理和包装

一句话原则：

- **前两部分先立住认知，第三部分开始必须同步手搓**

## 当前进度

- [x] 建立中文双线学习路线
- [x] 已明确多轮 tool agent 的第一版终局设计：中间轮做 tool use，最终轮输出 `<answer>`
- [x] 已完成 LeetCode 的最小数据 / prompt / tool 骨架
- [x] 已完成 `agent_core` 的最小 env / loop 骨架
- [x] 已完成真实数据下载与本地数据入口兼容：
  - `newfacade/LeetCodeDataset`
  - `LeetCodeDataset-train.jsonl`
  - `LeetCodeDataset-test.jsonl`
- [x] 已完成 `uv` 虚拟环境建立与 `datasets` 安装
- [x] 第三部分基本完成：主体骨架已搭好，`env.py` 已补上终局评测、非法 action 收口、工具异常处理与最大轮数终止
- [x] 第四部分基本完成：`types.py` 已有最小 `StepRecord / Trajectory` 结构，`loop.py` 已能稳定产出完整 trajectory
- [x] 已新增最小 `policy` 层骨架：`agent_core/policy.py`
- [x] 已完成 `debug_agent_loop.py` 的真实样本联调：
  - 第 0 步错误代码 `reward = 0`
  - 第 1 步修正代码 `reward = 1`
  - 总 reward = 1
- [ ] 当前新的进行中重点：`policy.py` 还只是规则版，尚未真正根据 `obs.messages` 中的 tool feedback 做决策
- [ ] 第五部分进行中：`rl_math.py` 已有最小函数骨架，但还没有和真实 trajectory 做系统联调
- [ ] 第一部分完成
- [ ] 第二部分完成
- [x] 第三部分完成
- [x] 第四部分完成
- [ ] 第五部分完成
- [ ] 第六部分完成
- [ ] 第七部分完成
- [ ] 第八部分完成

---

## 给新 Codex 的快速接手区

这一部分是给“新开一个 Codex 对话”准备的。目标是让新的助手不用重新读完整个仓库，也能快速知道项目背景、参考路线、当前进度与下一步。

### 1. 项目一句话

我在做一个 **LeetCode Code Agent 的最小可训练原型**：

- agent 可以多轮调用 tool
- 中间轮执行 `tool_call`
- 最终轮输出 `<answer>`
- 后面要把轨迹、奖励、advantage、训练接口逐步补齐

### 2. 参考项目

我的参考是两条线：

- **架构主线参考 `Agent-R1`**
  - 重点参考多步 agent-flow、step、trajectory、env-loop 的组织方式
- **数据与奖励主线参考 `code-r1` / `coder1`**
  - 重点参考 LeetCode 数据处理
  - 重点参考判题、reward、return、advantage、GAE 的设计

换句话说：

- `Agent-R1` 主要回答“多轮 agent 框架怎么搭”
- `code-r1 / coder1` 主要回答“LeetCode 任务怎么做数据、评测与奖励”

### 3. 当前自己项目的目录重点

当前最重要的是我自己项目里的：

- `agent/agent_core/env.py`
- `agent/agent_core/loop.py`
- `agent/agent_core/types.py`
- `agent/agent_core/rl_math.py`

当前参考目录还包括：

- `E:/code/project/Agent-R1`
- `E:/code/project/agentr1`
- `E:/code/project/coder1`

### 4. 当前已经完成到哪里

已经完成：

- 最小版多轮 agent 骨架已经有了
- `env.py` 已经补了终局闭环
- `env.py` 已经补了非法 action 收口
- `env.py` 已经补了工具异常处理
- `types.py` 已经开始承载 `StepRecord / Trajectory`
- `loop.py` 已经能稳定返回完整 `Trajectory`
- 本地 LeetCode 真实数据已经下载
- `debug_agent_loop.py` 已经能跑真实样本
- `agent_core/policy.py` 已经有最小 `BasePolicy / DebugRulePolicy`

正在进行：

- `policy.py` 需要从“按 step_idx 输出”推进到“读取 obs/messages 决策”
- `rl_math.py` 需要补最小 `return / advantage / GAE`

还没进入正式收口：

- 训练入口
- 更系统的 reward 组织
- 是否拆出独立 `trainer/`、`reward_loop/` 目录

### 5. 当前最重要的架构约定

当前默认约定是：

- `observation` 是 agent 在某一轮看到的输入
- `action` 是 agent 在该轮输出的行为
- 一个 `step` = `(observation, action, reward, next_observation, done, info)`
- 多个 `step` 串起来形成一个 `trajectory`
- `trajectory` 是后续 reward / return / advantage / trainer 的输入基础

### 6. 当前建议的新 Codex 工作顺序

如果新 Codex 接手，建议按这个顺序继续：

1. 先检查 `policy.py` 是否已经开始读取 `obs.messages`
2. 把 `DebugRulePolicy` 从“按 step_idx 输出”改成“根据 tool feedback 决策”
3. 再把 `rl_math.py` 和真实 trajectory 接起来
4. 再写最小 `compute_returns`
5. 再写最小 `compute_advantages`
6. 最后再补 `compute_gae_advantages`

### 7. 当前最值得优先完成的具体下一步

当前最建议直接继续写的是：

- 让 `policy.py` 真正利用 `obs.messages` 中的 tool feedback 决定下一轮 action

因为只有这一步稳定后，后面的：

- 多轮 decision
- 更真实的 trajectory
- return 计算
- advantage / GAE
- trainer 输入

才有统一的数据承载结构。

### 8. 如果新 Codex 要先读代码，优先读什么

建议优先阅读顺序：

1. 自己项目
   - `agent/agent_core/types.py`
   - `agent/agent_core/env.py`
   - `agent/agent_core/loop.py`
   - `agent/agent_core/policy.py`
   - `agent/agent_core/rl_math.py`
2. 架构参考
   - `Agent-R1/agent_r1/agent_flow/agent_flow.py`
   - `Agent-R1/agent_r1/agent_flow/agent_env_loop.py`
3. 奖励参考
   - `Agent-R1/agent_r1/reward_loop/reward_loop.py`
   - `Agent-R1/agent_r1/trainer/ppo/core_algos.py`

### 9. 可直接复制给新 Codex 的一句话说明

如果我新开一个 Codex 对话，可以直接这样说：

> 我在做一个 LeetCode Code Agent 的最小可训练原型。多轮 agent 架构主要参考 Agent-R1，LeetCode 数据处理、判题与奖励设计主要参考 coder1 / code-r1。当前我自己的 `env.py` 已基本收口，`types.py` 已有最小 `StepRecord / Trajectory`，下一步最优先是把 `loop.py` 补成能稳定产出完整 trajectory，然后继续补 `rl_math.py` 里的 return / advantage / GAE。请先结合 `learn.md` 和 `agent/agent_core` 代码理解当前状态，再继续实现下一步。

> 我在做一个 LeetCode Code Agent 的最小可训练原型。多轮 agent 架构主要参考 Agent-R1，LeetCode 数据处理、判题与奖励设计主要参考 coder1 / code-r1。当前我自己的 `env.py`、`loop.py`、`types.py` 已能稳定跑通真实样本，`debug_agent_loop.py` 已通过 `policy.py` 接入最小规则策略，下一步最优先是让 `policy.py` 真正读取 `obs.messages` 中的 tool feedback 决定下一轮 action，然后再继续补 `rl_math.py` 与真实 trajectory 的联调。请先结合 `learn.md` 和 `agent/agent_core` 代码理解当前状态，再继续实现下一步。
