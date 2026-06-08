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
- [ ] 自己实现最小版本：
  - `Observation`
  - `Action`
  - `Env.reset()`
  - `Env.step()`
  - `AgentEnvLoop`
- [ ] 先不要接模型，哪怕用假数据也行

### 本部分产出

- [ ] 我自己的最小 loop 伪代码
- [ ] 一个可以跑通 1 个 step 的 toy env

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

- [ ] 我自己的 trajectory 数据结构
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
- [ ] 第一部分完成
- [ ] 第二部分完成
- [ ] 第三部分完成
- [ ] 第四部分完成
- [ ] 第五部分完成
- [ ] 第六部分完成
- [ ] 第七部分完成
- [ ] 第八部分完成
