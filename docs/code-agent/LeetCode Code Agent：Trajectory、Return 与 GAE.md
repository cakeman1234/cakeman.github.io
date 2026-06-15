# LeetCode Code Agent：Trajectory、Return 与 GAE

`agent_core/rl_math.py` 负责把一条已经完成采样的轨迹转换为训练阶段可直接使用的数值目标。该文件不处理 prompt 组装、tool 调用或环境状态推进，只处理 reward、return、advantage 与 value target 的计算。

## 一、Trajectory 的位置

在当前项目中，一条 `Trajectory` 由多个 `StepRecord` 组成。每个 step 至少包含：

- 当前 step 的 `reward`
- 当前 step 是否结束的 `done`
- 当前 step 在轨迹中的顺序

对多轮 code agent 而言，一条轨迹通常对应一次完整求解过程。例如：

1. 第一轮生成代码并调用 `run_tests`
2. 第二轮根据测试反馈修改代码并再次调用 `run_tests`
3. 最后一轮输出 `<answer>` 并结束

`rl_math.py` 的输入不是原始 prompt，也不是 tool 的原始文本，而是已经整理完成的轨迹数据。

## 二、`compute_step_returns`

### 1. 定义

折扣回报 `return` 表示从当前 step 开始，未来能够获得的总折扣奖励。记第 \(t\) 步的 return 为 \(G_t\)，则

\[
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
\]

也可以写成递推形式：

\[
G_t = r_t + \gamma G_{t+1}
\]

其中：

- \(r_t\) 是第 \(t\) 步即时奖励
- \(\gamma\) 是折扣因子

### 2. 对应代码

```python
def compute_step_returns(
    trajectory: Trajectory,
    gamma: float = 1.0,
) -> list[float]:
    step_nums = len(trajectory.steps)
    returns = [0.0] * step_nums
    running_return = 0.0

    for i in range(step_nums - 1, -1, -1):
        reward = trajectory.steps[i].reward or 0.0
        running_return = reward + gamma * running_return
        returns[i] = running_return

    return returns
```

### 3. 代码含义

`compute_step_returns(trajectory, gamma)` 采用从后往前的递推方式计算：

- 最后一步的 return 等于最后一步的 reward
- 前一步的 return 等于当前 reward 加上下一个 return 的折扣值

这正是 Monte Carlo 风格的整条轨迹回报计算。

### 4. 在多轮 code agent 中的意义

如果一条三步轨迹的奖励是

\[
[0.0,\ 0.0,\ 1.0]
\]

则前两步的即时奖励虽然为 0，但它们的 return 不为 0，因为最后一步的成功会沿着轨迹向前传播。这样可以把“最终通过测试”的信用分配给前面的推理、修复与工具调用步骤。

## 三、`compute_step_advantages_from_returns`

### 1. 定义

advantage 用于表示某个动作相对基线表现得更好还是更差。最基本形式为

\[
A_t = G_t - b_t
\]

其中：

- \(G_t\) 是当前 step 的 return
- \(b_t\) 是当前 step 的 baseline

如果 baseline 取 value function，则有

\[
A_t = G_t - V(s_t)
\]

### 2. 对应代码

```python
def compute_step_advantages_from_returns(
    returns: list[float],
    baselines: list[float] | None = None,
) -> list[float]:
    if baselines is None:
        baselines = [0.0] * len(returns)

    if len(returns) != len(baselines):
        raise ValueError("returns and baselines must have the same length")

    return [ret - base for ret, base in zip(returns, baselines)]
```

### 3. 当前实现

`compute_step_advantages_from_returns(returns, baselines=None)` 的逻辑很直接：

- 若未提供 `baselines`，则默认全为 0
- 若提供了 `baselines`，则逐项执行 `return - baseline`

因此当前函数对应的是最基础的 advantage 构造方式。

### 4. 实际含义

若

\[
A_t > 0
\]

说明这一动作比基线更好，应在策略更新中被鼓励。

若

\[
A_t < 0
\]

说明这一动作比基线更差，应在策略更新中被抑制。

在多轮 tool agent 中，这一量决定了哪些 step 应被强化，哪些 step 应被削弱。

## 四、`compute_gae_advantages`

### 1. 目标

`compute_gae_advantages(rewards, values, dones, gamma, lam)` 用于根据：

- `rewards`
- `values`
- `dones`

计算两组量：

- 每一步的 GAE advantage
- 每一步对应的 return target

### 2. 对应代码

```python
def compute_gae_advantages(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[list[float], list[float]]:
    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("rewards, values, dones must have the same length")

    advantages = [0.0] * len(rewards)
    returns = [0.0] * len(rewards)
    next_advantage = 0.0
    next_value = 0.0

    for i in range(len(rewards) - 1, -1, -1):
        mask = 0.0 if dones[i] else 1.0
        delta = rewards[i] + gamma * next_value - values[i]
        next_advantage = delta + gamma * lam * mask * next_advantage
        advantages[i] = next_advantage
        returns[i] = advantages[i] + values[i]
        next_value = values[i]

    return advantages, returns
```

### 3. TD 残差

GAE 先构造一步 TD 残差：

\[
\delta_t = r_t + \gamma m_t V(s_{t+1}) - V(s_t)
\]

其中终止掩码 \(m_t\) 定义为

\[
m_t =
\begin{cases}
0, & \text{if done}_t = \text{True} \\
1, & \text{otherwise}
\end{cases}
\]

其中各项的含义直接写在公式里：

- \(r_t\) 是当前 step 立即得到的奖励。在 code agent 中，它可能来自最终是否通过评测、工具调用是否成功或格式是否合格。
- \(V(s_t)\) 是当前状态的价值估计，即模型认为“从当前状态继续做下去还能拿多少总收益”。
- \(V(s_{t+1})\) 是下一状态的价值估计，用于 bootstrap，即用下一状态的价值去近似后续尚未展开的未来回报。
- \(m_t\) 是终止掩码；若该步后轨迹结束，则 \(m_t = 0\)，表示不再从下一状态继续借 value。
- \(\gamma\) 是折扣因子，控制未来奖励的重要程度。

因此，\(\delta_t\) 可以理解为：当前这一步的真实结果，相比原先 value 估计，多出来了多少，或少掉了多少。

### 4. GAE 递推

GAE advantage 的递推形式为

\[
A_t^{\text{GAE}} = \delta_t + \gamma \lambda m_t A_{t+1}^{\text{GAE}}
\]

展开后可写为

\[
A_t^{\text{GAE}}
=
\delta_t
+ \gamma \lambda \delta_{t+1}
+ \gamma^2 \lambda^2 \delta_{t+2}
+ \cdots
\]

其中：

- \(\gamma\) 仍然控制远期奖励的折扣
- \(\lambda\) 控制 advantage 传播的长度。它越大，后续 step 的信息向前传得越远；它越小，advantage 越接近局部的一步 TD 估计。
- \(m_t\) 保证终止状态后不再继续递推。

这也解释了代码中的这一行：

```python
next_advantage = delta + gamma * lam * mask * next_advantage
```

它表示当前 step 的 advantage 由两部分组成：

- 当前 step 自己的 TD 残差 `delta`
- 后续 advantage 经过 `gamma * lam * mask` 折扣后的回传值

### 5. return target

得到 advantage 后，可以进一步构造 value 学习目标：

\[
\hat{G}_t = A_t^{\text{GAE}} + V(s_t)
\]

这就是代码中 `returns[i] = advantages[i] + values[i]` 的来源。

## 五、GAE 相比普通 advantage 的优势

普通 advantage 常写为

\[
A_t = G_t - V(s_t)
\]

该形式直接、清晰，但通常方差较大，因为 \(G_t\) 依赖整条未来轨迹，终局的小波动会影响前面所有 step。

GAE 的优势主要体现在三个方面。

### 1. 方差更低

GAE 不是直接使用整条 Monte Carlo return，而是利用一串 TD 残差递推 advantage。这样通常比直接使用

\[
G_t - V(s_t)
\]

更稳定。

### 2. 更适合长程信用分配

若只使用一步 TD，

\[
A_t \approx \delta_t
\]

则估计过于短视。多轮 code agent 的很多有效动作并不会立刻产生奖励，而是体现在后续：

- 是否获得有价值的测试反馈
- 是否定位到错误方向
- 是否为最终修复创造条件

GAE 能把后续信息平滑地传播回前面的 step。

### 3. 可以在偏差与方差之间折中

- 当 \(\lambda = 0\) 时，GAE 更接近一步 TD
- 当 \(\lambda \to 1\) 时，GAE 更接近长程回报

因此 GAE 提供了可调节的折中机制。

## 六、在 LeetCode 多轮 Code Agent 中的含义

设一条轨迹包含三步：

1. 生成第一版代码并调用测试工具
2. 依据反馈修复代码并再次调用工具
3. 输出最终答案，评测通过

若即时奖励为

\[
[0.0,\ 0.0,\ 1.0]
\]

则：

- 第三步直接拿到成功奖励
- 第一、二步虽然没有即时奖励，但它们是最终成功的必要前提

GAE 的作用就是把终局成功的部分信用合理分配回前面的工具调用、错误修复和中间决策步骤。这正是多轮 agent 训练中最核心的 credit assignment 问题。

## 七、当前实现与规范公式的对应关系

`rl_math.py` 当前已经具备最小骨架：

- `compute_step_returns`
- `compute_step_advantages_from_returns`
- `compute_gae_advantages`

其中前两者与数学定义基本直接对应。`compute_gae_advantages` 也已经具备反向递推结构，但从规范形式看，TD 残差更严谨的写法应为

\[
\delta_t = r_t + \gamma m_t V(s_{t+1}) - V(s_t)
\]

也就是在 `done=True` 时，不应继续引入下一状态价值。若后续继续完善该文件，应优先检查这一点，使代码与终止掩码的数学定义完全一致。

当前代码里对应的是：

```python
delta = rewards[i] + gamma * next_value - values[i]
```

它已经具备 GAE 的主干结构，但这一行还没有把 `mask` 乘到 `next_value` 上。因此当前实现更准确地说是“接近 GAE 的最小骨架”，而不是完全严格的终止态版本。

## 八、结构总结

当前 `rl_math.py` 可以理解为三层结构：

### 1. return 层

`compute_step_returns` 负责计算从当前 step 出发的未来总折扣回报。

### 2. 基础 advantage 层

`compute_step_advantages_from_returns` 负责基于 `return - baseline` 构造最直接的 advantage。

### 3. GAE 层

`compute_gae_advantages` 负责在存在 value 估计时，构造更平滑、更适合多步任务的 advantage 与 value target。

因此其整体链路为

\[
\text{trajectory} \rightarrow \text{return} \rightarrow \text{advantage} \rightarrow \text{GAE advantage}
\]

这一层的作用不是执行环境逻辑，而是把多轮交互轨迹转化为可训练的优化目标。
