# LeetCode Code Agent：数据、Prompt 与 Tool 骨架

## 一、整体结构

LeetCode code agent 的最小骨架可表示为：

`原始题目 -> 数据加工 -> prompt -> agent 生成代码 -> tool 执行测试 -> 返回反馈`

与这条链路对应的核心文件如下：

- `process_lc.py`：数据加工
- `prompt.py`：初始 prompt 构造
- `tool.py`：测试工具定义与执行
- `base.yaml`：recipe 配置入口

这四个文件构成了 LeetCode 任务从原始样本到 agent-env-tool 闭环的基础结构。

---

## 二、文件关系

`process_lc.py` 负责把原始 LeetCode 样本加工为统一的数据格式。  
`prompt.py` 负责定义模型在初始轮次看到的消息结构。  
`tool.py` 负责接收模型输出、执行测试并返回结构化反馈。  
`base.yaml` 负责将上述模块接入 recipe。

四者的关系可概括为：

- `process_lc.py` 定义样本形态
- `prompt.py` 定义输入形态
- `tool.py` 定义执行与反馈形态
- `base.yaml` 定义系统接入方式

---

## 三、`process_lc.py`

### 1. 功能

`process_lc.py` 的功能是将原始 LeetCode 数据样本转换为 Agent-R1 风格的数据格式。  
该文件不负责训练、不负责测试执行，只负责样本整理。

### 2. 核心字段

当前加工后的样本包含以下关键字段：

- `prompt`
- `reward_model`
- `extra_info`
- `env_kwargs`

其中最核心的是 `prompt` 与 `env_kwargs`。

### 3. `prompt`

`prompt` 表示模型初始输入。  
一道题目在进入 agent 之前，会被组织为一组 chat messages。

### 4. `extra_info`

`extra_info` 表示辅助信息，不直接输入模型。  
该字段通常包含：

- `question_id`
- `entry_point`
- `starter_code`
- `reference`
- `functional_test`

该部分主要用于调试、核对和离线分析。

### 5. `env_kwargs`

`env_kwargs` 表示环境与工具执行时使用的隐藏参数。  
当前最核心的内容包括：

- `test_code`
- `entry_point`
- `starter_code`

因此，`process_lc.py` 不仅构造 prompt，也同时构造后续 tool 执行所需的上下文。

### 6. 结构意义

经过该文件处理后，一道 LeetCode 题被拆分为两个部分：

- 面向模型的输入部分：`prompt`
- 面向环境与工具的执行部分：`env_kwargs`

这一拆分构成了后续 prompt 构造与 tool 执行的基础。

---

## 四、`prompt.py`

### 1. 功能

`prompt.py` 负责定义初始交互模板。  
其输出结果用于构造样本中的 `prompt` 字段。

### 2. 当前结构

当前文件主要包含：

- `LEETCODE_AGENT_SYSTEM_PROMPT`
- `LEETCODE_AGENT_USER_PROMPT`
- `build_agent_messages(question, starter_code)`

其中：

- system prompt 用于规定 agent 身份、任务边界和输出约束
- user prompt 用于提供题目内容与 starter code

### 3. 当前 prompt 的性质

当前 prompt 属于最小可用版本，其主要作用包括：

- 指定模型为 Python coding agent
- 提供题目描述
- 提供 starter code
- 要求输出完整 Python 解

该版本已经能够支持基本的 code generation 与弱多轮修复流程。

### 4. 当前 prompt 的边界

当前 `prompt.py` 尚未完整定义多轮 tool-use 协议。  
尚未明确规定的内容包括：

- 中间轮的 tool call 输出格式
- 最终轮的 `<answer>...</answer>` 输出格式

因此，当前 prompt 的定位是最小可用 coding prompt，而非完整多轮协议 prompt。

---

## 五、`tool.py`

### 1. 功能

`tool.py` 是当前 LeetCode code agent 骨架中的核心执行模块。  
该文件负责定义工具接口、执行测试、解析结果，并将执行反馈返回给 agent。

当前主要工具为：

- `run_tests`

### 2. 所处位置

在 agent-env-tool 闭环中，`tool.py` 位于模型输出与环境反馈之间。  
其作用是将模型给出的代码转化为可验证行为，并将验证结果重新组织为反馈文本。

该过程可表示为：

`模型输出代码 -> tool 执行测试 -> 反馈文本 -> next_obs`

### 3. 内部结构

当前 `tool.py` 可划分为以下几个层次。

#### 参数与结果结构

- `RunTestsArgs`
- `RunTestsResult`

该层负责统一输入输出结构，减少散乱参数传递。

#### 工具入口与参数整理

- `get_tool_schemas()`
- `parse_env_kwargs(...)`
- `build_run_tests_args(...)`
- `validate_run_tests_args(...)`

该层负责：

- 定义工具 schema
- 解析环境参数
- 合并模型输入与环境输入
- 检查必要字段

#### 代码提取与结构检查

- `extract_python_code(...)`
- `validate_code_structure(...)`
- `make_format_error_result(...)`

该层负责：

- 从模型输出中提取代码
- 检查代码是否可被 Python AST 解析
- 检查目标函数 `entry_point` 是否存在
- 在结构错误时构造 `format_error`

#### 执行与错误解析

- `build_execution_script(...)`
- `run_python_script(...)`
- `classify_error_type(...)`
- `extract_first_error_line(...)`
- `truncate_text(...)`
- `parse_test_feedback(...)`

该层负责：

- 拼接最终执行脚本
- 调用 Python 解释器执行
- 分类错误类型
- 抽取关键报错
- 控制反馈长度
- 生成结构化测试结果

#### 对外接口

- `format_tool_response(...)`
- `run_tests(...)`
- `dispatch_tool_call(...)`

该层负责生成最终工具返回值，并作为环境调用入口。

### 4. 关键设计点

#### `starter_code` 与 `entry_point`

二者分属不同层次：

- `starter_code`：生成阶段的提示信息
- `entry_point`：执行阶段的检查目标

前者服务于代码生成，后者服务于结构验证与测试调用。

#### 整体测试判定

当前 `parse_test_feedback()` 采用整体判定方式。  
一份 functional test 的结果暂表示为：

- 通过：`1/1`
- 失败：`0/1`

该设计对应最小可用版本，尚未展开为 case-level 统计。

#### 最终轮 `<answer>` 兼容

`extract_python_code()` 会优先提取 `<answer>...</answer>` 内部内容。  
这说明当前工具实现已经对最终轮单独输出 `<answer>` 的协议形式保持兼容。

该兼容性属于接口预留，不等于完整多轮协议已经实现。

### 5. 结构意义

`tool.py` 的核心意义在于：  
它将“语言形式的代码输出”转化为“可执行、可验证、可反馈”的环境行为。

在多轮修复型 code agent 中，tool 的作用不是单纯执行脚本，而是形成：

- 动作验证
- 错误定位
- 反馈回流

这一闭环。

---

## 六、`base.yaml`

### 1. 功能

`base.yaml` 是 recipe 配置入口。  
其主要作用是将数据、prompt、tool 和训练配置接入统一系统。

### 2. 位置

前述 Python 文件分别定义局部模块，`base.yaml` 负责系统级装配。  
因此，该文件更接近装配图而非算法逻辑本身。

### 3. 结构意义

在当前 LeetCode recipe 中：

- `process_lc.py` 提供数据侧定义
- `prompt.py` 提供输入侧定义
- `tool.py` 提供执行侧定义
- `base.yaml` 提供系统接入定义

因此，`base.yaml` 的意义在于收口，而不在于展开具体算法。

---

## 七、完整链路

这几个文件可以组织为以下完整链路：

### 1. 数据加工阶段

`process_lc.py` 读取原始题目并生成：

- `prompt`
- `env_kwargs`

### 2. 初始输入阶段

`prompt.py` 决定 `prompt` 的具体消息结构。  
模型在第 0 轮看到的内容由此产生。

### 3. 工具执行阶段

模型生成代码后，`tool.py` 接收：

- 模型输出的 `code`
- `env_kwargs` 中的测试参数

随后完成：

- 代码提取
- 结构检查
- 测试执行
- 结果解析
- 文本反馈生成

### 4. 系统接入阶段

`base.yaml` 将数据、prompt、tool 及相关配置接入 recipe。  
至此，LeetCode 任务具备进入 agent-env-tool 闭环的最小条件。

---

## 八、当前实现的性质

当前实现的性质可以概括为：

- 属于最小可用骨架
- 已经连通数据、prompt、tool、recipe 配置
- 尚未完整展开多轮 tool-use 协议

未完成部分主要包括：

- 中间轮 tool call 的标准化协议
- 最终轮 `<answer>` 的统一格式约束
- case-level 测试统计
- 更完整的 env 侧多轮状态组织

因此，当前阶段的重点不在于完整性，而在于基础闭环的建立。

---

## 九、结论

这组文件的核心作用不是支持普通代码生成任务，而是将 LeetCode 题目改造为可进入 agent-env-tool 闭环的任务形式。

其基本分工如下：

- `process_lc.py`：题目到样本的转换
- `prompt.py`：样本到初始输入的转换
- `tool.py`：模型输出到可验证反馈的转换
- `base.yaml`：局部模块到系统 recipe 的转换

由此形成的不是单纯的数据集，而是一套最小化的 code agent 任务骨架。

