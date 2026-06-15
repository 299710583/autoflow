# AutoFlow

AutoFlow 是一个面向授权安全评估、靶场测试和内部红队演练的多 Agent 渗透测试自动化框架原型。项目基于 LangGraph 组织工作流，通过 OpenAI-compatible LLM 完成规划、推理和工具调用决策，并使用 Docker 工具容器执行安全工具。Redis 用于运行时记忆与 LangGraph checkpoint，为中断恢复、多轮推进和后续审批流打基础。

> AutoFlow 仅用于已授权目标、靶场环境或内部安全评估。请勿用于未授权系统。

## 项目目标

AutoFlow 希望把一次安全评估拆成可追踪、可复盘、可扩展的自动化流程：

- 接收用户输入的目标、范围和测试意图。
- 自动收集端口、Web 页面、路径、API、安全头、技术栈等信息。
- 让 LLM 基于上下文和工具清单分析攻击面。
- 生成下一步测试计划，并调用容器内工具执行。
- 从工具结果中提取候选脆弱点。
- 针对候选脆弱点生成验证计划。
- 执行验证动作，更新漏洞状态。
- 汇总资产、证据、验证结果和建议，生成 Markdown 报告。

## 当前能力

- LangGraph 多阶段工作流。
- OpenAI-compatible LLM 接入。
- LLM function calling 工具调用循环。
- Docker 容器内工具执行，避免直接使用宿主机 shell。
- Redis runtime memory，保存 flow 状态、事件、观测结果、Finding 和验证计划。
- LangGraph Redis checkpointer，支持后续中断恢复和审批后继续执行。
- Web recon 页面结构采集，包括标题、链接、表单、脚本、robots、sitemap 等。
- 工具结果统一为 `ToolObservation`。
- 工具观测结果可提升为候选 `Finding`。
- 候选 Finding 可生成 `ValidationPlan`。
- ValidationExecutor 可执行部分验证动作并生成 `ValidationResult`。
- Markdown 报告输出。

## 工作流概览

```text
User Target / Prompt
  -> PlannerAgent
  -> DiscoveryAgent
      -> ReconAgent
      -> DiscoveryReasonerAgent
  -> ExecutorAgent
  -> VerifierAgent
  -> ValidationAgent
  -> ValidationExecutorAgent
  -> Strategy Loop
  -> ReporterAgent
```

核心数据流：

```text
Tool Output
  -> ToolObservation
  -> ToolSignal
  -> Candidate Finding
  -> ValidationPlan
  -> ValidationResult
  -> Report
```

## Agent 职责

### PlannerAgent

理解用户输入、授权范围和测试目标，创建初始 `AssessmentFlow`。Planner 负责确定顶层评估任务，而不是执行具体扫描或验证。

### DiscoveryAgent

发现阶段入口。它组合基础 recon 和 LLM 推理：

- `ReconAgent` 负责确定性扫描和信息采集。
- `DiscoveryReasonerAgent` 负责基于上下文分析攻击面并生成测试计划。

### ReconAgent

负责执行基础探测动作，例如端口扫描、Web 页面结构采集、技术栈识别等。它更接近工具编排器，不依赖 LLM 做复杂推理。

### DiscoveryReasonerAgent

负责真正的 LLM 攻击面分析。它会读取：

- 资产信息。
- Web recon 结果。
- 已有工具观测。
- 已有候选 Finding。
- Redis memory pack。
- 可用工具清单。

然后输出攻击面、优先级和 discovery 阶段的 `TestPlan`。

### ExecutorAgent

根据 `TestPlanAction` 调用容器内工具。工具真实风险以 `configs/tools.yaml` 中的 profile 为准，避免 LLM 错误标注风险。高风险动作后续应接入审批流。

### VerifierAgent

从工具输出中识别明确风险信号，将原始工具结果转换为候选 Finding。它负责“扫描结果是否值得进一步验证”的判断。

### ValidationAgent

根据候选 Finding 类型生成验证计划，例如：

- API 暴露验证。
- 目录 listing 验证。
- Debug endpoint 验证。
- 弱安全头验证。
- 公开配置文件验证。
- CORS 配置验证。

### ValidationExecutorAgent

执行验证计划中的动作，收集证据，并更新 Finding 状态。该阶段更接近“确认是否存在漏洞”，而不是单纯扫描。

### ReporterAgent

汇总资产、工具观测、候选 Finding、验证计划、验证结果和建议，生成 Markdown 报告。

## 工具调用模型

AutoFlow 不让 LLM 直接执行系统命令。LLM 只能看到工具 schema 和工具说明，并通过 function calling 发起工具调用。

```text
LLM
  -> tool_calls
  -> ToolDispatcher
  -> Docker Tool Container / WebRecon / ScriptRunner / Memory Tools
  -> tool result
  -> LLM continues
```

命令行工具、shell 动作和脚本动作都应在 Docker 工具容器中执行，不直接使用宿主机 shell。

## 工具类型

当前工具分为三类。

### 目标扫描与验证工具

- `nmap`
- `curl`
- `whatweb`
- `httpx`
- `nikto`
- `nuclei`
- `dirsearch`
- `gobuster`
- `ffuf`
- `feroxbuster`
- `naabu`
- `subfinder`
- `testssl.sh`
- `sslscan`
- `wafw00f`
- `sqlmap`
- `hydra`
- `medusa`
- `smbclient`
- `enum4linux`
- `smbmap`

### 源码与制品分析工具

- `trivy`
- `bandit`
- `gitleaks`
- `semgrep`

### 内置工具

- `web_recon_fetch_page`
- `run_shell__bounded_bash`
- `read_agent_memory`
- `list_known_targets`
- `search_observations`
- `run_script__security_headers_check`
- `run_script__api_endpoint_probe`
- `run_script__cors_probe`
- `run_script__debug_endpoint_probe`
- `run_script__directory_listing_probe`
- `run_script__public_config_probe`

## 记忆机制

AutoFlow 当前包含两类持久化能力。

### Redis Runtime Memory

用于保存评估过程中的运行时记忆：

```text
latest_state
memory_pack
events
observations
findings
validation_plans
```

这部分给 Agent 提供跨阶段上下文，让后续推理能看到前面扫描、观测和验证过程。

### LangGraph Redis Checkpointer

用于保存 LangGraph checkpoint。后续可以用于：

- 中断恢复。
- 审批后继续执行。
- 长任务失败后恢复。
- 多轮流程状态追踪。

Redis key 结构示例：

```text
autoflow:flow:{flow_id}:latest_state
autoflow:flow:{flow_id}:memory_pack
autoflow:flow:{flow_id}:events
autoflow:flow:{flow_id}:observations
autoflow:flow:{flow_id}:findings
autoflow:flow:{flow_id}:validation_plans
```

## 目录结构

```text
autoflow/
  agents/          Agent 实现
  api/             API 与后续前端接入
  artifacts/       原始证据与报告产物存储
  domain/          项目、任务、Finding 等领域模型
  executor/        Docker 工具执行、脚本执行、shell 执行
  flows/           AssessmentFlow 业务状态
  graph/           LangGraph 节点、边、构图与 checkpoint
  llm/             LLM 客户端
  memory/          Agent memory 与 Redis memory
  observations/    工具输出解析与风险信号提取
  policy/          风险策略与审批策略
  reporting/       报告生成
  tools/           LLM tool schema、dispatcher、manifest

configs/
  app.yaml             应用配置
  kali.yaml            Docker/Kali 执行环境配置
  policy.yaml          风险与审批策略
  tools.yaml           可执行工具 profile
  tool_manifest.yaml   暴露给 LLM 的工具说明
  tool_installs.yaml   容器缺失工具安装白名单

docker/
  autoflow-kali-tools/ 工具镜像定义

scripts/
  build_tool_image.py
  check_tool_image.py
  check_redis_connection.py
  check_redis_checkpoint.py
  test_llm_connection.py
  run_assessment.py
  run_stepwise_assessment.py

data/
  artifacts/       工具原始输出和中间证据
  reports/         Markdown 报告

docs/
  architecture.md
  agent-workflow.md
  kali-adapter.md
  safety-policy.md
```

## 环境要求

- Python 3.11+
- Docker
- Redis 或 Redis Stack
- 可用的 OpenAI-compatible LLM API

## 安装

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
```

复制环境变量文件：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

关键配置示例：

```env
LLM_MODEL=your-model
LLM_BASE_URL=https://your-llm-provider.example/v1
LLM_API_KEY=your_api_key

REDIS_ENABLED=true
REDIS_URL=redis://your-redis-host:6379/0
REDIS_KEY_PREFIX=autoflow

CHECKPOINT_BACKEND=redis
```

请不要把真实 API key 提交到仓库。

## 构建工具镜像

```bash
python scripts/build_tool_image.py --tag autoflow-kali-tools:latest
```

检查镜像内工具：

```bash
python scripts/check_tool_image.py --image autoflow-kali-tools:latest
```

## 基础检查

测试 LLM 连接：

```bash
python scripts/test_llm_connection.py
```

测试 Redis 连接：

```bash
python scripts/check_redis_connection.py
```

测试 Redis checkpoint：

```bash
python scripts/check_redis_checkpoint.py --thread-id autoflow-checkpoint-smoke
```

## 运行评估

分阶段运行，适合观察每一步输出：

```bash
python scripts/run_stepwise_assessment.py \
  --target http://target.example:3001 \
  --project demo-assessment \
  --max-rounds 2 \
  --output data/reports/demo-assessment.md
```

受控链路运行，适合验证工程流程：

```bash
python scripts/run_stepwise_assessment.py \
  --target http://target.example:3001 \
  --project controlled-e2e \
  --offline-planner \
  --execute-limit 2 \
  --validation-execute-limit 2 \
  --max-rounds 1 \
  --output data/reports/controlled-e2e.md
```

使用 LangGraph Redis checkpoint：

```bash
python scripts/run_assessment.py \
  --target http://target.example:3001 \
  --project checkpoint-demo \
  --checkpoint-backend redis \
  --thread-id checkpoint-demo \
  --output data/reports/checkpoint-demo.md
```

Windows PowerShell 可使用反引号换行：

```powershell
python scripts/run_stepwise_assessment.py `
  --target http://target.example:3001 `
  --project demo-assessment `
  --max-rounds 2 `
  --output data/reports/demo-assessment.md
```

## 测试

```bash
python -m pytest -q
```

## 输出结果

运行后主要产物包括：

```text
data/artifacts/
  工具原始输出、脚本输出、结构化结果

data/reports/
  Markdown 报告

Redis
  latest_state、memory_pack、events、observations、findings、validation_plans
```

## 安全边界

- 仅允许对授权范围内的目标执行任务。
- Discovery 阶段以只读和低风险动作优先。
- medium、high、critical 动作后续应接入审批流。
- 工具真实风险以 `configs/tools.yaml` 中的 profile 为准。
- 原始大输出写入 artifact，不直接塞入 LLM 上下文。
- 源码和制品扫描只访问受控目录。
- 可疑二进制默认只做类型识别，不默认解密、破解或暴力处理。

## 当前限制

- 长时间 LLM 多轮链路仍需要更强的可观测性。
- Checkpoint resume 需要继续产品化。
- 审批后继续执行还没有完全打通。
- Scheduler 并行调度仍处于后续规划阶段。
- 报告结构还需要继续向交付级渗透测试报告靠近。
- 高风险验证动作需要更细的策略、授权和审计设计。

## 相关文档

- `README-AUTOFLOW-ARCHITECTURE.md`：完整架构说明。
- `version0.6.md`：阶段版本说明。
- `docs/architecture.md`：架构设计。
- `docs/agent-workflow.md`：Agent 工作流。
- `docs/kali-adapter.md`：Kali / Docker 执行环境说明。
- `docs/safety-policy.md`：安全边界和使用规范。

