# AutoFlow

AutoFlow 是一个面向授权安全评估和靶场测试的多 Agent 渗透测试自动化框架原型。它基于 LangGraph 组织工作流，使用 LLM 做规划和推理，通过 Docker 工具容器执行安全工具，并用 Redis 保存运行时记忆和 checkpoint。


## 当前能力

- LangGraph 多阶段工作流。
- LLM Planner / DiscoveryReasoner / ValidationAgent。
- OpenAI-compatible LLM API。
- LLM function calling 工具调用。
- Docker 内执行 nmap、whatweb、nikto、nuclei、curl、feroxbuster、sqlmap 等工具。
- Web recon 页面结构采集。
- ToolObservation -> Candidate Finding。
- Candidate Finding -> ValidationPlan。
- 部分漏洞验证执行与 Finding 状态更新。
- Redis runtime memory。
- LangGraph Redis checkpointer。
- Markdown 报告输出。

## 工作流

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

核心产物流转：

```text
Tool Output
  -> ToolObservation
  -> Candidate Finding
  -> ValidationPlan
  -> ValidationResult
  -> Report
```

## 目录结构

```text
autoflow/
  agents/        Agent 实现
  graph/         LangGraph 工作流、节点、路由、checkpoint
  tools/         LLM tool schema、tool dispatcher、tool manifest
  executor/      Docker 工具执行、脚本执行、shell 执行
  memory/        Agent memory、Redis memory
  observations/  工具输出解析
  policy/        风险策略和审批
  artifacts/     原始证据和报告产物
  api/           后续前端/API 接入

configs/
  tools.yaml          可执行工具 profile
  tool_manifest.yaml  LLM 可读工具说明
  policy.yaml         风险和审批策略
  tool_installs.yaml  容器缺失工具安装白名单

scripts/
  build_tool_image.py
  check_tool_image.py
  check_redis_connection.py
  check_redis_checkpoint.py
  test_llm_connection.py
  run_assessment.py
  run_stepwise_assessment.py

data/
  artifacts/     工具原始输出
  reports/       Markdown 报告
```

## 环境配置

复制环境变量文件：

```powershell
Copy-Item .env.example .env
```

关键配置：

```env
LLM_MODEL=deepseek-v4-flash
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your_key

REDIS_ENABLED=true
REDIS_URL=redis://192.168.34.191:6379/0
REDIS_KEY_PREFIX=autoflow

CHECKPOINT_BACKEND=redis
```

## 安装

建议使用当前项目环境 `qwen-skills`：

```powershell
D:\Anaconda\path\envs\qwen-skills\python.exe -m pip install -e .
D:\Anaconda\path\envs\qwen-skills\python.exe -m pip install -e ".[dev]"
```

## Docker 工具镜像

构建工具镜像：

```powershell
D:\Anaconda\path\envs\qwen-skills\python.exe scripts/build_tool_image.py --tag autoflow-kali-tools:latest
```

检查镜像工具：

```powershell
D:\Anaconda\path\envs\qwen-skills\python.exe scripts/check_tool_image.py --image autoflow-kali-tools:latest
```

## 基础检查

测试 LLM：

```powershell
D:\Anaconda\path\envs\qwen-skills\python.exe scripts/test_llm_connection.py
```

测试 Redis：

```powershell
D:\Anaconda\path\envs\qwen-skills\python.exe scripts/check_redis_connection.py
```

测试 Redis checkpoint：

```powershell
D:\Anaconda\path\envs\qwen-skills\python.exe scripts/check_redis_checkpoint.py --thread-id autoflow-checkpoint-smoke
```

## 运行评估

分阶段运行，适合观察每一步：

```powershell
D:\Anaconda\path\envs\qwen-skills\python.exe scripts/run_stepwise_assessment.py `
  --target 192.168.34.191:3001 `
  --project juice-shop-demo `
  --max-rounds 2 `
  --output data/reports/juice-shop-demo.md
```

受控测试链路：

```powershell
D:\Anaconda\path\envs\qwen-skills\python.exe scripts/run_stepwise_assessment.py `
  --target 192.168.34.191:3001 `
  --project redis-memory-controlled-e2e `
  --offline-planner `
  --execute-limit 2 `
  --validation-execute-limit 2 `
  --max-rounds 1 `
  --output data/reports/redis-memory-controlled-e2e.md
```

LangGraph checkpoint 运行：

```powershell
D:\Anaconda\path\envs\qwen-skills\python.exe scripts/run_assessment.py `
  --target 192.168.34.191:3001 `
  --project checkpoint-demo `
  --checkpoint-backend redis `
  --thread-id checkpoint-demo `
  --output data/reports/checkpoint-demo.md
```

## 测试

```powershell
D:\Anaconda\path\envs\qwen-skills\python.exe -m pytest -q
```

## 文档

- `README-AUTOFLOW-ARCHITECTURE.md`：完整架构说明。
- `version0.6.md`：当前版本设计和阶段性说明。
- `PROJECT_PLAN.md`：早期项目规划。

## 当前限制

- 完整 LLM 多轮链路耗时较长，10 分钟超时不一定代表报错。
- checkpoint 已能写入 Redis，但审批后 resume 还需要继续完善。
- Scheduler 并行调度仍是预留能力。
- 报告还需要继续向正式渗透测试交付格式升级。
