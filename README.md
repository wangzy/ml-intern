<p align="center">
  <img src="frontend/public/smolagents.webp" alt="smolagents logo" width="160" />
</p>

# ML Intern

An ML intern that autonomously researches, writes, and ships good quality ML related code using the Hugging Face ecosystem — with deep access to docs, papers, datasets, and cloud compute.

## Quick Start

### Installation

```bash
git clone git@github.com:huggingface/ml-intern.git
cd ml-intern
uv sync
uv tool install -e .
```

#### That's it. Now `ml-intern` works from any directory:

```bash
ml-intern
```

Create a `.env` file in the project root (or export these in your shell):

```bash
ANTHROPIC_API_KEY=<your-anthropic-api-key> # if using anthropic models
OPENAI_API_KEY=<your-openai-api-key>     # if using openai models
DEEPSEEK_API_KEY=<your-deepseek-api-key> # if using deepseek models
HF_TOKEN=<your-hugging-face-token>
GITHUB_TOKEN=<github-personal-access-token> 
```
If no `HF_TOKEN` is set, the CLI will prompt you to paste one on first launch. To get a GITHUB_TOKEN follow the tutorial [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token).

### Usage

**Interactive mode** (start a chat session):

```bash
ml-intern
```

**Headless mode** (single prompt, auto-approve):

```bash
ml-intern "fine-tune llama on my dataset"
```

**Options:**

```bash
ml-intern --model anthropic/claude-opus-4-6 "your prompt"
ml-intern --model openai/gpt-5.5 "your prompt"
ml-intern --model deepseek/deepseek-v4-pro "your prompt"
ml-intern --max-iterations 100 "your prompt"
ml-intern --no-stream "your prompt"
```

## Sharing Traces

Every session is auto-uploaded to your **own private Hugging Face dataset**
in [Claude Code JSONL format](https://huggingface.co/changelog/agent-trace-viewer),
which the HF Agent Trace Viewer auto-detects so you can browse turns, tool
calls, and model responses directly on the Hub.

By default the dataset is named `{your-hf-username}/ml-intern-sessions` and is
**created private**. You can flip it to public from inside the CLI:

```bash
/share-traces            # show current visibility + dataset URL
/share-traces public     # publish (anyone can view)
/share-traces private    # lock it back down
```

You can also flip visibility from the dataset page on huggingface.co — the
agent honours whatever you set there for subsequent uploads.

To opt out entirely, set in your CLI config (e.g. `configs/cli_agent_config.json`
or `~/.config/ml-intern/cli_agent_config.json`):

```json
{ "share_traces": false }
```

To override the destination repo, set:

```json
{ "personal_trace_repo_template": "{hf_user}/my-custom-traces" }
```

The shared `smolagents/ml-intern-sessions` dataset is unrelated and only
receives anonymized telemetry rows used by the backend KPI scheduler.

## Supported Gateways

ML Intern currently supports one-way notification gateways from CLI sessions.
These gateways send out-of-band status updates; they do not accept inbound chat
messages.

### Slack

Slack notifications use the Slack Web API to post messages when the agent needs
approval, hits an error, or completes a turn. Create a Slack app with a bot token
that has `chat:write`, invite the bot to the target channel, then set:

```bash
SLACK_BOT_TOKEN=xoxb-...
SLACK_CHANNEL_ID=C...
```

The CLI automatically creates a `slack.default` destination when both variables
are present. Optional environment variables for the env-only default:

```bash
ML_INTERN_SLACK_NOTIFICATIONS=false
ML_INTERN_SLACK_DESTINATION=slack.ops
ML_INTERN_SLACK_AUTO_EVENTS=approval_required,error,turn_complete
ML_INTERN_SLACK_ALLOW_AGENT_TOOL=true
ML_INTERN_SLACK_ALLOW_AUTO_EVENTS=true
```

For a persistent user-level config, put overrides in
`~/.config/ml-intern/cli_agent_config.json` or point `ML_INTERN_CLI_CONFIG` at a
JSON file:

```json
{
  "messaging": {
    "enabled": true,
    "auto_event_types": ["approval_required", "error", "turn_complete"],
    "destinations": {
      "slack.ops": {
        "provider": "slack",
        "token": "${SLACK_BOT_TOKEN}",
        "channel": "${SLACK_CHANNEL_ID}",
        "allow_agent_tool": true,
        "allow_auto_events": true
      }
    }
  }
}
```

### Custom LiteLLM Gateway

Route all LLM calls through your own LiteLLM proxy or gateway by setting:

```bash
export LITELLM_API_BASE="http://localhost:4000"
export LITELLM_API_KEY="sk-your-gateway-key"
```

When `LITELLM_API_BASE` is set, every model — including `anthropic/`, `openai/`,
`deepseek/`, `bedrock/`, and HF Router IDs — is routed through that endpoint
with `LITELLM_API_KEY` as the auth token. The gateway receives the model name
with its provider prefix (e.g. `openai/gpt-5.5`) and handles upstream routing.
Provider-specific parameters like reasoning effort and thinking config are
forwarded as-is.

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         User/CLI                            │
└────────────┬─────────────────────────────────────┬──────────┘
             │ Operations                          │ Events
             ↓ (user_input, exec_approval,         ↑
      submission_queue  interrupt, compact, ...)  event_queue
             │                                          │
             ↓                                          │
┌────────────────────────────────────────────────────┐  │
│            submission_loop (agent_loop.py)         │  │
│  ┌──────────────────────────────────────────────┐  │  │
│  │  1. Receive Operation from queue             │  │  │
│  │  2. Route to handler (run_agent/compact/...) │  │  │
│  └──────────────────────────────────────────────┘  │  │
│                      ↓                             │  │
│  ┌──────────────────────────────────────────────┐  │  │
│  │         Handlers.run_agent()                 │  ├──┤
│  │                                              │  │  │
│  │  ┌────────────────────────────────────────┐  │  │  │
│  │  │  Agentic Loop (max 300 iterations)     │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  ┌──────────────────────────────────┐  │  │  │  │
│  │  │  │ Session                          │  │  │  │  │
│  │  │  │  ┌────────────────────────────┐  │  │  │  │  │
│  │  │  │  │ ContextManager             │  │  │  │  │  │
│  │  │  │  │ • Message history          │  │  │  │  │  │
│  │  │  │  │   (litellm.Message[])      │  │  │  │  │  │
│  │  │  │  │ • Auto-compaction (170k)   │  │  │  │  │  │
│  │  │  │  │ • Session upload to HF     │  │  │  │  │  │
│  │  │  │  └────────────────────────────┘  │  │  │  │  │
│  │  │  │                                  │  │  │  │  │
│  │  │  │  ┌────────────────────────────┐  │  │  │  │  │
│  │  │  │  │ ToolRouter                 │  │  │  │  │  │
│  │  │  │  │  ├─ HF docs & research     │  │  │  │  │  │
│  │  │  │  │  ├─ HF repos, datasets,    │  │  │  │  │  │
│  │  │  │  │  │  jobs, papers           │  │  │  │  │  │
│  │  │  │  │  ├─ GitHub code search     │  │  │  │  │  │
│  │  │  │  │  ├─ Sandbox & local tools  │  │  │  │  │  │
│  │  │  │  │  ├─ Planning               │  │  │  │  │  │
│  │  │  │  │  └─ MCP server tools       │  │  │  │  │  │
│  │  │  │  └────────────────────────────┘  │  │  │  │  │
│  │  │  └──────────────────────────────────┘  │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  ┌──────────────────────────────────┐  │  │  │  │
│  │  │  │ Doom Loop Detector               │  │  │  │  │
│  │  │  │ • Detects repeated tool patterns │  │  │  │  │
│  │  │  │ • Injects corrective prompts     │  │  │  │  │
│  │  │  └──────────────────────────────────┘  │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  Loop:                                 │  │  │  │
│  │  │    1. LLM call (litellm.acompletion)   │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    2. Parse tool_calls[]               │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    3. Approval check                   │  │  │  │
│  │  │       (jobs, sandbox, destructive ops) │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    4. Execute via ToolRouter           │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    5. Add results to ContextManager    │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    6. Repeat if tool_calls exist       │  │  │  │
│  │  └────────────────────────────────────────┘  │  │  │
│  └──────────────────────────────────────────────┘  │  │
└────────────────────────────────────────────────────┴──┘
```

### Agentic Loop Flow

```
User Message
     ↓
[Add to ContextManager]
     ↓
     ╔═══════════════════════════════════════════╗
     ║      Iteration Loop (max 300)             ║
     ║                                           ║
     ║  Get messages + tool specs                ║
     ║         ↓                                 ║
     ║  litellm.acompletion()                    ║
     ║         ↓                                 ║
     ║  Has tool_calls? ──No──> Done             ║
     ║         │                                 ║
     ║        Yes                                ║
     ║         ↓                                 ║
     ║  Add assistant msg (with tool_calls)      ║
     ║         ↓                                 ║
     ║  Doom loop check                          ║
     ║         ↓                                 ║
     ║  For each tool_call:                      ║
     ║    • Needs approval? ──Yes──> Wait for    ║
     ║    │                         user confirm ║
     ║    No                                     ║
     ║    ↓                                      ║
     ║    • ToolRouter.execute_tool()            ║
     ║    • Add result to ContextManager         ║
     ║         ↓                                 ║
     ║  Continue loop ─────────────────┐         ║
     ║         ↑                       │         ║
     ║         └───────────────────────┘         ║
     ╚═══════════════════════════════════════════╝
```

## Source Tree

```
├── agent/                          # Core agent logic (Python package)
│   ├── main.py                     # CLI entry point
│   ├── config.py                   # Config loading, env var substitution
│   ├── core/                       # Agent engine
│   │   ├── agent_loop.py           # Main agent loop (submission, LLM, tools)
│   │   ├── session.py              # Session, Event, ContextManager classes
│   │   ├── tools.py                # ToolRouter, ToolSpec, built-in tool registry
│   │   ├── llm_params.py           # LiteLLM kwargs resolution per provider
│   │   ├── model_switcher.py       # CLI /model command + probe orchestration
│   │   ├── effort_probe.py         # Reasoning effort probe cascade
│   │   ├── doom_loop.py            # Doom loop detection
│   │   ├── approval_policy.py      # Approval rules for sensitive operations
│   │   ├── cost_estimation.py      # Cost estimation for tool operations
│   │   ├── prompt_caching.py       # Anthropic prompt caching breakpoints
│   │   ├── hf_access.py            # HF API access helpers
│   │   ├── hf_router_catalog.py    # HF Router model catalog pre-warming
│   │   ├── hf_tokens.py            # Token resolution (env, HF CLI, etc.)
│   │   ├── redact.py               # Sensitive data redaction
│   │   ├── session_persistence.py  # Session save/load to MongoDB/file
│   │   ├── session_uploader.py     # Upload to HF datasets
│   │   └── telemetry.py            # Heartbeat saving, telemetry
│   ├── tools/                      # Tool implementations
│   │   ├── docs_tools.py           # HF documentation search/fetch
│   │   ├── papers_tool.py          # Academic paper search
│   │   ├── jobs_tool.py            # HF Jobs compute
│   │   ├── sandbox_tool.py         # Sandbox environment management
│   │   ├── sandbox_client.py       # Remote sandbox client
│   │   ├── dataset_tools.py        # Dataset inspection
│   │   ├── research_tool.py        # Sub-agent research
│   │   ├── plan_tool.py            # Task planning
│   │   ├── github_find_examples.py # GitHub code search
│   │   ├── github_list_repos.py    # GitHub repo listing
│   │   ├── github_read_file.py     # GitHub file reading
│   │   ├── hf_repo_files_tool.py   # HF repo file operations
│   │   ├── hf_repo_git_tool.py     # HF repo git operations
│   │   ├── notify_tool.py          # Slack notifications
│   │   ├── web_search_tool.py      # Web search
│   │   └── local_tools.py          # Local shell/file operations
│   ├── context_manager/            # Context window compaction
│   │   └── manager.py
│   ├── messaging/                  # Notification gateway (Slack)
│   │   ├── gateway.py
│   │   ├── slack.py
│   │   └── models.py
│   ├── prompts/                    # System prompts (YAML)
│   └── sft/                        # SFT dataset building
│       └── tagger.py
├── backend/                        # FastAPI web backend
│   ├── main.py                     # App setup, CORS, static files
│   ├── session_manager.py          # Multi-session management, SSE broadcasting
│   ├── dependencies.py             # Auth dependency injection
│   ├── user_quotas.py              # Per-user rate limiting
│   ├── kpis_scheduler.py           # In-process KPI rollup scheduler
│   ├── models.py                   # Pydantic API models
│   ├── start.sh                    # Production start script
│   └── routes/
│       ├── agent.py                # Agent SSE/WS/REST endpoints
│       └── auth.py                 # HF OAuth login/callback
├── frontend/                       # React SPA
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat/               # Chat message components
│   │   │   ├── CodePanel/          # Code editing panel
│   │   │   ├── Layout/             # App shell layout
│   │   │   ├── SessionSidebar/     # Session list sidebar
│   │   │   ├── WelcomeScreen/      # Landing/welcome screen
│   │   │   ├── SessionChat.tsx      # Main chat session view
│   │   │   ├── ClaudeCapDialog.tsx  # Claude capability dialog
│   │   │   ├── JobsUpgradeDialog.tsx # Jobs hardware upgrade dialog
│   │   │   └── YoloControl.tsx      # YOLO mode toggle
│   │   ├── hooks/                   # React hooks
│   │   │   ├── useAgentChat.ts      # AI SDK chat hook (SSE transport)
│   │   │   ├── useAuth.ts           # Auth state hook
│   │   │   └── useUserQuota.ts      # Quota tracking hook
│   │   ├── store/                   # Zustand stores
│   │   │   ├── agentStore.ts        # Agent state
│   │   │   ├── sessionStore.ts      # Session management
│   │   │   └── layoutStore.ts       # UI layout state
│   │   ├── lib/                     # Client libraries
│   │   │   ├── sse-chat-transport.ts # SSE transport for AI SDK
│   │   │   └── chat-message-store.ts # Client-side message store
│   │   ├── types/                   # TypeScript type definitions
│   │   └── utils/                   # Shared utilities
│   ├── vite.config.ts               # Vite config with API proxy
│   └── package.json
├── configs/                         # JSON config files
│   ├── cli_agent_config.json        # CLI defaults
│   └── frontend_agent_config.json   # Web UI defaults
├── scripts/                         # Utility scripts
│   ├── build_kpis.py
│   ├── build_sft.py
│   └── sweep_orphan_sandboxes.py
├── tests/
│   ├── unit/                        # 30 unit test files
│   └── integration/                 # 2 integration test files
├── Dockerfile                       # Multi-stage Docker build
├── pyproject.toml                   # Python project metadata and dependencies
└── README.md
```

## Events

The agent emits the following events via `event_queue`:

- `processing` - Starting to process user input
- `ready` - Agent is ready for input
- `assistant_chunk` - Streaming token chunk
- `assistant_message` - Complete LLM response text
- `assistant_stream_end` - Token stream finished
- `tool_call` - Tool being called with arguments
- `tool_output` - Tool execution result
- `tool_log` - Informational tool log message
- `tool_state_change` - Tool execution state transition
- `approval_required` - Requesting user approval for sensitive operations
- `turn_complete` - Agent finished processing
- `error` - Error occurred during processing
- `interrupted` - Agent was interrupted
- `compacted` - Context was compacted
- `undo_complete` - Undo operation completed
- `shutdown` - Agent shutting down

## Development

### Adding Built-in Tools

Edit `agent/core/tools.py`:

```python
def create_builtin_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="your_tool",
            description="What your tool does",
            parameters={
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "Parameter description"}
                },
                "required": ["param"]
            },
            handler=your_async_handler
        ),
        # ... existing tools
    ]
```

### Adding MCP Servers

Edit `configs/cli_agent_config.json` for CLI defaults, or
`configs/frontend_agent_config.json` for web-session defaults:

```json
{
  "model_name": "anthropic/claude-sonnet-4-5-20250929",
  "mcpServers": {
    "your-server-name": {
      "transport": "http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${YOUR_TOKEN}"
      }
    }
  }
}
```

Note: Environment variables like `${YOUR_TOKEN}` are auto-substituted from `.env`.
