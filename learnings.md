# mini-swe-agent Codebase Learnings

## Overview

**mini-swe-agent** is a minimal yet high-performing AI software engineering agent built by the Princeton & Stanford team behind SWE-bench. It achieves >74% on SWE-bench verified while being radically simple (~100 lines for the core agent).

### Key Design Philosophy

- **Minimal**: No custom tools beyond bash; executes via `subprocess.run`
- **Performant**: Fast startup, competitive benchmark scores
- **Deployable**: Works with local, Docker, Podman, Singularity, and cloud environments

---

## Core Architecture

The system follows a **protocol-based plugin design** with three main components:

### 1. Agents (`src/minisweagent/agents/`)

- `DefaultAgent` - Core ~100-line agent with the main control loop
- `InteractiveAgent` - Human-in-the-loop extension with modes: `human`, `confirm`, `yolo`

### 2. Models (`src/minisweagent/models/`)

- `LitellmModel` - Default, uses function calling via LiteLLM (supports 100+ LLM APIs)
- `LitellmTextbasedModel` - Regex-based action extraction from markdown
- `OpenRouterModel`, `PortkeyModel`, `RequestyModel` - Alternative API backends
- `RouletteModel`, `InterleavingModel` - Meta-models for model selection strategies

### 3. Environments (`src/minisweagent/environments/`)

- `LocalEnvironment` - Direct subprocess execution
- `DockerEnvironment` - Container-based execution
- `SingularityEnvironment` - HPC container support
- `BubblewrapEnvironment`, `SwerexModalEnvironment` - Additional sandboxing options

---

## Architectural Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  CLI (mini.py)                    │     Python API                          │
│  $ mini --config my.yaml          │     agent = DefaultAgent(model, env)    │
│                                   │     agent.run("fix the bug")            │
└───────────────────────────────────┴─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONFIGURATION LAYER                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                          │
│  │ default.yaml│  │  mini.yaml  │  │  CLI args   │   ← Merged recursively   │
│  └─────────────┘  └─────────────┘  └─────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AGENT CORE                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  DefaultAgent / InteractiveAgent                                      │  │
│  │                                                                       │  │
│  │  run(task):                                                           │  │
│  │    while True:                                                        │  │
│  │      ┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │  │
│  │      │ 1. Query LM │───▶│ 2. Execute  │───▶│ 3. Observe  │───┐       │  │
│  │      │   (Model)   │    │   (Environ) │    │   Format    │   │       │  │
│  │      └─────────────┘    └─────────────┘    └─────────────┘   │       │  │
│  │            ▲                                                  │       │  │
│  │            └──────────────────────────────────────────────────┘       │  │
│  │                         (loop until exit/submit/limit)                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
          │                              │                              │
          ▼                              ▼                              ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│       MODEL         │    │     ENVIRONMENT     │    │    TRAJECTORY       │
│   (Protocol)        │    │     (Protocol)      │    │    OUTPUT           │
├─────────────────────┤    ├─────────────────────┤    ├─────────────────────┤
│ • LitellmModel      │    │ • LocalEnvironment  │    │ • JSON file         │
│ • LitellmTextbased  │    │ • DockerEnvironment │    │ • Messages          │
│ • OpenRouterModel   │    │ • SingularityEnv    │    │ • Metadata          │
│ • PortkeyModel      │    │ • BubblewrapEnv     │    │ • Cost tracking     │
│ • RouletteModel     │    │ • SwerexModalEnv    │    │                     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────┐    ┌─────────────────────┐
│   EXTERNAL APIs     │    │  EXECUTION TARGET   │
├─────────────────────┤    ├─────────────────────┤
│ • OpenAI            │    │ • Local subprocess  │
│ • Anthropic         │    │ • Docker container  │
│ • OpenRouter        │    │ • Singularity       │
│ • Portkey           │    │ • Modal (cloud)     │
│ • Any LiteLLM API   │    │ • Bubblewrap        │
└─────────────────────┘    └─────────────────────┘
```

---

## Data Flow

```
Task Description
      │
      ▼
┌─────────────────┐
│ System Prompt   │  ← Jinja2 template rendering
│ + Task Context  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Model.query()  │────▶│  LLM API Call   │
│                 │◀────│  (litellm)      │
└────────┬────────┘     └─────────────────┘
         │
         │ Parse tool calls → actions
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Env.execute()   │────▶│ subprocess.run  │
│                 │◀────│ or docker exec  │
└────────┬────────┘     └─────────────────┘
         │
         │ {output, returncode, exception_info}
         ▼
┌─────────────────┐
│ Format & append │
│ to messages     │
└────────┬────────┘
         │
         ▼
   Exit condition?
   ├─ SUBMIT → Save trajectory, return
   ├─ LIMIT  → LimitsExceeded exception
   └─ NO     → Loop back to Model.query()
```

---

## Key Files

| Path                                       | Purpose                                          |
| ------------------------------------------ | ------------------------------------------------ |
| `src/minisweagent/__init__.py`             | Protocol definitions (Model, Environment, Agent) |
| `src/minisweagent/agents/default.py`       | Core agent implementation (~100 lines)           |
| `src/minisweagent/agents/interactive.py`   | Human-in-the-loop agent                          |
| `src/minisweagent/models/litellm_model.py` | Primary LLM interface                            |
| `src/minisweagent/environments/local.py`   | Local subprocess execution                       |
| `src/minisweagent/run/mini.py`             | CLI entry point                                  |
| `src/minisweagent/config/default.yaml`     | Default configuration                            |

---

## Core Agent Code Deep Dive

### The `run()` Method (`default.py` lines 77-97)

```python
def run(self, task: str = "", **kwargs) -> dict:
    """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
    self.extra_template_vars |= {"task": task, **kwargs}
    self.messages = []
    self.add_messages(
        self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
        self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
    )
    while True:
        try:
            self.step()
        except InterruptAgentFlow as e:
            self.add_messages(*e.messages)
        except Exception as e:
            self.handle_uncaught_exception(e)
            raise
        finally:
            self.save(self.config.output_path)
        if self.messages[-1].get("role") == "exit":
            break
    return self.messages[-1].get("extra", {})
```

### What `run()` Does:

1. **Initialize** — Stores the task in template vars, clears message history
2. **Build initial messages** — Renders system prompt and task description using Jinja2 templates
3. **Main loop** — Repeatedly calls `step()` until done:
   - `step()` = `query()` + `execute_actions()`
   - Catches `InterruptAgentFlow` (for controlled exits like submission)
   - Catches other exceptions and records them
   - Saves trajectory after every step (in `finally`)
   - Exits when the last message has `role="exit"`
4. **Return** — Returns the final message's `extra` dict (contains `exit_status`, `submission`)

### Supporting Methods

| Method              | Lines   | Purpose                                                       |
| ------------------- | ------- | ------------------------------------------------------------- |
| `step()`            | 99-101  | One agent turn: query LM → execute actions                    |
| `query()`           | 103-117 | Check limits, call model, track cost, add response to history |
| `execute_actions()` | 119-122 | Run each bash command via environment, format observations    |
| `save()`            | 147-155 | Serialize and write trajectory to JSON                        |

### Control Flow Diagram

```
run(task)
    │
    ├─► Initialize messages with system + task prompts
    │
    └─► while True:
            │
            ├─► step()
            │     ├─► query()
            │     │     ├─► Check cost/step limits
            │     │     ├─► model.query(messages)
            │     │     └─► Track cost, add to history
            │     │
            │     └─► execute_actions()
            │           ├─► env.execute(action) for each action
            │           └─► Format observations, add to history
            │
            ├─► Handle exceptions (InterruptAgentFlow, others)
            │
            ├─► save() trajectory
            │
            └─► if last message role == "exit": break

    return {exit_status, submission}
```

---

## Template Rendering

The `_render_template()` method processes Jinja2 templates and returns a string:

```python
def _render_template(self, template: str) -> str:
    return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())
```

### Where the rendered string goes:

In `run()`, the rendered strings become the **content** of messages:

```python
self.add_messages(
    self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
    self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
)
```

### Flow:

```
Template string (with {{ variables }})
        │
        ▼
_render_template()  ──► Substitutes variables like {{ task }}, {{ cwd }}, etc.
        │
        ▼
Plain string (rendered)
        │
        ▼
model.format_message(content=...)  ──► Wraps in message dict
        │
        ▼
add_messages()  ──► Appends to self.messages list
        │
        ▼
Sent to LLM in query()
```

### Example:

If `system_template` contains:

```
You are an AI assistant. The current directory is {{ cwd }}.
```

And `cwd` is `/home/user/project`, then `_render_template()` returns:

```
You are an AI assistant. The current directory is /home/user/project.
```

That string becomes the system message's content sent to the LLM.

---

## User Flow Example: "Fix the bug in login.py"

### Step 1: User starts the agent

```python
from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.environments.local import LocalEnvironment

agent = DefaultAgent(
    model=LitellmModel(model_name="claude-sonnet-4-20250514"),
    env=LocalEnvironment(),
    system_template="You are a coding assistant...",
    instance_template="Task: {{ task }}",
)
agent.run("Fix the bug in login.py")
```

### Step 2: `run()` initializes (lines 77-84)

```python
def run(self, task: str = "", **kwargs) -> dict:
    # Store task for template rendering
    self.extra_template_vars |= {"task": "Fix the bug in login.py", **kwargs}

    # Clear message history
    self.messages = []

    # Build initial messages
    self.add_messages(
        self.model.format_message(role="system", content="You are a coding assistant..."),
        self.model.format_message(role="user", content="Task: Fix the bug in login.py"),
    )
```

**State after Step 2:**

```python
self.messages = [
    {"role": "system", "content": "You are a coding assistant..."},
    {"role": "user", "content": "Task: Fix the bug in login.py"},
]
```

### Step 3: Enter the main loop (lines 85-97)

```python
    while True:
        try:
            self.step()  # ◄── This does the work
        except InterruptAgentFlow as e:
            self.add_messages(*e.messages)
        except Exception as e:
            self.handle_uncaught_exception(e)
            raise
        finally:
            self.save(self.config.output_path)  # Save after every step
        if self.messages[-1].get("role") == "exit":
            break
```

### Step 4: `step()` calls `query()` (lines 99-117)

```python
def step(self) -> list[dict]:
    return self.execute_actions(self.query())

def query(self) -> dict:
    # Check if we've exceeded limits
    if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
        raise LimitsExceeded(...)

    self.n_calls += 1  # Now n_calls = 1

    # Call the LLM with current messages
    message = self.model.query(self.messages)
    # Returns something like:
    # {
    #     "role": "assistant",
    #     "content": "I'll look at login.py first.",
    #     "extra": {
    #         "actions": [{"command": "cat login.py", "tool_call_id": "abc123"}],
    #         "cost": 0.003
    #     }
    # }

    self.cost += 0.003  # Track cost
    self.add_messages(message)  # Add to history
    return message
```

**State after query():**

```python
self.messages = [
    {"role": "system", "content": "You are a coding assistant..."},
    {"role": "user", "content": "Task: Fix the bug in login.py"},
    {"role": "assistant", "content": "I'll look at login.py first.", "extra": {...}},  # NEW
]
self.cost = 0.003
self.n_calls = 1
```

### Step 5: `execute_actions()` runs the command (lines 119-122)

```python
def execute_actions(self, message: dict) -> list[dict]:
    # Extract actions from the message
    actions = message.get("extra", {}).get("actions", [])
    # actions = [{"command": "cat login.py", "tool_call_id": "abc123"}]

    # Execute each action in the environment
    outputs = [self.env.execute(action) for action in actions]
    # env.execute() runs: subprocess.run("cat login.py", ...)
    # outputs = [{"output": "def login(user, pwd):\n    ...", "returncode": 0}]

    # Format as observation messages and add to history
    return self.add_messages(
        *self.model.format_observation_messages(message, outputs, self.get_template_vars())
    )
```

**State after execute_actions():**

```python
self.messages = [
    {"role": "system", "content": "You are a coding assistant..."},
    {"role": "user", "content": "Task: Fix the bug in login.py"},
    {"role": "assistant", "content": "I'll look at login.py first.", "extra": {...}},
    {"role": "user", "content": "def login(user, pwd):\n    ...", "extra": {"tool_call_id": "abc123"}},  # NEW
]
```

### Step 6: Loop continues...

Back to `while True:` → `step()` → `query()` → LLM sees the file content → responds with a fix.

### Step 7: Task completion

Eventually, the LLM decides it's done and returns a submission message. The model raises `InterruptAgentFlow` with an exit message:

```python
# Caught in run():
except InterruptAgentFlow as e:
    self.add_messages(*e.messages)
    # Adds: {"role": "exit", "content": "...", "extra": {"exit_status": "submitted", "submission": "Fixed the == bug"}}
```

**Exit check passes:**

```python
if self.messages[-1].get("role") == "exit":  # True!
    break
```

### Step 8: Return result (line 97)

```python
return self.messages[-1].get("extra", {})
# Returns: {"exit_status": "submitted", "submission": "Fixed the == bug"}
```

### Visual Summary

```
User calls agent.run("Fix the bug in login.py")
                │
                ▼
        ┌───────────────────┐
        │ Initialize        │
        │ messages = [      │
        │   system prompt,  │
        │   task prompt     │
        │ ]                 │
        └────────┬──────────┘
                 │
    ┌────────────▼────────────┐
    │      while True:        │◄─────────────────────┐
    │                         │                      │
    │  ┌─────────────────┐    │                      │
    │  │ query()         │    │                      │
    │  │  • Check limits │    │                      │
    │  │  • Call LLM     │    │                      │
    │  │  • Track cost   │    │                      │
    │  └────────┬────────┘    │                      │
    │           ▼             │                      │
    │  ┌─────────────────┐    │                      │
    │  │ execute_actions │    │                      │
    │  │  • Run bash cmd │    │                      │
    │  │  • Add output   │    │                      │
    │  └────────┬────────┘    │                      │
    │           ▼             │                      │
    │  ┌─────────────────┐    │      No              │
    │  │ role == "exit"? ├────┼──────────────────────┘
    │  └────────┬────────┘    │
    │           │ Yes         │
    └───────────┼─────────────┘
                ▼
        Return {exit_status, submission}
```

---

## How Model Query Works

### The `query()` method in `LitellmModel` (lines 80-93):

```python
def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
    # 1. Retry loop for transient failures
    for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
        with attempt:
            # 2. Call the LLM API
            response = self._query(self._prepare_messages_for_api(messages), **kwargs)

    # 3. Calculate cost
    cost_output = self._calculate_cost(response)
    GLOBAL_MODEL_STATS.add(cost_output["cost"])

    # 4. Extract the message from API response
    message = response.choices[0].message.model_dump()

    # 5. Parse tool calls into actions and attach metadata
    message["extra"] = {
        "actions": self._parse_actions(response),  # ◄── Key: extracts bash commands
        "response": response.model_dump(),
        **cost_output,
        "timestamp": time.time(),
    }
    return message
```

### The actual API call in `_query()` (lines 63-73):

```python
def _query(self, messages, **kwargs):
    return litellm.completion(
        model=self.config.model_name,
        messages=messages,
        tools=[BASH_TOOL],  # ◄── Tells LLM it can call bash
        **(self.config.model_kwargs | kwargs),
    )
```

### The `BASH_TOOL` definition:

```python
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}
```

This is the **only tool** the agent has — just bash. The LLM calls it via the standard OpenAI function-calling format.

---

## How the Agent Knows What to Run Next

**The key insight: The agent doesn't decide — the LLM does.**

The agent simply passes the **entire message history** to the LLM each time, and the LLM decides what bash command to run next based on:

1. The system prompt (instructions)
2. The task description
3. All previous commands and their outputs

### Flow diagram:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         self.messages                               │
├─────────────────────────────────────────────────────────────────────┤
│ [0] system:    "You are a coding assistant. Use bash to..."        │
│ [1] user:      "Task: Fix the bug in login.py"                     │
│ [2] assistant: "I'll read the file" + tool_call: cat login.py      │
│ [3] tool:      "def login(user, pwd):..."  (output of cat)         │
│ [4] assistant: "I see the bug" + tool_call: sed -i 's/...'         │
│ [5] tool:      "" (empty output, sed succeeded)                    │
│ [6] assistant: "Let me verify" + tool_call: cat login.py           │
│ [7] tool:      "def login(user, pwd):..." (fixed code)             │
│ ... and so on                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   model.query()       │
                    │                       │
                    │   Sends ALL messages  │
                    │   to LLM API          │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   LLM Response        │
                    │                       │
                    │   "Now I'll run the   │
                    │   tests to confirm"   │
                    │                       │
                    │   tool_call: pytest   │
                    └───────────────────────┘
```

### The loop in detail:

```python
# In DefaultAgent.run():
while True:
    self.step()  # query + execute
    # ...
    if self.messages[-1].get("role") == "exit":
        break

# step() does:
def step(self):
    return self.execute_actions(self.query())

# query() does:
def query(self):
    message = self.model.query(self.messages)  # ◄── Pass ENTIRE history
    self.add_messages(message)                  # ◄── Append LLM response
    return message

# execute_actions() does:
def execute_actions(self, message):
    outputs = [self.env.execute(action) for action in message["extra"]["actions"]]
    # Format outputs and append to self.messages
    return self.add_messages(*self.model.format_observation_messages(...))
```

### So the "next action" logic is:

| Step | Who decides | What happens                                   |
| ---- | ----------- | ---------------------------------------------- |
| 1    | Agent       | Sends full `self.messages` to LLM              |
| 2    | **LLM**     | Sees entire history, decides next bash command |
| 3    | Agent       | Parses tool calls → `actions` list             |
| 4    | Agent       | Executes each action via `env.execute()`       |
| 5    | Agent       | Appends output to `self.messages`              |
| 6    | Agent       | Loop back to step 1                            |

The agent is just a **dumb loop** — all intelligence comes from the LLM seeing the growing conversation history and deciding the next logical step.

---

## Key Takeaways

1. **Radical simplicity**: The entire agent loop is ~20 lines of code
2. **Protocol-based design**: Easy to swap models and environments via duck typing
3. **Single tool**: Only bash — no complex tool definitions
4. **Stateless execution**: Each bash command runs independently via `subprocess.run`
5. **LLM-driven**: The agent doesn't plan; it just relays history to the LLM and executes what it says
6. **Trajectory saving**: Every step is saved, enabling debugging and replay

---

## How Environments Work

### What is an Environment?

An **Environment** is the execution layer that sits between the agent's decision-making and the actual machine. It answers the question: **"Where and how do commands actually run?"**

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT                                    │
│                                                                 │
│   LLM says: "run `cat file.py`"                                │
│                          │                                      │
│                          ▼                                      │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  Environment.execute({"command": "cat file.py"})        │  │
│   └─────────────────────────────────────────────────────────┘  │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
   ┌────────────┐   ┌────────────┐   ┌────────────┐
   │   LOCAL    │   │   DOCKER   │   │ SINGULARITY│
   │            │   │            │   │            │
   │ Your Mac/  │   │ Isolated   │   │ HPC cluster│
   │ Linux box  │   │ container  │   │ container  │
   └────────────┘   └────────────┘   └────────────┘
```

### The Core Abstraction

The agent **doesn't know or care** where commands run. It just calls:

```python
output = self.env.execute({"command": "cat file.py"})
```

And gets back:

```python
{"output": "def foo():...", "returncode": 0, "exception_info": ""}
```

This separation is powerful because:

| Concern | Agent's Job | Environment's Job |
|---------|-------------|-------------------|
| **What** to run | ✅ Decides commands | ❌ Doesn't care |
| **Where** it runs | ❌ Doesn't care | ✅ Handles execution |
| **How** it runs | ❌ Doesn't care | ✅ subprocess, docker exec, etc. |
| **Isolation** | ❌ Doesn't care | ✅ Sandboxing, permissions |
| **Cleanup** | ❌ Doesn't care | ✅ Container lifecycle |

### Why Environments Matter

**1. Safety & Isolation** — When running untrusted LLM-generated code (like on SWE-bench), you don't want `rm -rf /` running on your host. Docker containers are throwaway.

**2. Reproducibility** — SWE-bench tasks need specific Python versions, dependencies, etc. Docker images provide consistent environments.

**3. Deployment Flexibility** — Same agent code works everywhere:

```python
# Development - fast iteration
agent = DefaultAgent(model, LocalEnvironment())

# CI/Testing - isolated
agent = DefaultAgent(model, DockerEnvironment(image="python:3.11"))

# HPC cluster - Singularity required
agent = DefaultAgent(model, SingularityEnvironment(image="swe-bench.sif"))
```

### The Three Responsibilities

Every environment must handle:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. EXECUTE                                                     │
│     Take a command dict → run it → return output dict           │
│     Also: detect task completion (COMPLETE_TASK_AND_SUBMIT...)  │
├─────────────────────────────────────────────────────────────────┤
│  2. PROVIDE CONTEXT                                             │
│     get_template_vars() → {"cwd": "/...", "system": "Darwin"}   │
│     Used in system prompts so LLM knows where it's running      │
├─────────────────────────────────────────────────────────────────┤
│  3. SERIALIZE                                                   │
│     serialize() → config snapshot for trajectory logging        │
│     Enables replay and debugging                                │
└─────────────────────────────────────────────────────────────────┘
```

---

### Environment Protocol

From `src/minisweagent/__init__.py:60-69`:

```python
class Environment(Protocol):
    config: Any

    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]: ...
    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...
    def serialize(self) -> dict: ...
```

Only **3 methods** required — very minimal interface.

---

### LocalEnvironment (Simplest Implementation)

From `local.py:23-53`:

```
┌─────────────────────────────────────────────────────────────────┐
│  execute(action)                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  action = {"command": "cat file.py", "tool_call_id": "abc123"} │
│                         │                                       │
│                         ▼                                       │
│  ┌─────────────────────────────────────────┐                   │
│  │  subprocess.run(                        │                   │
│  │      command,                           │                   │
│  │      shell=True,        ◄── Runs in sh  │                   │
│  │      cwd=cwd,           ◄── Working dir │                   │
│  │      timeout=30,        ◄── Default 30s │                   │
│  │      stdout=PIPE,                       │                   │
│  │      stderr=STDOUT,     ◄── Combined    │                   │
│  │  )                                      │                   │
│  └─────────────────────────────────────────┘                   │
│                         │                                       │
│                         ▼                                       │
│  return {                                                       │
│      "output": "def foo():...",   ◄── stdout+stderr            │
│      "returncode": 0,             ◄── Exit code                │
│      "exception_info": ""         ◄── Error details if any     │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

**Key characteristics:**
- Executes directly on host machine via `subprocess.run`
- Shell mode enabled (`shell=True`)
- Merges custom env vars with `os.environ`
- Handles exceptions gracefully (returns dict, doesn't crash)

---

### DockerEnvironment (Sandboxed)

From `docker.py:45-138`:

```
┌─────────────────────────────────────────────────────────────────┐
│  __init__()                                                     │
├─────────────────────────────────────────────────────────────────┤
│  _start_container():                                            │
│    docker run -d --name minisweagent-abc123 \                   │
│               -w /  \                                           │
│               --rm \                                            │
│               <image> \                                         │
│               sleep 2h        ◄── Keeps container alive         │
│                                                                 │
│    Stores container_id for later exec calls                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  execute(action)                                                │
├─────────────────────────────────────────────────────────────────┤
│  docker exec -w /workdir \                                      │
│              -e VAR1=val1 \      ◄── Forwarded env vars         │
│              <container_id> \                                   │
│              bash -lc "cat file.py"  ◄── Login shell            │
│                                                                 │
│  Same return format as LocalEnvironment                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  cleanup() / __del__()                                          │
├─────────────────────────────────────────────────────────────────┤
│  docker stop <container_id>  (or rm -f if timeout)              │
│  Runs in background to not block                                │
└─────────────────────────────────────────────────────────────────┘
```

**Key differences from Local:**
- Starts a long-running container on init
- Commands run via `docker exec` instead of direct subprocess
- Automatic cleanup when environment is garbage collected
- Configurable interpreter (`bash -lc` by default)

**DockerEnvironmentConfig options** (from `docker.py:15-42`):

```python
class DockerEnvironmentConfig(BaseModel):
    image: str                          # Required: Docker image to use
    cwd: str = "/"                      # Working directory in container
    env: dict[str, str] = {}            # Environment variables to set
    forward_env: list[str] = []         # Host env vars to forward
    timeout: int = 30                   # Command execution timeout (seconds)
    executable: str = "docker"          # Container runtime (docker/podman)
    run_args: list[str] = ["--rm"]      # Extra args for `docker run`
    container_timeout: str = "2h"       # How long container stays alive
    pull_timeout: int = 120             # Timeout for pulling images
    interpreter: list[str] = ["bash", "-lc"]  # Shell to run commands
```

**Example YAML config:**

```yaml
environment:
  type: minisweagent.environments.docker.DockerEnvironment
  image: python:3.11-slim
  cwd: /workspace
  timeout: 60
  env:
    PYTHONDONTWRITEBYTECODE: "1"
  forward_env:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
  run_args:
    - "--rm"
    - "--memory=4g"
    - "--cpus=2"
```

---

### Task Completion Detection

Both environments share `_check_finished()` (`local.py:55-66`):

```python
def _check_finished(self, output: dict):
    lines = output.get("output", "").lstrip().splitlines(keepends=True)
    if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
        submission = "".join(lines[1:])
        raise Submitted(...)
```

**The magic phrase:** When a command outputs `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as its first line, the environment raises `Submitted` — which the agent catches to exit the loop.

```
┌──────────────────────────────────────────────────────────┐
│  LLM decides task is done, runs:                         │
│                                                          │
│  echo "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"            │
│  echo "Fixed the bug by changing == to ==="              │
│                                                          │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  env.execute() sees magic string                         │
│                    │                                     │
│                    ▼                                     │
│  raises Submitted(exit_status="Submitted",               │
│                   submission="Fixed the bug...")         │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Agent catches InterruptAgentFlow                        │
│  Adds exit message → role="exit"                         │
│  Loop breaks, returns result                             │
└──────────────────────────────────────────────────────────┘
```

---

### Exception Hierarchy

From `exceptions.py`:

```
InterruptAgentFlow          ◄── Base class, caught in run()
    │
    ├── Submitted           ◄── Task completed successfully
    ├── LimitsExceeded      ◄── Cost/step limit hit
    ├── UserInterruption    ◄── Human stopped it
    ├── FormatError         ◄── LLM output parsing failed
    └── TimeoutError        ◄── Command timed out
```

All of these are **graceful exits** — they add messages to history and can trigger the exit condition.

---

### Template Variables from Environment

Both environments provide context via `get_template_vars()`:

| Variable | Source | Example |
|----------|--------|---------|
| `cwd` | Config | `/home/user/project` |
| `timeout` | Config | `30` |
| `system` | `platform.uname()` | `Darwin` |
| `node` | `platform.uname()` | `MacBook-Pro.local` |
| `PATH` | `os.environ` | `/usr/bin:/bin:...` |

These are available in Jinja2 templates (system prompt, etc.).

---

### Environment Comparison

| Aspect | LocalEnvironment | DockerEnvironment |
|--------|------------------|-------------------|
| Execution | Direct `subprocess.run` | `docker exec` into container |
| Isolation | None (host machine) | Full container sandbox |
| Startup | Instant | Container creation time |
| Cleanup | None needed | Stops/removes container |
| Use case | Development/trusted | SWE-bench/untrusted code |

The protocol design means you can swap `LocalEnvironment` for `DockerEnvironment` with zero code changes — just different config.

---

## Real Use Case: SWE-bench with Docker (Annotated End-to-End)

This is the primary use case for mini-swe-agent: solving real GitHub issues from the SWE-bench benchmark inside Docker containers.

### The Command

```bash
mini-swebench -i django__django-11099 -m anthropic/claude-sonnet-4-5-20250929
```

### Step 1: CLI Parses Arguments

File: `swebench_single.py:42-53`

```
$ mini-swebench -i django__django-11099 -m anthropic/claude-sonnet-4-5-20250929

Parses args:
  subset = "lite"
  instance_spec = "django__django-11099"
  model_name = "anthropic/claude-sonnet-4-5-20250929"
```

### Step 2: Load SWE-bench Instance from Hugging Face

File: `swebench_single.py:56-64`

```python
instances = load_dataset("princeton-nlp/SWE-Bench_Lite", split="dev")
instance = instances["django__django-11099"]
```

The instance is a dict containing:

```python
{
    "instance_id": "django__django-11099",
    "problem_statement": "UsernameValidator allows trailing newline...",
    "repo": "django/django",
    "base_commit": "abc123...",    # Exact commit to check out
    "patch": "diff --git...",       # Gold solution (hidden from agent)
    "test_patch": "diff --git...",  # Tests that validate the fix
    "FAIL_TO_PASS": "[\"test_...\"]",  # Tests that must go from fail → pass
    "PASS_TO_PASS": "[\"test_...\"]",  # Tests that must stay passing
    ...
}
```

### Step 3: Build Configuration

File: `swebench_single.py:66-77`

Loads and merges `swebench.yaml` + CLI overrides:

```yaml
agent:
  system_template: "You are a helpful assistant..."
  instance_template: "<pr_description>{{task}}</pr_description>..."
  step_limit: 250       # Max 250 LLM calls
  cost_limit: 3.0       # Max $3 per instance

environment:
  environment_class: docker   # ◄── Key: use Docker
  cwd: "/testbed"             # ◄── Working dir inside container
  timeout: 60

model:
  model_name: "anthropic/claude-sonnet-4-5-20250929"
  model_kwargs:
    temperature: 0.0
```

### Step 4: Create Docker Environment

File: `swebench.py:92-106` → `docker.py:46-99`

```
get_sb_environment(config, instance):

  4a. Compute Docker image name from instance_id:
      "django__django-11099"
       → "swebench/sweb.eval.x86_64.django_1776_django-11099:latest"
      (SWE-bench provides pre-built images with repo + all dependencies)

  4b. DockerEnvironment.__init__() calls _start_container():

      $ docker run -d \
          --name minisweagent-a1b2c3d4 \
          -w /testbed \
          --rm \
          swebench/sweb.eval.x86_64.django_1776_django-11099:latest \
          sleep 2h

      Returns: container_id = "abc123def456..."

  Container now running with:
    - Django repo checked out at the exact base_commit
    - All dependencies pre-installed
    - Working directory: /testbed
```

### Step 5: Create Model + Agent

File: `swebench_single.py:79-84`

```python
model = LitellmModel(model_name="anthropic/claude-sonnet-4-5-20250929")

agent = InteractiveAgent(
    model,
    env,              # ◄── The Docker environment
    system_template="You are a helpful assistant...",
    instance_template="<pr_description>{{task}}...",
    step_limit=250,
    cost_limit=3.0
)
```

### Step 6: Start Agent Run

File: `default.py:77-97`

```python
agent.run(instance["problem_statement"])
```

Renders templates and initializes messages:

```python
messages = [
    {"role": "system",  "content": "You are a helpful assistant..."},
    {"role": "user",    "content": "<pr_description>UsernameValidator allows..."}
]
```

Enters the main loop: `while True: self.step()`

### Step 7: Agent Loop (Multiple Iterations)

Each iteration: `query()` → LLM API call → `execute_actions()` → docker exec

```
Iteration 1: LLM explores the codebase
  LLM:  "I'll start by exploring the structure."
  cmd:  find /testbed -name '*.py' | head -20
  exec: docker exec -w /testbed abc123 bash -lc "find /testbed -name '*.py' | head -20"
  out:  /testbed/django/contrib/auth/validators.py ...

Iteration 2: LLM reads the relevant file
  LLM:  "Let me look at the validator file."
  cmd:  cat /testbed/django/contrib/auth/validators.py
  exec: docker exec -w /testbed abc123 bash -lc "cat ..."
  out:  class UsernameValidator: regex = r'^[\w.@+-]+$' ...

Iteration 3: LLM creates reproduction script
  cmd:  cat <<'EOF' > /testbed/reproduce.py ...

Iteration 4: LLM runs reproduction
  cmd:  cd /testbed && python reproduce.py

Iteration 5: LLM edits the source
  cmd:  sed -i "s/regex = r'^[\\w.@+-]+\$'/regex = r'^[\\w.@+-]+\\Z'/" validators.py

Iteration 6: LLM verifies fix
  cmd:  cd /testbed && python reproduce.py

... (all commands run via docker exec into the SAME container)
```

### Step 8: Task Completion

File: `docker.py:140-151`

LLM creates a patch and submits:

```
LLM runs: git diff -- django/contrib/auth/validators.py > patch.txt
LLM runs: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

Docker exec runs the command, output is:

```
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT           ◄── Magic string
diff --git a/django/contrib/auth/validators.py b/...
--- a/django/contrib/auth/validators.py
+++ b/django/contrib/auth/validators.py
@@ -10,7 +10,7 @@
-    regex = r'^[\w.@+-]+$'
+    regex = r'^[\w.@+-]+\Z'
```

`_check_finished()` sees the magic string → raises `Submitted(patch)`
→ Agent catches it → adds exit message → loop breaks

### Step 9: Agent Returns Result

```python
return {"exit_status": "Submitted", "submission": "diff --git a/django/..."}
```

### Step 10: Cleanup

```
- Trajectory saved to JSON file (every command + output recorded)
- env object goes out of scope → __del__() → docker stop abc123
- Container removed (--rm flag)
- Patch written to preds.json for SWE-bench evaluation
```

### Summary Timeline

```
User runs command
       │
       ▼
Load SWE-bench instance (problem + metadata)
       │
       ▼
Build config (swebench.yaml + CLI args)
       │
       ▼
┌──────────────────────────────────────────┐
│  docker run ... sleep 2h                 │  ◄── Container starts
│  (pre-built image with repo + deps)      │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  AGENT LOOP                              │
│  ┌────────────────────────────────────┐  │
│  │ LLM: "I'll explore the codebase"   │  │
│  │ docker exec: find /testbed ...     │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │ LLM: "Let me read the file"        │  │
│  │ docker exec: cat validators.py     │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │ LLM: "I'll fix the regex"          │  │
│  │ docker exec: sed -i 's/...'        │  │
│  └────────────────────────────────────┘  │
│  ┌────────────────────────────────────┐  │
│  │ LLM: "Submitting the fix"          │  │
│  │ docker exec: echo COMPLETE... &&   │  │
│  │              cat patch.txt         │  │
│  └────────────────────────────────────┘  │
│                    │                     │
│        _check_finished() sees magic      │
│        raises Submitted(patch)           │
└──────────────────────────────────────────┘
       │
       ▼
Agent returns {exit_status, submission}
       │
       ▼
┌──────────────────────────────────────────┐
│  docker stop ...                         │  ◄── Container dies
└──────────────────────────────────────────┘
       │
       ▼
Patch saved to preds.json for evaluation
```

---

## What is SWE-bench?

### Overview

**SWE-bench** (Software Engineering Bench) is a benchmark for evaluating AI systems on their ability to resolve **real-world GitHub issues**. Published at ICLR 2024 by the Princeton & Stanford team, it is the standard benchmark for measuring AI coding agent performance.

**Core idea:** Given a codebase and a natural-language issue description, an AI system must produce a **patch** (code diff) that resolves the described problem. The patch is then verified by running the repository's **actual test suite**.

### The 12 Repositories

| Repository | Domain |
|-----------|--------|
| `django/django` | Web framework |
| `sympy/sympy` | Symbolic mathematics |
| `scikit-learn/scikit-learn` | Machine learning |
| `matplotlib/matplotlib` | Data visualization |
| `astropy/astropy` | Astronomy |
| `psf/requests` | HTTP library |
| `pallets/flask` | Web microframework |
| `pytest-dev/pytest` | Testing framework |
| `pydata/xarray` | Labeled arrays |
| `pylint-dev/pylint` | Code linting |
| `sphinx-doc/sphinx` | Documentation |
| `mwaskom/seaborn` | Statistical visualization |

All are popular, well-maintained Python projects with good test coverage.

### What a SWE-bench Instance Contains

Each instance is a JSON object representing a single resolved GitHub issue + PR:

| Field | Description |
|-------|-------------|
| `instance_id` | e.g., `django__django-11099` |
| `problem_statement` | The full issue description text |
| `repo` | e.g., `django/django` |
| `base_commit` | Exact commit hash before the fix |
| `patch` | The gold-standard solution diff (hidden from agent) |
| `test_patch` | Test changes from the PR |
| `FAIL_TO_PASS` | Tests that must go from failing → passing |
| `PASS_TO_PASS` | Tests that must continue passing (regression) |

### How Evaluation Works

```
┌─────────────────────────────────────────────────────────────────┐
│  For each instance:                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Docker container with repo at base_commit                   │
│     (before the fix was applied)                                │
│                                                                 │
│  2. Apply test_patch (adds/modifies test files)                 │
│                                                                 │
│  3. Apply model's generated patch                               │
│                                                                 │
│  4. Run test suite:                                             │
│     - FAIL_TO_PASS tests: must now PASS  ◄── Proves fix works  │
│     - PASS_TO_PASS tests: must still PASS ◄── No regressions   │
│                                                                 │
│  5. Score:                                                      │
│     ✅ Resolved = ALL FAIL_TO_PASS pass AND ALL PASS_TO_PASS    │
│     ❌ Not resolved = any test fails                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Primary metric:

  Resolve Rate = (resolved instances / total instances) × 100%

  e.g., "74% on SWE-bench Verified" means 370 of 500 instances fixed
```

### The Three Subsets

| | Full | Lite | Verified |
|---|------|------|----------|
| **Instances** | 2,294 | 300 | 500 |
| **Curation** | Automated | Subsampled | Human-validated |
| **Difficulty labels** | No | No | Yes (easy/medium/hard) |
| **Purpose** | Complete coverage | Cost-efficient eval | Most reliable signal |
| **Status** | Original | Superseded | **Recommended** |

- **Full**: The original 2,294 instances. Some are infeasible or have flawed tests.
- **Lite**: 300 instances subsampled for cheaper evaluation. Largely superseded.
- **Verified**: 500 instances reviewed by 93 professional developers. Removes problematic samples. The current standard for leaderboards.

### Pre-built Docker Images

SWE-bench provides a three-layer Docker image architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: INSTANCE IMAGE  (one per task, ~2,294 total)         │
│  - Repo checked out at exact base_commit                        │
│  - test_patch applied                                           │
│  - Ready to run                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: ENVIRONMENT IMAGE  (~60 total)                       │
│  - Correct Python version                                       │
│  - All pip/conda dependencies installed                         │
│  - Repository clone                                             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: BASE IMAGE  (1 total)                                │
│  - Ubuntu 22.04                                                 │
│  - System packages + Miniconda                                  │
└─────────────────────────────────────────────────────────────────┘
```

Image naming in mini-swe-agent (`swebench.py:81-89`):

```
instance_id:  "django__django-11099"
                     ↓
docker image: "swebench/sweb.eval.x86_64.django_1776_django-11099:latest"
              (double underscores replaced with _1776_ for Docker compatibility)
```

### How mini-swe-agent Connects to SWE-bench

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   SWE-bench      │     │  mini-swe-agent  │     │   SWE-bench      │
│   Dataset        │────▶│  Agent           │────▶│   Evaluation     │
│                  │     │                  │     │                  │
│  problem_statement│     │  Reads problem   │     │  Applies patch   │
│  base_commit     │     │  Runs in Docker  │     │  Runs tests      │
│  Docker image    │     │  Produces patch  │     │  Scores result   │
└──────────────────┘     └──────────────────┘     └──────────────────┘
      INPUT                   AGENT                    SCORING
```

**mini-swe-agent's role**: Take the problem statement, work inside the Docker container, and produce a `git diff` patch. SWE-bench's evaluation harness then scores the patch separately.
