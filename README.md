<p align="center">
  <br />
  <strong><code>✦ aura</code></strong>
  <br />
  <em>Stop re-explaining yourself to every AI tool.</em>
  <br /><br />
  <a href="https://pypi.org/project/aura-ctx/"><img alt="PyPI" src="https://img.shields.io/pypi/v/aura-ctx?color=blue&label=PyPI" /></a>
  <a href="https://pepy.tech/projects/aura-ctx"><img alt="Downloads" src="https://static.pepy.tech/personalized-badge/aura-ctx?period=total&units=international_system&left_color=black&right_color=green&left_text=downloads" /></a>
  <a href="https://github.com/WozGeek/aura-ctx/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green" /></a>
  <a href="https://pypi.org/project/aura-ctx/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/aura-ctx" /></a>
  <a href="https://github.com/WozGeek/aura-ctx/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/WozGeek/aura-ctx?style=flat" /></a>
</p>

<p align="center">
  <a href="https://wozgeek.github.io/aura-ctx">Website</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#supported-tools">Supported Tools</a> •
  <a href="#commands">Commands</a> •
  <a href="#security">Security</a>
</p>

---

Define who you are — your stack, your style, your rules — **once**, in plain YAML files you own. aura serves that identity to Claude, ChatGPT, Cursor, and Gemini through the [Model Context Protocol](https://modelcontextprotocol.io). 100% local. No cloud. No lock-in.

<!-- TODO: Add terminal recording GIF here
     Record with: asciinema rec demo.cast && agg demo.cast demo.gif
     Show: pip install aura-ctx && aura quickstart (30 second flow) -->

### Highlights

- **30-second setup** — `pip install aura-ctx && aura quickstart` scans your machine, asks 5 questions, starts serving
- **14 templates** — `aura create -t frontend`, `data-scientist`, `founder`, `student`, and 10 more
- **Cross-tool** — Claude Desktop, ChatGPT Desktop, Cursor, Gemini CLI, any MCP client
- **Smart token delivery** — 3 levels (~50, ~500, ~1000+ tokens) so AI tools only load what they need
- **Secret scanning** — auto-detects leaked API keys before they reach an LLM, redacts on serve
- **File watcher** — `aura serve --watch` hot-reloads when you edit a YAML pack
- **Human-readable** — YAML files in `~/.aura/packs/`, git-friendly, fully editable
- **No cloud, no telemetry, no tracking** — everything stays on your machine

## Why aura

Every AI tool starts from scratch. Claude doesn't know what ChatGPT learned. Cursor doesn't know your writing style. Gemini has no idea what framework you prefer.

The industry is building solutions for this, but at the wrong layer:

| Layer | What it solves | Examples |
|-------|---------------|----------|
| **Memory** | What happened in past conversations | Mem0, Zep, DeltaMemory |
| **Context engineering** | What the AI should know right now | LACP, Claudesidian, OpenClaw |
| **Identity** | Who you are, across everything | **aura** |

Memory is session history. Context is prompt engineering. **Identity is who you are** — your stack, your style, your rules, your role — structured, portable, and owned by you.

aura is the identity layer.

## Who this is for

**You use multiple AI tools daily** — Claude for thinking, Cursor for coding, ChatGPT for drafting, Gemini for research. You're tired of re-explaining your stack and style to each one.

**You're a developer who values control** — you want your context in plain text files you can read, edit, and version-control. Not locked inside a platform.

**You're building with AI, not just using it** — you care about token efficiency, MCP, and how your tools talk to each other.

If you've ever pasted your coding style into a system prompt and wished it could just follow you everywhere — that's what aura does.

## Quick Start

```bash
pip install aura-ctx
aura quickstart
```

Here's what happens:

```
✦ aura quickstart

Step 1/5 — Scanning your machine...
  ✦ Detected 12 facts about your dev environment

Step 2/5 — Quick questions about you...
  What's your role? → Full-stack dev at Acme Corp
  How do you want AI to talk to you? → 1 (Direct, no fluff)
  What are you working on? → shipping v2 of our dashboard
  Any rules or pet peeves? → No corporate jargon, always use TypeScript
  What human languages? → English and French
  ✦ Created writer (2 facts, 3 rules)
  ✦ Created work (2 facts, 0 rules)

Step 3/5 — Configuring AI tools...
  ✦ Claude Desktop configured
  ✦ Cursor configured

Step 4/5 — Security audit...
  ✦ All clean — no secrets detected

Step 5/5 — Starting MCP server...
  ✦ http://localhost:3847/mcp
  Restart your AI tools — they know you now.
```

30 seconds. No Docker. No database. No cloud account.

## See it work

After running `aura quickstart`, open Claude Desktop:

```
You:    What do you know about me?

Claude: I don't have any information about you!
        Memory is turned off for your account.
```

Now restart Claude (so it connects to aura's MCP server):

```
You:    What do you know about me?

Claude: Here's what I know from your aura context:

        Role: CS student
        Editor: Cursor
        Stack: TypeScript, Python, React, FastAPI, Tailwind
        Projects: aura, kipedia, hotepia
        Style: Technical, precise, no hand-holding
        Rules: Always use TypeScript strict mode, no 'any'
```

Same question. Completely different answer. That's aura.

## How It Works

```
  You
   │
   ├── aura scan          Detects languages, frameworks, tools, projects
   ├── aura onboard       5 questions → writing style, role, rules
   ├── aura import        Pulls context from ChatGPT & Claude exports
   │
   ▼
  Context Packs (YAML)    ~/.aura/packs/developer.yaml
   │                      ~/.aura/packs/writer.yaml
   │                      ~/.aura/packs/work.yaml
   │
   ▼
  MCP Server              localhost:3847
   │
   ├──▶ Claude Desktop    (auto-configured)
   ├──▶ ChatGPT Desktop   (SSE)
   ├──▶ Cursor IDE        (auto-configured)
   └──▶ Gemini CLI        (auto-configured)
```

> **What's MCP?** The [Model Context Protocol](https://modelcontextprotocol.io) is an open standard that lets AI tools connect to local data sources. aura uses it so Claude, Cursor, and others can read your context without any custom integration.

### Context Packs

Your identity lives in scoped YAML files. Each pack covers a domain — development, writing, work, or anything custom:

```yaml
# ~/.aura/packs/developer.yaml
name: developer
scope: development
facts:
  - key: languages.primary
    value: [TypeScript, Python]
    type: skill
    confidence: high
  - key: editor
    value: Cursor
    type: preference
  - key: frameworks
    value: [Next.js, FastAPI, Tailwind, Supabase]
    type: skill
  - key: style.code
    value: "Explicit types, functional patterns, minimal comments"
    type: style
rules:
  - instruction: Always use TypeScript strict mode — no 'any'
    priority: 9
  - instruction: Dark theme by default, CSS variables for all colors
    priority: 8
  - instruction: Error handling with specific types, not generic catches
    priority: 7
```

You own these files. Human-readable. Git-friendly. They never leave your machine unless you choose otherwise.

### Three-Level Token Delivery

AI tools have limited context windows. aura serves your identity at the right depth:

| Level | MCP Tool | Tokens | When |
|-------|----------|--------|------|
| 1 | `get_identity_card` | ~50–100 | Auto-called at conversation start |
| 2 | `get_user_profile` | ~200–500 | When the AI needs more detail |
| 3 | `get_all_context` | ~1000+ | Only when explicitly asked |

The server instructs AI clients to start with the identity card and drill down only when needed. Most conversations never need the full dump.

## Supported Tools

| Tool | Setup | Transport |
|------|-------|-----------|
| **Claude Desktop** | `aura setup` — auto | Streamable HTTP |
| **Cursor IDE** | `aura setup` — auto | Streamable HTTP |
| **Gemini CLI** | `aura setup` — auto | SSE |
| **ChatGPT Desktop** | Developer Mode → add SSE URL | SSE |
| **Any MCP client** | Point to `localhost:3847` | HTTP or SSE |

```bash
aura setup   # writes config for all detected tools
aura serve   # starts MCP server on localhost:3847
```

<details>
<summary><strong>Claude Desktop</strong></summary>

Auto-configured by `aura setup`. Manual config:

```json
{
  "mcpServers": {
    "aura": { "url": "http://localhost:3847/mcp" }
  }
}
```
</details>

<details>
<summary><strong>Cursor IDE</strong></summary>

Auto-configured by `aura setup`. Manual config:

```json
{
  "mcpServers": {
    "aura": { "url": "http://localhost:3847/mcp" }
  }
}
```
</details>

<details>
<summary><strong>ChatGPT Desktop</strong></summary>

Settings → Connectors → Advanced → Developer Mode:

```
SSE URL: http://localhost:3847/sse
```
</details>

<details>
<summary><strong>Gemini CLI</strong></summary>

Auto-configured by `aura setup`. Manual config:

```json
{
  "mcpServers": {
    "aura": { "uri": "http://localhost:3847/sse" }
  }
}
```
</details>

## Commands

### Getting started

| Command | What it does |
|---------|-------------|
| `aura quickstart` | Full setup: scan → onboard → setup → audit → serve |
| `aura scan` | Auto-detect your stack from tools, repos, and config files |
| `aura onboard` | 5 questions to generate your context packs |
| `aura setup` | Auto-configure Claude Desktop, Cursor, Gemini |
| `aura serve` | Start the MCP server |
| `aura serve --watch` | Start with hot-reload on YAML changes |

### Managing packs

| Command | What it does |
|---------|-------------|
| `aura list` | List all context packs |
| `aura show <pack>` | Display a pack's contents |
| `aura add <pack> <key> <value>` | Add a fact without editing YAML |
| `aura edit <pack>` | Open a pack in `$EDITOR` |
| `aura create <n>` | Create a new empty pack |
| `aura create <n> -t <template>` | Create from a built-in template |
| `aura templates` | List all 14 available templates |
| `aura delete <pack>` | Delete a pack |
| `aura diff <a> <b>` | Compare two packs |

### Templates

14 built-in templates to get started fast. Each includes facts and AI interaction rules tailored to the profile.

**Stack-specific:** `frontend`, `backend`, `data-scientist`, `mobile`, `devops`, `ai-builder`

**Role-specific:** `founder`, `student`, `marketer`, `designer`

**General-purpose:** `developer`, `writer`, `researcher`, `work`

```bash
aura templates                         # list all available templates
aura create mydev -t frontend          # create a frontend dev pack
aura create research -t data-scientist # create a data science pack
aura create study -t student           # create a student pack
```

Every template is a starting point. Edit the generated YAML to match your actual stack and preferences.

### Health & maintenance

| Command | What it does |
|---------|-------------|
| `aura doctor` | Check pack health — bloat, stale facts, duplicates, secrets |
| `aura audit` | Scan packs for leaked API keys, tokens, credentials |
| `aura audit --fix` | Auto-redact critical secrets |
| `aura consolidate` | Merge duplicate facts, find contradictions across packs |
| `aura decay` | Remove expired facts based on type-aware TTL |

### Import & export

| Command | What it does |
|---------|-------------|
| `aura import -s chatgpt <file>` | Import from a ChatGPT data export |
| `aura import -s claude <file>` | Import from a Claude data export |
| `aura extract <file>` | Extract facts from conversations using a local LLM |
| `aura export <pack> -f system-prompt` | Universal LLM system prompt |
| `aura export <pack> -f cursorrules` | `.cursorrules` file |
| `aura export <pack> -f chatgpt` | ChatGPT custom instructions |
| `aura export <pack> -f claude` | Claude memory statements |

## Security

aura is local-first. Your context never leaves your machine.

```bash
aura serve                              # localhost only, open
aura serve --token my-secret            # require Bearer token
aura serve --packs developer,writer     # expose only specific packs
aura serve --read-only                  # block all writes via MCP
aura serve --watch                      # auto-reload on pack changes
```

**Secret detection** — `aura audit` scans every fact and rule for leaked credentials before they reach an LLM. Catches 30+ patterns: AWS keys, GitHub tokens, OpenAI/Anthropic API keys, Slack tokens, database URLs, private keys, Bearer tokens, and more. The MCP server scrubs critical secrets automatically at serve time — even if you forget to audit.

- Binds to `127.0.0.1` only — not reachable from the network
- Optional Bearer token auth (`--token` or `AURA_TOKEN` env var)
- Scoped serving — control which packs each tool sees
- Read-only mode — AI reads your context, never writes to it
- **No telemetry. No analytics. No cloud. No tracking.**

## Architecture

```
aura/
├── cli.py           # 22 commands (Typer + Rich)
├── schema.py        # ContextPack, Fact, Rule (Pydantic)
├── mcp_server.py    # FastAPI MCP server (HTTP + SSE)
├── scanner.py       # Machine scanner with incremental hashing
├── onboard.py       # Interactive onboarding
├── pack.py          # Pack CRUD + templates
├── audit.py         # Secret detection engine (30+ patterns)
├── scan_cache.py    # SHA-256 content hashing for fast re-scans
├── watcher.py       # File watcher for hot-reload
├── doctor.py        # Pack health checker
├── consolidate.py   # Dedup + contradiction detection
├── extractor.py     # LLM-based extraction (Ollama / OpenAI)
├── diff.py          # Pack comparison
├── setup.py         # Auto-config for Claude, Cursor, Gemini
├── exporters/       # system-prompt, cursorrules, chatgpt, claude
└── importers/       # ChatGPT + Claude data importers
```

7,800+ lines of Python · 151 tests · 22 commands · 14 templates · MIT license

## Roadmap

### Shipped

- [x] Machine scanner — languages, frameworks, tools, projects, git identity
- [x] Context packs with typed facts, confidence levels, sources
- [x] MCP server — resources, tools, prompt templates
- [x] Auto-config for Claude Desktop, Cursor, Gemini CLI
- [x] ChatGPT Desktop support via SSE
- [x] Token auth, scoped serving, read-only mode
- [x] Import from ChatGPT + Claude data exports
- [x] LLM-based extraction (Ollama, OpenAI)
- [x] Pack health checker + consolidation engine
- [x] Memory decay with type-aware TTL
- [x] Secret detection and auto-redaction
- [x] Incremental scan with content hashing
- [x] File watcher (`aura serve --watch`)
- [x] Three-level token delivery
- [x] 14 built-in templates (frontend, backend, data-scientist, mobile, devops, founder, student, marketer, designer, ai-builder)

### Next

- [ ] TypeScript / npm package — `npx aura-ctx`
- [ ] JSON Schema spec for context packs
- [ ] Usage-based fact priority
- [ ] Per-agent permissions
- [ ] Share via GitHub Gist
- [ ] GraphRAG local knowledge graph
- [ ] Cloud sync (opt-in, encrypted)
- [ ] Team sharing

## Contributing

```bash
git clone https://github.com/WozGeek/aura-ctx.git
cd aura-ctx
pip install -e ".[dev]"
pytest
```

**Good first issues:**

- **New export format** — add Windsurf, Continue.dev, or AGENTS.md support ([guide](CONTRIBUTING.md#adding-an-exporter))
- **New importer** — Gemini history export parsing
- **Pack templates** — create domain-specific starter packs (frontend, data-scientist, devops, writer)
- **JSON Schema** — publish `context-pack.schema.json` to formalize the pack format
- **Translations** — translate this README to French, Spanish, Portuguese, or Chinese

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

## License

[MIT](LICENSE) — © Enoch Afanwoubo
