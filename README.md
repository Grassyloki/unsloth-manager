# Unsloth Service Manager

A single-host control plane for running multiple [Unsloth Studio](https://unsloth.ai)
inference servers at once: an interactive curses TUI plus a CLI for starting and
stopping servers on dedicated ports as an unprivileged user, applying Unsloth's
own settings to headless launches, and benchmarking throughput.

A launch is **(model, profile, port)**, and every profile is *Unsloth's own*
— read live out of `studio.db`, out of the Studio venv's shipped tables, or
transcribed from Unsloth's published model guides. This tool invents no
settings of its own; what it does is show you which source won, before the load
and after it — and, when you change a value at launch, save it back into the
Studio profile it came from.

Built for and tested on a **2x NVIDIA Tesla V100-SXM2 32 GB** box running
CachyOS (Arch), serving GGUF models out of a shared Hugging Face cache.

## Design in brief

- **Settings belong to Unsloth, not to this tool.** It keeps no model or
  sampling profiles of its own (`local.env` only describes the machine). You
  edit them on the Studio settings page and in the Preset dropdown — or on the
  manager's own review screen at launch, which saves an edit back into the
  same Studio preset or per-model override, through **Studio's own settings
  API**. The manager never opens `studio.db` for writing; see
  [Editing a profile at launch](#editing-a-profile-at-launch). The one thing it
  does own is a transcription of Unsloth's *published* sampling guides — see
  [Where settings come from](#where-settings-come-from) for why that is
  necessary.
- **Servers run as a dedicated non-root user.** The manager runs as root and
  drops to `unsloth` for every server it starts.
- **Auth is mandatory.** Every Unsloth server requires a bearer token on `/v1`.
  This manager deliberately does *not* create or store keys — see
  [API keys](#api-keys).

## Contents

**Sections**

- [Design in brief](#design-in-brief)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Command reference](#command-reference)
- [Where settings come from](#where-settings-come-from)
- [Editing a profile at launch](#editing-a-profile-at-launch)
- [Context, and what "max" means](#context-and-what-max-means)
- [Idle auto-unload, and what a reload restores](#idle-auto-unload-and-what-a-reload-restores)
- [Keeping a server warm](#keeping-a-server-warm)
- [Reading settings back after a load](#reading-settings-back-after-a-load)
- [Tokens churned](#tokens-churned)
- [Instance groups](#instance-groups)
- [Ports and instance limit](#ports-and-instance-limit)
- [API keys](#api-keys)
- [Benchmark](#benchmark)
- [How servers are launched](#how-servers-are-launched)
- [Gotchas worth knowing](#gotchas-worth-knowing)
- [Connecting clients](#connecting-clients)
- [Notes for this box](#notes-for-this-box)
- [Tests](#tests)
- [History](#history)
- [License](#license)

**Files**

| File | Purpose |
|------|---------|
| `unsloth_manager.py` | Main entry point: TUI + CLI to start, stop, inspect, test and benchmark servers. |
| `unsloth_profiles.py` | Every source of "how should this model run": published guides, Unsloth's shipped defaults, Studio presets, per-model overrides — plus the payload shapes for saving an edit back to the last two. |
| `model_lib.py` | HF-cache scanning, GGUF variant detection, VRAM fit estimation, native context length. |
| `tui_lib.py` | Shared curses widgets (selector, prompts, coloured bars, screen plumbing). |
| `api_tester.sh` | Interactive whiptail client for poking any OpenAI-compatible endpoint. |
| `local.env.example` | Template for `local.env`, the git-ignored per-machine settings file. |
| `tests/test_logic.py` | Unit tests for the logic that guards user data. See [Tests](#tests). |

The manager runs on the system Python (3.10+) and imports nothing outside the
standard library. `unsloth_profiles.py` will use PyYAML if it happens to be
installed, only to read a model-specific YAML from Unsloth's own config tree;
without it that one lookup is skipped and the family table (plain JSON) is
used, which is what every model on this box resolves through anyway.

## Requirements

- Linux with NVIDIA drivers and `nvidia-smi` on `PATH`.
- The `unsloth` CLI installed system-wide (`/usr/bin/unsloth`).
- A service account (default `unsloth`) that owns `STUDIO_HOME` and the HF cache.
- Root, to drop privileges to that account.
- `whiptail`, `jq`, `curl`, `bc` for `api_tester.sh`.

## Setup

Every setting is an environment variable, and every one has a default chosen to
work on a fresh machine — so a clone runs without configuration.

For the handful that are per-machine, copy `local.env.example` to **`local.env`**
beside the script and edit it. It is read at startup and is git-ignored, so
nothing about your box ends up in the repository:

```bash
cp local.env.example local.env
$EDITOR local.env
```

A real environment variable always beats the file, so a systemd unit or a
one-off `UNSLOTH_MGR_LOG_KEEP=10 python unsloth_manager.py ...` still wins.
Only `UNSLOTH_MGR_*` names are read from it — a config file has no business
setting `PATH`, which is also why the Hugging Face token has its own
`UNSLOTH_MGR_HF_TOKEN` name. Everything after `=` is the value, so keep
comments on their own lines. Point `UNSLOTH_MGR_ENV_FILE` elsewhere to use a
different file. `api_tester.sh` reads the same file. Once it holds a key or
token, `chmod 600` it.

The two most likely to need setting are `UNSLOTH_MGR_PUBLIC_HOST` (if clients
reach the box by a name other than its hostname) and `UNSLOTH_MGR_HF_HOME` (if
the weights are not in the service account's `~/.cache/huggingface`).

| Variable | Default | Meaning |
|----------|---------|---------|
| `UNSLOTH_MGR_USER` | `unsloth` | Account the servers run as. |
| `UNSLOTH_MGR_STUDIO_HOME` | `~unsloth/.unsloth/studio` | Unsloth data root. Leave unset unless it is genuinely elsewhere — see the symlink item in [Gotchas](#gotchas-worth-knowing). |
| `UNSLOTH_MGR_STUDIO_DB` | `<STUDIO_HOME>/studio.db` | Where presets and per-model overrides are read from. |
| `UNSLOTH_MGR_STUDIO_ASSETS` | globbed under `<STUDIO_HOME>` | Unsloth's `assets/configs` (its shipped sampling tables). |
| `UNSLOTH_MGR_HF_HOME` | `~<user>/.cache/huggingface` | Model cache; passed to the child as `HF_HOME`. |
| `UNSLOTH_MGR_STATE_DIR` | `/root/.unsloth-pids` | State file directory: `state.json`, `groups.json`, `tokens.json`. |
| `UNSLOTH_MGR_LOG_DIR` | `/root/.unsloth-logs` | Per-model server logs + benchmark JSON. |
| `UNSLOTH_MGR_BASE_PORT` | `10001` | First port in the API pool. |
| `UNSLOTH_MGR_MAX_INSTANCES` | `8` | Pool size, and therefore the concurrent-server cap. |
| `UNSLOTH_MGR_GROUP_SLOTS` | `10` | Instance-group slots. |
| `UNSLOTH_MGR_BIND_HOST` | `0.0.0.0` | Server bind address. |
| `UNSLOTH_MGR_PUBLIC_HOST` | this machine's hostname | Hostname printed in URLs. Only affects what is printed. |
| `UNSLOTH_MGR_PUBLIC_SCHEME` | `http` | Scheme for those URLs. |
| `UNSLOTH_MGR_API_KEY` | unset | Key for `test` / `benchmark` / readouts, and the `auth` that `api_tester.sh` seeds its endpoints with on its first run. Default: read from the server's own log. |
| `UNSLOTH_MGR_HF_TOKEN` | unset | Hugging Face token for gated repos, passed to the servers as `HF_TOKEN`. A real `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` in the environment wins. |
| `UNSLOTH_MGR_BIN` | `unsloth` | Unsloth CLI to invoke. A bare name is looked up on the servers' fixed `PATH` (`/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`), not yours — use a full path for anything installed elsewhere. `doctor` checks that same `PATH`. |
| `UNSLOTH_MGR_PER_GPU_VRAM_GB` | auto-detected | Per-GPU VRAM used for quant sizing. |
| `UNSLOTH_MGR_STOP_TIMEOUT` | `30` | Seconds to wait for SIGTERM before SIGKILL. |
| `UNSLOTH_MGR_LOAD_TIMEOUT` | `900` | Default readiness wait for `start`. |
| `UNSLOTH_MGR_LOG_KEEP` | `3` | Rotated log generations to keep. |
| `UNSLOTH_MGR_ENV_FILE` | `local.env` beside the script | Where the per-machine settings above are read from. |

Then check all of it, including that presets, per-model overrides and
Unsloth's shipped default tables are all readable:

```bash
python unsloth_manager.py doctor
```

## Usage

Run with no arguments for the interactive TUI:

```bash
python unsloth_manager.py
```

The home screen lists each running server with what it is actually running —
the knobs that decide throughput and output quality. Anything left at Unsloth's
own default is dim; anything this manager changed is bright, so the line
answers "what am I not running stock?" at a glance:

```
Running: 2 model(s), 2 ready   all-time ↑18.9M ↓4.1M     ⏱ 21:40:03 · refresh 5s
  ● Qwen3.6-35B-A3B-MTP-GGUF UD-Q4_K_XL :10001 rdy g0   0/2sl            ↑820K ↓191K    load 9.3s
      256k  kv q4_0  mtp  tools ON  vis_off  | T1.0 P0.95 K20 M0.0 pres0.0 rep1.0
  ● Qwen3.8-27B-GGUF         UD-Q4_K_XL :10002 rdy g1   1/3sl  16 tok/s  ↑614K ↓121K    load 13s
      256k  kv q4_0  mtp  tools off  vis_off  warm | T1.0 P0.95 K20 M0.0 pres0.0 rep1.0
  Tokens  24h ↑412K ↓96.2K   7d ↑3.1M ↓702K   30d ↑11.4M ↓2.6M   365d ↑18.9M ↓4.1M
```

The first line is what a server *is* — identity, placement, and the facts that
change as it runs. `1/3sl` is **sessions: one of three decode slots busy right
now**, `8 tok/s` its recent decode rate, `↑614K ↓121K` the input and output
tokens it has processed since it started, and `load 9.3s` how long its most
recent model load took. Everything on the second line is configuration, which
does not move. `status` spells them out (`sessions 1/3  8.0 tok/s  last load
9.3s (idle reload)`), and the `settings` readout carries sessions and rate.

The figure in the header is the other half of that pair: every model's tokens,
all-time, for the box rather than for one server. The `Tokens` line under the
servers is the third: the same host-wide figure cut into the last 24 hours, 7
days, 30 days and year, so a large all-time number can be told apart from a
box that is still busy. All three are covered in
[Tokens churned](#tokens-churned).

The load time is what an idle unload costs: once Studio has freed the weights,
the next request waits that long before its first token, so the figure stays on
the line — dim — while the server sits unloaded. It is timed from Studio's
`Starting llama-server` log line to its `Loaded GGUF model via llama-server`,
which nothing else reports; the first attempt is the one kept, so a load that
needed a retry is charged its full cost. While a reload is under way the column
reads `loading 4s` in bright cyan instead — claimed only while a `llama-server`
process actually exists beneath the server, because a load that dies logs
nothing that says so. The log is read incrementally from where the last redraw
stopped, so a busy server's megabytes of request lines are never re-read.

Nothing above `llama-server` reports either: Unsloth's status carries the slot
*count* but never the occupancy, and no throughput at all. So the manager finds
`llama-server` — it is a direct child of the pid already tracked — and reads
its `/slots` and `/metrics`. Each is fetched at most once per redraw — a
3-second cache on `/slots`, 2.5 on `/metrics`, whose one response serves both
the rate and the [token totals](#tokens-churned) — and both carry a 0.6s
timeout, so the 5-second redraw never waits on them.

An idle-unloaded model still shows `0/2sl`, because with nothing loaded "no
sessions" is a fact rather than a guess, and the total is the slot count it
will come back with. That is deliberately distinct from *not being able to
tell*: a probe that fails against a live server, or a server started by a
version that recorded no slot count, shows nothing at all rather than claiming
zero. Throughput is dropped for an unloaded server too — a rate measured before
the unload is history, not status.

Throughput is Δtokens ÷ Δgeneration-seconds between probes, from llama.cpp's
own cumulative counters — the rate it actually decoded at, not an average
diluted by idle time. Two details matter. llama.cpp updates those counters only
when a request *completes*, so during one long stream they would hold the
previous request's rate for as long as it runs. While a slot is generating the
figure therefore comes from `/slots` instead: each busy slot's `n_decoded`
climbs token by token, and its growth between two redraws is the live rate,
averaged across the slots that advanced so it stays a per-stream figure like the
counters'. Once the server goes quiet the last completed rate is held rather
than dropping to zero. And a window must carry at least 5 tokens to count, because
a keep-warm ping is a one-token generation and would otherwise be what the
figure reported. On first sight of a server the cumulative counters seed it, so
a freshly opened screen shows a real number instead of a blank.

The trailing `!` on the context is a **drift flag**, in red: the live context
from `/slots` disagrees with what this manager launched with. A server can be
reloaded out from under the manager — by the Studio UI, by a `/v1` auto-switch,
by any `POST /api/inference/load` — and everything else on the line is what
*we* asked for, not necessarily what is running. When they disagree, the live
number is the one shown. `settings <model>` asks the server directly and is
always the authority.

The quant sits beside the model because it is half of what identifies a
running server — and because per-model overrides are keyed `<repo>:<variant>`,
it is also what decided the settings on the line below. Both columns size
themselves to what is actually on screen, so short names leave more room for
the settings; a quant-per-folder variant keeps its tag end (`M/UD-Q4_K_XL`).

It adapts rather than bloating. On a short terminal each server collapses to
one line and every value still at a default is dropped; on a narrow one the
order flips so the settings that cost the most when wrong (tools, then the
sampling pins) survive truncation, and what does not fit is elided whole —
never sliced mid-number. If the servers would push the menu off screen, the
list is capped with a `… N more` line, because the menu is the point of that
screen.

Values it cannot establish are omitted rather than defaulted: a server started
by an older version records less, and printing `kv f16` for one actually
running `q4_0` would be worse than printing nothing.

Start walks model → **quant** → profile → **review** → **server-side tools** → **idle behaviour** → port → GPUs. The review step
shows every sampling value and the load settings that matter, each with its
source, and lets you edit them and save the edits back to that profile — see
[Editing a profile at launch](#editing-a-profile-at-launch). The quant step matters
more than it looks: per-model overrides are keyed `<repo>:<variant>`, so the
quant decides which saved profile the launch reads. The picker marks the ones
Unsloth has settings for:

```
  Q4_K_M               17.2G  fits 1 GPU    profile
  Q6_K                 20.9G  fits 1 GPU    profile
  Q8_0                 28.2G  needs 2 GPUs
  auto  (best that fits: Q8_0)
```

Note what that example shows: `auto` picks the biggest quant that fits by
weight alone, which here is the one quant with *no* saved profile. Pick
explicitly when a profile exists.

The tools step is asked outright rather than left at Unsloth's default,
because the default silently costs every bit of the concurrency the slot count
promises. It shows the measured numbers and names the slot count this launch
will actually ask for:

```
Unsloth's server-side tools SERIALISE /v1.
With them on, this model's 3 decode slots run one request at a time.

Measured here, 3 concurrent requests:
   Qwen3.8-27B   tools on 7.1 tok/s   ->  off 48.5 tok/s
   Qwen3.6-35B   tools on 37.1 tok/s  ->  off 110.6 tok/s
Single request is ~3.5x slower with tools on, too.

  Tools OFF — full concurrency, no server-side web/code  (recommended)
  Tools ON  — server-side web search + code execution, serialised
```

The choice is recorded, so `restart` keeps it; `restart --tools/--no-tools`
changes it without retyping the rest.

Or use the CLI:

```bash
python unsloth_manager.py groups                          # the 10 instance-group slots
python unsloth_manager.py groups save <slot> [--name N]   # snapshot what is running
python unsloth_manager.py groups restore <slot>           # bring it all back
python unsloth_manager.py settings <model>                # every settings source, side by side
python unsloth_manager.py presets                         # what the Studio UI has saved
python unsloth_manager.py list [--variants]               # cached models, GGUF quants, VRAM fit
python unsloth_manager.py start <model> [--port N] [--gpus 0,1] [--dry-run]
python unsloth_manager.py start <model> --kv-cache-dtype q8_0 --save-profile  # ...and write it back to the profile
python unsloth_manager.py status                          # running servers, ports, GPU usage
python unsloth_manager.py logs <model> [-n N] [-f]        # tail a server log
python unsloth_manager.py test <model> --api-key KEY      # one streaming prompt + tok/s
python unsloth_manager.py benchmark [<model>] [--preset NAME ...]  # running: as-is; else a preset sweep
python unsloth_manager.py restart <model>                 # keeps port, GPUs, profile choices and flags
python unsloth_manager.py stop <model>
python unsloth_manager.py stop-all
python unsloth_manager.py doctor
```

A typical launch:

```bash
python unsloth_manager.py start Qwen3.8-27B-GGUF --port 10001
```

That takes Unsloth's own saved profile for this model *and quant* (context, KV
dtype, speculative mode, slots, GPU pin, vision) and Unsloth's **published**
thinking-mode sampling, prints both with their provenance, and then prints what
the server reports it actually loaded. `--dry-run` stops after the plan.

Model names accept a unique suffix, so `start Qwen3.8-27B-GGUF` resolves to
`unsloth/Qwen3.8-27B-GGUF`. Preset names match the same way, so
`--preset mtp+ngram` finds `MTP+Ngram 256k`. An ambiguous shorthand is rejected
rather than guessed at.

## Command reference

Everything the CLI exposes. `--help` on any subcommand prints the same thing.

| Command | What it does |
|---|---|
| `start <model>` | Start a server. The bulk of the options; see below. |
| `stop <model>` / `stop-all` | SIGTERM, escalating to a process-group SIGKILL after `STOP_TIMEOUT`. |
| `restart <model>` | Stop then start, keeping port, GPUs, quant, profile choices and every explicit flag it was started with. |
| `status` | Running servers, their settings, ports, uptime, token totals and GPU usage. |
| `list [--variants]` | Cached models and their all-time token totals; `--variants` adds each GGUF quant, its size and VRAM fit. |
| `settings <model>` | All four settings sources side by side, plus live state if it is running. |
| `presets` | Studio presets from `studio.db`, read-only. |
| `groups [list\|save\|restore\|show\|clear] [slot]` | Instance groups — see [Instance groups](#instance-groups). |
| `keepalive <model>` | Ping a server so Studio never idle-unloads it — see [Keeping a server warm](#keeping-a-server-warm). |
| `logs <model> [-n N] [-f]` | Tail a server log. |
| `test <model>` | One streaming prompt, with TTFT and tok/s. |
| `benchmark [<model>]` | Steady-state tok/s: a running server as it is, every running server, or a preset sweep on fresh loads — see [Benchmark](#benchmark). |
| `doctor` | Check the host can serve before anything is launched. |

### `start` options

**Identity and placement**

| Flag | Meaning |
|---|---|
| `--variant Q` | GGUF quant. Default: the best that fits. Also selects which per-model override applies. |
| `--port N` | API port, within the pool. Default: lowest free. |
| `--gpus 0,1` | `CUDA_VISIBLE_DEVICES`. Default: the profile's GPU pin, else Unsloth chooses. |

**Where settings come from** — see [Where settings come from](#where-settings-come-from)

| Flag | Meaning |
|---|---|
| `--sampling docs\|unsloth\|preset\|none` | Sampling source. Default `docs`. |
| `--mode thinking\|coding\|instruct` | Which published profile, with `--sampling docs`. Default `thinking`. |
| `--load-profile auto\|override\|preset\|none` | Load-config source. Default `auto`. |
| `--preset NAME` | Studio preset; `none` to skip. Default: whatever the UI has selected. |

**Load overrides** — each beats the profile

| Flag | Meaning |
|---|---|
| `--ctx N` | Context length; `0` asks for the largest that fits. |
| `--parallel N` | Decode slots. |
| `--tensor-parallel` / `--no-tensor-parallel` | Split by tensor instead of by layer. |
| `--kv-cache-dtype T` | `f32 f16 bf16 q8_0 q5_1 q5_0 q4_1 q4_0 iq4_nl`. |
| `--spec MODE` | `auto mtp dspark dflash ngram mtp+ngram off ngram-simple`. |
| `--spec-draft-n-max N` | Draft tokens per step (1-16); MTP/DSpark/DFlash only. |
| `--vision` / `--no-vision` | Load the companion mmproj. `--no-vision` frees ~0.9 GB. |
| `--n-batch N` / `--n-ubatch N` | llama-server `--batch-size` / `--ubatch-size`. |
| `--extra ARG` | Repeatable raw llama-server argument, e.g. `--extra=-ngl --extra=99`. |

**Sampling pins** — each beats the profile, and each is a *hard* override; see
[Pinning sampling is a hard override](#pinning-sampling-is-a-hard-override)

`--temperature` (0-2) · `--top-p` (0-1) · `--top-k` (-1..100) · `--min-p` (0-1)
· `--presence-penalty` (0-2) · `--repetition-penalty` (1-2)

**Server behaviour**

| Flag | Meaning |
|---|---|
| `--tools` / `--no-tools` | Unsloth's server-side web/code tools. Default on — and it serialises `/v1`. |
| `--keep-warm` / `--no-keep-warm` | Hold this server against Studio's idle unload. Default off. |
| `--wait N` | Seconds to wait for readiness; `0` returns immediately. Default `LOAD_TIMEOUT`. |
| `--save-profile` | Once the model has loaded, write the values given as flags back into the profile they override — the Studio preset and/or this quant's per-model override. See [Editing a profile at launch](#editing-a-profile-at-launch). |
| `--dry-run` | Print the plan and the command line, start nothing. Touches no state and saves nothing. |
| `--force` | Start despite a VRAM warning or an instance already running. |
| `--api-key K` | Key for the post-load readout. Default: read from the server's own log. |

### Other command options

| Flag | Meaning |
|---|---|
| `logs -n N` / `--lines N` | Lines to show (default 40). |
| `logs -f` / `--follow` | Follow the log as it grows. |
| `test --prompt "..."` | Prompt to send (default: a short built-in one). |
| `test --max-tokens N` | Generation cap (default 256). |
| `benchmark --preset NAME` | Sweep only. Repeatable; default is every preset in `studio.db`. |
| `benchmark --max-tokens N` | Generation cap per measured run (default 400). |
| `benchmark --gpus` / `--variant` | Sweep only: placement and quant, held constant across runs. |
| `settings --variant Q` | Which quant's profile to look up (default: best that fits). |
| `groups --name "..."` | Label for a saved group. |
| `groups --wait N` | Seconds to wait per model on restore (`0` = don't wait). |
| `restart --preset` / `--port` / `--gpus` / `--variant` / `--tools` / `--keep-warm` / `--wait` | Change one thing; everything else is kept. |
| `keepalive --interval N` / `--api-key` | Ping period (default: half the idle TTL) and token. |
| `test --api-key` / `benchmark --api-key` / `settings --api-key` | Bearer token. Default: read from the server's own log. |

### The TUI

`python unsloth_manager.py` with no arguments. The menu:

| Entry | |
|---|---|
| **Start Model** | model → quant → profile → review (edit, and optionally save back) → tools → idle behaviour → port → GPUs, then the launch plan. |
| **Instance Groups** | Restore / save / show / clear the ten slots. |
| **Model Settings** | `settings` for one model. |
| **Stop Model** | Stop one running server. |
| **Status** | Full `status` output. |
| **List Models** | Cached models and quants. |
| **Test Model** | One streaming prompt against a running server. |
| **Benchmark** | Every running server as it is, one running server (`*`), or a preset sweep for a model that is not running. |
| **API Tester** | Hands off to `api_tester.sh`. |
| **View Logs** | Tail a server log. |
| **Studio Presets** | What the Studio UI has saved. |
| **Environment Check** | `doctor`. |

Above the menu, each GPU gets one live line, refreshed with the rest of the
header and laid out like btop's GPU box:

```
GPU1 [████████████████░░░░] 26/32G   use ■■■■■■■■ 98%  82°C   pwr ■■■■···· 147/300W  Tesla V100-SXM2-32GB
```

Memory is split into this manager's servers (green) and everything else
(yellow). `use` is compute utilisation and `pwr` is draw against the card's
power limit; both meters shade green → yellow → red cell by cell, and the
temperature turns yellow at 70°C and red at 85°C. A sensor the card does not
report shows `-`. `status` prints the same figures, and all of them come from
the single `nvidia-smi` query the memory bars already made.

Every screen takes Up/Down/Home/End, Enter to choose, and Esc or `q` to back
out — cancelling any step of a flow abandons the whole action rather than
falling through to a launch.

## Where settings come from

Four sources can decide how a model runs. `settings <model>` prints all four
side by side, with the values each one holds, before you commit to any of them:

```bash
python unsloth_manager.py settings Qwen3.6-35B-A3B-MTP-GGUF
```

### 1. Unsloth's published guides — the sampling default

Transcribed into `unsloth_profiles.py` from the pages each entry names, and the
default for `start`. This is the one thing here that is not read live, and it
exists because **Unsloth's shipped defaults disagree with Unsloth's own
documentation** for these models:

| Qwen3.6 / Qwen3.8 | temp | top_p | top_k | min_p | presence | rep |
|---|---|---|---|---|---|---|
| docs, thinking (general) | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| docs, thinking (coding, 3.6 only) | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| docs, instruct / non-thinking | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 |
| **shipped `inference_defaults.json`** | **0.7** | **0.8** | **20** | **0.0** | **1.5** | 1.0 |

The shipped family row is the *Instruct* row. Both of these models run in
thinking mode, so serving them on Unsloth's own default sampling is serving
them wrong. Pick a row with `--mode thinking|coding|instruct`.

### 2. Unsloth's shipped defaults — what the server does on its own

`assets/configs/inference_defaults.json` plus any model-specific YAML, inside
the Studio venv. The server applies these to **any sampling field a request
omits**, resolved exactly the way `load_inference_config` does, so `settings`
can show you what you would get by pinning nothing. Note that Unsloth
deliberately never auto-applies `repetition_penalty` — it is manual-only, the
same as in the chat UI — and `settings` says so.

Ask for these instead of the published ones with `--sampling unsloth`.

### 3. Per-model launch overrides — the load default

`app_settings.openai_api_auto_switch_overrides` in `studio.db`, keyed
`<repo>:<variant>`: the load config the **Studio settings page** saves for one
model and one quant. Context, KV dtype, speculative mode, draft depth, slots,
batch sizes, tensor split, vision, GPU pin.

The catch that makes this manager's job necessary: Unsloth applies these
**only** when a `/v1` request auto-switches models. A direct
`POST /api/inference/load` — which is exactly what `unsloth studio run` does —
ignores them entirely. So a headless launch that does not read them itself
silently runs on nothing.

| override field | becomes |
|---|---|
| `custom_context_length` / `max_seq_length` | `--max-seq-length` |
| `n_parallel` | `--parallel` |
| `speculative_type` | `--speculative-type` (Unsloth's own vocabulary) |
| `spec_draft_n_max` | `--spec-draft-n-max`, for `mtp`/`dspark`/`dflash` only |
| `tensor_parallel` | `--tensor-parallel` |
| `kv_cache_dtype` | `--cache-type-k X --cache-type-v X` (llama-server passthrough) |
| `n_batch` / `n_ubatch` | `--batch-size` / `--ubatch-size` (passthrough) |
| `disable_vision` | `--no-mmproj` (passthrough) |
| `gpu_ids` | `CUDA_VISIBLE_DEVICES` |
| `llama_extra_args` | prepended to `--extra` |
| `gpu_memory_mode`, `gpu_layers`, `n_cpu_moe`, `chat_template_override` | **nothing** — no `studio run` flag carries them; the launch prints a `note:` saying so |

Use the long `--cache-type-k` spelling, not `-ctk`: the CLI's unknown-flag
passthrough parses a clustered short option, so `-ctk` arrives at llama-server
as `-ct` and the load dies with *"does not recognise the argument '-ct'"*.

### 4. Studio presets — global, and now opt-in

`chat_settings.customPresets` in `studio.db`: what the chat UI's Preset
dropdown saves, with `activePreset` recording the current selection. Two things
about presets that are easy to get wrong:

- **They are global, not per-model.** `chat_settings` is a flat key/value table
  with no model column, so a preset applies to whatever model is loaded. It is
  not attached to the model you were looking at when you saved it.
- **They carry no model, port or GPU.** Those are always the manager's job.

A preset supplies the load config only when no per-model override exists (or
with `--load-profile preset`), and its sampling is **never** pinned unless you
ask for it with `--sampling preset`. `presets` lists them and says so.

A preset's `disableVision` is honoured the same way an override's
`disable_vision` is — as `--no-mmproj`. Until it was, a preset that said
"vision off" launched with the projector attached, and the first idle reload,
reading the override where the same choice usually also lives, quietly took it
away again.

### Choosing, and being told

```bash
--sampling docs      # published profile for this family (default)
--sampling unsloth   # pin nothing; the server applies its own recommendation
--sampling preset    # the Studio preset's params
--sampling none      # pin nothing, and do not claim a profile

--load-profile auto      # per-model override, else the Studio preset (default)
--load-profile override  # per-model override, or fail if there is none
--load-profile preset    # the Studio preset only
--load-profile none      # nothing — Unsloth's defaults throughout
```

Every value stays overridable per launch — `--ctx`, `--parallel`,
`--tensor-parallel`, `--kv-cache-dtype`, `--spec`, `--spec-draft-n-max`,
`--vision/--no-vision`, `--n-batch`, `--n-ubatch`, and each of `--temperature`,
`--top-p`, `--top-k`, `--min-p`, `--presence-penalty`, `--repetition-penalty`.
Precedence is always **explicit flag > profile > Unsloth's default**, and the
launch plan prints the winning source in brackets next to every value. Add
`--save-profile` and the explicit values are written back into the profile they
beat, once the model has loaded — see
[Editing a profile at launch](#editing-a-profile-at-launch).

`--extra` passes raw arguments to llama-server (repeatable, GGUF only):
`--extra=-ngl --extra=99`.

### Pinning sampling is a hard override

`unsloth studio run --temperature 1.0` does not set a default — it writes
`UNSLOTH_SAMPLING_TEMPERATURE`, which the server treats as an **operator pin
that beats even a value a client sent explicitly**. The precedence inside
Unsloth is:

> operator pin → client's explicit value → per-model recommendation → schema default

So `--sampling docs` (the default) means clients cannot change temperature.
`--sampling unsloth` or `none` pins nothing, leaving clients free to set what
they like and the server's per-model recommendation to fill the rest. The
launch plan says which of these you are getting, in as many words.

### Speculative decoding is Unsloth's job, not ours

`unsloth studio run` has a first-class `--speculative-type` that takes
Unsloth's own vocabulary (`auto`, `mtp`, `dspark`, `dflash`, `ngram`,
`mtp+ngram`, `off`, `ngram-simple`) and does the llama-server translation
itself, including probing what the installed binary supports.

Do **not** emit llama-server's `--spec-type` by hand. Supplying it at all
suppresses Unsloth's capability-aware auto-emit, so a token this build does not
know silently *disables* speculative decoding instead of enabling it. (An
earlier version of this manager translated the tokens itself and probed
`llama-server --help` to do it; that whole layer is gone.)

`--spec-draft-n-max` is emitted only for the modes that actually launch a
drafter with a configurable depth. Ask for it with anything else and the launch
says it is being ignored rather than passing a flag nothing reads.

## Editing a profile at launch

Between choosing where settings come from and launching, the TUI shows what
the launch will actually run with — every sampling value and the load settings
that decide cost and behaviour — each next to the source it came from, and lets
you change any of them in place:

```
Review settings  (Enter edits a value)

ukisai/Swift-Qwen3.8-27B-GGUF   [Q4_K_L]
load from preset Qwen3.8-27B-Thinking   ·   sampling from preset Qwen3.8-27B-Thinking

Edits save to:  load     → preset "Qwen3.8-27B-Thinking" + the Q4_K_L override
                sampling → preset "Qwen3.8-27B-Thinking"
Sampling values are pins: they win over what a client sends.
────────────────────────────────────────────────────────
  Sampling
  * temperature     0.8               edited, was 1.0
    top_p           0.95              preset Qwen3.8-27B-Thinking
    ...
  Load
    context         262,144 (256k)    preset
  * kv cache        q4_0              edited, was q8_0
    speculative     mtp               preset
    ...
    Continue — the edits apply to this launch only
    Continue, and save to preset "Qwen3.8-27B-Thinking" + the Q4_K_L override once it has loaded
    Undo all edits
```

Numbers are typed (`256k` and `0` for fit-max work for the context); the KV
dtype, speculative mode, vision and tensor split are picked from a list. A value
out of the range `unsloth studio run` accepts is refused on the spot rather than
at launch, and setting a field back to what the profile already gave it is not
an edit.

An edit becomes the matching `start` flag, so **Continue** is exactly a launch
with those flags — and, like any flag, it is kept in the instance's saved
answers, which `restart` and instance groups replay. **Continue, and save**
also writes the edits back into the profile each one came from, so the *next*
start of this model reads them from there, and so does Studio's own idle
reload. On the CLI it is the same thing spelled `--save-profile`:

```bash
python unsloth_manager.py start Swift-Qwen3.8-27B-GGUF --variant Q4_K_L \
    --preset Qwen3.8-27B-Thinking --load-profile preset --sampling preset \
    --temperature 0.8 --kv-cache-dtype q4_0 --save-profile
```

The launch plan shows the save before anything happens:

```
profile save: after the model loads, through Studio's own settings API
  preset "Qwen3.8-27B-Thinking"
    kv cache        q8_0              ->  q4_0
    temperature     1.0               ->  0.8
  override ukisai/Swift-Qwen3.8-27B-GGUF:Q4_K_L   — what an idle reload rebuilds from
    kv cache        q8_0              ->  q4_0
```

### Where each edit goes

| the value came from | a save writes it to |
|---|---|
| a Studio preset — load settings | that preset **and** this quant's per-model override |
| the per-model override, or nothing, under `--load-profile auto`/`override` | this quant's per-model override (created if there is none) |
| a Studio preset — sampling | that preset |
| the published docs table, or Unsloth's own defaults — sampling | **nowhere**: kept on this instance only, and the launch says so |
| anything under `--load-profile none` | **nowhere**: kept on this instance only |

The override is written alongside the preset because an idle reload reads
**only** the override (see [Idle auto-unload](#idle-auto-unload-and-what-a-reload-restores)):
an edit saved to the preset alone would be undone the first time the server
sat idle for five minutes. So whenever a save goes to Studio, the override is
made to reproduce this launch's load config in full — which also ends the job
of keeping a preset and its quant's override in step by hand. Overrides hold no
sampling, and do not need to: sampling pins ride the server's environment,
which no reload touches.

Sampling from the published docs table has no profile to go back to — that
table is a transcription of Unsloth's guides, not a setting. An edit to it stays
an explicit flag on this instance: `restart` and a saved group keep it, and no
other launch of the model sees it. The review screen and the launch plan both
mark it **instance only** rather than letting it look saved. To make it a saved
setting, launch that model from a Studio preset.

One consequence of `--load-profile auto` worth knowing: it reads the per-model
override *before* the preset, so a save for a quant that had no override creates
one, and from then on `auto` loads that quant from the (identical) override. The
plan notes it when it happens.

### How the save is made

- **Through Studio's API, not the database.** A preset goes through
  `POST /api/chat/settings/compare-and-set` and an override through
  `PUT /api/settings/openai-auto-switch/overrides` — the routes Studio's own
  settings page uses — so Studio's validation, locking and cache invalidation
  all apply. Nothing in this manager opens `studio.db` for writing.
- **After the model loads, through the new server.** Those routes need a live
  server and its key, and the key is printed only once the load completes. The
  useful consequence: a profile is never rewritten with settings that failed to
  load. If the load dies or times out, or `--wait 0` means it is never
  confirmed, nothing is saved and the launch says so; the edits stay on the
  instance. `--dry-run` shows the save and does not make it.
- **Refused, not merged, on a conflict.** The preset entry and the override row
  must still be exactly what the launch read. If either was changed in Studio in
  the meantime, that part is not written, and the message says why.
- **The whole row goes back.** Studio's override save *replaces* the row —
  anything left out is cleared, a GPU pin included — so the payload is the
  stored row with only the changed fields swapped in (extra llama args and the
  server-tuning fields Studio carries over on its own). A preset keeps every
  field this manager does not model: max tokens, system prompt, reasoning
  budget and the rest.
- **Read back.** Studio's normaliser drops a value it cannot use rather than
  refusing the save, so a 200 proves little. The saved entry is read back, and
  anything that did not stick is named.

```
Profile save

  preset "Qwen3.8-27B-Thinking": saved kv cache, temperature
  override ukisai/Swift-Qwen3.8-27B-GGUF:Q4_K_L: saved kv cache
```

A successful save removes those values from the instance's saved answers, so
`restart` and a group saved afterwards read them from the profile — where a
later edit in Studio still reaches them — instead of freezing this launch's
copy. A group saved *before* the edit already names the preset, so its next
restore picks the new values up too.

## Context, and what "max" means

Both target models are natively 262144 tokens, read straight from the GGUF
header (`{arch}.context_length`) with no server running.

There are two different things "max context" can mean, and they behave very
differently:

- **Fit-max** — send no `--max-seq-length` at all. Unsloth starts at the GGUF's
  native length and caps it to what its own architecture-aware estimator says
  fits on the selected GPUs. You get the largest context that fits, and nothing
  ever spills to CPU. This is what you get with `--load-profile none`, or when
  no profile names a context.
- **Pinned** — send an explicit number. Unsloth honours it verbatim and turns on
  llama.cpp's `--fit`, which will offload layers to CPU rather than shrink the
  window. You get the full context, possibly much slower.

A saved per-model override naming `262144` is a deliberate pin, so it is sent
verbatim. Everything else defaults to fit-max. The launch plan says which one
you are getting, and the post-load readout says what the server settled on:

```
    context    : 262,144 (256k)   of native 262,144 (256k)   (fits up to 190,464 here)
```

## Idle auto-unload, and what a reload restores

Studio frees a GGUF after `openai_api_auto_unload_idle_seconds` of inactivity
and reloads it on the next `/v1` request. This is on for this box (300s), and
it has a consequence worth stating plainly:

> **A reload does not replay the command line the manager launched with.** It
> rebuilds a `LoadRequest` from the per-model override in `studio.db`
> (`resolve_override_for_load` → `model_override_load_kwargs`).

So anything the manager set that the override does not *also* carry is dropped
the first time a server sits idle — silently, with no restart and no error.

Two things are safe regardless:

- **Sampling.** The pins ride `UNSLOTH_SAMPLING_*` in the server's environment,
  which no reload touches.
- **Anything that came from the override in the first place.** With the default
  `--load-profile auto` the manager derives its flags *from* that same override,
  so a reload reproduces the launch exactly. Verified on this box by diffing the
  first and last `llama-server` command lines across a real unload/reload cycle:
  identical for Qwen3.6-35B, and for Qwen3.8-27B differing only in `--no-mmproj`
  → `--no-mmproj-auto`, which llama-server parses as the same option.

Everything else drifts, so `start` checks and says so before launching:

```
idle reload: WARNING — after 5m idle Unsloth frees this model and reloads
             it from studio.db, not from this command line:
               context         fit-max           ->  262,144
               kv cache        -                 ->  q4_0
               speculative     -                 ->  mtp
```

The check compares each field a reload actually re-supplies. It stays quiet
when the idle timeout is 0, and when nothing would change it says so in one
line rather than nothing at all. `--parallel` is deliberately exempt: it is a
server-wide startup default, so a reload that omits `n_parallel` still lands on
the value the manager passed.

Vision is compared by what each side *yields*. A launch that says nothing about
it still attaches the projector — Unsloth does that on its own when the
snapshot ships an mmproj — so an override with `disable_vision` is drift even
against a launch that never mentioned vision. Missing exactly that case is how a
preset launch ran for hours with the mmproj attached and then lost it to an
idle reload unannounced. With no projector on disk there is nothing to compare.

The fix the warning suggests — make the override match — is one keystroke on
the review screen: **Continue, and save** writes the launch's load config into
the override. See [Editing a profile at launch](#editing-a-profile-at-launch).

The readout also compares what the server reports against what the manager
launched with, and says so when they disagree — a server can be reloaded out
from under this tool by the Studio UI, a `/v1` auto-switch, or any
`POST /api/inference/load`:

```
WARNING — this is not what the manager launched. Something reloaded it
          (the Studio UI, a /v1 auto-switch, or any /api/inference/load):
            context       262,144           ->  180,992
            kv cache      q4_0              ->  f16 (default)
            speculative   mtp               ->  ngram
          `restart` puts it back on the launch settings.
```

A setting that *disappeared* counts: launching with `q4_0` and finding the
server on its `f16` default is exactly the case this exists to catch. The home
screen carries the short version — a red `!` on the context.

When a server is idle-unloaded, the settings readout says that rather than
reporting empty fields:

```
state      : idle-unloaded — Unsloth freed the weights after 5m idle.
             The next /v1 request reloads it; VRAM is free until then.
```

## Keeping a server warm

Studio's idle unload is **global**. `openai_api_auto_unload_idle_seconds`,
`model_memory_keep_resident` and the auto-switch toggle all live in
`app_settings`, apply to every model in a `STUDIO_HOME`, and none has an
environment override — so "don't unload *this* one" cannot be expressed through
Unsloth's settings at all. (The manager writes back to Studio only to save a
profile you edited at launch; a global unload policy is not that.)

It is done from outside instead. `start --keep-warm` (and the TUI's **Idle
behaviour** step) launches a small detached process that pings that one server
just often enough that it never goes idle:

```bash
python unsloth_manager.py start <model> --keep-warm
python unsloth_manager.py restart <model> --keep-warm
python unsloth_manager.py keepalive <model>        # run one in the foreground
```

The interval is half the configured TTL, clamped to 30–600s, so two pings fit
in every idle window. Only a **POST to an inference path** stamps activity in
Studio's tracker (`LlamaKeepWarmMiddleware`), so the ping is a real one-token
completion — with `enable_tools: false` and `tool_choice: "none"`, because
otherwise the ping itself would serialise `/v1` behind the tool machinery (on a
tools-on server only the second one works; see [Gotchas](#gotchas-worth-knowing)).

The pinger is recorded in the state file and killed by `stop` / `stop-all`
before the server it was holding: a survivor would keep POSTing at a dead port
and, with auto-switch on, could even reload the model it was meant to be
keeping warm. It also exits on its own once the server leaves the state file,
so it cannot outlive what it was defending. `status` shows `keep-warm`
(or `keep-warm(DEAD)`), the home screen a green `warm`.

Other models keep getting reclaimed as usual — that is the point of doing it
per-server rather than turning the global setting off.

Measured here: with the 300s TTL in force, a server held this way ran 10m53s
with `sessions 0/3` and no unload, its pings landing 150s apart at ~200ms each.

## Reading settings back after a load

Once a server is ready, `start` (and `settings`, for an already-running model)
asks it `GET /api/inference/status` and prints what it actually loaded: the
effective, maximum and native context, the KV dtype, the speculative mode plus
any fallback reason, the slot count it settled on versus the one requested, the
capabilities it advertises, and the per-model sampling recommendation it holds.

That endpoint needs a bearer token. The manager still creates and stores no
keys — it reads back the one `unsloth studio run` minted and printed into the
log it was already capturing. Pass `--api-key` to use a different one. If no
key can be found, the readout is skipped and says so; the load is unaffected.

## Tokens churned

The home screen carries three token figures, and they are deliberately
different things. Each server's line shows what **that server** has processed
since it started; the summary line at the top shows what **this box** has
processed all-time — every model, models that are not running now included, so
stopping a server never makes the number fall; and the `Tokens` line shows that
same host-wide total cut into **windows**, so an all-time figure that took a
year to build is not mistaken for a box that is busy today.

```
Running: 2 model(s), 2 ready   all-time ↑18.9M ↓4.1M     ⏱ 21:40:03 · refresh 5s
  ● Qwen3.8-27B-GGUF  UD-Q4_K_XL :10002 rdy g1  1/3sl  16 tok/s  ↑614K ↓121K
  Tokens  24h ↑412K ↓96.2K   7d ↑3.1M ↓702K   30d ↑11.4M ↓2.6M   365d ↑18.9M ↓4.1M
```

The window line is always drawn, running or not — the question it answers is
about the box, not about what happens to be loaded — and it is always one line.
A terminal too narrow for all four windows drops them from the wide end rather
than letting the edge slice a figure in half, because half of `↑18.9M` reads
as a real, smaller number.

`status` prints both per server, in full, and adds the host line:

```
      tokens  this run ↑412,004 in ↓96,211 out   ·   all-time ↑9,204,551 in ↓2,001,884 out
  Tokens, all-time: ↑18,912,004 in · ↓4,102,119 out across 7 model(s), 5 not running
```

`list` carries the all-time pair for any model that has ever served a request,
running or not, and the run figures beside it for the ones that are.

**Where the numbers come from.** `llama-server` counts the tokens it processes,
and the manager reads its `/metrics` on the same poll that already fetches the
decode rate — one request, not two. But those counters live and die with that
process: on this box an idle unload throws it away after five minutes, and the
reload starts again at zero. So the totals are kept here instead, in
`STATE_DIR/tokens.json`, banked from the difference each poll sees:

- The llama-server's **pid** decides what "new" means. The same process means
  the difference since the last read; a different one — or none on record —
  means the whole counter, because that process started at zero. Without that
  check an idle reload would either double-count the new server or drop it.
- Because the counters are cumulative, a gap in polling costs nothing as long
  as the process survives it: the next look catches the whole gap up. What is
  lost is a `llama-server` that loaded, worked and idle-unloaded *entirely*
  between two looks — so the totals are complete for a box where the TUI is
  left open or `status` is run now and then, and conservative otherwise.
- `stop` takes one last reading before it signals, so an orderly stop banks
  everything; a `kill -9` from outside costs at most the last poll.
- The file is kept separate from `state.json` on purpose. It is written by
  every poll that sees traffic, and a counter file that goes missing or lands
  half-written must never be able to cost a server its record. It is written by
  atomic replace, and a read that fails starts the counting over rather than
  failing the command.

**What the two figures actually count.** Both are `llama-server`'s own: `↓` is
tokens predicted, `↑` is prompt tokens *evaluated*. A request whose prefix was
already in the KV cache is not re-evaluated, so the input figure is the work
the server did, not the size of what the client sent — it reads low against a
provider-style bill, and it is the honest number for "what has this box
chewed through". Both only move when a request **completes**, which is the same
caveat the throughput figure carries: a stream in flight adds nothing until it
ends.

**How the windows are kept.** The all-time counters cannot be cut into windows
after the fact: a banked difference means nothing without the moment it was
banked at. So the same differences are also stamped into buckets in
`tokens.json`, host-wide rather than per model, and a window is the buckets it
reaches summed:

- **Hour buckets for the last 30 days, day buckets out to a year.** A 24h
  figure summed from day buckets would be off by most of a day; a year of hour
  buckets would be 8760 entries. Hours older than 30 days are folded into their
  day on the next save, days older than 366 are dropped, so the file stays at a
  few hundred entries and does not grow without bound.
- **A bucket that straddles the start of a window counts whole.** Only the
  365-day figure is ever rounded by more than an hour, and there it is worth
  less than a third of a percent. These are trend figures; the sampling behind
  them is not precise to the minute either.
- **A difference lands in the hour of the poll that noticed it**, not spread
  over the time it actually accumulated. With the home screen open that is a
  5-second resolution; a box polled once a day banks a day's work into one
  hour. The window totals are right either way, the shape within them is not.
- **History starts when a version that keeps it first polls.** An existing
  `tokens.json` carries all-time counters and no buckets, so the windows read
  `↑0 ↓0` until traffic is seen — the all-time figures are untouched and keep
  counting from where they were.

There is no reset command. Delete `STATE_DIR/tokens.json` — with nothing
running, or accepting that each live server's current counters are then banked
in full on the next poll — and counting starts over.

## Instance groups

A group is a named snapshot of what is running **and every answer each launch
was given** — model, quant, port, GPUs, load profile, sampling profile and
mode, tools, and any explicit overrides. Ten slots, stored in
`STATE_DIR/groups.json`. The point is bringing a box back after a reboot in one
command.

```bash
python unsloth_manager.py groups save 1 --name "default set"
python unsloth_manager.py groups restore 1
python unsloth_manager.py groups            # list all ten slots
python unsloth_manager.py groups show 1     # every saved answer
python unsloth_manager.py groups clear 1
```

Saving prints the restore command, plus the systemd and cron lines, so the
service is copy-pasteable:

```
Restore after a reboot with:
    python /root/unsloth-manager/unsloth_manager.py groups restore 1

As a systemd unit (After=network-online.target, Type=oneshot):
    ExecStart=python /root/unsloth-manager/unsloth_manager.py groups restore 1

Or from cron:
    @reboot python /root/unsloth-manager/unsloth_manager.py groups restore 1 >> /root/.unsloth-logs/group-restore.log 2>&1
```

Four things make it safe to run unattended:

- **Idempotent.** A model already running is skipped, so a `@reboot` job that
  fires twice, or a service restarted by hand, does nothing the second time.
- **Partial failure is survivable.** One member failing does not abandon the
  rest; the summary names what failed and the exit status is 1, so a service
  reports the failure without having lost the models that did come up.
- **Ports are pinned to what was actually serving**, not to what was asked for.
  A launch that auto-assigned stored `None`, and re-auto-assigning would not
  reproduce the layout.
- **Answers, not resolved settings.** A restore replays the same choices
  through `start`, so it picks up whatever `studio.db` says today rather than
  freezing a command line that Unsloth's own idle-reload would contradict five
  minutes later.

Groups are also the second screen in the TUI (**Instance Groups**), which
restores, saves, shows and clears slots. And because a menu-driven launch is
otherwise hard to turn into a cron job, every `start` now prints itself as a
command:

```
this launch as a command:
  python .../unsloth_manager.py start unsloth/Qwen3.8-27B-GGUF --variant UD-Q4_K_XL --port 10002 --gpus 1 --no-tools
```

Only what you actually chose appears — a value a profile supplied is implied by
`--load-profile` and is deliberately not frozen into the line.

The same answers are what `restart` replays, explicit flags included: a
`--ctx`, a sampling pin kept on the instance because it had no profile to be
saved into, an `--extra`. A value that `--save-profile` wrote back is dropped
from the answers once the save lands, so from then on it comes from the profile
like everything else.

## Ports and instance limit

**The API lives on ports 10001-10008, and nowhere else.** Eight ports, eight
concurrent servers — the pool size *is* the instance cap, so there is no second
limit that could disagree with it.

`start` takes the lowest free port in the pool. `--port` is accepted only for a
port inside it; anything else is refused rather than bound somewhere no client
knows to look. A ninth `start` is refused too, and says which eight are holding
the slots:

```
Error: already running 8 of 8 instances (ports 10001-10008).
       org/model-a :10001, org/model-b :10002, ...
       Stop one first, or raise UNSLOTH_MGR_MAX_INSTANCES.
```

That is deliberately a different message from an exhausted pool, because the
fix is different — a port held by something that is not this manager reports
which one, so you go looking for the squatter instead of stopping a server:

```
Error: no free port in 10001-10008: 7 in use by this manager, and 10008 is
       held by another process.
```

Move or resize the pool with `UNSLOTH_MGR_BASE_PORT` and
`UNSLOTH_MGR_MAX_INSTANCES`; they are always contiguous.

Note this is the *manager's* pool only. Unsloth's own web UI unit still listens
on 8888, and `unsloth studio stop` would stop that too — use this tool's `stop`
/ `stop-all`, which only touch what it started.

## API keys

Unsloth Studio always requires a bearer token on `/v1`. **This manager does not
create, store, or rotate keys**, by design:

- `unsloth studio run` mints a fresh key at every launch and prints it. The
  manager captures the server's output to its log, so the key is there:
  `grep 'API Key' /root/.unsloth-logs/<model>.log` (or wherever `UNSLOTH_MGR_LOG_DIR` points)
- Keys live in `STUDIO_HOME/auth/auth.db` and are **not** tied to a single
  server. Any key already in that database authenticates against every server
  the manager starts, on any port. One long-lived key is usually what you want.

`test`, `benchmark` and the settings readout need a key, and they now find it
themselves: an explicit `--api-key` wins, then `UNSLOTH_MGR_API_KEY`, then the
key this manager's own log already recorded for that server. Nothing new is
stored anywhere. For `api_tester.sh`, put the key in each endpoint's `auth`
field. On its first run it writes `.api_tester.json` (mode 600, git-ignored)
with one endpoint per port in the pool, already carrying
`UNSLOTH_MGR_API_KEY` if that is set. The file is never regenerated after
that.

One race worth knowing about: the log line the manager watches for to call a
load "ready" is printed *before* the startup banner carrying the key, so the
post-load readout waits briefly for the key to appear rather than reporting it
missing.

## Benchmark

```bash
python unsloth_manager.py benchmark                         # every running server, as it is
python unsloth_manager.py benchmark <running model>         # that server, as it is
python unsloth_manager.py benchmark <model> --preset "Default 1" --preset "MTP+Ngram 256k"
```

The command answers one of two questions, decided by whether the model is
already running.

**A running server is measured as it is, and left running.** Its settings are
whatever it was launched with, or whatever Studio has since reloaded it with,
and no flag of this command chose them. So the saved result carries them: the
profile, quant, port, GPUs, context (live, as well as launched), KV type, slots,
spec mode, tools, and sampling pins. With no model at all, every running server
is measured in turn, in port order. `--preset`, `--gpus` and `--variant` are
refused here, because they describe a fresh load; stop the server first to
sweep it.

A running server's model may not be loaded when its turn comes. It may have
been idle-unloaded, or swapped out by a client that asked that server for a
different model. The warm-up brings it back, gets the load timeout rather than
the stall one, and its wait is reported in the `reload` column. The run says
which case it was: swapping a model back in interrupts whoever was using the
other one. It also names any field that the reload restores from `studio.db`
differently from the launch, since the reload is what gets measured
([Idle auto-unload](#idle-auto-unload-and-what-a-reload-restores)).

**A model that is not running gets a preset sweep.** With no `--preset` it runs
**every** preset in `studio.db`, which is the question the sweep exists to
answer: which of your presets is actually fastest on this model. Each run uses
`--load-profile preset --sampling none`, so the preset is genuinely the only
thing that differs: a per-model override would otherwise supply the same load
config to every run and flatten the comparison. For each one it starts the
server, waits for readiness, measures, then hard-stops the server. Other servers
already running are listed and left alone; they share the host, and on a shared
GPU their traffic is in the numbers.

Either way, a warm-up plus N measured generations with distinct prompts (so
prefix caching cannot collapse the batch). It reports:

- **Steady tok/s**: best sustained rate over a sliding window (default 5s), so
  one mid-stream stall does not sink the number. `n/a` when the generation was
  too small to distinguish a sustained rate from a burst.
- **Mean tok/s**: overall including stalls, for contrast.
- TTFT, chunk count, stall count, load (sweep) or reload (running) time, and
  per-GPU VRAM attributed to that server's own processes, not to everything
  the manager runs.

Results print as a table and are saved as timestamped JSON in the log dir
(`bench-<model>-<stamp>.json` for a sweep, `bench-<model|running>-live-<stamp>.json`
for running servers).

**Runs that shared the server are flagged `shared`, never silently averaged
in.** After each measured run the benchmark reads what the server log gained
during it. Any request other than its own, or any `Starting llama-server` (a
model load), marks the run. A keep-warm ping counts too. The log is the only view
that sees all of it. `llama-server`'s `/slots` shows requests decoding at that
instant, but not one queued behind Unsloth's tool layer, nor one waiting on a
model switch. A client that asks the same server for another model makes Studio
unload yours, load theirs, and load yours back. That cost a measured run over a
minute of TTFT while `/slots` reported the server idle.

Benchmark requests name the server's own model, matched by repo id on
`/v1/models`, and never simply the first entry there. Studio lists every model
it could serve on that endpoint, and asking for any of them swaps it in. On an
idle-unloaded server the first entry can be some other cached model, and
requesting it loads that model onto the server. `test` and the keep-warm pinger
look the id up the same way.

They then send it as `repo:QUANT` (e.g. `ukisai/Swift-Qwen3.8-27B-GGUF:Q4_K_L`),
pinned to the quant the server was launched with. `/v1/models` lists only the
bare repo id, and naming that on an idle-unloaded server lets Studio reload
whichever quant *it* prefers. A quant with no saved override loads on estimator
defaults: Qwen3.8-27B launched as Q4_K_L / q8_0 KV / 262k came back from a
bare-id reload as Q4_K_M / f16 KV / 210k, and ran the GPU out of memory on the
first long prompt. With the quant named, the reload takes that quant's override;
on a loaded server the same id is answered by the resident model, no reload.
**Point your own clients at `repo:QUANT` too** — the bare id is what they will
copy from `/v1/models`, and it is the one that drifts.

They also send `enable_tools: false` and `tool_choice: "none"`, as `test`
does. With Unsloth's
server-side tools on, a model asked for a long technical answer stops to run web
searches partway through. That puts network round-trips into TTFT, and on a
live server it queues other clients behind the benchmark. The numbers describe
the model, not the tool layer. What tools on costs a server is covered in
[Gotchas](#gotchas-worth-knowing).

Two honest caveats about the numbers:

- Counts are **streamed SSE chunks**, not tokenizer tokens. In practice
  llama.cpp emits one content piece per token, but they are not guaranteed
  equal, so the column is labelled `chunks`.
- `ignore_eos` is not honoured through Unsloth, so runs cannot be pinned to a
  fixed length. The benchmark prompt deliberately asks for a long answer; a
  model that stops early yields a smaller sample and may report `n/a`.

## How servers are launched

The manager runs as root and starts each server with `subprocess.Popen(...,
user=, group=, extra_groups=)`, which drops privileges between fork and exec.
There is no `su`/`runuser` wrapper in between, so the pid the manager tracks is
the real process. Each server gets its own session (`start_new_session`), which
is what makes a clean process-group kill possible when SIGTERM is not enough.

`unsloth studio run` re-execs into the Studio venv, so the pid can change. Both
are tracked: Unsloth records the serving pid under `STUDIO_HOME`, and the
manager prefers that once it appears.

Stopping sends SIGTERM first — Studio traps it to shut its `llama-server`
children down, which is what avoids orphaned GPU memory — and escalates to a
process-group SIGKILL only after `STOP_TIMEOUT`.

Note that `unsloth studio stop` is blunt: it stops *every* server in a
STUDIO_HOME, including the web UI. Use `stop`/`stop-all` here instead to stop
only what this manager started.

GPU placement uses `CUDA_VISIBLE_DEVICES` (from `--gpus`) rather than
llama.cpp's `--device`, because it constrains the whole process — torch
included — and Unsloth's own GPU selection then only sees the allowed cards.
With no `--gpus`, Unsloth chooses. Note that Unsloth's fitter pins the
*smallest subset* of the pool that fits, so `--gpus 0,1` on a model that fits
one card will legitimately land on one card.

## Gotchas worth knowing

These cost real debugging time; they are documented here so they do not have to
be rediscovered.

- **Do not point `UNSLOTH_MGR_STUDIO_HOME` at a symlink-resolved path.**
  `unsloth studio run` re-execs into the Studio venv and decides whether it
  arrived by comparing `sys.prefix` to `STUDIO_HOME/unsloth_studio`. Unsloth
  resolves the env var but *not* its own default, so if `~/.unsloth` is a
  symlink (to, say, `/srv/unsloth`) the two spellings
  disagree, the child never believes it is in the venv, and it re-execs itself
  forever — same pid, flat memory, one core pinned at 100%, no error message.
  The manager therefore leaves the variable unset unless you override it, and
  `doctor` fails the check if an override is not its own realpath.

- **`/api/health` is liveness only, not readiness.** It answers 200 as soon as
  uvicorn binds, which can be many minutes before the model is loaded; requests
  in that gap come back **404**. The endpoint carries no model state and
  `/v1/models` needs a bearer token, so the manager confirms the load from the
  server's own log, which prints its banner only after loading finishes.

- **Reasoning models stream into `reasoning_content`, not `content`.** Qwen3.6
  MTP and similar send `content: ""` with the real output in
  `reasoning_content`. A client that reads only `content` sees an empty
  response and zero throughput. `test` and `benchmark` count both.

- **A fresh API key is minted per launch**, but old keys keep working — they
  live in `STUDIO_HOME/auth/auth.db`, not in the server.

- **Unsloth's server-side tools are on by default, and they serialise `/v1`.**
  This is the single biggest performance trap here. With tools on, concurrent
  requests queue behind each other no matter how many decode slots
  `--parallel` bought you, and single-request throughput drops too. Measured on
  this box, 3 concurrent 160-token requests:

  | | slots | tools on | tools off |
  |---|---|---|---|
  | Qwen3.8-27B | 3 | 7.1 tok/s, TTFT 2.0/26.7/39.8s | **48.5 tok/s**, TTFT 0.42s x3 |
  | Qwen3.6-35B | 2 | 37.1 tok/s, TTFT 2.2/10.3/11.6s | **110.6 tok/s**, 2 parallel + 1 queued |

  The staggered TTFTs are the tell: with tools on, request *n* gets its first
  token only after request *n-1* finishes. `llama-server` itself is blameless —
  probing its internal port directly gives 48.0 tok/s for the same 3-way test,
  matching the tools-off number exactly. The serialisation is entirely in the
  tool layer above it.

  Single request, Qwen3.8-27B: 9.1 tok/s with tools, 32.7 without, 39.7 straight
  to `llama-server`.

  So pass `--no-tools` unless you specifically want server-side web search and
  code execution. `start` warns when tools are on and `--parallel` is above 1,
  because the slot count is otherwise a promise the server will not keep, and
  the TUI asks outright. A client can opt out per request, but not with
  `enable_tools: false`: a server started with tools on runs `--enable-tools`,
  a policy that outranks the request field and forces the tool loop anyway.
  Only `tool_choice: "none"` turns it off per request. A *streamed* request
  that gets the forced loop without Studio's `X-Unsloth-Events: 1` confirm
  channel is refused outright — HTTP 400 `confirm_tool_calls requires
  stream=true and the X-Unsloth-Events: 1 header` — which is what broke `test`
  and `benchmark` on tools-on servers until they started sending
  `tool_choice: "none"`. Most OpenAI-compatible clients send neither field,
  which is why the server-level flag is the one that matters. Verified: with the server started `--no-tools`, three
  plain concurrent requests that send nothing special get TTFT 1.47s each and
  50.0 tok/s aggregate.

- **Per-model overrides are keyed by quant, not by model.** The key is
  `<repo>:<variant>`, so which quant a launch picks silently decides which
  profile it reads. `settings` lists every quant that has one, and the launch
  plan names the key it used plus any other quant carrying an override.

- **Studio's per-model override save replaces the row; it does not patch it.**
  `PUT /api/settings/openai-auto-switch/overrides` stores exactly the fields it
  is sent (bar extra llama args and the server-tuning fields, which it carries
  over itself), so a client that sends only the field it changed clears the
  rest — a `gpu_ids` pin, a chat template, `disable_vision`. The manager's save
  sends the stored row back with the edit swapped in. Any other client that
  writes overrides has to do the same.

- **A pinned sampling flag is not a default, it is an override.** See
  [Pinning sampling is a hard override](#pinning-sampling-is-a-hard-override).
  This is the single easiest thing to get wrong here, because the flag is
  spelled exactly like a default would be.

- **`-ctk` does not survive the CLI passthrough.** `unsloth studio run` parses
  it as a clustered short option and llama-server receives `-ct`, killing the
  load with *"does not recognise the argument '-ct'"*. Use the long
  `--cache-type-k` / `--cache-type-v` spellings, which the manager now emits.

- **Unsloth's shipped sampling defaults live in the Studio venv**, not in
  `studio.db`: `unsloth_studio/lib/python*/site-packages/studio/backend/assets/configs/`.
  A venv rebuilt or moved elsewhere costs `settings` half its comparison, which
  is why `doctor` checks that path is readable. Override with
  `UNSLOTH_MGR_STUDIO_ASSETS`.

## Connecting clients

The base URL is `http://<host>:<port>/v1` and requests need
`Authorization: Bearer <key>`. Ask a running server what it calls the loaded
model — Unsloth serves a sanitised alias:

```bash
curl -s http://gpubox.example.com:10001/v1/models -H "Authorization: Bearer $KEY" | jq -r '.data[].id'
```

`/v1` is a base URL, not a GET-able path; health checks should hit
`/api/health` (no token) or `/v1/models` (token required).

## Notes for this box

- The GGUF quant is the main VRAM lever. `list --variants` shows each quant's
  size and whether it fits one GPU, needs both, or is too large — the estimate
  budgets 1.25x the weights for KV cache, compute buffers and CUDA context.
  Without `--variant`, the manager picks the best quality that fits.
- `tensorParallel` splits by tensor instead of by layer. It only helps dense
  models that genuinely span both cards; MoE models usually do not gain, and it
  is a no-op on a single GPU.
- **Decode slots do not divide the context here.** Unsloth launches
  `llama-server` with `--kv-unified`, so the slots share one KV pool and each
  reports the *full* context: with `-c 262144 --parallel 3`, `/slots` shows
  `n_ctx=262144` on all three. Raising `--parallel` costs about 1.6% more KV
  per slot (Unsloth's own estimator: 4.65 GiB at 1 slot, 4.94 at 3, for
  Qwen3.8-27B at 262k/q4_0) — not a division. What the slots do share is the
  pool, so three simultaneous *long* contexts still compete for it; three
  ordinary requests do not. Lowering `--parallel` is therefore close to
  useless as a VRAM lever — see the ordering in
  [Gotchas](#gotchas-worth-knowing).
- Companion GGUFs (`mmproj-*` vision projectors, `mtp-*` prediction modules)
  are not servable variants and are filtered out of `list`. Unsloth picks them
  up on its own from the same directory.

## Tests

```bash
python3 -m unittest discover -s tests
```

Stdlib `unittest`. The tests need no server, GPU or network, and take well
under a second. They cover the logic where a quiet regression costs data or
a wrong number, not the plumbing:

- **Override and preset payloads.** A Studio override save *replaces* the
  row, so every field the edit did not touch (`gpu_ids`, the chat template,
  the manual-offload layers) has to be echoed back. A preset write-back has
  to carry through every key the chat UI owns.
- **Save routing.** Which profile each edited value goes to, and the
  override sync a preset save always brings along.
- **Reload drift.** What an idle reload would silently change, the vision
  cases included.
- **Quant naming.** `list` and the header lookups share one rule, and
  companion GGUFs (projectors, MTP and draft modules) are never offered as
  quants.
- **Throughput and token history.** The steady-rate window, and the
  hour→day rollup, pruning and windowed totals behind the token line.

Every setting is pointed at a throwaway directory before the manager is
imported: state, logs and `studio.db`, and `UNSLOTH_MGR_ENV_FILE` is set to
`/dev/null`. A run never reads `local.env`, never touches the real
`tokens.json`, and never opens Studio's database. A test that needs Studio
rows writes them to its own temporary `studio.db`.

## History

unsloth-manager started as a fork of
[vllm-manager](https://github.com/Grassyloki/vllm-manager) and is now far ahead
of it. The curses widgets in `tui_lib.py` are still shared with it.

## License

MIT — see [LICENSE](LICENSE).

Note that this is a separate program: it reads Unsloth Studio's configuration
files and invokes its CLI, but contains none of its code, so Studio's AGPL does
not extend here.
