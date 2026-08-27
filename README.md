# Unsloth Service Manager

A single-host control plane for running multiple [Unsloth Studio](https://unsloth.ai)
inference servers at once: an interactive curses TUI plus a CLI for starting and
stopping servers on dedicated ports as an unprivileged user, applying Unsloth's
own settings to headless launches, and benchmarking throughput.

It is the Unsloth counterpart to the sibling
[vllm-manager](https://github.com/Grassyloki/vllm-manager), but a launch here is
**(model, profile, port)**, and every profile is *Unsloth's own* — read live out
of `studio.db`, out of the Studio venv's shipped tables, or transcribed from
Unsloth's published model guides. This tool invents no settings of its own; what
it does is show you which source won, before the load and after it.

Built for and tested on a **2x NVIDIA Tesla V100-SXM2 32 GB** box running
CachyOS (Arch), serving GGUF models out of a shared Hugging Face cache.

## How this differs from vllm-manager

- **Settings belong to Unsloth, not to this tool.** There is no `profiles.toml`.
  You edit them on the Studio settings page and in the Preset dropdown; the
  manager reads them and never writes to `studio.db`. The one thing it does own
  is a transcription of Unsloth's *published* sampling guides — see
  [Where settings come from](#where-settings-come-from) for why that is
  necessary.
- **Servers run as a dedicated non-root user.** The manager runs as root and
  drops to `unsloth` for every server it starts.
- **Auth is mandatory.** Every Unsloth server requires a bearer token on `/v1`.
  This manager deliberately does *not* create or store keys — see
  [API keys](#api-keys).

## Contents

**Sections**

- [How this differs from vllm-manager](#how-this-differs-from-vllm-manager)
- [Requirements](#requirements)
- [Usage](#usage)
- [Command reference](#command-reference)
- [Where settings come from](#where-settings-come-from)
- [Context, and what "max" means](#context-and-what-max-means)
- [Idle auto-unload, and what a reload restores](#idle-auto-unload-and-what-a-reload-restores)
- [Keeping a server warm](#keeping-a-server-warm)
- [Reading settings back after a load](#reading-settings-back-after-a-load)
- [Instance groups](#instance-groups)
- [Ports and instance limit](#ports-and-instance-limit)
- [API keys](#api-keys)
- [Benchmark](#benchmark)
- [How servers are launched](#how-servers-are-launched)
- [Gotchas worth knowing](#gotchas-worth-knowing)
- [Connecting clients](#connecting-clients)
- [Configuration](#configuration)
- [Notes for this box](#notes-for-this-box)
- [License](#license)

**Files**

| File | Purpose |
|------|---------|
| `unsloth_manager.py` | Main entry point: TUI + CLI to start, stop, inspect, test and benchmark servers. |
| `unsloth_profiles.py` | Every source of "how should this model run": published guides, Unsloth's shipped defaults, Studio presets, per-model overrides. |
| `model_lib.py` | HF-cache scanning, GGUF variant detection, VRAM fit estimation, native context length. |
| `tui_lib.py` | Shared curses widgets (selector, prompts, coloured bars, screen plumbing). |
| `api_tester.sh` | Interactive whiptail client for poking any OpenAI-compatible endpoint. |
| `.env.example` | Template for `local.env`, the git-ignored per-machine settings file. |

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

Check all of it, including that presets, per-model overrides and Unsloth's
shipped default tables are all readable:

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
  ● Qwen3.6-35B-A3B-MTP-GGUF UD-Q4_K_XL :10001 rdy g0   0/2sl
      256k  kv q4_0  mtp  tools ON  no-vis  | T1.0 P0.95 K20 M0.0 pres0.0 rep1.0
  ● Qwen3.8-27B-GGUF         UD-Q4_K_XL :10002 rdy g1   0/3sl  16 tok/s
      256k  kv q4_0  mtp  tools off  no-vis  warm | T1.0 P0.95 K20 M0.0 pres0.0 rep1.0
```

The first line is what a server *is* — identity, placement, and the two facts
that change second to second. `1/3sl` is **sessions: one of three decode slots
busy right now**, and `8 tok/s` its recent decode rate. Everything on the
second line is configuration, which does not move. `status` spells both out
(`sessions 1/3  8.0 tok/s`), as does the `settings` readout.

Nothing above `llama-server` reports either: Unsloth's status carries the slot
*count* but never the occupancy, and no throughput at all. So the manager finds
`llama-server` — it is a direct child of the pid already tracked — and reads
its `/slots` and `/metrics`. Both probes share one 3-second cache and a 0.6s
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
diluted by idle time. Two details matter. llama.cpp updates those counters when
a request *completes*, so the number lags a long stream rather than tracking it
live, and the last real rate is held rather than dropping to zero the moment a
server goes quiet. And a window must carry at least 5 tokens to count, because
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

Start walks model → **quant** → profile → **server-side tools** → **idle behaviour** → port → GPUs. The quant step matters
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
python unsloth_manager.py status                          # running servers, ports, GPU usage
python unsloth_manager.py logs <model> [-n N] [-f]        # tail a server log
python unsloth_manager.py test <model> --api-key KEY      # one streaming prompt + tok/s
python unsloth_manager.py benchmark <model> [--preset NAME ...] --api-key KEY
python unsloth_manager.py restart <model>                 # keeps port, GPUs and profile choices
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
| `restart <model>` | Stop then start, keeping port, GPUs, quant and profile choices. |
| `status` | Running servers, their settings, ports, uptime and GPU usage. |
| `list [--variants]` | Cached models; `--variants` adds each GGUF quant, its size and VRAM fit. |
| `settings <model>` | All four settings sources side by side, plus live state if it is running. |
| `presets` | Studio presets from `studio.db`, read-only. |
| `groups [list\|save\|restore\|show\|clear] [slot]` | Instance groups — see [Instance groups](#instance-groups). |
| `keepalive <model>` | Ping a server so Studio never idle-unloads it — see [Keeping a server warm](#keeping-a-server-warm). |
| `logs <model> [-n N] [-f]` | Tail a server log. |
| `test <model>` | One streaming prompt, with TTFT and tok/s. |
| `benchmark <model>` | Steady-state tok/s per preset — see [Benchmark](#benchmark). |
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
| `--dry-run` | Print the plan and the command line, start nothing. Touches no state. |
| `--force` | Start despite a VRAM warning or an instance already running. |
| `--api-key K` | Key for the post-load readout. Default: read from the server's own log. |

### Other command options

| Flag | Meaning |
|---|---|
| `logs -n N` / `--lines N` | Lines to show (default 40). |
| `logs -f` / `--follow` | Follow the log as it grows. |
| `test --prompt "..."` | Prompt to send (default: a short built-in one). |
| `test --max-tokens N` | Generation cap (default 256). |
| `benchmark --preset NAME` | Repeatable; default is every preset in `studio.db`. |
| `benchmark --max-tokens N` | Generation cap per measured run (default 400). |
| `benchmark --gpus` / `--variant` | Placement and quant, held constant across runs. |
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
| **Start Model** | model → quant → profile → tools → idle behaviour → port → GPUs, then the launch plan. |
| **Instance Groups** | Restore / save / show / clear the ten slots. |
| **Model Settings** | `settings` for one model. |
| **Stop Model** | Stop one running server. |
| **Status** | Full `status` output. |
| **List Models** | Cached models and quants. |
| **Test Model** | One streaming prompt against a running server. |
| **Benchmark** | Measure steady-state tok/s across presets. |
| **API Tester** | Hands off to `api_tester.sh`. |
| **View Logs** | Tail a server log. |
| **Studio Presets** | What the Studio UI has saved. |
| **Environment Check** | `doctor`. |

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
launch plan prints the winning source in brackets next to every value.

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
Unsloth's settings at all, and this manager does not write to `studio.db`.

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
completion — with `enable_tools: false`, because otherwise the ping itself
would serialise `/v1` behind the tool machinery.

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
  `grep 'API Key' /root/.unsloth-logs/<model>.log`
- Keys live in `STUDIO_HOME/auth/auth.db` and are **not** tied to a single
  server. Any key already in that database authenticates against every server
  the manager starts, on any port. One long-lived key is usually what you want.

`test`, `benchmark` and the settings readout need a key, and they now find it
themselves: an explicit `--api-key` wins, then `UNSLOTH_MGR_API_KEY`, then the
key this manager's own log already recorded for that server. Nothing new is
stored anywhere. For `api_tester.sh`, put the key in each endpoint's `auth`
field.

One race worth knowing about: the log line the manager watches for to call a
load "ready" is printed *before* the startup banner carrying the key, so the
post-load readout waits briefly for the key to appear rather than reporting it
missing.

## Benchmark

```bash
python unsloth_manager.py benchmark <model> --preset "Default 1" --preset "MTP+Ngram 256k"
```

With no `--preset` it runs **every** preset in `studio.db`, which is the
question the command exists to answer: which of your presets is actually
fastest on this model. Each run uses `--load-profile preset --sampling none`,
so the preset is genuinely the only thing that differs: a per-model override
would otherwise supply the same load config to every run and flatten the
comparison. For each one it starts the server, waits for readiness,
runs a warm-up plus N measured generations with distinct prompts (so prefix
caching cannot collapse the batch), then hard-stops the server. It reports:

- **Steady tok/s**: best sustained rate over a sliding window (default 5s), so
  one mid-stream stall does not sink the number. `n/a` when the generation was
  too small to distinguish a sustained rate from a burst.
- **Mean tok/s**: overall including stalls, for contrast.
- TTFT, chunk count, stall count, load time, and per-GPU VRAM attributed to the
  server.

Results print as a table and are saved as timestamped JSON in the log dir. The
model must not already be running — benchmark needs to own the load.

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
  the TUI asks outright. A client can also opt out per request with
  `enable_tools: false` (which is what the Studio UI sends), but most
  OpenAI-compatible clients will not — which is why the server-level flag is
  the one that matters. Verified: with the server started `--no-tools`, three
  plain concurrent requests that send nothing special get TTFT 1.47s each and
  50.0 tok/s aggregate.

- **Per-model overrides are keyed by quant, not by model.** The key is
  `<repo>:<variant>`, so which quant a launch picks silently decides which
  profile it reads. `settings` lists every quant that has one, and the launch
  plan names the key it used plus any other quant carrying an override.

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

## Configuration

Every setting is an environment variable, and every one has a default chosen to
work on a fresh machine — so a clone runs without configuration.

For the handful that are per-machine, copy `.env.example` to **`local.env`**
beside the script and edit it. It is read at startup and is git-ignored, so
nothing about your box ends up in the repository:

```bash
cp .env.example local.env
$EDITOR local.env
```

A real environment variable always beats the file, so a systemd unit or a
one-off `UNSLOTH_MGR_LOG_KEEP=10 python unsloth_manager.py ...` still wins.
Only `UNSLOTH_MGR_*` names are read from it — a config file has no business
setting `PATH`. Point `UNSLOTH_MGR_ENV_FILE` elsewhere to use a different file.

The two most likely to need setting are `UNSLOTH_MGR_PUBLIC_HOST` (if clients
reach the box by a name other than its hostname) and `UNSLOTH_MGR_HF_HOME` (if
the weights are not in the service account's `~/.cache/huggingface`).

| Variable | Default | Meaning |
|----------|---------|---------|
| `UNSLOTH_MGR_USER` | `unsloth` | Account the servers run as. |
| `UNSLOTH_MGR_STUDIO_HOME` | `~unsloth/.unsloth/studio` | Unsloth data root. Leave unset unless it is genuinely elsewhere — see the gotcha above. |
| `UNSLOTH_MGR_STUDIO_DB` | `<STUDIO_HOME>/studio.db` | Where presets and per-model overrides are read from. |
| `UNSLOTH_MGR_STUDIO_ASSETS` | globbed under `<STUDIO_HOME>` | Unsloth's `assets/configs` (its shipped sampling tables). |
| `UNSLOTH_MGR_HF_HOME` | `~<user>/.cache/huggingface` | Model cache; passed to the child as `HF_HOME`. |
| `UNSLOTH_MGR_STATE_DIR` | `/root/.unsloth-pids` | State file directory. |
| `UNSLOTH_MGR_LOG_DIR` | `/root/.unsloth-logs` | Per-model server logs + benchmark JSON. |
| `UNSLOTH_MGR_BASE_PORT` | `10001` | First port in the API pool. |
| `UNSLOTH_MGR_MAX_INSTANCES` | `8` | Pool size, and therefore the concurrent-server cap. |
| `UNSLOTH_MGR_GROUP_SLOTS` | `10` | Instance-group slots. |
| `UNSLOTH_MGR_BIND_HOST` | `0.0.0.0` | Server bind address. |
| `UNSLOTH_MGR_PUBLIC_HOST` | this machine's hostname | Hostname printed in URLs. Only affects what is printed. |
| `UNSLOTH_MGR_PUBLIC_SCHEME` | `http` | Scheme for those URLs. |
| `UNSLOTH_MGR_API_KEY` | unset | Key for `test` / `benchmark` / readouts. Default: read from the server's own log. |
| `UNSLOTH_MGR_BIN` | `unsloth` | Unsloth CLI to invoke. |
| `UNSLOTH_MGR_PER_GPU_VRAM_GB` | auto-detected | Per-GPU VRAM used for quant sizing. |
| `UNSLOTH_MGR_STOP_TIMEOUT` | `30` | Seconds to wait for SIGTERM before SIGKILL. |
| `UNSLOTH_MGR_LOAD_TIMEOUT` | `900` | Default readiness wait for `start`. |
| `UNSLOTH_MGR_LOG_KEEP` | `3` | Rotated log generations to keep. |
| `UNSLOTH_MGR_ENV_FILE` | `local.env` beside the script | Where the per-machine settings above are read from. |

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

## License

MIT — see [LICENSE](LICENSE).

Note that this is a separate program: it reads Unsloth Studio's configuration
files and invokes its CLI, but contains none of its code, so Studio's AGPL does
not extend here.
