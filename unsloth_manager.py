#!/usr/bin/env python3
"""
unsloth_manager.py — control plane for Unsloth Studio inference servers.

Runs as root, serves as the `unsloth` user. Each launch is a
(model, profile, port) triple: `unsloth studio run` is started, dropped to the
unsloth user, and tracked in a state file so several models can be served at
once on their own ports.

Nothing here invents settings. Four sources decide how a model runs, and
`settings <model>` prints all four side by side before you commit to one:

  * Unsloth's PUBLISHED profile (unsloth.ai/docs) — sampling, transcribed into
    unsloth_profiles.py. The default, because Unsloth's shipped family table
    gives Qwen3.6/3.8 the Instruct row while both models run in thinking mode.
  * Unsloth's SHIPPED family defaults — what the server applies on its own to
    any field a request omits, if this manager pins nothing.
  * Unsloth's PER-MODEL override — the load config (context, KV dtype,
    speculative mode, slots, GPU pin, vision) the Studio settings page saves
    per model. Unsloth applies it only when /v1 auto-switches models, so a
    headless launch has to read it itself.
  * Studio PRESETS — the chat UI's Preset dropdown. Global; no model or port.

    python unsloth_manager.py                       # interactive TUI
    python unsloth_manager.py settings <model>      # all four sources
    python unsloth_manager.py start <model> --port 7101
    python unsloth_manager.py status

Unsloth Studio always requires a bearer token on /v1. This manager does not
create or store keys: `unsloth studio run` mints one per launch and prints it
into the server log, which is where `test`, `benchmark` and the post-load
readout pick it up. Override with --api-key or UNSLOTH_MGR_API_KEY.
"""
from __future__ import annotations

import argparse
import grp
import json
import os
import pwd
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import NoReturn

import model_lib as ml
import tui_lib as tui
import unsloth_profiles as up

# =============================================================================
# CONFIG
# =============================================================================

# Settings come from the environment. A checkout is meant to be usable as-is,
# so nothing host-specific is baked in here: an optional `local.env` beside
# this script supplies the values for one machine without being committed.
#
# Precedence is the usual one -- a real environment variable beats the file --
# so a systemd unit or a one-off `UNSLOTH_MGR_... = ...` on the command line
# still wins. Loaded into os.environ before anything reads it, which is what
# lets every os.environ.get in this project (and in unsloth_profiles) see it
# without knowing the file exists.
ENV_FILE = os.environ.get("UNSLOTH_MGR_ENV_FILE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "local.env")


def _load_env_file(path: str) -> None:
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, val = line.partition("=")
        key = key.strip()
        # Only our own namespace: a config file has no business setting PATH.
        if not sep or not key.startswith("UNSLOTH_MGR_"):
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ.setdefault(key, val)


_load_env_file(ENV_FILE)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# The account the servers run as. Never root: llama-server loads untrusted
# GGUFs and Studio exposes a network service.
RUN_USER = _env("UNSLOTH_MGR_USER", "unsloth")

# Unsloth's own data root (auth.db, studio.db, logs, per-port pid files).
#
# Left unset by default ON PURPOSE, and the default below matches the exact
# path spelling Unsloth derives for itself (~/.unsloth/studio, unresolved).
#
# `unsloth studio run` re-execs into the Studio venv and then decides whether
# it has arrived by comparing sys.prefix against STUDIO_HOME/unsloth_studio.
# _resolve_studio_home() calls .resolve() on UNSLOTH_STUDIO_HOME but NOT on the
# default, so exporting a symlink-resolved path (e.g. /srv/unsloth/studio
# for a ~/.unsloth symlink) makes those two spellings disagree: the child never
# believes it is in the venv, re-execs again, and spins forever at 100% CPU on
# the same pid. Only export the variable when it is a deliberate override.
_STUDIO_HOME_OVERRIDE = os.environ.get("UNSLOTH_MGR_STUDIO_HOME", "").strip()


def _default_studio_home() -> str:
    try:
        return os.path.join(pwd.getpwnam(RUN_USER).pw_dir, ".unsloth", "studio")
    except KeyError:
        return "/home/unsloth/.unsloth/studio"


STUDIO_HOME = _STUDIO_HOME_OVERRIDE or _default_studio_home()

def _default_hf_home() -> str:
    """The service account's Hugging Face cache.

    The servers run as RUN_USER, so its cache is the one that matters -- not
    root's, and not this process's HF_HOME.
    """
    try:
        home = pwd.getpwnam(RUN_USER).pw_dir
    except KeyError:
        home = os.path.expanduser("~")
    return os.path.join(home, ".cache", "huggingface")


# Where the weights are. Passed to the child as HF_HOME.
HF_HOME = _env("UNSLOTH_MGR_HF_HOME", "") or _default_hf_home()

def _default_studio_db() -> str:
    return os.path.join(STUDIO_HOME, "studio.db")


# Studio's own database, where the UI saves its presets. Read-only here.
STUDIO_DB = _env("UNSLOTH_MGR_STUDIO_DB", "") or _default_studio_db()

STATE_DIR = _env("UNSLOTH_MGR_STATE_DIR", "/root/.unsloth-pids")
STATE_FILE = os.path.join(STATE_DIR, "state.json")
LOG_DIR = _env("UNSLOTH_MGR_LOG_DIR", "/root/.unsloth-logs")

# The API port pool. Every managed server binds one of these and nothing else,
# so the set of ports a client may ever need to know is fixed and small: with
# the defaults, /v1 lives on 10001-10008 and nowhere else.
#
# The pool size IS the instance cap -- one server, one port -- so there is no
# second limit that could disagree with it.
BASE_PORT = int(_env("UNSLOTH_MGR_BASE_PORT", "10001"))
MAX_INSTANCES = int(_env("UNSLOTH_MGR_MAX_INSTANCES", "8"))
if MAX_INSTANCES < 1:
    raise SystemExit("UNSLOTH_MGR_MAX_INSTANCES must be at least 1")
PORT_POOL = range(BASE_PORT, BASE_PORT + MAX_INSTANCES)
PORT_POOL_LABEL = (f"{PORT_POOL.start}-{PORT_POOL[-1]}" if MAX_INSTANCES > 1
                   else str(PORT_POOL.start))
BIND_HOST = _env("UNSLOTH_MGR_BIND_HOST", "0.0.0.0")
# Only ever used to print URLs, so the machine's own name is a safe default;
# set it when clients reach this box by some other name.
PUBLIC_HOST = _env("UNSLOTH_MGR_PUBLIC_HOST", "") or socket.gethostname()
PUBLIC_SCHEME = _env("UNSLOTH_MGR_PUBLIC_SCHEME", "http")

UNSLOTH_BIN = _env("UNSLOTH_MGR_BIN", "unsloth")

# Key used only by `test` / `benchmark` to talk to a running server.
API_KEY = _env("UNSLOTH_MGR_API_KEY", "")

STOP_TIMEOUT_SEC = int(_env("UNSLOTH_MGR_STOP_TIMEOUT", "30"))
LOAD_TIMEOUT_SEC = int(_env("UNSLOTH_MGR_LOAD_TIMEOUT", "900"))
LOG_KEEP = int(_env("UNSLOTH_MGR_LOG_KEEP", "3"))

_PER_GPU_VRAM_ENV = os.environ.get("UNSLOTH_MGR_PER_GPU_VRAM_GB")

TEST_PROMPT = "Briefly explain what a buffer overflow is."

# Long-form on purpose. A short answer finishes in a fraction of a second on a
# fast model, which leaves too small a sample to separate a sustained rate from
# a burst — and `ignore_eos` is not honoured here, so the only way to get a
# long generation is to ask for one.
BENCH_PROMPT = (
    "Write a detailed technical explanation of how TCP congestion control "
    "works, covering slow start, congestion avoidance, fast retransmit and "
    "fast recovery. Aim for at least 500 words."
)

BENCH_WARMUP = 1
BENCH_RUNS = 3
BENCH_MAX_TOKENS = 400
BENCH_WINDOW = 5.0        # sliding window (s) for the steady-state tok/s metric
BENCH_FREEZE_TIMEOUT = 90  # no-token gap that aborts a run

# Below these, a generation is too small to have a steady state at all. Both
# must fail it for the sample to be junk: 400 chunks over 0.4s is a real rate,
# while 13 chunks over 20ms is transport batching. Guarding on sample count as
# well as span keeps a fast small model from being reported as unmeasurable.
STEADY_MIN_TOKENS = 16
STEADY_MIN_SPAN = 0.25    # seconds between first and last chunk

_C_GREEN, _C_YELLOW = tui.C_GREEN, tui.C_YELLOW
_C_DIM, _C_CYAN = tui.C_DIM, tui.C_CYAN


# =============================================================================
# SMALL HELPERS
# =============================================================================

def _die(msg: str, code: int = 1) -> NoReturn:
    print(f"  Error: {msg}")
    sys.exit(code)


def _ensure_dirs():
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def _safe_name(repo_id: str) -> str:
    """Filesystem-safe form of a repo id, for log file names."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", repo_id)


def _log_path(repo_id: str) -> str:
    return os.path.join(LOG_DIR, f"{_safe_name(repo_id)}.log")


def _rotate_log(path: str):
    """Keep the last LOG_KEEP launches so a crash loop stays diagnosable."""
    if not os.path.exists(path):
        return
    for i in range(LOG_KEEP - 1, 0, -1):
        older, newer = f"{path}.{i}", f"{path}.{i - 1}" if i > 1 else path
        if os.path.exists(newer):
            try:
                os.replace(newer, older)
            except OSError:
                pass


def _has_command(name: str) -> bool:
    return shutil.which(name) is not None


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def _pid_alive(pid: int) -> bool:
    """True if the process exists and has not exited.

    A zombie answers kill(0) successfully, so the signal probe alone reports a
    server we just killed as still running whenever the manager is its parent
    (start and stop in one process, e.g. benchmark). Check the state field too.
    """
    if pid < 2:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
        # The comm field is parenthesised and may itself contain spaces, so
        # state is the first field after the final ')'.
        state = data[data.rfind(")") + 1:].split()[0]
        return state != "Z"
    except (OSError, IndexError):
        return True


def _reap_children():
    """Clear zombies left by servers this process started and then killed."""
    try:
        while os.waitpid(-1, os.WNOHANG)[0]:
            pass
    except ChildProcessError:
        pass
    except OSError:
        pass


def _profile_label(entry: dict) -> str:
    """How a running server was configured, for a one-line display.

    Reads what `start` recorded rather than re-deriving it: the profile that
    was in force at launch is not necessarily what studio.db holds now.
    """
    load = entry.get("load_source") or ""
    # The stored source is a full studio.db key ("<repo>:<variant>"); the repo
    # is already the line's subject, so only the quant carries information.
    if load.startswith("override "):
        key = load[len("override "):]
        load = f"override :{key.rsplit(':', 1)[-1]}" if ":" in key else "override"
    if not load or load == "none":
        load = "unsloth defaults"
    sampling = entry.get("sampling") or "docs"
    mode = entry.get("mode") or ""
    if sampling == "docs" and mode:
        sampling = f"docs/{mode}"
    return f"{load} + sampling {sampling}"


def _fmt_uptime(started: float) -> str:
    secs = max(0, int(time.time() - started))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# =============================================================================
# STATE
# =============================================================================
# One JSON file, keyed by repo id. Holds what Unsloth's own per-port pid files
# cannot: which profile and which GPUs a server was launched with.

def _load_state() -> dict:
    if not os.path.isfile(STATE_FILE):
        return {"models": {}}
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"models": {}}
    state.setdefault("models", {})
    return state


def _save_state(state: dict):
    _ensure_dirs()
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def _prune_dead(state: dict) -> tuple[dict, list[str]]:
    """Drop entries whose process is gone, so `status` never lies."""
    dead = [name for name, e in state["models"].items()
            if not _pid_alive(int(e.get("pid", 0)))]
    for name in dead:
        state["models"].pop(name, None)
    if dead:
        _save_state(state)
    return state, dead


def _get_running() -> dict:
    state, _ = _prune_dead(_load_state())
    return state["models"]


def _next_port(state: dict) -> int:
    """The lowest free port in the pool.

    Never walks past the pool: the pool is the whole contract with clients, and
    a server quietly bound outside it would be one nothing knows to look for.
    """
    taken = {int(e["port"]) for e in state["models"].values()}
    for port in PORT_POOL:
        if port not in taken and _port_free(port):
            return port
    # Distinguish "the manager is full" from "something else took the ports",
    # because the fix is completely different.
    blocked = [p for p in PORT_POOL if p not in taken and not _port_free(p)]
    if blocked:
        _die(f"no free port in {PORT_POOL_LABEL}: "
             f"{len(taken)} in use by this manager, and "
             f"{', '.join(str(p) for p in blocked)} "
             f"{'is' if len(blocked) == 1 else 'are'} held by another process.")
    _die(f"all {MAX_INSTANCES} instance slots are in use "
         f"(ports {PORT_POOL_LABEL}). Stop a server first, or raise "
         f"UNSLOTH_MGR_MAX_INSTANCES.")


# =============================================================================
# INSTANCE GROUPS
# =============================================================================
# A named snapshot of what is running and exactly how it was launched, so a
# box can be brought back to a known state after a reboot with one command.
#
# What is saved is the ANSWERS, not the resolved settings: the model, quant,
# port, GPUs, profile choices, tools, and every explicit override. Restoring
# replays them through the same `start` path, so a group picks up whatever
# studio.db says today rather than freezing a command line that Unsloth's own
# idle-reload would contradict five minutes later.

GROUP_SLOTS = int(_env("UNSLOTH_MGR_GROUP_SLOTS", "10"))
GROUPS_FILE = os.path.join(STATE_DIR, "groups.json")

# Every input `cmd_start` reads, with the value that means "not given". A group
# member is exactly this dict, so adding a launch option here is all it takes
# for groups to carry it.
LAUNCH_DEFAULTS: dict = {
    "model": "", "variant": "", "port": None, "gpus": "",
    "preset": None, "load_profile": "auto", "sampling": "docs",
    "mode": up.DEFAULT_DOC_MODE, "tools": None,
    "ctx": None, "parallel": None, "tensor_parallel": None,
    "kv_cache_dtype": None, "spec": None, "spec_draft_n_max": None,
    "vision": None, "n_batch": None, "n_ubatch": None, "extra": None,
    **{k: None for k in up.SAMPLING_FIELDS},
}


def _launch_args(args) -> dict:
    """The answers this launch was given, normalised for storage."""
    out = {}
    for key, default in LAUNCH_DEFAULTS.items():
        val = getattr(args, key, default)
        out[key] = default if val is None and default is not None else val
    return out


def _launch_namespace(member: dict) -> argparse.Namespace:
    """Rebuild a `start` namespace from a stored member.

    Unknown keys are dropped and missing ones defaulted, so a group written by
    an older version still restores instead of raising AttributeError deep in
    the resolver.
    """
    ns = dict(LAUNCH_DEFAULTS)
    ns.update({k: v for k, v in member.items() if k in LAUNCH_DEFAULTS})
    # Not saved: these are per-invocation, not part of the group's identity.
    ns.update(api_key="", dry_run=False, force=False, wait=None)
    return argparse.Namespace(**ns)


def _load_groups() -> dict:
    try:
        with open(GROUPS_FILE) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    slots = data.get("slots")
    return slots if isinstance(slots, dict) else {}


def _save_groups(slots: dict):
    _ensure_dirs()
    tmp = GROUPS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"slots": slots}, f, indent=2)
    os.replace(tmp, GROUPS_FILE)


def _check_slot(slot: int) -> int:
    if not 1 <= slot <= GROUP_SLOTS:
        _die(f"slot {slot} is out of range (1-{GROUP_SLOTS})")
    return slot


def _group_cli(action: str, slot: int) -> str:
    return (f"python {os.path.abspath(__file__)} groups {action} {slot}")


def _fmt_member(mem: dict) -> str:
    bits = [f"port {mem.get('port') or 'auto'}"]
    if mem.get("variant"):
        bits.append(mem["variant"])
    if mem.get("gpus"):
        bits.append(f"gpu {mem['gpus']}")
    bits.append(f"{mem.get('load_profile', 'auto')}/{mem.get('sampling', 'docs')}")
    tools = mem.get("tools")
    bits.append("tools off" if tools is False
                else ("tools on" if tools else "tools default"))
    return ", ".join(bits)


# =============================================================================
# UNSLOTH PID FILES
# =============================================================================
# `unsloth studio run` re-execs into the Studio venv, so the pid we spawn is not
# always the pid that ends up serving. Unsloth records the real one in
# STUDIO_HOME/studio-<port>.pid; prefer that when it shows up.

def _studio_pid_file(port: int) -> str:
    return os.path.join(STUDIO_HOME, f"studio-{port}.pid")


def _read_studio_pid(port: int) -> int:
    path = _studio_pid_file(port)
    try:
        with open(path) as f:
            first = f.readline().strip()
    except OSError:
        return 0
    return int(first) if first.isdigit() else 0


# =============================================================================
# GPU
# =============================================================================

_gpu_cache: list[dict] = []


def _query_gpus(timeout: float = 6.0) -> list[dict]:
    if not _has_command("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout).stdout
    except (subprocess.SubprocessError, OSError):
        return []

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "total_mb": int(parts[2]),
                "used_mb": int(parts[3]),
            })
        except ValueError:
            continue
    return gpus


def _gpus(refresh: bool = False) -> list[dict]:
    global _gpu_cache
    if refresh or not _gpu_cache:
        _gpu_cache = _query_gpus()
    return _gpu_cache


def _per_gpu_vram_gb() -> int:
    if _PER_GPU_VRAM_ENV:
        return int(_PER_GPU_VRAM_ENV)
    gpus = _gpus()
    if not gpus:
        return 32
    return max(1, min(g["total_mb"] for g in gpus) // 1024)


def _n_gpus() -> int:
    return len(_gpus()) or 1


def _managed_pids() -> set[int]:
    """Every pid this manager owns, plus their process-group children."""
    pids = set()
    for e in _get_running().values():
        pid = int(e.get("pid", 0))
        if pid:
            pids.add(pid)
    return pids


def _gpu_usage_split(timeout: float = 6.0) -> list[dict]:
    """Per-GPU memory split into 'ours' vs 'other', for the TUI bars.

    nvidia-smi reports compute-apps per GPU; anything whose pid sits in a
    process group we launched counts as ours.
    """
    gpus = _gpus(refresh=True)
    if not gpus:
        return []

    ours = _managed_pids()
    our_pgids = set()
    for pid in ours:
        try:
            our_pgids.add(os.getpgid(pid))
        except (ProcessLookupError, PermissionError):
            pass

    per_gpu_ours = {g["index"]: 0 for g in gpus}
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout).stdout
        uuid_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout).stdout
    except (subprocess.SubprocessError, OSError):
        out = uuid_out = ""

    uuid_to_index = {}
    for line in uuid_out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                uuid_to_index[parts[1]] = int(parts[0])
            except ValueError:
                pass

    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx = uuid_to_index.get(parts[0])
        if idx is None:
            continue
        try:
            pid, mb = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        mine = pid in ours
        if not mine:
            try:
                mine = os.getpgid(pid) in our_pgids
            except (ProcessLookupError, PermissionError):
                mine = False
        if mine:
            per_gpu_ours[idx] = per_gpu_ours.get(idx, 0) + mb

    split = []
    for g in gpus:
        mine = min(per_gpu_ours.get(g["index"], 0), g["used_mb"])
        split.append({
            "index": g["index"],
            "name": g["name"],
            "total_mb": g["total_mb"],
            "used_mb": g["used_mb"],
            "ours_mb": mine,
            "other_mb": max(0, g["used_mb"] - mine),
        })
    return split


def _read_mem_gb() -> tuple[int, int]:
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                k, _, rest = line.partition(":")
                info[k] = int(rest.split()[0])
        return info["MemTotal"] // 1048576, info["MemAvailable"] // 1048576
    except (OSError, KeyError, ValueError, IndexError):
        return 0, 0


# -- bars --------------------------------------------------------------------

def _stacked_bar(segvals, total, width=20):
    """[(chars, attr_key), ...] for a two-tone usage bar."""
    if total <= 0:
        return [("░" * width, _C_DIM)]
    out, used = [], 0
    for val, attr in segvals:
        n = int(round(width * max(0, val) / total))
        n = min(n, width - used)
        if n > 0:
            out.append(("█" * n, attr))
            used += n
    if used < width:
        out.append(("░" * (width - used), _C_DIM))
    return out


def _gpu_bar_line(g, width=20):
    import curses
    segs = _stacked_bar(
        [(g["ours_mb"], _C_GREEN), (g["other_mb"], _C_YELLOW)],
        g["total_mb"], width)
    line = [("  ", 0),
            (f"GPU{g['index']} ", curses.A_BOLD),
            ("[", curses.color_pair(_C_DIM))]
    for text, attr in segs:
        line.append((text, curses.color_pair(attr)))
    line.append(("] ", curses.color_pair(_C_DIM)))
    line.append((f"{g['used_mb'] // 1024}/{g['total_mb'] // 1024}G ",
                 curses.color_pair(_C_DIM)))
    line.append((g["name"][:22], curses.color_pair(_C_DIM)))
    return line


def _ram_bar_line(used_gb, total_gb, width=20):
    import curses
    segs = _stacked_bar([(used_gb, _C_CYAN)], total_gb, width)
    line = [("  ", 0), ("RAM  ", curses.A_BOLD),
            ("[", curses.color_pair(_C_DIM))]
    for text, attr in segs:
        line.append((text, curses.color_pair(attr)))
    line.append(("] ", curses.color_pair(_C_DIM)))
    line.append((f"{used_gb}/{total_gb}G", curses.color_pair(_C_DIM)))
    return line


# =============================================================================
# PRIVILEGE DROP + LAUNCH
# =============================================================================

def _run_user_ids() -> tuple[int, int, list[int], str]:
    """(uid, gid, supplementary gids, home) for RUN_USER."""
    try:
        pw = pwd.getpwnam(RUN_USER)
    except KeyError:
        _die(f"user {RUN_USER!r} does not exist on this host")
    extra = [g.gr_gid for g in grp.getgrall() if pw.pw_name in g.gr_mem]
    if pw.pw_gid not in extra:
        extra.append(pw.pw_gid)
    return pw.pw_uid, pw.pw_gid, sorted(set(extra)), pw.pw_dir


@dataclass
class Launch:
    """One resolved launch: every source merged, with a record of who won.

    `sources` maps each decided field to the source that supplied it
    (doc/unsloth/override/preset/flag), which is what lets the pre-launch
    readout show why a value is what it is instead of just asserting it.
    """
    model: ml.ModelInfo
    preset: up.Preset | None = None
    override: up.ModelOverride = field(default_factory=up.ModelOverride)

    port: int = 0
    gpus: str = ""                       # "" = let Unsloth choose
    variant: str = ""

    # -- load config ---------------------------------------------------------
    # ctx 0 means "send no --max-seq-length", which is how you ask Unsloth for
    # the largest context that fits: it starts at the GGUF's native length and
    # caps to VRAM with its own estimator. A non-zero value is honoured
    # verbatim, and llama.cpp's --fit then offloads layers to CPU rather than
    # shrink it -- a full window at a possible cost in speed.
    ctx: int = 0
    parallel: int | None = None
    tensor_parallel: bool | None = None
    kv_cache_dtype: str | None = None
    spec_mode: str | None = None
    spec_draft_n_max: int | None = None
    vision: bool | None = None           # False -> --no-mmproj
    n_batch: int | None = None
    n_ubatch: int | None = None

    # -- sampling ------------------------------------------------------------
    sampling: dict = field(default_factory=dict)
    sampling_source: str = "docs"        # docs | unsloth | preset | none
    doc: up.DocProfile | None = None

    load_source: str = "none"            # override | preset | none
    tools: bool | None = None
    api_only: bool = True
    extra: list[str] = field(default_factory=list)
    sources: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def preset_name(self) -> str:
        return self.preset.name if self.preset else ""

    @property
    def native_ctx(self) -> int | None:
        return ml.native_context(self.model, self.variant)


def _child_env(gpus: str, extra_env: dict[str, str], home: str) -> dict[str, str]:
    """A clean environment for the server, not root's inherited one."""
    env = {
        "HOME": home,
        "USER": RUN_USER,
        "LOGNAME": RUN_USER,
        "SHELL": "/usr/bin/bash",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "HF_HOME": HF_HOME,
        # Our stdout is a log file, not a tty, so CPython would block-buffer
        # the child's output and `logs` would show nothing for minutes during
        # the exact load the user is trying to watch.
        "PYTHONUNBUFFERED": "1",
    }
    # Pin the visible devices rather than passing llama.cpp device flags: it
    # constrains the whole process (torch included), not just the llama-server
    # hop, and Unsloth's auto GPU selection then sees only the allowed cards.
    # Left unset when no --gpus was given, so Unsloth picks for itself.
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    # See the STUDIO_HOME comment: only a deliberate override is exported.
    if _STUDIO_HOME_OVERRIDE:
        env["UNSLOTH_STUDIO_HOME"] = _STUDIO_HOME_OVERRIDE
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "NO_PROXY", "no_proxy"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    env.update(extra_env)
    return env


def _build_cmd(lb: Launch) -> list[str]:
    """Turn a resolved Launch into an `unsloth studio run` command line.

    Two kinds of argument come out of here. Most are first-class `studio run`
    flags. The rest ride the documented pass-through: unknown flags reach
    llama-server and last-wins-override Unsloth's own value, which is the only
    way to reach knobs with no CLI flag of their own (the KV cache dtype, the
    vision opt-out, batch sizes).
    """
    m = lb.model
    label = f"mgr-{_safe_name(m.repo_id)}"
    if lb.preset_name:
        label += f"-{_safe_name(lb.preset_name)}"

    argv = [
        UNSLOTH_BIN, "studio", "run",
        "--model", m.repo_id,
        "--host", BIND_HOST,
        "--port", str(lb.port),
        # Never open a public tunnel from a managed launch. --no-secure keeps
        # the raw bind that this manager's clients connect to.
        "--no-cloudflare",
        "--no-secure",
        "--api-key-name", label[:60],
    ]

    if lb.api_only:
        argv.append("--api-only")
    if lb.tools is not None:
        argv.append("--enable-tools" if lb.tools else "--disable-tools")
    if lb.parallel is not None:
        argv += ["--parallel", str(lb.parallel)]
    # Omitted entirely for fit-max: --max-seq-length 0 is the CLI default and
    # means "native length, capped to VRAM by Unsloth's own fitter".
    if lb.ctx:
        argv += ["--max-seq-length", str(lb.ctx)]

    if m.is_gguf:
        if lb.variant:
            argv += ["--gguf-variant", lb.variant]
        if lb.tensor_parallel is not None:
            argv.append("--tensor-parallel" if lb.tensor_parallel
                        else "--no-tensor-parallel")
        # Unsloth's OWN vocabulary, translated by Unsloth. Emitting
        # llama-server's --spec-type by hand would be worse than useless:
        # supplying it at all suppresses Unsloth's capability-probed auto-emit,
        # so a token this build does not know disables speculative decoding
        # instead of enabling it.
        if lb.spec_mode and lb.spec_mode != "auto":
            argv += ["--speculative-type", lb.spec_mode]
        # Only the drafter modes read it; elsewhere it is silently ignored.
        if lb.spec_draft_n_max and lb.spec_mode in up.DRAFT_N_MAX_MODES:
            argv += ["--spec-draft-n-max", str(lb.spec_draft_n_max)]

    for key in up.SAMPLING_FIELDS:
        val = lb.sampling.get(key)
        if val is not None:
            flag = "--" + key.replace("_", "-")
            argv += [flag, str(int(val) if key in up.INT_FIELDS else val)]

    # -- llama-server pass-through -------------------------------------------
    if m.is_gguf:
        if lb.kv_cache_dtype:
            # Long form, NOT -ctk/-ctv: the CLI's unknown-flag passthrough
            # parses a clustered short option, so "-ctk" reaches llama-server
            # as "-ct" and the load dies with "does not recognise '-ct'".
            argv += ["--cache-type-k", lb.kv_cache_dtype,
                     "--cache-type-v", lb.kv_cache_dtype]
        if lb.n_batch:
            argv += ["--batch-size", str(lb.n_batch)]
        if lb.n_ubatch:
            argv += ["--ubatch-size", str(lb.n_ubatch)]
        # Unsloth attaches a companion mmproj on its own; --no-mmproj is how
        # its own settings page spells "disable vision", and it recognises the
        # flag in pass-through args. Worth real VRAM on these models: the
        # projectors here are ~0.85 GB that a text-only server never touches.
        if lb.vision is False:
            argv.append("--no-mmproj")

    argv += lb.extra
    return argv


def _spawn(argv: list[str], env: dict[str, str], log_path: str,
           home: str) -> subprocess.Popen:
    """Start the server as RUN_USER in its own session.

    Popen's user=/group= drop privileges between fork and exec, so the pid we
    get back is the real process (no runuser/su wrapper in the way).
    start_new_session gives it its own process group, which is what makes a
    clean group-kill possible when SIGTERM is not enough.
    """
    uid, gid, extra, _ = _run_user_ids()
    logf = open(log_path, "a", buffering=1)
    logf.write(f"\n=== launch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    logf.write(f"$ {' '.join(argv)}\n\n")
    logf.flush()
    try:
        return subprocess.Popen(
            argv,
            cwd=home,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            user=uid,
            group=gid,
            extra_groups=extra,
        )
    finally:
        logf.close()


# =============================================================================
# HTTP
# =============================================================================

def _http(url: str, *, method: str = "GET", data: bytes | None = None,
          api_key: str = "", timeout: float = 10.0):
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    return urllib.request.urlopen(req, timeout=timeout)


_READY_CACHE: dict[int, tuple[float, bool]] = {}
_READY_TTL = 4.0

# /api/health answers 200 as soon as uvicorn binds, which is seconds (for a
# small GGUF) to many minutes (for a large one) BEFORE the model is loaded —
# requests sent in that gap come back 404. The endpoint carries no model state,
# and /v1/models needs a bearer token this manager deliberately does not hold,
# so the load is confirmed from the server's own log instead. Studio prints its
# banner only after the load completes.
_LOADED_MARKERS = (
    "Loaded GGUF model via llama-server",
    '"/api/inference/load", "status_code": 200',
    "API Key:",
)


def _model_loaded(log_path: str) -> bool:
    """True once the server log shows the model finished loading.

    The log is rotated per launch, so what it contains belongs to this run.
    """
    if not log_path:
        return True          # caller has no log to check; liveness is all we have
    try:
        with open(log_path, errors="replace") as f:
            text = f.read()
    except OSError:
        return False
    return any(marker in text for marker in _LOADED_MARKERS)


def _probe_ready(port: int, timeout: float = 0.6, log_path: str = "") -> bool:
    """Serving and model-loaded, not merely listening."""
    now = time.time()
    hit = _READY_CACHE.get(port)
    if hit and now - hit[0] < _READY_TTL:
        return hit[1]
    ok = False
    try:
        with _http(f"http://127.0.0.1:{port}/api/health", timeout=timeout) as r:
            ok = r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        ok = False
    if ok:
        ok = _model_loaded(log_path)
    _READY_CACHE[port] = (now, ok)
    return ok


def _endpoint_url(port: int) -> str:
    return f"{PUBLIC_SCHEME}://{PUBLIC_HOST}:{port}/v1"


# Unsloth mints a key at every launch and prints it into the banner this
# manager captures. The key is not a secret this tool created or stores -- it
# is already sitting in a root-owned log file -- so reading it back is what
# makes a settings readout work without asking for one.
_LOG_KEY_RE = re.compile(r"API Key:\s*(sk-[A-Za-z0-9._-]+)")


def _log_api_key(log_path: str) -> str:
    """The key the CURRENT run printed, or "".

    Last match wins: the log is rotated per launch, but a run that reloaded
    prints more than once and only the newest key is certain to be live.
    """
    if not log_path or not os.path.isfile(log_path):
        return ""
    try:
        with open(log_path, errors="replace") as f:
            hits = _LOG_KEY_RE.findall(f.read())
    except OSError:
        return ""
    return hits[-1] if hits else ""


def _resolve_api_key(explicit: str = "", log_path: str = "") -> str:
    """An explicit key, then the environment, then the server's own log."""
    key = explicit or API_KEY or _log_api_key(log_path)
    if not key:
        _die("no API key. Unsloth requires a bearer token on /v1.\n"
             "         Pass --api-key, or set UNSLOTH_MGR_API_KEY.\n"
             "         The key a server printed at startup is in its log:\n"
             f"         grep 'API Key' {LOG_DIR}/<model>.log")
    return key


def _inference_status(port: int, api_key: str, timeout: float = 10.0) -> dict:
    """GET /api/inference/status — what the server says it actually loaded.

    The richest readout available: the effective, max and native context, the
    KV dtype, the speculative mode plus any fallback reason, the slot count,
    and the per-model sampling block the server will apply to a request that
    omits a field. Returns {} rather than raising, because a readout failing
    must never look like the load failing.
    """
    if not api_key:
        return {}
    try:
        with _http(f"http://127.0.0.1:{port}/api/inference/status",
                   api_key=api_key, timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _print_effective(port: int, log_path: str, api_key: str = "",
                     wait_key: float = 0.0) -> bool:
    """Print what the server reports it is actually running. False if it could
    not be asked (no key, or the endpoint did not answer).

    `wait_key` exists because readiness and the key race: the log line this
    manager watches for ("Loaded GGUF model via llama-server") is printed
    BEFORE the startup banner carrying the key, so a readout taken the instant
    a load completes finds no key at all.
    """
    key = api_key or API_KEY or _log_api_key(log_path)
    deadline = time.monotonic() + wait_key
    while not key and time.monotonic() < deadline:
        time.sleep(0.5)
        key = _log_api_key(log_path)
    st = _inference_status(port, key)
    if not st:
        return False

    # Idle auto-unload frees the GGUF but leaves the server up, so every field
    # below comes back null. That is a normal resting state, not a failure, and
    # reporting it as "model: ?" reads like something broke.
    if not st.get("active_model") and not st.get("loading"):
        ttl = up.idle_unload_seconds(STUDIO_DB)
        print("\n  Effective settings (from the server)\n")
        if ttl:
            mins = f"{ttl // 60}m" if ttl % 60 == 0 else f"{ttl}s"
            print(f"    state      : idle-unloaded — Unsloth freed the weights "
                  f"after {mins} idle.")
            print(f"                 The next /v1 request reloads it; VRAM is "
                  f"free until then.")
        else:
            print("    state      : no model loaded (and idle auto-unload is "
                  "off, so something\n                 unloaded it "
                  "deliberately).")
        return True

    ctx = st.get("context_length")
    native = st.get("native_context_length")
    ceiling = st.get("max_context_length")
    print("\n  Effective settings (from the server)\n")
    print(f"    model      : {st.get('active_model') or '?'}"
          + (f"  [{st['gguf_variant']}]" if st.get("gguf_variant") else ""))
    line = f"    context    : {_fmt_ctx(ctx)}"
    if native:
        line += f"   of native {_fmt_ctx(native)}"
    if ceiling and ctx and ceiling != ctx:
        line += f"   (fits up to {_fmt_ctx(ceiling)} here)"
    print(line)
    if st.get("cache_type_kv"):
        print(f"    kv cache   : {st['cache_type_kv']}")
    slots = st.get("parallel_slots")
    asked = st.get("requested_parallel_slots")
    if slots is not None:
        print(f"    slots      : {slots}"
              + (f"   (asked for {asked})" if asked and asked != slots else ""))
    spec = st.get("speculative_type")
    if spec:
        line = f"    speculative: {spec}"
        if st.get("spec_draft_n_max"):
            line += f", draft-n-max {st['spec_draft_n_max']}"
        print(line)
        if st.get("spec_fallback_reason"):
            print(f"                 fell back: {st['spec_fallback_reason']}")
    caps = [name for name, on in (("vision", st.get("is_vision")),
                                  ("tools", st.get("supports_tools")),
                                  ("reasoning", st.get("supports_reasoning")))
            if on]
    if caps:
        line = f"    supports   : {', '.join(caps)}"
        if st.get("supports_reasoning") and st.get("reasoning_style"):
            line += f"   (style: {st['reasoning_style']}"
            levels = st.get("reasoning_effort_levels") or []
            if levels:
                line += f", levels {'/'.join(levels)}"
            line += ")"
        print(line)

    # The per-model recommendation the server holds. It applies only to fields
    # a request omits AND that no UNSLOTH_SAMPLING_* pin already claims, so
    # anything this launch pinned outranks what is printed here.
    inf = st.get("inference")
    if isinstance(inf, dict):
        vals = {k: inf[k] for k in up.SAMPLING_FIELDS if inf.get(k) is not None}
        print(f"    server's own recommendation: "
              f"{up.fmt_sampling(vals) or 'none'}")
    return True


def _served_model_id(port: int, api_key: str) -> str:
    """Ask the server what it calls the loaded model.

    Unsloth serves a sanitised alias, not the repo id or the .gguf path, so the
    id has to come from /v1/models rather than being guessed.
    """
    try:
        with _http(f"http://127.0.0.1:{port}/v1/models",
                   api_key=api_key, timeout=10) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            _die("API key rejected by the server (401/403). "
                 "Check --api-key / UNSLOTH_MGR_API_KEY.")
        _die(f"GET /v1/models failed: HTTP {e.code}")
    except (urllib.error.URLError, OSError, ValueError) as e:
        _die(f"GET /v1/models failed: {e}")

    ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
    if not ids:
        _die("server reported no models on /v1/models")
    return ids[0]


# =============================================================================
# LAUNCH RESOLUTION
# =============================================================================

def _resolve_model(name: str) -> ml.ModelInfo:
    m = ml.find_model(HF_HOME, name)
    if m is None:
        _die(f"model {name!r} not found in {ml.hub_dir(HF_HOME)}\n"
             f"         Run `list` to see what is cached "
             f"(a shorthand that matches two repos is rejected).")
    return m


def _resolve_preset(name: str | None) -> up.Preset | None:
    """Pick the Studio preset to launch with, or None.

    "none" opts out. No flag at all means "whatever the Studio UI currently
    has selected", which keeps a headless launch and the UI in agreement.
    """
    if name is not None and name.strip().lower() in ("none", "off", ""):
        return None

    presets = up.load_presets(STUDIO_DB)
    if not presets:
        if name:
            _die(f"no presets found in {STUDIO_DB}; cannot honour "
                 f"--preset {name!r}")
        return None

    want = name or up.active_preset_name(STUDIO_DB)
    if not want:
        return None

    preset = up.find_preset(presets, want)
    if preset is None:
        have = ", ".join(f'"{n}"' for n in sorted(presets))
        if name:
            _die(f"preset {name!r} not found. Have: {have}")
        # The UI's active preset was renamed or deleted; say so rather than
        # silently launching with different settings than the UI shows.
        print(f"  Note: the Studio UI's active preset {want!r} is not in "
              f"{os.path.basename(STUDIO_DB)}; ignoring it.")
        return None
    return preset


def _resolve_sampling(lb: Launch, args) -> None:
    """Decide what sampling, if anything, the launch pins.

    Pinning is not free: `unsloth studio run --temperature ...` writes
    UNSLOTH_SAMPLING_*, which the server treats as an OPERATOR pin that beats
    even a value a client sent explicitly. Pinning nothing leaves Unsloth's own
    per-model recommendation in charge for fields a request omits, while
    letting a request override any of them.
    """
    source = (getattr(args, "sampling", None) or "docs").strip().lower()
    lb.sampling_source = source

    if source in ("none", "unsloth"):
        # Same wire effect (no flags), different intent, so the readout can say
        # which one you asked for.
        pass
    elif source == "preset":
        if lb.preset and lb.preset.sampling:
            lb.sampling.update(lb.preset.sampling)
            for key in lb.preset.sampling:
                lb.sources[key] = f"preset {lb.preset.name}"
        else:
            lb.notes.append(
                "--sampling preset: no preset sampling available; pinning "
                "nothing, so Unsloth's own recommendation applies")
    elif source == "docs":
        mode = (getattr(args, "mode", None) or up.DEFAULT_DOC_MODE)
        doc = up.doc_profile(lb.model.repo_id, mode)
        if doc is None:
            modes = up.doc_modes(lb.model.repo_id)
            if modes:
                _die(f"no published {mode!r} profile for {lb.model.repo_id}. "
                     f"Have: {', '.join(modes)}")
            lb.notes.append(
                f"no published profile for {lb.model.repo_id}; pinning "
                f"nothing, so Unsloth's own recommendation applies")
        else:
            lb.doc = doc
            lb.sampling.update(doc.values)
            for key in doc.values:
                lb.sources[key] = f"docs {doc.label}/{doc.mode}"
    else:
        _die(f"unknown --sampling {source!r}; "
             f"use docs, unsloth, preset or none")

    # An explicit flag beats every source, including a published profile.
    for key in up.SAMPLING_FIELDS:
        val = getattr(args, key, None)
        if val is None:
            continue
        lo, hi = up.SAMPLING_RANGES[key]
        if not lo <= val <= hi:
            _die(f"--{key.replace('_', '-')} {val} is outside the range "
                 f"`unsloth studio run` accepts ({lo}..{hi})")
        lb.sampling[key] = int(val) if key in up.INT_FIELDS else val
        lb.sources[key] = "flag"


def _apply_load_profile(lb: Launch, args) -> None:
    """Fill the load config from the chosen profile source.

    Precedence is fixed: an explicit flag beats the profile, the profile beats
    Unsloth's default. What varies is WHICH profile -- Unsloth's own per-model
    override (its richest and most specific), a Studio preset, or nothing.
    """
    want = (getattr(args, "load_profile", None) or "auto").strip().lower()
    if want not in ("auto", "override", "preset", "none"):
        _die(f"unknown --load-profile {want!r}; "
             f"use auto, override, preset or none")

    ov = lb.override
    if want in ("auto", "override") and ov:
        lb.load_source = f"override {ov.key}"
        # An explicit context in the override is a deliberate pin, so it is
        # sent verbatim rather than downgraded to fit-max.
        pairs = (("ctx", ov.context_length), ("parallel", ov.n_parallel),
                 ("tensor_parallel", ov.tensor_parallel),
                 ("kv_cache_dtype", ov.kv_cache_dtype),
                 ("spec_mode", ov.speculative_type),
                 ("spec_draft_n_max", ov.spec_draft_n_max),
                 ("n_batch", ov.n_batch), ("n_ubatch", ov.n_ubatch))
        for name, val in pairs:
            if val is not None:
                setattr(lb, name, val)
                lb.sources[name] = "override"
        if ov.disable_vision:
            lb.vision = False
            lb.sources["vision"] = "override"
        if ov.gpu_ids:
            lb.gpus = ",".join(str(g) for g in ov.gpu_ids)
            lb.sources["gpus"] = "override"
        if ov.llama_extra_args:
            lb.extra = list(ov.llama_extra_args) + lb.extra
            lb.sources["extra"] = "override"
        for unsupported, why in (
                (ov.gpu_memory_mode == "manual", "gpu_memory_mode=manual"),
                (ov.gpu_layers is not None, f"gpu_layers={ov.gpu_layers}"),
                (ov.n_cpu_moe is not None, f"n_cpu_moe={ov.n_cpu_moe}"),
                (bool(ov.chat_template_override), "chat_template_override")):
            if unsupported:
                lb.notes.append(
                    f"override field not applied ({why}): no `studio run` flag "
                    f"carries it")
        return

    if want == "override" and not ov:
        _die(f"--load-profile override: no per-model override saved for "
             f"{lb.model.repo_id}"
             + (f":{lb.variant}" if lb.variant else "")
             + f" in {os.path.basename(STUDIO_DB)}")

    if want in ("auto", "preset") and lb.preset:
        p = lb.preset
        lb.load_source = f"preset {p.name}"
        pairs = (("ctx", p.effective_context), ("parallel", p.n_parallel),
                 ("tensor_parallel", p.tensor_parallel),
                 ("kv_cache_dtype", p.kv_cache_dtype),
                 ("spec_mode", p.speculative_type),
                 ("spec_draft_n_max", p.spec_draft_n_max),
                 ("n_batch", p.n_batch), ("n_ubatch", p.n_ubatch))
        for name, val in pairs:
            if val is not None:
                setattr(lb, name, val)
                lb.sources[name] = "preset"


def _resolve_launch(m: ml.ModelInfo, args) -> Launch:
    """Merge the published profile, Unsloth's stored settings and the flags.

    Order of business matters: the variant is settled first because Unsloth's
    per-model overrides are keyed "<repo>:<variant>", so the wrong quant reads
    the wrong profile.
    """
    lb = Launch(model=m)

    # -- variant: explicit, else the best quant that fits --------------------
    lb.variant = (getattr(args, "variant", "") or "").strip()
    if m.is_gguf and not lb.variant and m.variants:
        lb.variant = ml.suggest_variant(m, _per_gpu_vram_gb(), _n_gpus())
    if lb.variant:
        lb.sources["variant"] = ("flag" if getattr(args, "variant", "")
                                 else "best that fits")

    lb.preset = _resolve_preset(getattr(args, "preset", None))
    lb.override = up.model_override(STUDIO_DB, m.repo_id, lb.variant)

    _apply_load_profile(lb, args)
    _resolve_sampling(lb, args)

    # -- command-line overrides ----------------------------------------------
    for attr, name in (("ctx", "ctx"), ("parallel", "parallel"),
                       ("tensor_parallel", "tensor_parallel"),
                       ("kv_cache_dtype", "kv_cache_dtype"),
                       ("n_batch", "n_batch"), ("n_ubatch", "n_ubatch")):
        val = getattr(args, attr, None)
        if val is not None and val != "":
            setattr(lb, name, val)
            lb.sources[name] = "flag"
    if getattr(args, "spec", None):
        mode = up.parse_spec_mode(args.spec)
        if mode is None:
            _die(f"unknown --spec {args.spec!r}; "
                 f"use one of {', '.join(up.SPEC_MODES)}")
        lb.spec_mode = mode
        lb.sources["spec_mode"] = "flag"
    if getattr(args, "spec_draft_n_max", None) is not None:
        lb.spec_draft_n_max = args.spec_draft_n_max
        lb.sources["spec_draft_n_max"] = "flag"
    if getattr(args, "vision", None) is not None:
        lb.vision = args.vision
        lb.sources["vision"] = "flag"
    if getattr(args, "tools", None) is not None:
        lb.tools = args.tools
    gpus = (getattr(args, "gpus", "") or "").strip()
    if gpus:
        lb.gpus = gpus
        lb.sources["gpus"] = "flag"
    lb.extra = lb.extra + list(getattr(args, "extra", None) or [])

    if lb.spec_draft_n_max and lb.spec_mode not in up.DRAFT_N_MAX_MODES:
        lb.notes.append(
            f"spec-draft-n-max {lb.spec_draft_n_max} ignored: only "
            f"{'/'.join(up.DRAFT_N_MAX_MODES)} launch a drafter with a "
            f"configurable depth")

    return lb


# =============================================================================
# SETTINGS READOUT
# =============================================================================

def _fmt_ctx(n: int | None) -> str:
    if not n:
        return "-"
    return f"{n:,}" if n < 1000 else f"{n:,} ({n / 1024:.0f}k)"


def _reload_drift(lb: Launch) -> list[tuple[str, str, str]]:
    """Fields a Studio idle-reload would restore differently from this launch.

    Idle auto-unload frees the GGUF and the next request reloads it -- but the
    reload rebuilds its LoadRequest from the per-model override in studio.db,
    not from the command line this manager started with. Anything we set that
    the override does not also carry is dropped the first time the server sits
    idle, silently and without a restart.

    Returns (field, "what we launched with", "what a reload restores").
    """
    if not up.idle_unload_seconds(STUDIO_DB):
        return []                      # nothing ever reloads it out from under us
    ov = lb.override
    labels = {"ctx": "context", "kv_cache_dtype": "kv cache",
              "spec_mode": "speculative", "spec_draft_n_max": "draft-n-max",
              "parallel": "slots", "n_batch": "batch", "n_ubatch": "ubatch",
              "tensor_parallel": "tensor-parallel"}

    def shown(val, field):
        if val is None:
            return "-"
        if field == "ctx":
            return "fit-max" if not val else f"{int(val):,}"
        if field == "tensor_parallel":
            return "on" if val else "off"
        return str(val)

    drift = []
    for field, attr in up.RELOAD_FIELDS:
        ours = getattr(lb, field)
        theirs = up.reload_context(ov) if field == "ctx" else getattr(ov, attr)
        # An override that says nothing about a field leaves it to the server's
        # own default, which for the ones set at startup (--parallel) is still
        # what we asked for. Only a field we set and the override contradicts,
        # or one we set and it drops, is real drift.
        if theirs is None:
            if ours in (None, 0, False):
                continue
            # `--parallel` is a server-wide startup default, so a reload that
            # omits n_parallel still lands on ours.
            if field == "parallel":
                continue
            drift.append((labels[field], shown(ours, field), "Unsloth default"))
            continue
        if ours != theirs:
            drift.append((labels[field], shown(ours, field), shown(theirs, field)))

    # Vision is its own flag, and Unsloth only ever stores the OFF direction:
    # an override with no disable_vision means a reload reattaches the mmproj.
    # Both directions are drift, so compare what each side actually yields.
    if lb.vision is not None:
        reload_vision = not ov.disable_vision
        if lb.vision != reload_vision:
            drift.append(("vision",
                          "on" if lb.vision else "off",
                          "on (mmproj reattached)" if reload_vision else "off"))
    return drift


def _print_reload_note(lb: Launch) -> None:
    """Say what an idle reload will do, whenever it can do something else."""
    ttl = up.idle_unload_seconds(STUDIO_DB)
    if not ttl:
        return
    drift = _reload_drift(lb)
    mins = f"{ttl // 60}m" if ttl % 60 == 0 else f"{ttl}s"
    if not drift:
        print(f"\n    idle reload: after {mins} idle Unsloth frees this model and "
              f"reloads it\n                 from studio.db — which matches this "
              f"launch, so nothing changes.")
        return
    print(f"\n    idle reload: WARNING — after {mins} idle Unsloth frees this "
          f"model and reloads\n                 it from studio.db, not from this "
          f"command line:")
    for field, ours, theirs in drift:
        print(f"                   {field:<16}{ours:<18}->  {theirs}")
    # studio.db is the only thing a reload consults, so the fix is always to
    # change studio.db, match it, or stop the reload from happening.
    print(f"                 studio.db decides what a reload restores. Either "
          f"save the values you\n                 want on the Studio settings "
          f"page for {lb.model.repo_id}\n                 (quant "
          f"{lb.variant or '-'}), launch with settings that already match it, "
          f"or\n                 set the idle timeout to 0 in Studio to stop "
          f"reloads entirely.")


def _sampling_report(m: ml.ModelInfo, lb: Launch | None = None) -> list[str]:
    """Every sampling source for a model, side by side.

    The point is comparison: Unsloth's shipped family table and its published
    guide disagree for these models, and only one of them is what you meant.
    """
    out = []
    defaults = up.unsloth_defaults(STUDIO_HOME, m.repo_id)
    modes = up.doc_modes(m.repo_id)

    for mode in modes:
        doc = up.doc_profile(m.repo_id, mode)
        mark = "*" if lb and lb.doc is doc else " "
        tail = f"   # {doc.note}" if doc.note else ""
        out.append(f"    {mark} docs / {mode:<9}: "
                   f"{up.fmt_sampling(doc.values)}{tail}")
    if modes:
        out.append(f"      source          : {up.doc_profile(m.repo_id, modes[0]).url}")
    else:
        out.append("      docs            : no published profile for this model")

    mark = "*" if lb and lb.sampling_source in ("unsloth", "none") else " "
    if defaults.values:
        out.append(f"    {mark} unsloth shipped : "
                   f"{up.fmt_sampling(defaults.applied)}"
                   f"   [{defaults.source} {defaults.detail}]".rstrip())
        skipped = {k: v for k, v in defaults.values.items()
                   if k not in up.UNSLOTH_AUTO_FIELDS}
        if skipped:
            out.append(f"      not auto-applied: {up.fmt_sampling(skipped)} "
                       f"(manual-only in Unsloth, like the chat UI)")
    else:
        out.append(f"    {mark} unsloth shipped : unreadable "
                   f"({defaults.detail or 'no data'})")
    return out


def _load_report(m: ml.ModelInfo, variant: str,
                 preset: up.Preset | None) -> list[str]:
    """The load config each stored source would supply."""
    out = []
    ov = up.model_override(STUDIO_DB, m.repo_id, variant)
    if ov:
        bits = [f"ctx {_fmt_ctx(ov.context_length)}"]
        for label, val in (("kv", ov.kv_cache_dtype),
                           ("spec", ov.speculative_type),
                           ("draft-n-max", ov.spec_draft_n_max),
                           ("parallel", ov.n_parallel)):
            if val is not None:
                bits.append(f"{label} {val}")
        if ov.tensor_parallel:
            bits.append("tensor-parallel")
        if ov.disable_vision:
            bits.append("vision off")
        if ov.gpu_ids:
            bits.append(f"gpu {','.join(str(g) for g in ov.gpu_ids)}")
        out.append(f"      unsloth per-model: {', '.join(bits)}")
        out.append(f"        key            : {ov.key}")
    else:
        out.append("      unsloth per-model: none saved for this model/quant")

    if preset:
        bits = [f"ctx {_fmt_ctx(preset.effective_context)}"]
        for label, val in (("kv", preset.kv_cache_dtype),
                           ("spec", preset.speculative_type),
                           ("parallel", preset.n_parallel)):
            if val is not None:
                bits.append(f"{label} {val}")
        if preset.tensor_parallel is not None:
            bits.append("tensor-parallel" if preset.tensor_parallel
                        else "no tensor-parallel")
        out.append(f"      studio preset    : {preset.name} — {', '.join(bits)}")
        out.append(f"        sampling       : "
                   f"{up.fmt_sampling(preset.sampling) or 'model defaults'}")
    else:
        out.append("      studio preset    : none selected")
    return out


def _manager_cli(args, lb: Launch) -> str:
    """This launch as a one-line `start` command.

    The TUI asks questions; this is the answer sheet, so a session driven by
    menus can be turned into a cron job or a group without reverse-engineering
    which flags it implied.
    """
    out = [f"python {os.path.abspath(__file__)} start {lb.model.repo_id}"]
    if lb.variant:
        out.append(f"--variant {lb.variant}")
    if lb.port:
        out.append(f"--port {lb.port}")
    if lb.gpus:
        out.append(f"--gpus {lb.gpus}")
    lp = getattr(args, "load_profile", "auto") or "auto"
    if lp != "auto":
        out.append(f"--load-profile {lp}")
    sm = getattr(args, "sampling", "docs") or "docs"
    if sm != "docs":
        out.append(f"--sampling {sm}")
    md = getattr(args, "mode", up.DEFAULT_DOC_MODE) or up.DEFAULT_DOC_MODE
    if md != up.DEFAULT_DOC_MODE:
        out.append(f"--mode {md}")
    if lb.preset_name:
        out.append(f'--preset "{lb.preset_name}"')
    if lb.tools is not None:
        out.append("--tools" if lb.tools else "--no-tools")
    # Only what the caller actually typed: a value the profile supplied is
    # already implied by --load-profile, and repeating it would freeze a
    # number that studio.db is meant to keep supplying.
    for flag, attr in (("--ctx", "ctx"), ("--parallel", "parallel"),
                       ("--kv-cache-dtype", "kv_cache_dtype"),
                       ("--spec", "spec"),
                       ("--spec-draft-n-max", "spec_draft_n_max"),
                       ("--n-batch", "n_batch"), ("--n-ubatch", "n_ubatch")):
        val = getattr(args, attr, None)
        if val is not None and val != "":
            out.append(f"{flag} {val}")
    if getattr(args, "tensor_parallel", None) is not None:
        out.append("--tensor-parallel" if args.tensor_parallel
                   else "--no-tensor-parallel")
    if getattr(args, "vision", None) is not None:
        out.append("--vision" if args.vision else "--no-vision")
    for key in up.SAMPLING_FIELDS:
        val = getattr(args, key, None)
        if val is not None:
            out.append(f"--{key.replace('_', '-')} {val}")
    for extra in (getattr(args, "extra", None) or []):
        out.append(f"--extra={extra}")
    return " ".join(out)


def _print_launch_plan(lb: Launch, argv: list[str], args=None) -> None:
    """What this launch will actually do, before it does it."""
    m = lb.model
    native = lb.native_ctx or up.doc_native_context(m.repo_id)

    print(f"\n  Starting {m.repo_id}")
    print(f"    variant : {lb.variant or '(model default)'}")
    print(f"    port    : {lb.port}")
    print(f"    gpus    : {lb.gpus or 'auto (Unsloth chooses)'}"
          + (f"  [{lb.sources['gpus']}]" if "gpus" in lb.sources else ""))
    print(f"    user    : {RUN_USER}")

    print(f"\n    load profile: {lb.load_source}")
    if lb.ctx:
        pinned = ("= native, pinned" if native and lb.ctx >= native
                  else "pinned")
        print(f"      context        : {_fmt_ctx(lb.ctx)} {pinned}"
              f"   [{lb.sources.get('ctx', 'default')}]")
        print(f"                       llama.cpp --fit may offload layers to "
              f"CPU if it does not fit")
    else:
        print(f"      context        : fit-max — Unsloth starts at the native "
              f"{_fmt_ctx(native)} and caps to VRAM")
    for label, val in (("kv cache", lb.kv_cache_dtype),
                       ("speculative", lb.spec_mode),
                       ("draft-n-max", lb.spec_draft_n_max),
                       ("slots", lb.parallel),
                       ("batch", lb.n_batch), ("ubatch", lb.n_ubatch)):
        if val is not None:
            key = {"kv cache": "kv_cache_dtype", "speculative": "spec_mode",
                   "draft-n-max": "spec_draft_n_max", "slots": "parallel",
                   "batch": "n_batch", "ubatch": "n_ubatch"}[label]
            src = lb.sources.get(key, "")
            print(f"      {label:<15}: {val}" + (f"   [{src}]" if src else ""))
    if lb.tensor_parallel is not None:
        print(f"      tensor-parallel: {'on' if lb.tensor_parallel else 'off'}"
              f"   [{lb.sources.get('tensor_parallel', '')}]")
    if lb.vision is not None:
        print(f"      vision (mmproj): {'on' if lb.vision else 'off'}"
              f"   [{lb.sources.get('vision', '')}]")

    print(f"\n    sampling: {lb.sampling_source}"
          + (f" — {lb.doc.describe()}" if lb.doc else ""))
    if lb.sampling:
        print(f"      pinned         : {up.fmt_sampling(lb.sampling)}")
        print(f"      note           : a pin is a hard operator override — it "
              f"wins even over a value a client sends")
    else:
        d = up.unsloth_defaults(STUDIO_HOME, m.repo_id)
        print(f"      pinned         : nothing")
        if d.applied:
            print(f"      server applies : {up.fmt_sampling(d.applied)}"
                  f"   [unsloth {d.source}]")
        else:
            # No recommendation reachable, so every field falls all the way
            # through to ChatCompletionRequest's own defaults. Naming them
            # beats printing a phrase the reader would have to go look up.
            print(f"      server applies : "
                  f"{up.fmt_sampling(up.SCHEMA_DEFAULTS)}   [schema defaults]")
    print("\n    other sources for comparison:")
    for line in _sampling_report(m, lb):
        print("  " + line)
    for line in _load_report(m, lb.variant, lb.preset):
        print(line)
    # Overrides are keyed per quant, so the variant choice silently decides
    # which profile is read. Say when another quant carries one too.
    others = [v for v in m.variant_names()
              if v != lb.variant and up.model_override(STUDIO_DB, m.repo_id, v).key]
    if others:
        print(f"      other quants with an override: {', '.join(others)}")

    # Measured on this box, both models: server-side tools serialise /v1 --
    # every request waits for the one before it, and single-request throughput
    # drops ~3.5x as well. llama-server's slots are fine; the tool layer above
    # them is what queues. Worth saying at launch, because the slot count in
    # the plan above is otherwise a promise the server will not keep.
    slots = lb.parallel if lb.parallel is not None else 4
    if lb.tools is False:
        # Say it plainly rather than leaving the absence of a warning to carry
        # the message: this is the setting that decides whether the slot count
        # above is real.
        print(f"\n    tools:       off — {slots} slot"
              f"{'s' if slots != 1 else ''} decode concurrently. No "
              f"server-side web\n                 search or code execution.")
    elif slots > 1:
        print(f"\n    tools:       WARNING — Unsloth's server-side tools are ON "
              f"(the default), and they\n                 serialise /v1: "
              f"{slots} slots will decode one request at a time.\n"
              f"                 Measured here: 3 concurrent requests ran "
              f"6.8x slower with tools\n                 on than off, and "
              f"single-request throughput was 3.5x lower.\n"
              f"                 Pass --no-tools unless you actually want "
              f"server-side web/code\n                 execution; clients can "
              f"still send enable_tools: false per request.")

    _print_reload_note(lb)

    for note in lb.notes:
        print(f"    note    : {note}")
    print(f"\n    $ {' '.join(argv)}")
    if args is not None:
        print(f"\n    this launch as a command:\n      {_manager_cli(args, lb)}")


# =============================================================================
# COMMANDS
# =============================================================================

def cmd_doctor(args):
    """Check the host can actually serve before anything is launched."""
    print("\n  Environment check\n")
    ok = True

    def report(label: str, good: bool, detail: str = ""):
        nonlocal ok
        ok = ok and good
        mark = "ok  " if good else "FAIL"
        print(f"    [{mark}] {label}" + (f"  — {detail}" if detail else ""))

    report("running as root", os.geteuid() == 0,
           "needed to drop privileges to " + RUN_USER)

    try:
        uid, _gid, _extra, home = _run_user_ids()
        report(f"user {RUN_USER} exists", True, f"uid={uid} home={home}")
        report(f"{RUN_USER} home is readable", os.path.isdir(home), home)
    except SystemExit:
        report(f"user {RUN_USER} exists", False)

    report("unsloth CLI on PATH", _has_command(UNSLOTH_BIN),
           shutil.which(UNSLOTH_BIN) or "not found")
    report("STUDIO_HOME exists", os.path.isdir(STUDIO_HOME), STUDIO_HOME)
    # An override that is not its own realpath re-creates the re-exec loop:
    # Unsloth resolves UNSLOTH_STUDIO_HOME but compares it to an unresolved
    # sys.prefix, so the child never sees itself as in-venv.
    if _STUDIO_HOME_OVERRIDE:
        real = os.path.realpath(_STUDIO_HOME_OVERRIDE)
        report("STUDIO_HOME override is symlink-free",
               real == _STUDIO_HOME_OVERRIDE.rstrip("/"),
               f"resolves to {real} — servers would re-exec forever")
    report("HF cache exists", os.path.isdir(ml.hub_dir(HF_HOME)),
           ml.hub_dir(HF_HOME))
    report("nvidia-smi present", _has_command("nvidia-smi"))

    gpus = _gpus(refresh=True)
    report("GPUs visible", bool(gpus),
           ", ".join(f"GPU{g['index']} {g['name']}" for g in gpus) or "none")

    models = ml.scan_models(HF_HOME)
    report("models cached", bool(models), f"{len(models)} repo(s)")

    report("studio.db readable", os.path.isfile(STUDIO_DB), STUDIO_DB)
    presets = up.load_presets(STUDIO_DB)
    active = up.active_preset_name(STUDIO_DB)
    report("presets found", bool(presets),
           ", ".join(f'"{n}"' for n in sorted(presets)) or
           "none saved in the Studio UI")

    overrides = up.load_overrides(STUDIO_DB)
    report("per-model overrides readable", True,
           f"{len(overrides)} saved" if overrides
           else "none saved on the Studio settings page")

    # Unsloth's shipped sampling tables live inside the Studio venv, not in
    # studio.db, so a venv rebuilt elsewhere silently costs the comparison
    # half its readout.
    assets = up.studio_assets_dir(STUDIO_HOME)
    pinned = os.environ.get("UNSLOTH_MGR_STUDIO_ASSETS", "").strip()
    report("Unsloth default tables readable", bool(assets),
           assets or (f"UNSLOTH_MGR_STUDIO_ASSETS={pinned} is not a directory"
                      if pinned else f"no assets/configs under {STUDIO_HOME}"))

    print(f"\n    active preset : {active or '(none selected)'}")
    print(f"    state         : {STATE_FILE}")
    print(f"    logs          : {LOG_DIR}")
    print(f"    api ports     : {PORT_POOL_LABEL} "
          f"({MAX_INSTANCES} instance{'s' if MAX_INSTANCES != 1 else ''} max)")
    print(f"    api key       : "
          f"{'set via UNSLOTH_MGR_API_KEY' if API_KEY else 'not set (read from each server log)'}")

    print("\n  " + ("All checks passed." if ok else "Some checks failed.") + "\n")
    if not ok:
        sys.exit(1)


def cmd_list(args):
    models = ml.scan_models(HF_HOME)
    if not models:
        print(f"\n  No models in {ml.hub_dir(HF_HOME)}\n")
        return

    running = _get_running()
    per_gpu, n_gpus = _per_gpu_vram_gb(), _n_gpus()

    print(f"\n  Models in {ml.hub_dir(HF_HOME)}\n")
    for m in models:
        mark = "*" if m.repo_id in running else " "
        print(f"  {mark} {m.repo_id}")
        if m.is_empty:
            print("      (no weights on disk — incomplete download)")
        else:
            best = (ml.suggest_variant(m, per_gpu, n_gpus)
                    if m.is_gguf and m.variants else "")
            print(f"      {m.kind:<5} {ml.human_size(m.size_bytes):>7} on disk"
                  + (f"    default quant: {best}" if best else ""))

        if args.variants and m.variants:
            for v in m.variant_names():
                size = m.variants[v]
                note = ml.fit_note(size, per_gpu, n_gpus)
                print(f"        - {v:<16} {ml.human_size(size):>7}  {note}")

        if m.repo_id in running:
            e = running[m.repo_id]
            state = ("ready" if _probe_ready(e["port"], log_path=e.get("log", ""))
                     else "loading")
            print(f"      running: port {e['port']} "
                  f"[{_profile_label(e)}] "
                  f"{state}  {_endpoint_url(e['port'])}")
    print()
    if not args.variants:
        print("  (--variants to list GGUF quants and their VRAM fit)\n")


def cmd_start(args):
    _ensure_dirs()
    m = _resolve_model(args.model)
    if m.is_empty:
        _die(f"{m.repo_id} has no weights on disk (incomplete download). "
             f"Re-download it before serving.")
    lb = _resolve_launch(m, args)

    state, _ = _prune_dead(_load_state())

    if m.repo_id in state["models"] and not args.force:
        e = state["models"][m.repo_id]
        _die(f"{m.repo_id} is already running on port {e['port']} "
             f"(pid {e['pid']}). Use --force to start another instance, "
             f"or `restart`.")

    # The cap is checked before the port is chosen so a full manager says so
    # plainly, rather than reporting it as an exhausted port pool.
    if len(state["models"]) >= MAX_INSTANCES:
        running = ", ".join(
            f"{name} :{e['port']}"
            for name, e in sorted(state["models"].items(),
                                  key=lambda kv: kv[1]["port"]))
        _die(f"already running {len(state['models'])} of {MAX_INSTANCES} "
             f"instances (ports {PORT_POOL_LABEL}).\n"
             f"         {running}\n"
             f"         Stop one first, or raise UNSLOTH_MGR_MAX_INSTANCES.")

    if args.port:
        if args.port not in PORT_POOL:
            _die(f"--port {args.port} is outside the API port pool "
                 f"({PORT_POOL_LABEL}). Clients only look there; widen the "
                 f"pool with UNSLOTH_MGR_BASE_PORT / "
                 f"UNSLOTH_MGR_MAX_INSTANCES rather than binding outside it.")
        held = next((n for n, e in state["models"].items()
                     if int(e["port"]) == args.port), "")
        if held:
            _die(f"port {args.port} is already serving {held}")
    lb.port = args.port or _next_port(state)
    if not _port_free(lb.port):
        _die(f"port {lb.port} is already in use by another process")

    # -- pre-flight: do the requested GPUs exist, and is there room? ----------
    gpus = _gpus(refresh=True)
    known = {str(g["index"]) for g in gpus}
    want_gpus = [g.strip() for g in lb.gpus.split(",") if g.strip()]
    missing = [g for g in want_gpus if g not in known]
    if missing and known:
        _die(f"--gpus asks for {','.join(missing)}; "
             f"this host has {','.join(sorted(known))}")

    if m.is_gguf and lb.variant:
        if lb.variant not in m.variants:
            have = ", ".join(m.variant_names()) or "none"
            _die(f"variant {lb.variant!r} is not in the cache for "
                 f"{m.repo_id}. Have: {have}")
        weight = m.variants[lb.variant]
        # With no --gpus the model may land on any card, so budget against
        # everything free rather than a subset we did not actually pin.
        pool = [g for g in gpus
                if not want_gpus or str(g["index"]) in want_gpus]
        free_mb = sum(g["total_mb"] - g["used_mb"] for g in pool)
        need_mb = int(weight / (1024 ** 2) * ml.FIT_OVERHEAD)
        if gpus and need_mb > free_mb:
            where = lb.gpus or "all GPUs"
            print(f"\n  Warning: {lb.variant} needs roughly "
                  f"{need_mb // 1024}G but only {free_mb // 1024}G is free on "
                  f"{where}.")
            if not args.force:
                _die("refusing to start. Re-run with --force to try anyway.")

    # -- launch ---------------------------------------------------------------
    _, _, _, home = _run_user_ids()
    env = _child_env(lb.gpus, {}, home)
    argv = _build_cmd(lb)
    log_path = _log_path(m.repo_id)

    _print_launch_plan(lb, argv, args)
    print(f"    log     : {log_path}")
    # Before the rotate, not after: --dry-run must touch nothing. Rotating here
    # would rename a RUNNING server's log out from under it -- the server keeps
    # writing to the renamed inode, so `logs` finds nothing and the readout
    # loses the API key it reads from that file.
    if getattr(args, "dry_run", False):
        print("\n  --dry-run: nothing started.\n")
        return
    _rotate_log(log_path)

    try:
        proc = _spawn(argv, env, log_path, home)
    except (OSError, PermissionError) as e:
        _die(f"could not launch: {e}")

    # Unsloth re-execs; give it a moment to record the real pid per port.
    pid = proc.pid
    for _ in range(20):
        time.sleep(0.25)
        real = _read_studio_pid(lb.port)
        if real and _pid_alive(real):
            pid = real
            break
        if proc.poll() is not None:
            break

    if proc.poll() is not None and not _pid_alive(pid):
        print(f"\n  Server exited immediately (status {proc.returncode}).")
        _print_log_tail(log_path, 30)
        sys.exit(1)

    # Enough to reproduce this exact server. A restart that silently dropped
    # the profile choices would come back on different settings than the ones
    # the launch plan just printed.
    state["models"][m.repo_id] = {
        "pid": pid,
        "spawn_pid": proc.pid,
        "port": lb.port,
        "preset": lb.preset_name,
        "variant": lb.variant,
        "gpus": lb.gpus,
        "started": time.time(),
        "log": log_path,
        "studio_home": STUDIO_HOME,
        "sampling": lb.sampling_source,
        "mode": lb.doc.mode if lb.doc else "",
        "load_profile": getattr(args, "load_profile", "auto"),
        "load_source": lb.load_source,
        "ctx": lb.ctx,
        "spec": lb.spec_mode or "",
        # None means "left at Unsloth's default (on)"; store it as such so a
        # restart cannot silently turn tools back on for a server started
        # without them.
        "tools": lb.tools,
        # Every answer this launch was given, so an instance group can replay
        # it verbatim. Stored as inputs, not as the settings they resolved to.
        "launch_args": _launch_args(args),
        # The RESOLVED values, for the home screen. Stored rather than
        # re-derived because that header redraws every few seconds and must
        # not do file or HTTP work to say what a server is running.
        "pinned": dict(lb.sampling),
        "kv": lb.kv_cache_dtype,
        "parallel": lb.parallel,
        "vision": lb.vision,
    }
    _save_state(state)

    print(f"    pid     : {pid}")

    wait = args.wait if args.wait is not None else LOAD_TIMEOUT_SEC
    if wait <= 0:
        print(f"\n  Launched. {_endpoint_url(lb.port)}\n")
        return

    print(f"\n  Waiting up to {wait}s for the model to load ...", end="", flush=True)
    deadline = time.time() + wait
    while time.time() < deadline:
        if not _pid_alive(pid):
            print(" server died.\n")
            _print_log_tail(log_path, 30)
            state, _ = _prune_dead(_load_state())
            sys.exit(1)
        _READY_CACHE.pop(lb.port, None)
        if _probe_ready(lb.port, timeout=1.0, log_path=log_path):
            print(" ready.")
            # The plan said what we asked for; this says what the server did
            # with it -- the context its fitter settled on, the speculative
            # mode that survived, and the sampling it will apply on its own.
            if not _print_effective(lb.port, log_path,
                                    getattr(args, "api_key", "") or "",
                                    wait_key=20.0):
                print("\n  (could not read back effective settings: no API "
                      "key found in the log)")
            print(f"\n  {m.repo_id} -> {_endpoint_url(lb.port)}\n")
            return
        time.sleep(2)
        print(".", end="", flush=True)

    listening = False
    try:
        with _http(f"http://127.0.0.1:{lb.port}/api/health", timeout=2.0) as r:
            listening = r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        pass
    stage = ("serving, but the model has not finished loading" if listening
             else "not listening yet")
    print(f"\n\n  Still starting after {wait}s ({stage}); leaving it running.")
    print(f"  Follow with: python {os.path.basename(__file__)} "
          f"logs {m.repo_id} -f\n")


def _print_log_tail(path: str, n: int = 25):
    if not os.path.exists(path):
        return
    print(f"\n  --- last {n} lines of {path} ---")
    try:
        with open(path, errors="replace") as f:
            lines = f.readlines()[-n:]
        for line in lines:
            print("  " + line.rstrip())
    except OSError as e:
        print(f"  (could not read log: {e})")
    print()


def _signal_and_wait(name: str, pid: int, timeout: int) -> bool:
    """SIGTERM, then escalate to a group kill. True if it exited cleanly.

    Studio traps SIGTERM to shut its llama-server children down, so the polite
    signal is what avoids orphaned GPU memory. The group kill is the backstop.
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        print(f"  No permission to signal pid {pid}")
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        # Reap as we go: when this process is the server's parent, an unreaped
        # exit lingers as a zombie and would read as "still running".
        _reap_children()
        if not _pid_alive(pid):
            return True
        time.sleep(0.5)

    print(f"  {name} did not stop within {timeout}s; killing process group.")
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    for _ in range(20):
        _reap_children()
        if not _pid_alive(pid):
            return True
        time.sleep(0.25)
    return not _pid_alive(pid)


def cmd_stop(args):
    state, _ = _prune_dead(_load_state())
    m = ml.find_model(HF_HOME, args.model)
    key = m.repo_id if m else args.model

    if key not in state["models"]:
        running = ", ".join(sorted(state["models"])) or "none"
        _die(f"{key} is not running. Running: {running}")

    e = state["models"][key]
    print(f"\n  Stopping {key} (pid {e['pid']}, port {e['port']}) ...")
    ok = _signal_and_wait(key, int(e["pid"]), STOP_TIMEOUT_SEC)

    state["models"].pop(key, None)
    _save_state(state)

    # Unsloth leaves its per-port record behind on a hard kill.
    stale = _studio_pid_file(int(e["port"]))
    if os.path.exists(stale) and not _pid_alive(int(e["pid"])):
        try:
            os.unlink(stale)
        except OSError:
            pass

    print("  Stopped.\n" if ok else "  Stop may have failed; check status.\n")


def cmd_stop_all(args):
    running = _get_running()
    if not running:
        print("\n  Nothing running.\n")
        return
    print(f"\n  Stopping {len(running)} server(s) ...")
    for name in list(running):
        # One server refusing to stop must not strand the others, and cmd_stop
        # exits the process on error.
        try:
            cmd_stop(argparse.Namespace(model=name))
        except SystemExit:
            print(f"  Could not stop {name}; continuing.")


def cmd_restart(args):
    running = _get_running()
    m = ml.find_model(HF_HOME, args.model)
    key = m.repo_id if m else args.model

    # Carry the running server's placement AND its profile choices forward: a
    # restart that dropped either would come back on different hardware or
    # different settings than the plan the first launch printed.
    plan = dict(preset=args.preset, port=args.port,
                gpus=getattr(args, "gpus", "") or "",
                variant=getattr(args, "variant", "") or "",
                sampling=getattr(args, "sampling", None),
                mode=getattr(args, "mode", None),
                tools=getattr(args, "tools", None),
                load_profile=getattr(args, "load_profile", None))
    if key in running:
        e = running[key]
        if plan["preset"] is None:
            plan["preset"] = e.get("preset") or None
        if plan["port"] is None:
            plan["port"] = int(e["port"])
        for name, stored in (("gpus", "gpus"), ("variant", "variant")):
            if not plan[name]:
                plan[name] = e.get(stored, "")
        for name in ("sampling", "mode", "load_profile"):
            if plan[name] is None:
                plan[name] = e.get(name) or None
        # `or None` would turn a stored False back into "Unsloth default", so
        # tools is carried forward on presence, not on truth.
        if plan["tools"] is None:
            plan["tools"] = e.get("tools")
        cmd_stop(argparse.Namespace(model=key))
        time.sleep(1.5)

    cmd_start(argparse.Namespace(
        model=key, wait=args.wait, force=False, dry_run=False, api_key="",
        sampling=plan["sampling"] or "docs",
        mode=plan["mode"] or up.DEFAULT_DOC_MODE,
        load_profile=plan["load_profile"] or "auto",
        tools=plan["tools"],
        preset=plan["preset"], port=plan["port"], gpus=plan["gpus"],
        variant=plan["variant"]))


def cmd_status(args):
    state, dead = _prune_dead(_load_state())
    running = state["models"]

    print()
    if dead:
        print(f"  Pruned {len(dead)} dead entr(ies): {', '.join(dead)}\n")

    if not running:
        print("  No servers running.")
    else:
        print(f"  {len(running)} server(s) running\n")
        for name, e in sorted(running.items(), key=lambda kv: kv[1]["port"]):
            ready = _probe_ready(int(e["port"]), log_path=e.get("log", ""))
            print(f"  {'●' if ready else '○'} {name}")
            print(f"      {_profile_label(e)}"
                  + (f"  ·  variant {e['variant']}" if e.get("variant") else ""))
            if e.get("ctx"):
                print(f"      context {int(e['ctx']):,} pinned")
            print(f"      port {e['port']}  pid {e['pid']}  gpus {e.get('gpus', '-')}"
                  f"  up {_fmt_uptime(float(e.get('started', time.time())))}")
            print(f"      {'ready' if ready else 'loading'}   "
                  f"{_endpoint_url(int(e['port']))}")
            print()

    gsplit = _gpu_usage_split()
    if gsplit:
        print("  GPU memory")
        for g in gsplit:
            pct = 100 * g["used_mb"] // max(1, g["total_mb"])
            print(f"    GPU{g['index']} {g['name'][:24]:<24} "
                  f"{g['used_mb'] // 1024:>2}/{g['total_mb'] // 1024:>2}G "
                  f"({pct:>3}%)  ours {g['ours_mb'] // 1024}G  "
                  f"other {g['other_mb'] // 1024}G")

    tot, avail = _read_mem_gb()
    if tot:
        print(f"    RAM  {tot - avail}/{tot}G used")
    print()


def cmd_logs(args):
    m = ml.find_model(HF_HOME, args.model)
    key = m.repo_id if m else args.model
    path = _log_path(key)
    if not os.path.exists(path):
        _die(f"no log at {path}")

    if args.follow:
        try:
            subprocess.run(["tail", "-n", str(args.lines), "-f", path])
        except KeyboardInterrupt:
            print()
    else:
        _print_log_tail(path, args.lines)


# -- streaming client shared by test and benchmark ---------------------------

def _stream_completion(port: int, model_id: str, api_key: str, prompt: str,
                       max_tokens: int, freeze_timeout: float) -> dict:
    """One streaming chat completion, timing every token.

    Returns token_times (monotonic stamps), ttft, text and a stall count.
    Raises RuntimeError if the stream goes quiet for freeze_timeout.
    """
    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()

    start = time.monotonic()
    token_times: list[float] = []
    chunks: list[str] = []
    reasoning: list[str] = []
    stalls = 0
    last = start

    with _http(f"http://127.0.0.1:{port}/v1/chat/completions",
               method="POST", data=body, api_key=api_key,
               timeout=freeze_timeout) as r:
        for raw in r:
            now = time.monotonic()
            if now - last > freeze_timeout:
                raise RuntimeError(f"stream stalled for {freeze_timeout}s")
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                # Reasoning models (Qwen3.6 MTP and friends) send their output
                # as reasoning_content with content set to "". Counting only
                # content would time those runs as producing nothing at all.
                piece = delta.get("content")
                think = delta.get("reasoning_content")
                if piece or think:
                    if token_times and now - token_times[-1] > 2.0:
                        stalls += 1
                    token_times.append(now)
                    if piece:
                        chunks.append(piece)
                    if think:
                        reasoning.append(think)
            last = now

    return {
        "token_times": token_times,
        "ttft": (token_times[0] - start) if token_times else 0.0,
        "total": (token_times[-1] - start) if token_times else 0.0,
        "text": "".join(chunks),
        "reasoning": "".join(reasoning),
        "stalls": stalls,
    }


def _steady_tps(token_times: list[float], window: float) -> float:
    """Best sustained tokens/sec over a sliding window.

    A mid-stream stall drags the mean down and hides what the server actually
    sustains, so the headline number comes from the best window instead.

    Only near-full windows count. Streamed chunks can carry several tokens and
    arrive back-to-back, so a handful of them spanning milliseconds yields an
    enormous instantaneous rate that is a burst, not a sustained one. A
    generation shorter than one window has no steady state to report at all,
    and gets the honest overall rate instead.
    """
    n = len(token_times)
    span_all = token_times[-1] - token_times[0] if n >= 2 else 0.0
    # Too few tokens, or all of them delivered in one burst: there is no
    # sustained rate here. Report nothing rather than a number like 600 tok/s
    # derived from 13 chunks that landed 20ms apart. Callers show "n/a".
    if n < STEADY_MIN_TOKENS or span_all < STEADY_MIN_SPAN:
        return 0.0
    if span_all < window:
        return (n - 1) / span_all

    best, left = 0.0, 0
    for right in range(1, n):
        while token_times[right] - token_times[left] > window:
            left += 1
        span = token_times[right] - token_times[left]
        if span >= window * 0.8:
            best = max(best, (right - left) / span)
    return best or (n - 1) / span_all


def cmd_test(args):
    running = _get_running()
    m = ml.find_model(HF_HOME, args.model)
    key = m.repo_id if m else args.model
    if key not in running:
        _die(f"{key} is not running")

    port = int(running[key]["port"])
    api_key = _resolve_api_key(args.api_key, running[key].get("log", ""))
    if not _probe_ready(port, timeout=2.0,
                        log_path=running[key].get("log", "")):
        _die(f"server on port {port} is not ready yet")

    model_id = _served_model_id(port, api_key)
    prompt = args.prompt or TEST_PROMPT

    print(f"\n  {key}  (served as {model_id})")
    print(f"  {_endpoint_url(port)}\n")
    print(f"  > {prompt}\n")

    try:
        res = _stream_completion(port, model_id, api_key, prompt,
                                 args.max_tokens, BENCH_FREEZE_TIMEOUT)
    except urllib.error.HTTPError as e:
        _die(f"request failed: HTTP {e.code} {e.read().decode(errors='replace')[:200]}")
    except (urllib.error.URLError, OSError, RuntimeError) as e:
        _die(f"request failed: {e}")

    if res["reasoning"] and not res["text"].strip():
        # The budget went entirely to reasoning; show it rather than a blank.
        print("  [reasoning only — the answer needs a larger --max-tokens]\n")
        print("  " + res["reasoning"].replace("\n", "\n  ").strip())
    else:
        if res["reasoning"]:
            print(f"  [+{len(res['reasoning'])} chars of reasoning]\n")
        print("  " + res["text"].replace("\n", "\n  ").strip())
    n = len(res["token_times"])
    steady = _steady_tps(res["token_times"], BENCH_WINDOW)
    mean = n / res["total"] if res["total"] > 0 else 0.0
    steady_s = f"{steady:.1f} tok/s" if steady else "n/a (too short)"
    print(f"\n  {n} tokens   TTFT {res['ttft']:.2f}s   "
          f"steady {steady_s}   mean {mean:.1f} tok/s\n")


def cmd_benchmark(args):
    """Load each (model, preset), measure steady-state tok/s, then stop it."""
    _ensure_dirs()
    m = _resolve_model(args.model)
    api_key = _resolve_api_key(args.api_key)

    have = up.load_presets(STUDIO_DB)
    # No --preset means "compare every preset", which is the question the
    # benchmark exists to answer. Fall back to a single unpresetted run.
    names = args.preset or (sorted(have) if have else [""])
    missing = [n for n in names if n and up.find_preset(have, n) is None]
    if missing:
        _die(f"unknown preset(s): {', '.join(missing)}. "
             f"Have: {', '.join(sorted(have)) or 'none'}")

    if m.repo_id in _get_running():
        _die(f"{m.repo_id} is already running. Stop it first — benchmark "
             f"needs to own the load so the numbers mean something.")

    results = []
    print(f"\n  Benchmarking {m.repo_id}: {len(names)} preset(s)")
    print(f"  (preset load config only — no per-model override, no sampling "
          f"pins, so nothing but the preset differs between runs)\n")

    for pname in names:
        label = pname or "(no preset)"
        print(f"  --- preset '{label}' " + "-" * 40)
        t_load = time.monotonic()
        try:
            # Only the preset varies. The per-model override is held out
            # (--load-profile preset) or it would supply the same load config
            # to every run and flatten the comparison, and sampling is pinned
            # to nothing so it cannot differ between runs either.
            cmd_start(argparse.Namespace(
                model=m.repo_id, preset=(pname or "none"), port=None,
                gpus=args.gpus, variant=args.variant,
                load_profile="preset", sampling="none",
                mode=up.DEFAULT_DOC_MODE, api_key=args.api_key,
                dry_run=False, wait=LOAD_TIMEOUT_SEC, force=False))
        except SystemExit:
            print(f"  preset '{label}': failed to start; skipping.\n")
            results.append({"preset": label, "error": "start failed"})
            continue
        load_secs = time.monotonic() - t_load

        entry = _get_running().get(m.repo_id)
        if not entry:
            results.append({"preset": label, "error": "vanished after start"})
            continue
        port = int(entry["port"])

        try:
            model_id = _served_model_id(port, api_key)
            for _ in range(BENCH_WARMUP):
                _stream_completion(port, model_id, api_key, BENCH_PROMPT,
                                   64, BENCH_FREEZE_TIMEOUT)

            runs = []
            for i in range(BENCH_RUNS):
                # Distinct prompts so prefix caching cannot collapse the runs.
                prompt = f"{BENCH_PROMPT} (variation {i + 1})"
                res = _stream_completion(port, model_id, api_key, prompt,
                                         args.max_tokens, BENCH_FREEZE_TIMEOUT)
                runs.append(res)
                s = _steady_tps(res["token_times"], BENCH_WINDOW)
                print(f"    run {i + 1}: {len(res['token_times'])} chunks  "
                      + (f"steady {s:.1f} tok/s" if s else "steady n/a (too short)"))

            peak = [g["ours_mb"] for g in _gpu_usage_split()]
            steady = max(_steady_tps(r["token_times"], BENCH_WINDOW) for r in runs)
            mean_tps = sum(len(r["token_times"]) for r in runs) / max(
                1e-9, sum(r["total"] for r in runs))
            results.append({
                "preset": label,
                "steady_tps": round(steady, 2),
                "mean_tps": round(mean_tps, 2),
                "ttft": round(sum(r["ttft"] for r in runs) / len(runs), 3),
                # Streamed content pieces, not tokenizer tokens: the SSE
                # deltas are what we can actually time.
                "chunks": sum(len(r["token_times"]) for r in runs),
                "stalls": sum(r["stalls"] for r in runs),
                "load_secs": round(load_secs, 1),
                "peak_vram_mb": peak,
            })
        except (urllib.error.HTTPError, urllib.error.URLError,
                OSError, RuntimeError) as e:
            print(f"    failed: {e}")
            results.append({"preset": label, "error": str(e)})
        finally:
            cmd_stop(argparse.Namespace(model=m.repo_id))
            time.sleep(2)

    # -- report ---------------------------------------------------------------
    print(f"\n  Results for {m.repo_id}\n")
    print(f"    {'preset':<20} {'steady':>10} {'mean':>9} {'TTFT':>7} "
          f"{'chunks':>7} {'load':>7}")
    print("    " + "-" * 60)
    for r in results:
        if "error" in r:
            print(f"    {r['preset']:<20} {'FAILED':>10}  {r['error'][:30]}")
            continue
        steady = (f"{r['steady_tps']:>8.1f}/s" if r["steady_tps"]
                  else f"{'n/a':>10}")
        print(f"    {r['preset']:<20} {steady} "
              f"{r['mean_tps']:>8.1f}/s {r['ttft']:>6.2f}s "
              f"{r['chunks']:>7} {r['load_secs']:>6.0f}s")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(LOG_DIR, f"bench-{_safe_name(m.repo_id)}-{stamp}.json")
    try:
        with open(out, "w") as f:
            json.dump({"model": m.repo_id, "results": results,
                       "when": stamp}, f, indent=2)
        print(f"\n  Saved {out}\n")
    except OSError as e:
        print(f"\n  (could not save results: {e})\n")


# -- presets -----------------------------------------------------------------

def cmd_presets(args):
    """Show what the Studio UI has saved. Read-only: the UI owns these."""
    presets = up.load_presets(STUDIO_DB)
    active = up.active_preset_name(STUDIO_DB)

    if not presets:
        print(f"\n  No presets saved in {STUDIO_DB}")
        print("  Create one in the Unsloth Studio UI (Preset dropdown).\n")
        return

    print(f"\n  Presets in {STUDIO_DB}\n")
    for name in sorted(presets):
        p = presets[name]
        mark = "*" if name == active else " "
        print(f"  {mark} {name}")

        load_bits = []
        if p.effective_context:
            load_bits.append(f"ctx {p.effective_context}")
        if p.n_parallel is not None:
            load_bits.append(f"parallel {p.n_parallel}")
        if p.tensor_parallel is not None:
            load_bits.append("tensor-parallel"
                             if p.tensor_parallel else "no tensor-parallel")
        if p.kv_cache_dtype:
            load_bits.append(f"kv-cache {p.kv_cache_dtype}")
        if p.spec_draft_n_max:
            load_bits.append(f"draft-n-max {p.spec_draft_n_max}")
        print(f"      load    : {', '.join(load_bits) or 'Unsloth defaults'}")

        spec = p.speculative_type or "auto"
        print(f"      spec    : {spec}"
              + ("  (Unsloth decides)" if spec == "auto" else ""))
        print(f"      sampling: {up.fmt_sampling(p.sampling) or 'model defaults'}"
              + ("   — pinned only with --sampling preset" if p.sampling else ""))

        # These are chat-UI concerns with no `studio run` equivalent; saying so
        # is better than letting someone assume a headless launch honours them.
        skipped = []
        if p.max_tokens:
            skipped.append(f"maxTokens {p.max_tokens}")
        if p.system_prompt:
            skipped.append(f"systemPrompt ({len(p.system_prompt)} chars)")
        if skipped:
            print(f"      not applied (per-request, not load-time): "
                  f"{', '.join(skipped)}")
        print()

    print(f"  * = selected in the Studio UI; used when --preset is omitted.")
    print(f"  Presets are global and carry no model/port/GPU — pass those as flags.")
    print(f"  Preset sampling is NOT pinned by default; `settings <model>` shows "
          f"what is.\n")


def cmd_groups(args):
    """Save, restore and inspect named snapshots of what is running."""
    action = args.action
    slots = _load_groups()

    if action == "list":
        print(f"\n  Instance groups in {GROUPS_FILE}\n")
        for n in range(1, GROUP_SLOTS + 1):
            g = slots.get(str(n))
            if not g:
                print(f"    {n:>2}. (empty)")
                continue
            when = time.strftime("%Y-%m-%d %H:%M",
                                 time.localtime(g.get("saved", 0)))
            members = g.get("members", [])
            print(f"    {n:>2}. {g.get('name') or '(unnamed)'}"
                  f"   {len(members)} model(s)   saved {when}")
            for mem in members:
                print(f"        {mem.get('model', '?')}")
                print(f"          {_fmt_member(mem)}")
        print(f"\n  Restore one with: {_group_cli('restore', 1)}\n")
        return

    slot = _check_slot(args.slot)
    key = str(slot)

    if action == "clear":
        if key not in slots:
            print(f"\n  Slot {slot} is already empty.\n")
            return
        name = slots[key].get("name") or "(unnamed)"
        del slots[key]
        _save_groups(slots)
        print(f"\n  Cleared slot {slot} ({name}).\n")
        return

    if action == "show":
        g = slots.get(key)
        if not g:
            _die(f"slot {slot} is empty")
        print(f"\n  Slot {slot}: {g.get('name') or '(unnamed)'}\n")
        for mem in g.get("members", []):
            print(f"    {mem.get('model', '?')}")
            for k in sorted(mem):
                if k != "model" and mem[k] not in (None, "", []):
                    print(f"      {k:<18}{mem[k]}")
            print()
        print(f"  {_group_cli('restore', slot)}\n")
        return

    if action == "save":
        state, _ = _prune_dead(_load_state())
        running = state["models"]
        if not running:
            _die("nothing is running; start the models you want in the group "
                 "first.")
        members = []
        for name, e in sorted(running.items(), key=lambda kv: kv[1]["port"]):
            mem = dict(e.get("launch_args") or {})
            # A server started before groups existed (or by an older version)
            # has no stored answers; rebuild what we can from the state entry
            # rather than refusing to save it.
            if not mem.get("model"):
                mem = dict(LAUNCH_DEFAULTS)
                mem.update(model=name, variant=e.get("variant", ""),
                           gpus=e.get("gpus", ""), preset=e.get("preset") or None,
                           load_profile=e.get("load_profile") or "auto",
                           sampling=e.get("sampling") or "docs",
                           mode=e.get("mode") or up.DEFAULT_DOC_MODE,
                           tools=e.get("tools"))
            # Always the port it is ACTUALLY on, not the one that was asked
            # for: an auto-assigned launch stored None, and a restore that
            # re-auto-assigns would not reproduce this layout.
            mem["model"] = name
            mem["port"] = int(e["port"])
            members.append(mem)

        slots[key] = {
            "name": args.name or f"group {slot}",
            "saved": time.time(),
            "members": members,
        }
        _save_groups(slots)
        print(f"\n  Saved {len(members)} model(s) to slot {slot} "
              f"({slots[key]['name']}):\n")
        for mem in members:
            print(f"    {mem['model']}")
            print(f"      {_fmt_member(mem)}")
        print(f"\n  Restore after a reboot with:\n")
        print(f"    {_group_cli('restore', slot)}\n")
        print(f"  As a systemd unit (After=network-online.target, "
              f"Type=oneshot):\n")
        print(f"    ExecStart={_group_cli('restore', slot)}\n")
        print(f"  Or from cron:\n")
        print(f"    @reboot {_group_cli('restore', slot)} "
              f">> {LOG_DIR}/group-restore.log 2>&1\n")
        return

    if action == "restore":
        g = slots.get(key)
        if not g:
            _die(f"slot {slot} is empty. Save one with "
                 f"`groups save {slot}`.")
        members = g.get("members", [])
        if not members:
            _die(f"slot {slot} holds no models")

        state, _ = _prune_dead(_load_state())
        running = set(state["models"])
        print(f"\n  Restoring slot {slot} ({g.get('name') or 'unnamed'}): "
              f"{len(members)} model(s)\n")

        started, skipped, failed = [], [], []
        for mem in members:
            name = mem.get("model", "")
            if name in running:
                # Idempotent on purpose: a @reboot cron that fires twice, or a
                # service restarted by hand, must not try to double-start.
                print(f"  - {name}: already running; skipping.")
                skipped.append(name)
                continue
            ns = _launch_namespace(mem)
            ns.wait = args.wait if args.wait is not None else LOAD_TIMEOUT_SEC
            print(f"  - {name}: starting on port {ns.port or 'auto'} ...")
            try:
                cmd_start(ns)
                started.append(name)
            except SystemExit as exc:
                # One member failing must not abandon the rest: a reboot
                # restore is worth partially completing.
                print(f"    failed (exit {exc.code}).")
                failed.append(name)

        print(f"\n  Slot {slot}: {len(started)} started, {len(skipped)} "
              f"already running, {len(failed)} failed.")
        if failed:
            print(f"    failed: {', '.join(failed)}")
        print(f"\n  This restore as a command (safe to re-run; already-running "
              f"models are skipped):\n")
        print(f"    {_group_cli('restore', slot)}\n")
        if failed:
            sys.exit(1)
        return

    _die(f"unknown groups action {action!r}")


def cmd_settings(args):
    """Every source that could decide how this model runs, side by side.

    The question this answers is which numbers a launch would actually use,
    and where each one came from -- which matters because Unsloth's shipped
    family table and its own published guide disagree for the Qwen3.6/3.8
    families, and only the guide knows the model is in thinking mode.
    """
    m = _resolve_model(args.model)
    variant = (getattr(args, "variant", "") or "").strip()
    if m.is_gguf and not variant and m.variants:
        variant = ml.suggest_variant(m, _per_gpu_vram_gb(), _n_gpus())

    native = ml.native_context(m, variant) or up.doc_native_context(m.repo_id)
    print(f"\n  {m.repo_id}")
    print(f"    variant        : {variant or '(model default)'}"
          + (f"   {ml.human_size(m.variants[variant])}"
             if variant in m.variants else ""))
    print(f"    native context : {_fmt_ctx(native)}"
          f"   (a launch gets this capped to VRAM unless a profile pins one)")

    print("\n  Sampling\n")
    for line in _sampling_report(m):
        print(line)
    print(f"\n    Default for `start` is the published thinking profile. "
          f"Change it with\n    --sampling docs|unsloth|preset|none, or "
          f"--mode thinking|coding|instruct.")

    preset = _resolve_preset(None)
    print("\n  Load config\n")
    for line in _load_report(m, variant, preset):
        print(line)

    # Which quant a launch reads its profile from is decided by the variant,
    # so a tuned override on a quant you did not pick is invisible otherwise.
    tuned = [v for v in m.variant_names()
             if up.model_override(STUDIO_DB, m.repo_id, v).key]
    if tuned:
        print(f"      quants with a saved override: {', '.join(tuned)}"
              + ("" if variant in tuned else
                 f"   (this lookup used {variant or 'none'})"))

    running = _get_running().get(m.repo_id)
    if running:
        print(f"\n  Running on port {running['port']} — asking the server what "
              f"it actually loaded")
        if not _print_effective(running["port"], running.get("log", ""),
                                getattr(args, "api_key", "") or ""):
            print("    (no API key found in the log; cannot read it back)")
    print()


# =============================================================================
# TUI
# =============================================================================

def _tui_pick_model(stdscr, *, only_running=False, title="Select model"):
    import curses
    running = _get_running()
    if only_running:
        names = sorted(running)
        if not names:
            tui.pause(stdscr, "Nothing running. Press Enter ...")
            return None
        items = [(f"{n}   port {running[n]['port']}  "
                  f"[{_profile_label(running[n])}]", 0)
                 for n in names]
    else:
        models = ml.scan_models(HF_HOME)
        if not models:
            tui.pause(stdscr, f"No models in {ml.hub_dir(HF_HOME)}. Enter ...")
            return None
        names = [m.repo_id for m in models]
        items = []
        for m in models:
            live = m.repo_id in running
            label = (f"{'* ' if live else '  '}{m.repo_id}   "
                     f"{ml.human_size(m.size_bytes)}")
            items.append((label, curses.color_pair(_C_GREEN) if live else 0))

    idx = tui.select(stdscr, title, items)
    return names[idx] if idx >= 0 else None


def _tui_pick_variant(stdscr, repo_id: str):
    """Choose a GGUF quant. Returns "" for auto, a variant name, or None on
    cancel. Models with nothing to choose between skip straight through.

    Two things earn a column here. Fit, because the quant is the main VRAM
    lever on this box. And whether Unsloth has a saved profile for that quant,
    because overrides are keyed "<repo>:<variant>" -- picking a different quant
    silently picks a different profile, or none at all.
    """
    import curses
    m = ml.find_model(HF_HOME, repo_id)
    if m is None or not m.is_gguf or not m.variants:
        return ""

    per_gpu, n_gpus = _per_gpu_vram_gb(), _n_gpus()
    auto = ml.suggest_variant(m, per_gpu, n_gpus)
    names = m.variant_names()

    items = []
    for v in names:
        size = m.variants[v]
        note = ml.fit_note(size, per_gpu, n_gpus)
        tuned = "profile" if up.model_override(STUDIO_DB, repo_id, v).key else ""
        label = (f"  {v:<18}{ml.human_size(size):>8}  {note:<14}{tuned}")
        # Green for a quant Unsloth has a saved profile for: that is the one
        # whose settings this manager can actually reproduce.
        items.append((label, curses.color_pair(_C_GREEN) if tuned else 0))
    items.append((f"  auto  (best that fits: {auto or 'n/a'})", 0))

    header = [(f"{repo_id}", curses.A_BOLD),
              (f"{n_gpus} x {per_gpu}G  ·  'profile' = Unsloth has saved "
               f"settings for that quant", curses.color_pair(_C_DIM))]
    idx = tui.select(stdscr, "Quant", items, header=header)
    if idx < 0:
        return None
    return "" if idx == len(names) else names[idx]


def _planned_slots(repo_id: str, variant: str, load_profile: str,
                   preset: str) -> int:
    """Decode slots this launch will ask for, for the tools warning.

    Cheap re-derivation of what _apply_load_profile would settle on, so the
    warning can name a real number instead of "your slots". 4 is `unsloth
    studio run`'s own default when nothing supplies one.
    """
    if load_profile in ("auto", "override"):
        ov = up.model_override(STUDIO_DB, repo_id, variant)
        if ov.n_parallel:
            return ov.n_parallel
        if load_profile == "override":
            return 4
    if load_profile in ("auto", "preset"):
        presets = up.load_presets(STUDIO_DB)
        p = up.find_preset(presets, preset) if preset not in ("", "none") else None
        if p is None and preset in ("", "none"):
            p = presets.get(up.active_preset_name(STUDIO_DB))
        if p is not None and p.n_parallel:
            return p.n_parallel
    return 4


def _tui_pick_tools(stdscr, repo_id: str, variant: str, load_profile: str,
                    preset: str):
    """Server-side tools on or off. Returns True/False, or None on cancel.

    Asked explicitly rather than left at Unsloth's default because the default
    quietly costs every bit of the concurrency the slot count advertises --
    measured on this box, not inferred.
    """
    import curses
    slots = _planned_slots(repo_id, variant, load_profile, preset)
    warn = curses.color_pair(_C_YELLOW) | curses.A_BOLD
    dim = curses.color_pair(_C_DIM)
    header = [
        ("Unsloth's server-side tools SERIALISE /v1.", warn),
        (f"With them on, this model's {slots} decode slots run one request at "
         f"a time.", dim),
        ("", 0),
        ("Measured here, 3 concurrent requests:", dim),
        ("   Qwen3.8-27B   tools on 7.1 tok/s   ->  off 48.5 tok/s", dim),
        ("   Qwen3.6-35B   tools on 37.1 tok/s  ->  off 110.6 tok/s", dim),
        ("Single request is ~3.5x slower with tools on, too.", dim),
    ]
    items = [
        ("  Tools OFF — full concurrency, no server-side web/code  "
         "(recommended)", curses.color_pair(_C_GREEN)),
        ("  Tools ON  — server-side web search + code execution, "
         "serialised", 0),
    ]
    idx = tui.select(stdscr, "Server-side tools", items, header=header)
    if idx < 0:
        return None
    return idx == 1


def _tui_pick_preset(stdscr):
    """Choose a Studio preset, or none. Returns "none" to opt out, or None on cancel."""
    presets = up.load_presets(STUDIO_DB)
    active = up.active_preset_name(STUDIO_DB)
    if not presets:
        return "none"
    names = sorted(presets)
    items = []
    for n in names:
        p = presets[n]
        mark = "*" if n == active else " "
        items.append((f"{mark} {n:<22} ctx {str(p.effective_context or '-'):<8}"
                      f"spec {p.speculative_type or 'auto'}", 0))
    items.append(("  (no preset — Unsloth defaults)", 0))
    idx = tui.select(stdscr, "Preset  (* = active in the Studio UI)", items)
    if idx < 0:
        return None
    return "none" if idx == len(names) else names[idx]


# (load_profile, sampling, mode, label). The combinations worth offering
# without making the menu a settings editor; anything finer is a CLI job.
_TUI_PROFILES = (
    ("auto", "docs", "thinking",
     "Unsloth's per-model profile + published thinking sampling"),
    ("auto", "docs", "coding",
     "Unsloth's per-model profile + published coding sampling"),
    ("none", "docs", "thinking",
     "Fit-max context + published thinking sampling"),
    ("auto", "unsloth", "thinking",
     "Unsloth's per-model profile, pin no sampling"),
    ("none", "none", "thinking",
     "Unsloth defaults throughout — pin nothing at all"),
)


def _tui_pick_profile(stdscr):
    """Choose where settings come from. Returns (load, sampling, mode, preset)
    or None on cancel. The last entry drops through to the preset picker."""
    items = [(f"  {label}", 0) for _l, _s, _m, label in _TUI_PROFILES]
    items.append(("  Studio preset ...", 0))
    idx = tui.select(stdscr, "Settings profile", items)
    if idx < 0:
        return None
    if idx < len(_TUI_PROFILES):
        load, sampling, mode, _ = _TUI_PROFILES[idx]
        return load, sampling, mode, "none"
    preset = _tui_pick_preset(stdscr)
    if preset is None:
        return None
    return "preset", "preset", up.DEFAULT_DOC_MODE, preset


def _tui_act_start(stdscr):
    name = _tui_pick_model(stdscr, title="Start which model?")
    if not name:
        return
    # Before the profile, not after: the quant decides which per-model
    # override the launch will read, so choosing it second would mean choosing
    # a profile without knowing which one is on offer.
    variant = _tui_pick_variant(stdscr, name)
    if variant is None:
        return
    picked = _tui_pick_profile(stdscr)
    if picked is None:
        return
    load, sampling, mode, preset = picked
    # After the profile: the slot count the warning quotes comes from it.
    tools = _tui_pick_tools(stdscr, name, variant, load, preset)
    if tools is None:
        return
    port_s = tui.text(stdscr, f"Port (blank = auto, {PORT_POOL_LABEL}): ")
    port = int(port_s) if port_s.isdigit() else None
    gpus = tui.text(stdscr, "GPUs, e.g. 0 or 0,1 (blank = profile / auto): ")
    tui.run_cmd(stdscr, cmd_start, argparse.Namespace(
        model=name, preset=preset, port=port, gpus=gpus, variant=variant,
        load_profile=load, sampling=sampling, mode=mode, tools=tools,
        api_key="", dry_run=False, wait=None, force=False))


def _tui_act_settings(stdscr):
    name = _tui_pick_model(stdscr, title="Settings for which model?")
    if not name:
        return
    variant = _tui_pick_variant(stdscr, name)
    if variant is None:
        return
    tui.run_cmd(stdscr, cmd_settings, argparse.Namespace(
        model=name, variant=variant, api_key=""))


_EFFECTIVE_CACHE: dict[tuple, dict] = {}


def _effective_settings(name: str, entry: dict) -> dict:
    """What a running server is actually configured with, for display.

    Prefers what `start` recorded. Falls back to re-resolving the answers it
    stored, for a server launched before those fields existed -- cached per
    server run, since the header redraws constantly.

    A field that cannot be established is left absent rather than defaulted:
    printing "kv f16" for a server actually running q4_0 is worse than
    printing nothing at all.
    """
    if "pinned" in entry:
        return {"ctx": entry.get("ctx"), "kv": entry.get("kv"),
                "parallel": entry.get("parallel"), "spec": entry.get("spec"),
                "tools": entry.get("tools"), "vision": entry.get("vision"),
                "pinned": entry.get("pinned") or {}}

    key = (name, entry.get("started"), entry.get("pid"))
    if key in _EFFECTIVE_CACHE:
        return _EFFECTIVE_CACHE[key]

    # Only keys the entry actually carries: an absent "ctx" means we do not
    # know the context, which must not render as "fit" (a claim of its own).
    out = {k: entry[k] for k in ("ctx", "spec", "tools") if k in entry}
    args = entry.get("launch_args")
    if args:
        try:
            m = ml.find_model(HF_HOME, args.get("model") or name)
            if m is not None:
                lb = _resolve_launch(m, _launch_namespace(args))
                out.update(ctx=lb.ctx, kv=lb.kv_cache_dtype,
                           parallel=lb.parallel, spec=lb.spec_mode,
                           tools=lb.tools, vision=lb.vision,
                           pinned=dict(lb.sampling))
        except (SystemExit, OSError, ValueError, KeyError):
            # A model since deleted, or a preset since renamed. The header is
            # not the place to fail; show what the state file already knows.
            pass
    _EFFECTIVE_CACHE[key] = out
    return out


def _settings_segments(name: str, entry: dict, terse: bool = False,
                       width: int = 0) -> list:
    """Compact (text, attr) segments describing what a server is actually
    running, for the home screen.

    Only what is worth a glance: the knobs that decide throughput and output
    quality. Anything left at Unsloth's own default is drawn dim; anything
    this manager changed is drawn bright, so the line answers "what am I not
    running stock?" without having to know the defaults.

    `terse` drops everything still at a default, for when the screen is too
    short to give each model two lines.
    """
    import curses
    hot = curses.color_pair(_C_YELLOW)          # differs from Unsloth's default
    cool = curses.color_pair(_C_DIM)            # at the default
    ok = curses.color_pair(_C_GREEN)
    eff = _effective_settings(name, entry)
    segs: list = []

    parts: dict = {}

    def add(slot, text, attr, notable=True):
        if terse and not notable:
            return
        parts[slot] = (text + "  ", attr)

    # -- load knobs ----------------------------------------------------------
    ctx = int(eff.get("ctx") or 0)
    if ctx:
        add("ctx", f"{ctx // 1024}k" if ctx >= 1024 else str(ctx), hot)
    elif "ctx" in eff:
        add("ctx", "fit", cool, notable=False)

    kv = eff.get("kv")
    if kv and kv != "f16":
        add("kv", f"kv {kv}", hot)
    elif "kv" in eff:
        add("kv", "kv f16", cool, notable=False)

    par = eff.get("parallel")
    if par:
        add("sl", f"{par}sl", hot if par != 4 else cool, notable=par != 4)
    elif "parallel" in eff:
        add("sl", "4sl", cool, notable=False)

    spec = eff.get("spec") or ""
    if spec and spec != "auto":
        add("spec", spec, hot)

    # Tools ON is the notable state here, not off: it silently serialises /v1.
    tools = eff.get("tools")
    if tools is False:
        add("tools", "tools off", ok)
    else:
        add("tools", "tools ON", hot)

    if eff.get("vision") is False:
        add("vis", "no-vis", hot)

    # -- sampling ------------------------------------------------------------
    if "pinned" in eff:
        pinned = eff.get("pinned") or {}
        if not pinned:
            add("samp", "| sampling: unsloth", cool, notable=False)
        else:
            rec = up.unsloth_defaults(STUDIO_HOME, name).applied
            labels = {"temperature": "T", "top_p": "P", "top_k": "K",
                      "min_p": "M", "presence_penalty": "pres",
                      "repetition_penalty": "rep"}
            bits: list = []
            for key in up.SAMPLING_FIELDS:
                val = pinned.get(key)
                if val is None:
                    continue
                shown = int(val) if key in up.INT_FIELDS else val
                # Bright when the pin actually changes what the server would
                # have done on its own; dim when we pinned the same number.
                differs = key not in rec or rec[key] != val
                if terse and not differs:
                    continue
                bits.append((f"{labels[key]}{shown} ", hot if differs else cool))
            if bits:
                parts["samp"] = [("| ", cool)] + bits

    # Truncation is what decides what a narrow line shows, so terse leads with
    # the settings that cost the most when wrong: tools serialises /v1, and
    # the sampling pins change what the model writes.
    # Logical order reads best; importance order only earns its keep when the
    # line is going to be cut, so choose by whether it actually fits.
    logical = ("ctx", "kv", "sl", "spec", "tools", "vis", "samp")
    by_impact = ("tools", "samp", "kv", "sl", "ctx", "spec", "vis")
    def _group(slot):
        val = parts.get(slot)
        return [] if val is None else (val if isinstance(val, list) else [val])

    total = sum(len(t) for slot in logical for t, _ in _group(slot))
    order = by_impact if (width and total > width) else logical
    used = 0
    for slot in order:
        group = _group(slot)
        if not group:
            continue
        # Fit whole values, never partial ones -- "pres0." is worse than no
        # presence penalty shown. The sampling run is a list of independent
        # values, so it may fill the remaining space partially; a single
        # segment is all-or-nothing.
        size = sum(len(t) for t, _ in group)
        if not width or used + size <= width:
            used += size
            segs.extend(group)
            continue
        if len(group) > 1:
            taken = []
            for text, attr in group:
                if used + len(text) > width:
                    break
                used += len(text)
                taken.append((text, attr))
            # A lone "| " prefix with no values after it says nothing.
            if len(taken) > 1:
                segs.extend(taken)
        segs.append(("…", curses.color_pair(_C_DIM)))
        break
    return segs


def _tui_group_items():
    """(label, attr) per slot, for the group menu."""
    import curses
    slots = _load_groups()
    items = []
    for n in range(1, GROUP_SLOTS + 1):
        g = slots.get(str(n))
        if not g:
            items.append((f"  {n:>2}. (empty)", curses.color_pair(_C_DIM)))
            continue
        when = time.strftime("%m-%d %H:%M", time.localtime(g.get("saved", 0)))
        members = g.get("members", [])
        ports = ",".join(str(m.get("port")) for m in members if m.get("port"))
        items.append((f"  {n:>2}. {(g.get('name') or 'unnamed')[:24]:<26}"
                      f"{len(members)} model(s)  :{ports:<14}{when}",
                      curses.color_pair(_C_GREEN)))
    return items


def _tui_act_groups(stdscr):
    """Instance groups: snapshot what is running, restore it in one action."""
    import curses
    while True:
        actions = [("  Restore a group  (start everything it holds)", 0),
                   ("  Save the running servers to a slot", 0),
                   ("  Show a group's saved answers", 0),
                   ("  Clear a slot", 0)]
        header = [("A group remembers every answer a launch was given —",
                   curses.color_pair(_C_DIM)),
                  ("model, quant, port, GPUs, profile, sampling and tools —",
                   curses.color_pair(_C_DIM)),
                  ("so a reboot can be undone with one command.",
                   curses.color_pair(_C_DIM))]
        choice = tui.select(stdscr, "Instance groups", actions, header=header)
        if choice < 0:
            return

        titles = ("Restore which group?", "Save to which slot?",
                  "Show which group?", "Clear which slot?")
        idx = tui.select(stdscr, titles[choice], _tui_group_items())
        if idx < 0:
            continue
        slot = idx + 1

        if choice == 0:
            tui.run_cmd(stdscr, cmd_groups, argparse.Namespace(
                action="restore", slot=slot, name="", wait=None))
        elif choice == 1:
            name = tui.text(stdscr, f"Name for slot {slot} (blank = default): ")
            tui.run_cmd(stdscr, cmd_groups, argparse.Namespace(
                action="save", slot=slot, name=name, wait=None))
        elif choice == 2:
            tui.run_cmd(stdscr, cmd_groups, argparse.Namespace(
                action="show", slot=slot, name="", wait=None))
        else:
            tui.run_cmd(stdscr, cmd_groups, argparse.Namespace(
                action="clear", slot=slot, name="", wait=None))


def _tui_act_stop(stdscr):
    name = _tui_pick_model(stdscr, only_running=True, title="Stop which model?")
    if not name:
        return
    tui.run_cmd(stdscr, cmd_stop, argparse.Namespace(model=name))


def _tui_act_status(stdscr):
    tui.run_cmd(stdscr, cmd_status, argparse.Namespace())


def _tui_act_list(stdscr):
    tui.run_cmd(stdscr, cmd_list, argparse.Namespace(variants=True))


def _tui_act_test(stdscr):
    name = _tui_pick_model(stdscr, only_running=True, title="Test which model?")
    if not name:
        return
    key = tui.text(stdscr, "API key (blank = from the server log): ")
    tui.run_cmd(stdscr, cmd_test, argparse.Namespace(
        model=name, api_key=key, prompt=None, max_tokens=256))


def _tui_act_benchmark(stdscr):
    name = _tui_pick_model(stdscr, title="Benchmark which model?")
    if not name:
        return
    # Quant before presets, as in the start flow: it belongs to the model, and
    # every preset in the run is measured against the one quant.
    variant = _tui_pick_variant(stdscr, name)
    if variant is None:
        return
    presets = up.load_presets(STUDIO_DB)
    chosen = None
    if presets:
        names = sorted(presets)
        picks = tui.select(stdscr, "Presets to benchmark",
                           [(n, 0) for n in names], multi=True)
        if not picks:
            return
        chosen = [names[i] for i in picks]
    key = tui.text(stdscr, "API key (blank = from the server log): ")
    tui.run_cmd(stdscr, cmd_benchmark, argparse.Namespace(
        model=name, preset=chosen, api_key=key, gpus="", variant=variant,
        max_tokens=BENCH_MAX_TOKENS))


def _tui_act_logs(stdscr):
    name = _tui_pick_model(stdscr, title="Logs for which model?")
    if not name:
        return
    tui.run_cmd(stdscr, cmd_logs, argparse.Namespace(
        model=name, lines=40, follow=False))


def _tui_act_presets(stdscr):
    tui.run_cmd(stdscr, cmd_presets, argparse.Namespace())


def _tui_act_doctor(stdscr):
    tui.run_cmd(stdscr, cmd_doctor, argparse.Namespace())


def _tui_act_api_tester(stdscr):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "api_tester.sh")
    if not os.path.exists(script):
        tui.pause(stdscr, "api_tester.sh not found. Enter ...")
        return
    tui.shell_out(stdscr, ["bash", script])


def _tui_main(stdscr):
    import curses

    curses.curs_set(0)
    stdscr.keypad(True)
    tui.init_colors()

    if tui.too_small(stdscr, min_h=10, min_w=50):
        curses.endwin()
        print("Terminal too small (need at least 50x10).")
        print("Use CLI commands instead: python unsloth_manager.py --help")
        return

    # How many menu rows must stay visible whatever else is on screen.
    _MENU_MIN_ROWS = 6

    MENU_ACTIONS = [
        ("Start Model",     _tui_act_start),
        ("Instance Groups", _tui_act_groups),
        ("Model Settings",  _tui_act_settings),
        ("Stop Model",      _tui_act_stop),
        ("Status",          _tui_act_status),
        ("List Models",     _tui_act_list),
        ("Test Model",      _tui_act_test),
        ("Benchmark",       _tui_act_benchmark),
        ("API Tester",      _tui_act_api_tester),
        ("View Logs",       _tui_act_logs),
        ("Studio Presets",  _tui_act_presets),
        ("Environment Check", _tui_act_doctor),
        ("Quit",            None),
    ]

    refresh_ms = 5000

    def _build_header():
        running = _get_running()
        header = []
        ts = time.strftime("%H:%M:%S")
        every = f"{refresh_ms // 1000}s"

        if running:
            n_ready = sum(1 for e in running.values()
                          if _probe_ready(int(e["port"]),
                                          log_path=e.get("log", "")))
            header.append((
                f"Running: {len(running)} model(s), {n_ready} ready"
                f"     ⏱ {ts} · refresh {every}",
                curses.A_BOLD | curses.color_pair(_C_GREEN)))
            # Two lines per model reads far better, but the menu below must
            # still fit: on a short terminal collapse to one line and drop
            # every value that is still at a default.
            # Budget rows before drawing: the menu is the point of this
            # screen, so a full complement of servers must not push it off.
            rows, cols = stdscr.getmaxyx()
            gpu_rows = (len(_gpus()) + 1) if _gpus() else 1
            # title, blanks, section headers, gpu bars, legend, ram, separator
            chrome = 11 + gpu_rows
            avail = max(1, rows - chrome - _MENU_MIN_ROWS)
            two_line = len(running) * 2 <= avail
            # Narrow terminals drop the values that are already at a default
            # too, so what survives truncation is what actually differs.
            terse = not two_line or cols < 100
            per = 2 if two_line else 1
            shown = sorted(running.items(), key=lambda x: x[1]["port"])
            hidden = max(0, len(shown) - max(1, avail // per))
            if hidden:
                shown = shown[:max(1, avail // per) - 1]
                hidden = len(running) - len(shown)
            # Size the two identity columns to what is actually on screen, so
            # short names leave more room for the settings that follow.
            def _short(n):
                return n.split("/")[-1]

            def _quant(e):
                q = e.get("variant") or "-"
                # A quant can be "dir/TAG" for a quant-per-folder repo; the tag
                # is the identifying half, so keep that end when it is long.
                return q[-12:] if len(q) > 12 else q

            namew = min(30, max(14, max((len(_short(n)) for n, _ in shown),
                                        default=14)))
            quantw = min(12, max(4, max((len(_quant(e)) for _, e in shown),
                                        default=4)))
            for name, entry in shown:
                ready = _probe_ready(int(entry["port"]),
                                     log_path=entry.get("log", ""))
                dattr = (curses.color_pair(_C_GREEN if ready else _C_CYAN)
                         | curses.A_BOLD)
                short = _short(name)
                quant = _quant(entry)
                line = [
                    ("  ", 0),
                    ("●", dattr),
                    (f" {short[:namew]:<{namew}} ", 0),
                    (f"{quant:<{quantw}} ", curses.color_pair(_C_CYAN)),
                    (f":{entry['port']} ", curses.color_pair(_C_CYAN)),
                    (f"{'rdy' if ready else 'load'} ", dattr),
                    (f"g{entry.get('gpus') or '-':<3} ",
                     curses.color_pair(_C_DIM)),
                ]
                if two_line:
                    header.append(line)
                    header.append([("      ", 0)]
                                  + _settings_segments(name, entry, terse,
                                                       cols - 8))
                else:
                    prefix = sum(len(t) for t, _ in line)
                    header.append(line + [("  ", 0)]
                                  + _settings_segments(name, entry, True,
                                                       cols - prefix - 5))
            if hidden:
                header.append((f"    … {hidden} more — Status lists them all",
                               curses.color_pair(_C_DIM)))
        else:
            header.append((
                f"Running: none     ⏱ {ts} · refresh {every}",
                curses.A_DIM))

        header.append(("", 0))
        gsplit = _gpu_usage_split()
        if gsplit:
            header.append(("GPU memory (live, actual):", curses.A_BOLD))
            for g in gsplit:
                header.append(_gpu_bar_line(g))
            header.append([
                ("    ", 0),
                ("█", curses.color_pair(_C_GREEN)),
                (" unsloth   ", curses.color_pair(_C_DIM)),
                ("█", curses.color_pair(_C_YELLOW)),
                (" other   ", curses.color_pair(_C_DIM)),
                ("░", curses.color_pair(_C_DIM)),
                (" free", curses.color_pair(_C_DIM)),
            ])
        else:
            header.append(("GPU memory: nvidia-smi unavailable", curses.A_DIM))

        tot, avail = _read_mem_gb()
        if tot:
            header.append(_ram_bar_line(tot - avail, tot))
        return header

    while True:
        header = _build_header()
        menu_items = [(label, 0) for label, _ in MENU_ACTIONS]
        menu_items[-1] = ("Quit", curses.A_DIM)

        def _refresh():
            return _build_header(), menu_items

        idx = tui.select(stdscr, "Unsloth Service Manager", menu_items,
                         header=header, refresh_cb=_refresh,
                         refresh_ms=refresh_ms)

        if idx == -1 or idx == len(MENU_ACTIONS) - 1:
            break
        action = MENU_ACTIONS[idx][1]
        if action:
            action(stdscr)


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Unsloth Studio service manager — serve, manage, benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Servers run as the {RUN_USER!r} user; this manager must run as root.
Run with no arguments for the interactive TUI.

The API lives on ports {PORT_POOL_LABEL}: at most {MAX_INSTANCES} servers,
one per port, and nothing ever binds outside that range.

Settings come from Unsloth, not from here. `settings <model>` prints all
four sources side by side: Unsloth's published profile (the default for
sampling), Unsloth's shipped family defaults, its per-model load override,
and the Studio preset. Nothing is pinned that you did not ask for.

Unsloth requires a bearer token on /v1. This tool does not create or store
keys: one is minted per launch and recorded in the server log ("API Key:"),
which is where `test`, `benchmark` and the settings readout find it.
Override with --api-key or UNSLOTH_MGR_API_KEY.
""")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("list", help="models in the HF cache + status")
    sp.add_argument("--variants", action="store_true",
                    help="list GGUF quants and their VRAM fit")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("presets", help="Unsloth Studio presets from studio.db")
    sp.set_defaults(func=cmd_presets)

    sp = sub.add_parser("groups",
                        help="save/restore a named set of running servers")
    sp.add_argument("action", nargs="?", default="list",
                    choices=("list", "save", "restore", "show", "clear"),
                    help="default: list")
    sp.add_argument("slot", nargs="?", type=int, default=0,
                    help=f"slot number, 1-{GROUP_SLOTS}")
    sp.add_argument("--name", default="",
                    help="label for the saved group")
    sp.add_argument("--wait", type=int,
                    help="seconds to wait per model on restore "
                         "(0 = don't wait)")
    sp.set_defaults(func=cmd_groups)

    sp = sub.add_parser("settings",
                        help="what every source says about a model")
    sp.add_argument("model")
    sp.add_argument("--variant", default="",
                    help="GGUF quant to look up (default: best that fits)")
    sp.add_argument("--api-key", default="",
                    help="read live settings back from a running server")
    sp.set_defaults(func=cmd_settings)

    sp = sub.add_parser("start", help="start a server")
    sp.add_argument("model")
    sp.add_argument("--port", type=int,
                    help=f"API port, {PORT_POOL_LABEL} "
                         f"(default: lowest free in the pool)")
    sp.add_argument("--gpus", default="",
                    help="CUDA devices, e.g. 0 or 0,1 "
                         "(default: the profile's GPU pin, else Unsloth chooses)")
    sp.add_argument("--variant",
                    help="GGUF quant (default: best that fits)")

    g = sp.add_argument_group("profiles")
    g.add_argument("--sampling", default="docs",
                   choices=("docs", "unsloth", "preset", "none"),
                   help="where sampling comes from. docs = Unsloth's published "
                        "profile (default); unsloth/none = pin nothing and let "
                        "the server apply its own per-model recommendation")
    g.add_argument("--mode", default=up.DEFAULT_DOC_MODE,
                   choices=("thinking", "coding", "instruct"),
                   help="which published profile, when --sampling docs "
                        "(default: thinking)")
    g.add_argument("--load-profile", dest="load_profile", default="auto",
                   choices=("auto", "override", "preset", "none"),
                   help="where the load config comes from. auto = Unsloth's "
                        "per-model override if one exists, else the Studio "
                        "preset (default)")
    g.add_argument("--preset",
                   help='Studio preset name; "none" to skip. '
                        "Default: whatever the Studio UI has selected.")

    g = sp.add_argument_group("load overrides (beat any profile)")
    g.add_argument("--ctx", type=int,
                   help="context length; 0 asks for the largest that fits")
    g.add_argument("--parallel", type=int,
                   help="decode slots; they share one KV pool (--kv-unified), "
                        "so this is not a context divider")
    g.add_argument("--tensor-parallel", dest="tensor_parallel",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="split by tensor instead of by layer")
    g.add_argument("--kv-cache-dtype", dest="kv_cache_dtype",
                   help=f"one of {', '.join(up.VALID_KV_DTYPES)}")
    g.add_argument("--spec", help=f"speculative decoding: "
                                  f"{'|'.join(up.SPEC_MODES)}")
    g.add_argument("--spec-draft-n-max", dest="spec_draft_n_max", type=int,
                   help="draft tokens per step (1-16); MTP/DSpark/DFlash only")
    g.add_argument("--vision", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="load the companion mmproj (--no-vision frees ~0.9G "
                        "on a text-only server)")
    g.add_argument("--n-batch", dest="n_batch", type=int,
                   help="llama-server --batch-size (logical batch)")
    g.add_argument("--n-ubatch", dest="n_ubatch", type=int,
                   help="llama-server --ubatch-size (physical batch)")

    g = sp.add_argument_group("sampling pins (beat any profile)")
    for name in up.SAMPLING_FIELDS:
        lo, hi = up.SAMPLING_RANGES[name]
        g.add_argument("--" + name.replace("_", "-"), dest=name,
                       type=int if name in up.INT_FIELDS else float,
                       help=f"{lo}..{hi}")

    sp.add_argument("--tools", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Unsloth's server-side web/code tools (default: on)")
    sp.add_argument("--extra", action="append", metavar="ARG",
                    help="repeatable raw llama-server arg, e.g. --extra=-ngl "
                         "--extra=99")
    sp.add_argument("--api-key", default="",
                    help="key for the post-load readout (default: from the log)")
    sp.add_argument("--wait", type=int,
                    help="seconds to wait for readiness (0 = don't wait)")
    sp.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="print the plan and the command line, start nothing")
    sp.add_argument("--force", action="store_true",
                    help="start despite a VRAM warning or a running instance")
    sp.set_defaults(func=cmd_start)

    sp = sub.add_parser("stop", help="stop a server")
    sp.add_argument("model")
    sp.set_defaults(func=cmd_stop)

    sp = sub.add_parser("stop-all", help="stop every managed server")
    sp.set_defaults(func=cmd_stop_all)

    sp = sub.add_parser("restart",
                        help="stop then start, keeping port/GPUs/profile")
    sp.add_argument("model")
    sp.add_argument("--preset", help="Studio preset (default: keep)")
    sp.add_argument("--port", type=int,
                    help=f"API port, {PORT_POOL_LABEL} (default: keep)")
    sp.add_argument("--gpus", default="", help="CUDA devices (default: keep)")
    sp.add_argument("--variant", default="",
                    help="GGUF quant (default: keep)")
    sp.add_argument("--tools", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="server-side tools (default: keep what it was "
                         "started with)")
    sp.add_argument("--wait", type=int,
                    help="seconds to wait for readiness (0 = don't wait)")
    sp.set_defaults(func=cmd_restart)

    sp = sub.add_parser("status", help="running servers + GPU usage")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("logs", help="show a server log")
    sp.add_argument("model")
    sp.add_argument("-n", "--lines", type=int, default=40,
                    help="lines to show (default: 40)")
    sp.add_argument("-f", "--follow", action="store_true",
                    help="follow the log as it grows")
    sp.set_defaults(func=cmd_logs)

    sp = sub.add_parser("test", help="one streaming prompt + tok/s")
    sp.add_argument("model")
    sp.add_argument("--api-key", default="",
                    help="bearer token (default: from the server log)")
    sp.add_argument("--prompt", help="prompt to send (default: a short one)")
    sp.add_argument("--max-tokens", type=int, default=256,
                    help="generation cap (default: 256)")
    sp.set_defaults(func=cmd_test)

    sp = sub.add_parser("benchmark",
                        help="measure steady-state tok/s per preset")
    sp.add_argument("model")
    sp.add_argument("--preset", action="append",
                    help="repeatable; default is every preset in studio.db")
    sp.add_argument("--gpus", default="", help="CUDA devices for every run")
    sp.add_argument("--variant", default="",
                    help="GGUF quant (default: best that fits)")
    sp.add_argument("--api-key", default="",
                    help="bearer token (default: from the server log)")
    sp.add_argument("--max-tokens", type=int, default=BENCH_MAX_TOKENS,
                    help=f"generation cap per run (default: {BENCH_MAX_TOKENS})")
    sp.set_defaults(func=cmd_benchmark)

    sp = sub.add_parser("doctor", help="check the host can serve")
    sp.set_defaults(func=cmd_doctor)

    args = p.parse_args()

    if getattr(args, "cmd", "") == "groups" and args.action != "list" \
            and not args.slot:
        p.error(f"groups {args.action} needs a slot number (1-{GROUP_SLOTS})")

    if not args.cmd:
        tui.launch(_tui_main)
        return

    args.func(args)


if __name__ == "__main__":
    main()
