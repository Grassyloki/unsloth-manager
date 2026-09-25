"""
unsloth_profiles.py — every source of "how should this model be run".

Four sources feed a launch, none of which this module writes to:

  1. DOC PROFILES (this file). Sampling settings transcribed from Unsloth's
     own published model guides. These exist because Unsloth's *shipped*
     defaults disagree with its *documented* ones: the family table gives
     Qwen3.6/3.8 the Instruct row (temp 0.7, top_p 0.8, presence 1.5) while
     the guides say a thinking-mode model wants temp 1.0, top_p 0.95,
     presence 0.0. Serving a thinking model on the Instruct row is the bug
     this table fixes.

  2. UNSLOTH FAMILY DEFAULTS. assets/configs/inference_defaults.json inside
     the Studio venv, plus any model-specific YAML, plus default.yaml —
     resolved exactly the way `load_inference_config` does. This is what the
     server applies on its own to any field a request omits.

  3. STUDIO PRESETS. chat_settings.customPresets in studio.db: what the chat
     UI's Preset dropdown saves. Global — the table has no model column — and
     carrying no model, port or GPU.

  4. PER-MODEL OVERRIDES. app_settings.openai_api_auto_switch_overrides in
     studio.db, keyed "<repo>:<variant>": the load config the Studio settings
     page saves per model (context, KV dtype, speculative mode, slots, GPU
     pin, vision). Unsloth applies these ONLY when a /v1 request auto-switches
     models — never on the direct /api/inference/load that `unsloth studio
     run` performs — so a headless launch has to read them itself.

Everything here READS. Studio owns both files, and every read is a read-only
SQLite connection. The one write this project makes -- an explicit "save to
profile" from `start` -- is not done here and never touches the database
directly: the manager sends it through Studio's own settings API, so Studio's
validation, locking and cache invalidation all apply. What lives here is only
the knowledge of Studio's field names that the payloads need (WRITING BACK,
at the end).
"""
from __future__ import annotations

import copy
import glob
import json
import os
import sqlite3
from dataclasses import dataclass, field

# =============================================================================
# SAMPLING VOCABULARY
# =============================================================================

# Our field names, which are also `unsloth studio run`'s flag names with the
# underscores swapped for dashes.
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p",
                   "presence_penalty", "repetition_penalty")

# The five Unsloth's server auto-applies from its own recommendation. It
# deliberately leaves repetition_penalty alone (manual-only, matching the chat
# UI, which never auto-fills it), so a family table's 1.0 never actually
# reaches llama-server unless something pins it.
UNSLOTH_AUTO_FIELDS = ("temperature", "top_p", "top_k", "min_p",
                       "presence_penalty")

# Static schema defaults from ChatCompletionRequest, used for any field with no
# pin and no recommendation. Kept here so a readout can name the number that
# actually applies rather than printing a blank.
SCHEMA_DEFAULTS = {
    "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.01,
    "repetition_penalty": 1.0, "presence_penalty": 0.0,
}

# Ranges `unsloth studio run` accepts. A value outside these is rejected by the
# CLI, so catching it here turns a launch failure into a message.
SAMPLING_RANGES = {
    "temperature": (0.0, 2.0), "top_p": (0.0, 1.0), "top_k": (-1, 100),
    "min_p": (0.0, 1.0), "repetition_penalty": (1.0, 2.0),
    "presence_penalty": (0.0, 2.0),
}

INT_FIELDS = ("top_k",)


def fmt_sampling(values: dict) -> str:
    """One-line rendering in the field order above, skipping absent fields."""
    bits = []
    for key in SAMPLING_FIELDS:
        val = values.get(key)
        if val is None:
            continue
        short = {"temperature": "temp", "presence_penalty": "pres",
                 "repetition_penalty": "rep"}.get(key, key)
        bits.append(f"{short} {int(val) if key in INT_FIELDS else val}")
    return ", ".join(bits)


# =============================================================================
# 1. DOC PROFILES — transcribed from Unsloth's published guides
# =============================================================================
# Sourced by hand from the pages named in `url`, read 2026-08-26. Update them
# together: the url is what makes a stale number checkable.
#
# Patterns match the way Unsloth's own family lookup does — the repo id
# lowercased with the org stripped — and are tried longest-first, so "qwen3.8"
# is consulted before a hypothetical "qwen3".


@dataclass(frozen=True)
class DocProfile:
    pattern: str                 # substring of the normalised repo id
    mode: str                    # thinking | coding | instruct
    label: str
    values: dict
    url: str
    note: str = ""

    def describe(self) -> str:
        return f"{self.label} ({self.mode})"


_Q3_THINKING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                    presence_penalty=0.0, repetition_penalty=1.0)
_Q3_CODING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                  presence_penalty=0.0, repetition_penalty=1.0)
_Q3_INSTRUCT = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
                    presence_penalty=1.5, repetition_penalty=1.0)

_Q36_URL = "https://unsloth.ai/docs/models/qwen3.6"
_Q38_URL = "https://unsloth.ai/docs/models/qwen3.8"

DOC_PROFILES: tuple[DocProfile, ...] = (
    DocProfile("qwen3.8", "thinking", "Qwen3.8", _Q3_THINKING, _Q38_URL),
    DocProfile("qwen3.8", "instruct", "Qwen3.8", _Q3_INSTRUCT, _Q38_URL,
               "non-thinking; the model runs in thinking mode by default"),
    DocProfile("qwen3.6", "thinking", "Qwen3.6", _Q3_THINKING, _Q36_URL,
               "general tasks"),
    DocProfile("qwen3.6", "coding", "Qwen3.6", _Q3_CODING, _Q36_URL,
               "precise coding tasks"),
    DocProfile("qwen3.6", "instruct", "Qwen3.6", _Q3_INSTRUCT, _Q36_URL,
               "non-thinking; the model runs in thinking mode by default"),
)

DEFAULT_DOC_MODE = "thinking"

# Native context both guides publish for these families. Only used to say so in
# a readout — what a launch actually gets is read from the GGUF header and then
# capped by Unsloth's own fitter.
DOC_NATIVE_CONTEXT = {"qwen3.6": 262144, "qwen3.8": 262144}


def normalise_model_id(model_id: str) -> str:
    """Lowercase, org stripped — the same normalisation Unsloth's family
    lookup applies before matching patterns."""
    low = (model_id or "").lower()
    return low.split("/", 1)[1] if "/" in low else low


def doc_profile(model_id: str, mode: str = DEFAULT_DOC_MODE) -> DocProfile | None:
    """The published profile for this model in `mode`, or None if we have no
    entry for the family (in which case nothing should be pinned)."""
    norm = normalise_model_id(model_id)
    want = (mode or DEFAULT_DOC_MODE).strip().lower()
    for p in sorted(DOC_PROFILES, key=lambda p: -len(p.pattern)):
        if p.pattern in norm and p.mode == want:
            return p
    return None


def doc_modes(model_id: str) -> list[str]:
    """Modes we have a published profile for, best-known first."""
    norm = normalise_model_id(model_id)
    hit = ""
    for p in sorted(DOC_PROFILES, key=lambda p: -len(p.pattern)):
        if p.pattern in norm:
            hit = p.pattern
            break
    if not hit:
        return []
    order = {"thinking": 0, "coding": 1, "instruct": 2}
    modes = [p.mode for p in DOC_PROFILES if p.pattern == hit]
    return sorted(set(modes), key=lambda m: order.get(m, 9))


def doc_native_context(model_id: str) -> int | None:
    norm = normalise_model_id(model_id)
    for pattern in sorted(DOC_NATIVE_CONTEXT, key=len, reverse=True):
        if pattern in norm:
            return DOC_NATIVE_CONTEXT[pattern]
    return None


# =============================================================================
# 2. UNSLOTH'S OWN FAMILY / MODEL DEFAULTS
# =============================================================================
# These live inside the Studio venv, not in studio.db. Reading them is what
# lets a readout say "Unsloth would have applied X" next to what we pinned.

_ASSETS_ENV = "UNSLOTH_MGR_STUDIO_ASSETS"
_assets_cache: dict[str, str] = {}


def studio_assets_dir(studio_home: str) -> str:
    """assets/configs inside the Studio venv, or "" when it cannot be found.

    Globbed on the python version because the venv is built against whatever
    interpreter Unsloth installed with, which is not this process's.
    """
    override = os.environ.get(_ASSETS_ENV, "").strip()
    if override:
        return override if os.path.isdir(override) else ""
    if studio_home in _assets_cache:
        return _assets_cache[studio_home]
    pattern = os.path.join(
        studio_home, "unsloth_studio", "lib", "python*", "site-packages",
        "studio", "backend", "assets", "configs")
    hits = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    found = hits[-1] if hits else ""
    _assets_cache[studio_home] = found
    return found


def _load_family_table(assets: str) -> tuple[list[str], dict]:
    path = os.path.join(assets, "inference_defaults.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return [], {}
    patterns = data.get("patterns")
    families = data.get("families")
    if not isinstance(patterns, list) or not isinstance(families, dict):
        return [], {}
    return [p for p in patterns if isinstance(p, str)], families


def unsloth_family_params(studio_home: str, model_id: str) -> dict:
    """Unsloth's family recommendation for this model, or {}.

    The JSON's `patterns` list is already ordered longest-match-first, and the
    first hit wins — matching get_family_inference_params exactly, so this
    cannot disagree with what the server will do.
    """
    assets = studio_assets_dir(studio_home)
    if not assets:
        return {}
    patterns, families = _load_family_table(assets)
    norm = normalise_model_id(model_id)
    for pattern in patterns:
        if pattern in norm:
            params = families.get(pattern)
            if params:
                return dict(params)
    return {}


def _load_yaml_inference(path: str) -> dict:
    """The `inference:` block of a Studio YAML, or {} when PyYAML is absent.

    Optional on purpose: this manager runs on the system interpreter with no
    third-party packages, and every model that matters here resolves through
    the JSON family table instead. A missing parser degrades the readout, not
    the launch.
    """
    try:
        import yaml
    except ImportError:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    block = data.get("inference") if isinstance(data, dict) else None
    return dict(block) if isinstance(block, dict) else {}


def _model_specific_yaml(assets: str, model_id: str) -> str:
    """Path of a model's own YAML under model_defaults/, or ""."""
    root = os.path.join(assets, "model_defaults")
    if not os.path.isdir(root):
        return ""
    names = {model_id, model_id.split("/")[-1]}
    for name in names:
        if not name:
            continue
        hits = glob.glob(os.path.join(root, "**", name.replace("/", "_") + ".yaml"),
                         recursive=True)
        if hits:
            return hits[0]
    return ""


@dataclass
class UnslothDefaults:
    """What the server will apply to a request that omits a sampling field."""
    values: dict = field(default_factory=dict)
    source: str = "none"          # model-yaml | family | default.yaml | none
    detail: str = ""

    @property
    def applied(self) -> dict:
        """Only the five the server actually auto-applies."""
        return {k: v for k, v in self.values.items() if k in UNSLOTH_AUTO_FIELDS}


_defaults_cache: dict[tuple[str, str], "UnslothDefaults"] = {}


def unsloth_defaults(studio_home: str, model_id: str) -> UnslothDefaults:
    """Cached wrapper: the TUI header compares against this on every redraw,
    and the tables it reads only change when Studio is reinstalled."""
    key = (studio_home, model_id)
    if key not in _defaults_cache:
        _defaults_cache[key] = _unsloth_defaults_uncached(studio_home, model_id)
    return _defaults_cache[key]


def _unsloth_defaults_uncached(studio_home: str, model_id: str) -> UnslothDefaults:
    """Resolve sampling the way load_inference_config does.

    Priority: the model's own YAML, then the family table, then default.yaml.
    Reported alongside the source so a readout can say where a surprising
    number came from.
    """
    assets = studio_assets_dir(studio_home)
    if not assets:
        return UnslothDefaults(detail="Studio assets directory not found")

    own_yaml = _model_specific_yaml(assets, model_id)
    own = _load_yaml_inference(own_yaml) if own_yaml else {}
    family = unsloth_family_params(studio_home, model_id)
    fallback = _load_yaml_inference(
        os.path.join(assets, "model_defaults", "default.yaml"))

    values: dict = {}
    for key in SAMPLING_FIELDS:
        if own_yaml and isinstance(own.get(key), (int, float)):
            values[key] = own[key]
        elif key in family:
            values[key] = family[key]
        elif key in fallback:
            values[key] = fallback[key]

    if own_yaml and own:
        return UnslothDefaults(values, "model-yaml", os.path.basename(own_yaml))
    if family:
        norm = normalise_model_id(model_id)
        patterns, _ = _load_family_table(assets)
        hit = next((p for p in patterns if p in norm), "")
        return UnslothDefaults(values, "family", hit)
    if values:
        return UnslothDefaults(values, "default.yaml", "no family match")
    return UnslothDefaults(detail="no defaults readable")


# =============================================================================
# SPECULATIVE DECODING
# =============================================================================
# `unsloth studio run --speculative-type` takes Unsloth's OWN vocabulary and
# does the llama-server translation itself, so nothing here maps onto
# llama-server's --spec-type tokens. Doing that by hand is actively harmful:
# supplying any --spec-type suppresses Unsloth's capability-aware auto-emit, so
# a token this build does not know silently disables speculative decoding.

SPEC_MODES = ("auto", "mtp", "dspark", "dflash", "ngram", "mtp+ngram", "off",
              "ngram-simple")

# Modes that launch a drafter with a configurable depth. --spec-draft-n-max is
# ignored for anything else, so emitting it there only misleads.
DRAFT_N_MAX_MODES = ("mtp", "dspark", "dflash")

# Legacy spellings the UI used to store, plus llama-server's own token names,
# which an older studio.db can still hold.
_LEGACY_SPEC = {
    "default": "auto", "draft-mtp": "mtp", "ngram-mod": "ngram",
    "draft-dspark": "dspark", "draft-dflash": "dflash", "none": "off",
}


def parse_spec_mode(value: str) -> str | None:
    """Strict version of :func:`canonical_spec_mode`, for user input.

    Reading a stored value leniently is right -- a typo in studio.db should
    land on Unsloth's safe auto path rather than fail a launch. Reading a
    command-line flag leniently is not: silently turning `--spec bogus` into
    "auto" hides the typo behind a launch that looks like it worked.
    """
    v = (value or "").strip().lower()
    if v in SPEC_MODES:
        return v
    if v in _LEGACY_SPEC:
        return _LEGACY_SPEC[v]
    return None


def canonical_spec_mode(value) -> str | None:
    """Normalise a stored speculativeType into Unsloth's own vocabulary.

    Unknown strings collapse to "auto", matching Unsloth's resolver: a typo
    should land on the safe capability-aware path, not reach the CLI as
    garbage and fail the launch.
    """
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if not v:
        return None
    if v in SPEC_MODES:
        return v
    if v in _LEGACY_SPEC:
        return _LEGACY_SPEC[v]
    pieces = [_LEGACY_SPEC.get(p.strip(), p.strip()) for p in v.split(",")]
    pieces = [p for p in pieces if p]
    has_mtp = "mtp" in pieces
    has_ngram = "ngram" in pieces
    if has_mtp and has_ngram:
        return "mtp+ngram"
    if has_mtp:
        return "mtp"
    if has_ngram:
        return "ngram"
    return "auto"


# =============================================================================
# STUDIO DATABASE ACCESS
# =============================================================================

def _read_setting(studio_db: str, table: str, key: str):
    """One value_json row, read-only.

    This database belongs to a running Studio; the manager has no business
    locking or writing it, and a Studio mid-write must not raise here.
    """
    if not os.path.isfile(studio_db):
        return None
    uri = f"file:{os.path.abspath(studio_db)}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(
            f"select value_json from {table} where key = ?", (key,)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return None


# =============================================================================
# 3. STUDIO PRESETS (chat_settings.customPresets)
# =============================================================================

PRESETS_KEY = "customPresets"
ACTIVE_PRESET_KEY = "activePreset"


@dataclass
class Preset:
    name: str
    # -- loadConfig: applied when the model is loaded ------------------------
    context_length: int | None = None      # customContextLength
    max_seq_length: int | None = None
    kv_cache_dtype: str | None = None
    speculative_type: str | None = None
    spec_draft_n_max: int | None = None
    n_parallel: int | None = None
    n_batch: int | None = None
    n_ubatch: int | None = None
    tensor_parallel: bool | None = None
    # Only the OFF direction, as in a per-model override: the UI writes
    # disableVision false on every preset, which means "Unsloth's default",
    # not a request to attach the projector.
    disable_vision: bool | None = None
    # -- params: the chat UI's sampling ------------------------------------
    sampling: dict = field(default_factory=dict)
    # Recorded for display only: no `studio run` flag carries these.
    max_tokens: int | None = None
    system_prompt: str = ""

    @property
    def effective_context(self) -> int | None:
        """The context the preset asks for. The UI writes the chosen value to
        customContextLength and usually leaves maxSeqLength null."""
        return self.context_length or self.max_seq_length


def _opt_int(d: dict, key: str) -> int | None:
    v = d.get(key)
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _opt_float(d: dict, key: str) -> float | None:
    v = d.get(key)
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# The UI's camelCase params -> our field names.
_PARAM_KEYS = {
    "temperature": "temperature", "topP": "top_p", "topK": "top_k",
    "minP": "min_p", "repetitionPenalty": "repetition_penalty",
    "presencePenalty": "presence_penalty",
}


def _sampling_from_params(params: dict) -> dict:
    out = {}
    for src, dst in _PARAM_KEYS.items():
        val = _opt_float(params, src)
        if val is not None:
            out[dst] = int(val) if dst in INT_FIELDS else val
    return out


def _preset_from_json(entry: dict) -> Preset | None:
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    load = entry.get("loadConfig") or {}
    params = entry.get("params") or {}
    if not isinstance(load, dict):
        load = {}
    if not isinstance(params, dict):
        params = {}
    return Preset(
        name=name,
        context_length=_opt_int(load, "customContextLength"),
        max_seq_length=_opt_int(load, "maxSeqLength"),
        kv_cache_dtype=(load.get("kvCacheDtype") or None),
        speculative_type=canonical_spec_mode(load.get("speculativeType")),
        spec_draft_n_max=_opt_int(load, "specDraftNMax"),
        n_parallel=_opt_int(load, "nParallel"),
        n_batch=_opt_int(load, "nBatch"),
        n_ubatch=_opt_int(load, "nUbatch"),
        tensor_parallel=(None if load.get("tensorParallel") is None
                         else bool(load["tensorParallel"])),
        disable_vision=(True if load.get("disableVision") is True else None),
        sampling=_sampling_from_params(params),
        max_tokens=_opt_int(params, "maxTokens"),
        system_prompt=str(params.get("systemPrompt") or ""),
    )


def load_presets(studio_db: str) -> dict[str, Preset]:
    """Every custom preset saved in the Studio UI, keyed by name."""
    raw = _read_setting(studio_db, "chat_settings", PRESETS_KEY)
    if not isinstance(raw, list):
        return {}
    out: dict[str, Preset] = {}
    for entry in raw:
        if isinstance(entry, dict):
            p = _preset_from_json(entry)
            if p is not None:
                out[p.name] = p
    return out


def raw_preset(studio_db: str, name: str) -> dict | None:
    """One preset exactly as Studio stored it, for a write-back to start from
    and to compare against: the parsed Preset drops every field this manager
    does not model, and a save must carry those through untouched."""
    raw = _read_setting(studio_db, "chat_settings", PRESETS_KEY)
    if not isinstance(raw, list):
        return None
    for entry in raw:
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    return None


def active_preset_name(studio_db: str) -> str:
    raw = _read_setting(studio_db, "chat_settings", ACTIVE_PRESET_KEY)
    return raw if isinstance(raw, str) else ""


def find_preset(presets: dict[str, Preset], name: str) -> Preset | None:
    """Exact name, then unique case-insensitive match, then unique substring.

    Preset names are free text from the UI ("MTP+Ngram 256k"), so exact
    matching alone makes them painful to type on a command line. An ambiguous
    shorthand returns None rather than guessing.
    """
    if name in presets:
        return presets[name]
    low = name.strip().lower()
    hits = [p for n, p in presets.items() if n.lower() == low]
    if len(hits) == 1:
        return hits[0]
    hits = [p for n, p in presets.items() if low in n.lower()]
    if len(hits) == 1:
        return hits[0]
    return None


# =============================================================================
# 4. PER-MODEL OVERRIDES (app_settings.openai_api_auto_switch_overrides)
# =============================================================================

OVERRIDES_KEY = "openai_api_auto_switch_overrides"

VALID_KV_DTYPES = ("f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1",
                   "q4_0", "iq4_nl")


@dataclass
class ModelOverride:
    """One row of Unsloth's per-model launch config.

    Field names are Unsloth's own, so a value read here can be compared with
    what the Studio settings page shows without a translation step in between.
    """
    key: str = ""                          # the studio.db key it came from
    context_length: int | None = None      # custom_context_length / max_seq_length
    kv_cache_dtype: str | None = None
    speculative_type: str | None = None
    spec_draft_n_max: int | None = None
    n_parallel: int | None = None
    n_batch: int | None = None
    n_ubatch: int | None = None
    tensor_parallel: bool | None = None
    disable_vision: bool | None = None
    gpu_ids: list[int] = field(default_factory=list)
    gpu_memory_mode: str | None = None
    gpu_layers: int | None = None
    n_cpu_moe: int | None = None
    llama_extra_args: list[str] = field(default_factory=list)
    chat_template_override: str | None = None

    def __bool__(self) -> bool:
        return bool(self.key)


def _override_from_json(key: str, raw: dict) -> ModelOverride:
    def _int(name):
        return _opt_int(raw, name)

    kv = raw.get("kv_cache_dtype")
    kv = kv if isinstance(kv, str) and kv.lower() in VALID_KV_DTYPES else None
    gpu_ids = raw.get("gpu_ids")
    ids: list[int] = []
    if isinstance(gpu_ids, (list, tuple)):
        for g in gpu_ids:
            if isinstance(g, bool):
                continue
            try:
                n = int(g)
            except (TypeError, ValueError):
                continue
            if n >= 0 and n not in ids:
                ids.append(n)
    extra = raw.get("llama_extra_args")
    extra = [str(a) for a in extra] if isinstance(extra, (list, tuple)) else []
    tmpl = raw.get("chat_template_override")

    return ModelOverride(
        key=key,
        # max_seq_length wins where both are set; they only collide in a legacy
        # or hand-written row. Mirrors resolve_fit_max_seq_length.
        context_length=_int("max_seq_length") or _int("custom_context_length"),
        kv_cache_dtype=kv.lower() if kv else None,
        speculative_type=canonical_spec_mode(raw.get("speculative_type")),
        spec_draft_n_max=_int("spec_draft_n_max"),
        n_parallel=_int("n_parallel"),
        n_batch=_int("n_batch"),
        n_ubatch=_int("n_ubatch"),
        tensor_parallel=(True if raw.get("tensor_parallel") is True else None),
        disable_vision=(True if raw.get("disable_vision") is True else None),
        gpu_ids=ids,
        gpu_memory_mode=(raw.get("gpu_memory_mode")
                         if raw.get("gpu_memory_mode") in ("auto", "manual")
                         else None),
        gpu_layers=_int("gpu_layers"),
        n_cpu_moe=_int("n_cpu_moe"),
        llama_extra_args=extra,
        chat_template_override=tmpl if isinstance(tmpl, str) and tmpl.strip() else None,
    )


# =============================================================================
# IDLE AUTO-UNLOAD
# =============================================================================
# Studio frees a GGUF after N idle seconds and reloads it on the next request.
# The reload does NOT replay the command line `unsloth studio run` was started
# with: it rebuilds a LoadRequest from the per-model override in studio.db
# (routes/inference.py -> resolve_override_for_load -> model_override_load_kwargs).
#
# So anything this manager set that the override does not also carry is at risk
# of being silently dropped the first time a server sits idle. Sampling is the
# exception: those ride UNSLOTH_SAMPLING_* in the server's environment, which
# no reload can touch.

IDLE_UNLOAD_KEY = "openai_api_auto_unload_idle_seconds"
MIN_IDLE_UNLOAD_SECONDS = 60


def idle_unload_seconds(studio_db: str) -> int:
    """Effective idle-unload TTL in seconds; 0 when off.

    Enabled values have a 60s floor, matching Studio: a tiny TTL tears the
    model down between turns of an active chat.
    """
    raw = _read_setting(studio_db, "app_settings", IDLE_UNLOAD_KEY)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0
    seconds = int(raw)
    return 0 if seconds <= 0 else max(MIN_IDLE_UNLOAD_SECONDS, seconds)


# What a reload restores, as (our Launch field, override attribute). Mirrors
# model_override_load_kwargs; only these reach the rebuilt LoadRequest.
RELOAD_FIELDS = (
    ("ctx", "context_length"),
    ("kv_cache_dtype", "kv_cache_dtype"),
    ("spec_mode", "speculative_type"),
    ("spec_draft_n_max", "spec_draft_n_max"),
    ("parallel", "n_parallel"),
    ("n_batch", "n_batch"),
    ("n_ubatch", "n_ubatch"),
    ("tensor_parallel", "tensor_parallel"),
)


def reload_context(override: "ModelOverride") -> int | None:
    """The context a reload sends, mirroring resolve_fit_max_seq_length.

    Under manual GPU memory with auto layers, llama.cpp's --fit owns sizing and
    the load sends the explicit pin (or 0 to hand sizing over) instead.
    """
    if override.gpu_memory_mode == "manual" and override.gpu_layers is None:
        return override.context_length or 0
    return override.context_length


def load_overrides(studio_db: str) -> dict[str, dict]:
    raw = _read_setting(studio_db, "app_settings", OVERRIDES_KEY)
    return raw if isinstance(raw, dict) else {}


def override_lookup_keys(repo_id: str, variant: str = "") -> list[str]:
    """The keys a load tries, in Unsloth's own order.

    Variant-qualified before bare: the settings page keys a row by the quant it
    loaded, and reading the bare repo id first would let an older whole-repo
    row shadow a freshly saved per-quant one.
    """
    ordered = [f"{repo_id}:{variant}" if variant else None, repo_id]
    return [k for k in ordered if k]


def raw_override(studio_db: str, key: str) -> dict:
    """The stored row under exactly `key` ({} when absent), unparsed."""
    row = load_overrides(studio_db).get(key) if key else None
    return dict(row) if isinstance(row, dict) else {}


def model_override(studio_db: str, repo_id: str,
                   variant: str = "") -> ModelOverride:
    """The per-model launch config Unsloth has stored, or an empty one.

    Falls back to a case-insensitive match, because repo ids and quants are
    case-insensitive in practice ("Q4_K_M" and "q4_k_m" name one file) and the
    Studio frontend lowercases before storing. An ambiguous fold matches
    nothing rather than replaying another model's settings.
    """
    table = load_overrides(studio_db)
    if not table:
        return ModelOverride()
    for key in override_lookup_keys(repo_id, variant):
        row = table.get(key)
        if isinstance(row, dict) and row:
            return _override_from_json(key, row)
        folded = [k for k, v in table.items()
                  if isinstance(k, str) and k.casefold() == key.casefold()
                  and isinstance(v, dict) and v]
        if len(folded) == 1:
            return _override_from_json(folded[0], table[folded[0]])
    return ModelOverride()


# =============================================================================
# WRITING BACK (payloads for Studio's own settings API)
# =============================================================================
# `start --save-profile` writes an edited launch back into the profile it came
# from. The manager sends these through Studio's HTTP API rather than into
# studio.db, so Studio validates, normalises, locks and invalidates its caches
# exactly as it does for its own settings page. Only the payload shapes live
# here, because they are Studio's field names, not process logic.
#
# Field names on our side are Launch's: ctx, kv_cache_dtype, spec_mode,
# spec_draft_n_max, parallel, n_batch, n_ubatch, tensor_parallel, vision, plus
# SAMPLING_FIELDS.

# The load fields a profile can hold, in the order a readout lists them.
PROFILE_LOAD_FIELDS = ("ctx", "kv_cache_dtype", "spec_mode", "spec_draft_n_max",
                       "parallel", "n_batch", "n_ubatch", "tensor_parallel",
                       "vision")

# Launch field -> (override key, preset loadConfig key), for the ones that map
# one-to-one. ctx, spec_mode, tensor_parallel and vision each need a rule.
_PLAIN_LOAD_KEYS = {
    "kv_cache_dtype": ("kv_cache_dtype", "kvCacheDtype"),
    "spec_draft_n_max": ("spec_draft_n_max", "specDraftNMax"),
    "parallel": ("n_parallel", "nParallel"),
    "n_batch": ("n_batch", "nBatch"),
    "n_ubatch": ("n_ubatch", "nUbatch"),
}

# Every override field a PUT REPLACES. Studio's save is not a patch: anything
# left out of the payload is cleared -- a gpu_ids pin, a chat template, a
# manual-offload layer count -- so the stored row is echoed back in full and
# only the edited fields change. llama_extra_args, the server-tuning four and
# the reasoning pair are deliberately NOT echoed: Studio carries those over by
# itself when a payload omits them (mirrors_* left false), and echoing the
# flags would re-validate ones since denylisted and 400 an unrelated save.
_OVERRIDE_REPLACED = ("max_seq_length", "custom_context_length",
                      "kv_cache_dtype", "mlx_kv_bits", "speculative_type",
                      "spec_draft_n_max", "n_parallel", "n_batch", "n_ubatch",
                      "chat_template_override", "gpu_memory_mode", "gpu_layers",
                      "n_cpu_moe", "gpu_ids", "gpu_index_kind")


def override_payload(key: str, row: dict, values: dict) -> dict:
    """The PUT /api/settings/openai-auto-switch/overrides body that stores
    `row` with `values` applied, under `key`.

    An all-default result is a valid payload: Studio reads a row with no
    usable fields as a removal, which is exactly what "every value back at
    Unsloth's default" means.
    """
    body: dict = {"model_id": key}
    for name in _OVERRIDE_REPLACED:
        if row.get(name) is not None:
            body[name] = row[name]
    # Booleans with a False default: omitting them clears them, so they are
    # always sent.
    body["tensor_parallel"] = row.get("tensor_parallel") is True
    body["disable_vision"] = row.get("disable_vision") is True

    for fld, val in values.items():
        if fld == "ctx":
            # max_seq_length outranks custom_context_length on read, so an
            # edit must clear it or the old number keeps winning. 0 is
            # fit-max: no pin at all.
            body.pop("max_seq_length", None)
            body.pop("custom_context_length", None)
            if val:
                body["custom_context_length"] = int(val)
        elif fld == "spec_mode":
            # Absent already means auto; storing the word adds nothing.
            body.pop("speculative_type", None)
            if val and val != "auto":
                body["speculative_type"] = val
        elif fld == "tensor_parallel":
            body["tensor_parallel"] = bool(val)
        elif fld == "vision":
            body["disable_vision"] = val is False
        elif fld in _PLAIN_LOAD_KEYS:
            name = _PLAIN_LOAD_KEYS[fld][0]
            body.pop(name, None)
            if val is not None:
                body[name] = val
    return body


# Our sampling names -> the chat UI's camelCase, the inverse of _PARAM_KEYS.
_PARAM_NAMES = {dst: src for src, dst in _PARAM_KEYS.items()}


def preset_with_values(raw: dict, values: dict) -> dict:
    """A copy of stored preset `raw` with `values` applied.

    Everything else in the entry -- max tokens, system prompt, seed, the
    reasoning budget, whichever loadConfig keys this manager does not model --
    comes through untouched, because the chat UI owns them.
    """
    out = copy.deepcopy(raw)
    params = out.get("params")
    if not isinstance(params, dict):
        params = out["params"] = {}
    load = out.get("loadConfig")
    load = dict(load) if isinstance(load, dict) else {}
    touched_load = False

    for fld, val in values.items():
        if fld in SAMPLING_FIELDS:
            # The UI stores every sampling number as a float, topK included.
            params[_PARAM_NAMES[fld]] = None if val is None else float(val)
            if fld == "min_p" and val is not None:
                # "server-default" makes the UI ignore minP entirely.
                params["minPMode"] = "custom"
            continue
        touched_load = True
        if fld == "ctx":
            load["customContextLength"] = int(val) if val else None
            load["maxSeqLength"] = None
        elif fld == "spec_mode":
            # The UI stores null, not "auto", for "let Unsloth decide".
            load["speculativeType"] = val if val and val != "auto" else None
        elif fld == "tensor_parallel":
            load["tensorParallel"] = bool(val)
        elif fld == "vision":
            load["disableVision"] = val is False
        elif fld in _PLAIN_LOAD_KEYS:
            load[_PLAIN_LOAD_KEYS[fld][1]] = val
    if touched_load:
        out["loadConfig"] = load
    return out


def override_load_values(ov: "ModelOverride") -> dict:
    """What a stored override supplies for each PROFILE_LOAD_FIELDS entry, in
    Launch terms, for a before/after readout."""
    return {
        "ctx": reload_context(ov) or 0,
        "kv_cache_dtype": ov.kv_cache_dtype,
        "spec_mode": ov.speculative_type,
        "spec_draft_n_max": ov.spec_draft_n_max,
        "parallel": ov.n_parallel,
        "n_batch": ov.n_batch,
        "n_ubatch": ov.n_ubatch,
        "tensor_parallel": ov.tensor_parallel,
        "vision": False if ov.disable_vision else None,
    }


def preset_values(p: "Preset") -> dict:
    """Load fields and sampling a stored preset supplies, in Launch terms."""
    out = {
        "ctx": p.effective_context or 0,
        "kv_cache_dtype": p.kv_cache_dtype,
        "spec_mode": p.speculative_type,
        "spec_draft_n_max": p.spec_draft_n_max,
        "parallel": p.n_parallel,
        "n_batch": p.n_batch,
        "n_ubatch": p.n_ubatch,
        "tensor_parallel": p.tensor_parallel,
        "vision": False if p.disable_vision else None,
    }
    out.update({k: p.sampling.get(k) for k in SAMPLING_FIELDS})
    return out


def parse_preset(entry: dict) -> "Preset | None":
    """A stored preset entry as a Preset -- for reading back what a save wrote."""
    return _preset_from_json(entry) if isinstance(entry, dict) else None


def parse_override(key: str, row: dict) -> "ModelOverride":
    """A stored override row as a ModelOverride -- for reading back a save."""
    return _override_from_json(key, row) if isinstance(row, dict) and row \
        else ModelOverride()
