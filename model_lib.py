"""
model_lib.py — Hugging Face cache and GGUF geometry for unsloth_manager.py.

Owns three things:

  1. Cache scanning: turn `models--org--repo` directories into usable
     (repo_id, kind, GGUF variants, on-disk size) records.
  2. GGUF variant parsing and VRAM fit estimation for the local GPUs.
  3. Enough GGUF header parsing to read a model's native context length
     without loading it.

Everything about *how* a model should be run — Unsloth's published sampling
profiles, its family defaults, Studio presets, and per-model launch overrides
— lives in unsloth_profiles.py instead.
"""
from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field

# =============================================================================
# HUGGING FACE CACHE
# =============================================================================

# Companion GGUFs that ship beside the weights but cannot be served on their
# own: vision projectors and multi-token-prediction / draft modules. Unsloth
# picks these up itself from the same directory, so listing them as loadable
# variants would only offer a launch that cannot work.
_NON_VARIANT_GGUF = re.compile(
    r"^(mmproj|mtp-|draft-)|-mmproj", re.IGNORECASE)

# A directory whose name is itself a quant tag: big repos ship one folder per
# quant with sharded files inside, and that folder name is what
# `--gguf-variant` expects.
_QUANT_DIR_RE = re.compile(
    r"^(?:UD-)?(?:IQ\d+[A-Z0-9_]*|Q\d+[A-Z0-9_]*|BF16|F16|F32)$",
    re.IGNORECASE)

# How deep to look for weights inside a snapshot. One level of nesting covers
# the quant-per-folder layout; deeper trees are collection repos, not models.
_SCAN_DEPTH = 2

# `Qwen3.6-27B-UD-Q4_K_XL.gguf` -> UD-Q4_K_XL
# `Model-Q8_0-00001-of-00003.gguf` -> Q8_0   (shards collapse to one variant)
_GGUF_VARIANT_RE = re.compile(
    r"-(?P<variant>"
    r"(?:UD-)?(?:IQ\d+[A-Z0-9_]*|Q\d+[A-Z0-9_]*|BF16|F16|F32)"
    r")"
    r"(?:-\d{5}-of-\d{5})?"
    r"\.gguf$",
    re.IGNORECASE,
)


@dataclass
class ModelInfo:
    repo_id: str                  # "unsloth/Qwen3.6-27B-MTP-GGUF"
    cache_dir: str                # .../hub/models--unsloth--Qwen3.6-27B-MTP-GGUF
    snapshot: str                 # resolved snapshot directory (may be "")
    kind: str = "hf"              # "gguf" or "hf"
    variants: dict[str, int] = field(default_factory=dict)  # variant -> bytes
    size_bytes: int = 0           # total on-disk size of the blobs

    @property
    def is_gguf(self) -> bool:
        return self.kind == "gguf"

    @property
    def is_empty(self) -> bool:
        """No weights on disk — an aborted or not-yet-started download.

        The cache directory exists (HF creates it up front) but holds nothing
        servable, so this must be caught before a launch rather than after
        Unsloth fails to find a file.
        """
        return self.size_bytes == 0 and not self.variants

    def variant_names(self) -> list[str]:
        """Variants smallest-first, which is also roughly worst-to-best quality."""
        return sorted(self.variants, key=lambda v: (self.variants[v], v))


def hub_dir(hf_home: str) -> str:
    return os.path.join(hf_home, "hub")


def _resolve_snapshot(cache_dir: str) -> str:
    """Pick the snapshot a download actually points at.

    Prefer whatever `refs/main` names; fall back to the newest snapshot dir so a
    repo fetched by revision (no refs/main) is still usable.
    """
    snaps = os.path.join(cache_dir, "snapshots")
    if not os.path.isdir(snaps):
        return ""

    ref = os.path.join(cache_dir, "refs", "main")
    if os.path.isfile(ref):
        try:
            with open(ref) as f:
                rev = f.read().strip()
            cand = os.path.join(snaps, rev)
            if os.path.isdir(cand):
                return cand
        except OSError:
            pass

    try:
        entries = [os.path.join(snaps, d) for d in os.listdir(snaps)]
    except OSError:
        return ""
    dirs = [d for d in entries if os.path.isdir(d)]
    if not dirs:
        return ""
    return max(dirs, key=lambda d: os.stat(d).st_mtime)


def _real_size(path: str) -> int:
    """Size of a snapshot entry, following the symlink into blobs/."""
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def _iter_gguf(snapshot: str):
    """(relative dir, filename, absolute path) for GGUFs up to _SCAN_DEPTH."""
    for root, dirs, files in os.walk(snapshot):
        rel = os.path.relpath(root, snapshot)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth >= _SCAN_DEPTH:
            dirs[:] = []
        for name in files:
            if name.lower().endswith(".gguf"):
                yield ("" if rel == "." else rel, name,
                       os.path.join(root, name))


def _scan_snapshot(snapshot: str) -> tuple[str, dict[str, int]]:
    """Classify a snapshot as GGUF or HF-format and collect its variants.

    Sharded GGUFs are summed under one variant name, so `variants[v]` is the
    full weight size you need to fit, not the size of the first shard.
    """
    variants: dict[str, int] = {}
    saw_gguf = False

    for reldir, name, path in _iter_gguf(snapshot):
        saw_gguf = True
        if _NON_VARIANT_GGUF.match(name):
            continue

        if reldir and _QUANT_DIR_RE.match(os.path.basename(reldir)):
            # Quant-per-folder layout: the folder is the variant, and the
            # shards inside it all belong to that one variant.
            variant = os.path.basename(reldir)
        else:
            m = _GGUF_VARIANT_RE.search(name)
            # A single-file GGUF with an unrecognised tag still deserves to be
            # listed; name it after the file so it can at least be selected.
            variant = m.group("variant") if m else name[: -len(".gguf")]
            if reldir:
                variant = f"{reldir}/{variant}"

        variants[variant] = variants.get(variant, 0) + _real_size(path)

    return ("gguf" if saw_gguf else "hf"), variants


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            total += _real_size(os.path.join(root, f))
    return total


def scan_models(hf_home: str) -> list[ModelInfo]:
    """Every model in the HF cache, sorted by repo id."""
    root = hub_dir(hf_home)
    if not os.path.isdir(root):
        return []

    out: list[ModelInfo] = []
    for entry in sorted(os.listdir(root)):
        if not entry.startswith("models--"):
            continue
        cache_dir = os.path.join(root, entry)
        if not os.path.isdir(cache_dir):
            continue

        repo_id = entry[len("models--"):].replace("--", "/")
        snapshot = _resolve_snapshot(cache_dir)
        kind, variants = _scan_snapshot(snapshot) if snapshot else ("hf", {})

        out.append(ModelInfo(
            repo_id=repo_id,
            cache_dir=cache_dir,
            snapshot=snapshot,
            kind=kind,
            variants=variants,
            size_bytes=_dir_size(os.path.join(cache_dir, "blobs")),
        ))
    return out


def find_model(hf_home: str, name: str) -> ModelInfo | None:
    """Look a model up by exact repo id, then by unique case-insensitive suffix.

    The suffix match is what makes `start Qwen3.6-27B-MTP-GGUF` work without
    typing the org. Ambiguous shorthands deliberately return None rather than
    guessing which of two repos you meant.
    """
    models = scan_models(hf_home)
    for m in models:
        if m.repo_id == name:
            return m

    low = name.lower()
    hits = [m for m in models
            if m.repo_id.lower() == low or m.repo_id.split("/")[-1].lower() == low]
    if len(hits) == 1:
        return hits[0]

    hits = [m for m in models if low in m.repo_id.lower()]
    if len(hits) == 1:
        return hits[0]
    return None


def human_size(n: int) -> str:
    if n <= 0:
        return "-"
    size = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024 or unit == "T":
            return (f"{size:.0f}{unit}" if unit in ("B", "K")
                    else f"{size:.1f}{unit}")
        size /= 1024.0
    return f"{size:.1f}T"


# =============================================================================
# SIZING
# =============================================================================

# llama.cpp needs headroom above the weights for the KV cache, compute buffers
# and CUDA context. 1.25x is the rule of thumb that has held on this box.
FIT_OVERHEAD = 1.25


def fits(variant_bytes: int, total_vram_gb: int) -> bool:
    if not variant_bytes:
        return True
    need_gb = (variant_bytes / (1024 ** 3)) * FIT_OVERHEAD
    return need_gb <= total_vram_gb


def fit_note(variant_bytes: int, per_gpu_gb: int, n_gpus: int) -> str:
    """One-word verdict on whether a variant can be served here."""
    if not variant_bytes:
        return ""
    weights_gb = variant_bytes / (1024 ** 3)
    need = weights_gb * FIT_OVERHEAD
    if need <= per_gpu_gb:
        return "fits 1 GPU"
    if need <= per_gpu_gb * n_gpus:
        return f"needs {n_gpus} GPUs"
    return "too large"


def suggest_variant(m: ModelInfo, per_gpu_gb: int, n_gpus: int) -> str:
    """Best-quality variant that still fits in total VRAM, else the smallest."""
    if not m.variants:
        return ""
    total = per_gpu_gb * n_gpus
    by_size = sorted(m.variants.items(), key=lambda kv: kv[1])
    ok = [name for name, size in by_size if fits(size, total)]
    return ok[-1] if ok else by_size[0][0]



# =============================================================================
# GGUF HEADER
# =============================================================================
# Just enough of the GGUF key/value block to read a model's native context
# length, which is what "max context" means before any VRAM capping. Unsloth
# reads the same key ({arch}.context_length) the same way; this is a local
# copy so a readout does not need a running server to answer the question.

_GGUF_MAGIC = 0x46554747          # b"GGUF" as a little-endian u32

# Byte width of every fixed-size GGUF value type. Strings (8) and arrays (9)
# are variable and handled inline.
_GGUF_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
               10: 8, 11: 8, 12: 8}


def _gguf_skip(f, vtype: int) -> bool:
    """Advance past one value. False means the type is unknown or the file is
    malformed, at which point the caller must stop rather than resync onto
    whatever bytes come next."""
    if vtype == 8:                                     # STRING
        raw = f.read(8)
        if len(raw) < 8:
            return False
        n = struct.unpack("<Q", raw)[0]
        if n > 1 << 30:
            return False
        f.seek(n, 1)
        return True
    if vtype == 9:                                     # ARRAY
        head = f.read(12)
        if len(head) < 12:
            return False
        atype, alen = struct.unpack("<IQ", head)
        if alen > 1 << 30:
            return False
        if atype == 8:
            for _ in range(alen):
                raw = f.read(8)
                if len(raw) < 8:
                    return False
                f.seek(struct.unpack("<Q", raw)[0], 1)
            return True
        if atype not in _GGUF_FIXED:
            return False
        f.seek(_GGUF_FIXED[atype] * alen, 1)
        return True
    if vtype in _GGUF_FIXED:
        f.seek(_GGUF_FIXED[vtype], 1)
        return True
    return False


def gguf_native_context(path: str) -> int | None:
    """`{arch}.context_length` from a GGUF header, or None.

    The architecture is learned from general.architecture, which GGUF writes
    before the arch-namespaced keys, so one forward pass is enough. Returns
    None for a non-GGUF, an unreadable file, or a zero/absent value — a real
    context length is positive, and treating garbage as a length would publish
    a window no fitter ever agreed to.
    """
    arch = None
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            if len(head) < 24:
                return None
            magic, _version, _tensors, kv_count = struct.unpack("<IIQQ", head)
            if magic != _GGUF_MAGIC:
                return None
            for _ in range(kv_count):
                raw = f.read(8)
                if len(raw) < 8:
                    return None
                klen = struct.unpack("<Q", raw)[0]
                if klen > 1 << 20:
                    return None
                key = f.read(klen).decode("utf-8", "replace")
                raw = f.read(4)
                if len(raw) < 4:
                    return None
                vtype = struct.unpack("<I", raw)[0]

                if vtype == 8 and key == "general.architecture":
                    n = struct.unpack("<Q", f.read(8))[0]
                    if n > 1 << 22:
                        return None
                    arch = f.read(n).decode("utf-8", "replace")
                elif (arch and vtype in (4, 10)
                      and key == f"{arch}.context_length"):
                    width = 4 if vtype == 4 else 8
                    val = struct.unpack("<I" if vtype == 4 else "<Q",
                                        f.read(width))[0]
                    return val if val > 0 else None
                elif not _gguf_skip(f, vtype):
                    return None
    except (OSError, struct.error, UnicodeDecodeError):
        return None
    return None


def variant_gguf_path(m: ModelInfo, variant: str) -> str:
    """A file belonging to `variant`, for header reads. Any shard will do —
    every shard of one model carries the same architecture metadata."""
    if not m.snapshot or not variant:
        return ""
    for reldir, name, path in _iter_gguf(m.snapshot):
        if _NON_VARIANT_GGUF.match(name):
            continue
        if reldir and _QUANT_DIR_RE.match(os.path.basename(reldir)):
            found = os.path.basename(reldir)
        else:
            hit = _GGUF_VARIANT_RE.search(name)
            found = hit.group("variant") if hit else name[: -len(".gguf")]
            if reldir:
                found = f"{reldir}/{found}"
        if found == variant:
            return path
    return ""


def native_context(m: ModelInfo, variant: str = "") -> int | None:
    """Native context of a cached model, from whichever GGUF is servable."""
    if not m.is_gguf:
        return None
    path = variant_gguf_path(m, variant) if variant else ""
    if not path:
        for name in m.variant_names():
            path = variant_gguf_path(m, name)
            if path:
                break
    return gguf_native_context(path) if path else None
