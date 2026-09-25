"""Unit tests for the pure logic that guards user data.

    python3 -m unittest discover -s tests

Stdlib only, no server, no GPU, no network. Everything the manager reads or
writes is pointed at a throwaway directory BEFORE it is imported: its settings
are module globals resolved at import, and a test run must never read this
box's local.env, bank into the real tokens.json, or open the real studio.db.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sqlite3
import sys
import tempfile
import unittest

_TMP = tempfile.TemporaryDirectory(prefix="unsloth-mgr-tests-")
_ROOT = _TMP.name
os.environ.update({
    "UNSLOTH_MGR_ENV_FILE": os.devnull,
    "UNSLOTH_MGR_STATE_DIR": os.path.join(_ROOT, "state"),
    "UNSLOTH_MGR_LOG_DIR": os.path.join(_ROOT, "logs"),
    "UNSLOTH_MGR_STUDIO_DB": os.path.join(_ROOT, "studio.db"),
    "UNSLOTH_MGR_HF_HOME": os.path.join(_ROOT, "hf"),
})
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_lib as ml                 # noqa: E402
import unsloth_manager as um           # noqa: E402
import unsloth_profiles as up          # noqa: E402

HOUR, DAY = 3600, 86400


def _studio_db(idle: int | None = None, presets: list | None = None,
               overrides: dict | None = None) -> None:
    """(Re)write the throwaway studio.db with just the rows the code reads."""
    path = um.STUDIO_DB
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    for table in ("app_settings", "chat_settings"):
        conn.execute(f"create table {table} (key text primary key, value_json text)")
    rows = [("app_settings", up.IDLE_UNLOAD_KEY, idle),
            ("app_settings", up.OVERRIDES_KEY, overrides),
            ("chat_settings", up.PRESETS_KEY, presets)]
    for table, key, val in rows:
        if val is not None:
            conn.execute(f"insert into {table} values (?, ?)", (key, json.dumps(val)))
    conn.commit()
    conn.close()


def _touch(path: str, size: int = 1) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size)
    return path


def _model(snapshot: str = "", kind: str = "hf") -> ml.ModelInfo:
    return ml.ModelInfo(repo_id="org/Test-GGUF", cache_dir="", snapshot=snapshot,
                        kind=kind)


def _launch(**kw) -> um.Launch:
    kw.setdefault("model", _model())
    return um.Launch(**kw)


# =============================================================================
# Variant naming: one rule, shared by `list` and the header lookups
# =============================================================================

class VariantNaming(unittest.TestCase):
    def test_names(self):
        cases = [
            (("", "Qwen3.6-27B-UD-Q4_K_XL.gguf"), "UD-Q4_K_XL"),
            (("", "Model-Q8_0-00001-of-00003.gguf"), "Q8_0"),
            (("", "Model-IQ2_XXS.gguf"), "IQ2_XXS"),
            (("", "Model-BF16.gguf"), "BF16"),
            (("UD-Q4_K_XL", "Model-UD-Q4_K_XL-00001-of-00002.gguf"), "UD-Q4_K_XL"),
            (("extra", "Model-Q4_K_M.gguf"), "extra/Q4_K_M"),
            (("", "weird-name.gguf"), "weird-name"),
        ]
        for args, want in cases:
            with self.subTest(args=args):
                self.assertEqual(ml._variant_of(*args), want)

    def test_companions_are_not_variants(self):
        for name in ("mmproj-F16.gguf", "Model-mmproj-f16.gguf",
                     "mtp-Model-Q8_0.gguf", "draft-Model-Q4_0.gguf"):
            with self.subTest(name=name):
                self.assertIsNone(ml._variant_of("", name))

    def test_scan_and_lookup_agree(self):
        """Every variant `list` offers can be found again by name -- the drift
        between two copies of the rule that _variant_of exists to prevent."""
        with tempfile.TemporaryDirectory() as snap:
            _touch(os.path.join(snap, "M-Q4_K_M.gguf"), 10)
            _touch(os.path.join(snap, "mmproj-F16.gguf"), 99)
            _touch(os.path.join(snap, "Q8_0", "M-Q8_0-00001-of-00002.gguf"), 20)
            _touch(os.path.join(snap, "Q8_0", "M-Q8_0-00002-of-00002.gguf"), 22)
            _touch(os.path.join(snap, "misc", "M-F16.gguf"), 5)
            kind, variants = ml._scan_snapshot(snap)
            self.assertEqual(kind, "gguf")
            self.assertEqual(variants, {"Q4_K_M": 10, "Q8_0": 42, "misc/F16": 5})
            m = _model(snap, "gguf")
            for v in variants:
                with self.subTest(variant=v):
                    path = ml.variant_gguf_path(m, v)
                    self.assertTrue(path)
                    rel = os.path.relpath(os.path.dirname(path), snap)
                    self.assertEqual(ml._variant_of("" if rel == "." else rel,
                                                    os.path.basename(path)), v)
            self.assertEqual(ml.variant_gguf_path(m, "mmproj-F16"), "")
            self.assertTrue(ml.has_mmproj(m))

    def test_hf_snapshot(self):
        with tempfile.TemporaryDirectory() as snap:
            _touch(os.path.join(snap, "model.safetensors"))
            self.assertEqual(ml._scan_snapshot(snap), ("hf", {}))


# =============================================================================
# Override PUT: Studio REPLACES the row, so everything unedited must echo back
# =============================================================================

class OverridePayload(unittest.TestCase):
    ROW = {"custom_context_length": 65536, "kv_cache_dtype": "q8_0",
           "n_parallel": 3, "gpu_ids": [1], "chat_template_override": "{{x}}",
           "gpu_memory_mode": "manual", "gpu_layers": 40, "n_cpu_moe": 2,
           "tensor_parallel": True, "llama_extra_args": ["--foo"]}

    def test_no_edit_echoes_the_row(self):
        row = copy.deepcopy(self.ROW)
        body = up.override_payload("org/M:Q4_K_M", row, {})
        self.assertEqual(body["model_id"], "org/M:Q4_K_M")
        for k in ("custom_context_length", "kv_cache_dtype", "n_parallel",
                  "gpu_ids", "chat_template_override", "gpu_memory_mode",
                  "gpu_layers", "n_cpu_moe"):
            self.assertEqual(body[k], self.ROW[k], k)
        self.assertIs(body["tensor_parallel"], True)
        self.assertIs(body["disable_vision"], False)
        # Studio carries these over itself; echoing them can 400 a save.
        self.assertNotIn("llama_extra_args", body)
        self.assertEqual(row, self.ROW, "input row must not be mutated")

    def test_edit_keeps_unrelated_fields(self):
        body = up.override_payload("k", self.ROW, {"kv_cache_dtype": "q4_0"})
        self.assertEqual(body["kv_cache_dtype"], "q4_0")
        self.assertEqual(body["gpu_ids"], [1])
        self.assertEqual(body["chat_template_override"], "{{x}}")

    def test_ctx(self):
        row = {"max_seq_length": 8192, "custom_context_length": 4096}
        body = up.override_payload("k", row, {"ctx": 32768})
        self.assertNotIn("max_seq_length", body)   # it would outrank the edit
        self.assertEqual(body["custom_context_length"], 32768)
        body = up.override_payload("k", row, {"ctx": 0})
        self.assertNotIn("max_seq_length", body)
        self.assertNotIn("custom_context_length", body)

    def test_spec_vision_tp_plain(self):
        row = {"speculative_type": "mtp", "n_batch": 512}
        body = up.override_payload("k", row, {"spec_mode": "auto"})
        self.assertNotIn("speculative_type", body)
        body = up.override_payload("k", {}, {"spec_mode": "ngram"})
        self.assertEqual(body["speculative_type"], "ngram")
        self.assertIs(up.override_payload("k", {}, {"vision": False})["disable_vision"], True)
        self.assertIs(up.override_payload("k", {"disable_vision": True},
                                          {"vision": True})["disable_vision"], False)
        self.assertIs(up.override_payload("k", {"tensor_parallel": True},
                                          {"tensor_parallel": False})["tensor_parallel"], False)
        self.assertEqual(up.override_payload("k", {}, {"parallel": 2})["n_parallel"], 2)
        self.assertNotIn("n_batch", up.override_payload("k", row, {"n_batch": None}))

    def test_sampling_is_not_an_override_field(self):
        body = up.override_payload("k", {}, {"temperature": 0.7})
        self.assertNotIn("temperature", body)

    def test_all_default_is_a_valid_removal(self):
        self.assertEqual(up.override_payload("k", {}, {}),
                         {"model_id": "k", "tensor_parallel": False,
                          "disable_vision": False})


class OverrideParsing(unittest.TestCase):
    def test_ints_are_lenient(self):
        ov = up._override_from_json("k", {
            "n_parallel": "3", "n_batch": True, "n_ubatch": "x",
            "max_seq_length": None, "custom_context_length": 4096,
            "kv_cache_dtype": "Q8_0", "gpu_ids": [1, "0", 1, -1, True, "z"],
            "chat_template_override": "  "})
        self.assertEqual(ov.n_parallel, 3)
        self.assertIsNone(ov.n_batch)       # a bool is not a count
        self.assertIsNone(ov.n_ubatch)
        self.assertEqual(ov.context_length, 4096)
        self.assertEqual(ov.kv_cache_dtype, "q8_0")
        self.assertEqual(ov.gpu_ids, [1, 0])
        self.assertIsNone(ov.chat_template_override)

    def test_max_seq_length_wins(self):
        ov = up._override_from_json("k", {"max_seq_length": 8192,
                                          "custom_context_length": 4096})
        self.assertEqual(ov.context_length, 8192)

    def test_reload_context(self):
        manual = up.ModelOverride(key="k", gpu_memory_mode="manual")
        self.assertEqual(up.reload_context(manual), 0)      # --fit sizes it
        self.assertIsNone(up.reload_context(up.ModelOverride(key="k")))


# =============================================================================
# Preset write-back: the chat UI owns every key this manager does not model
# =============================================================================

class PresetWithValues(unittest.TestCase):
    RAW = {"name": "P", "systemPrompt": "be brief", "maxTokens": 4096,
           "params": {"temperature": 0.6, "minPMode": "server-default",
                      "seed": 7},
           "loadConfig": {"customContextLength": 8192, "maxSeqLength": 4096,
                          "gpuMemoryMode": "auto", "nParallel": 2}}

    def test_untouched_keys_survive(self):
        raw = copy.deepcopy(self.RAW)
        out = up.preset_with_values(raw, {"temperature": 1.0})
        self.assertEqual(raw, self.RAW, "input preset must not be mutated")
        self.assertEqual(out["systemPrompt"], "be brief")
        self.assertEqual(out["maxTokens"], 4096)
        self.assertEqual(out["params"]["seed"], 7)
        self.assertEqual(out["loadConfig"], self.RAW["loadConfig"])
        self.assertEqual(out["params"]["temperature"], 1.0)

    def test_sampling_spelling(self):
        out = up.preset_with_values(self.RAW, {"top_k": 20, "min_p": 0.05})
        self.assertEqual(out["params"]["topK"], 20.0)
        self.assertIsInstance(out["params"]["topK"], float)
        self.assertEqual(out["params"]["minP"], 0.05)
        self.assertEqual(out["params"]["minPMode"], "custom")
        out = up.preset_with_values(self.RAW, {"min_p": None})
        self.assertIsNone(out["params"]["minP"])
        self.assertEqual(out["params"]["minPMode"], "server-default")

    def test_load_fields(self):
        out = up.preset_with_values(self.RAW, {"ctx": 32768, "spec_mode": "auto",
                                               "vision": False, "parallel": 4})
        load = out["loadConfig"]
        self.assertEqual(load["customContextLength"], 32768)
        self.assertIsNone(load["maxSeqLength"])
        self.assertIsNone(load["speculativeType"])
        self.assertIs(load["disableVision"], True)
        self.assertEqual(load["nParallel"], 4)
        self.assertEqual(load["gpuMemoryMode"], "auto")
        self.assertIsNone(up.preset_with_values(self.RAW, {"ctx": 0})
                          ["loadConfig"]["customContextLength"])

    def test_sampling_only_adds_no_load_config(self):
        out = up.preset_with_values({"name": "P"}, {"temperature": 0.5})
        self.assertNotIn("loadConfig", out)
        self.assertEqual(out["params"], {"temperature": 0.5})

    def test_bad_params_replaced(self):
        out = up.preset_with_values({"name": "P", "params": "junk"},
                                    {"top_p": 0.9})
        self.assertEqual(out["params"], {"topP": 0.9})


# =============================================================================
# Reload drift: what an idle reload would silently change
# =============================================================================

class ReloadDrift(unittest.TestCase):
    def setUp(self):
        _studio_db(idle=300)

    def test_nothing_reloads_when_idle_unload_is_off(self):
        _studio_db(idle=0)
        lb = _launch(ctx=32768, kv_cache_dtype="q4_0")
        self.assertEqual(um._reload_drift(lb), [])

    def test_dropped_and_contradicted(self):
        ov = up.ModelOverride(key="k", kv_cache_dtype="q8_0")
        lb = _launch(ctx=32768, kv_cache_dtype="q4_0", override=ov)
        self.assertEqual(sorted(um._reload_drift(lb)), [
            ("context", "32,768", "Unsloth default"),
            ("kv cache", "q4_0", "q8_0"),
        ])

    def test_matching_and_unset_are_not_drift(self):
        ov = up.ModelOverride(key="k", context_length=32768, kv_cache_dtype="q4_0")
        lb = _launch(ctx=32768, kv_cache_dtype="q4_0", override=ov)
        self.assertEqual(um._reload_drift(lb), [])
        self.assertEqual(um._reload_drift(_launch()), [])

    def test_parallel_is_a_startup_default(self):
        self.assertEqual(um._reload_drift(_launch(parallel=3)), [])
        ov = up.ModelOverride(key="k", n_parallel=1)
        self.assertEqual(um._reload_drift(_launch(parallel=3, override=ov)),
                         [("slots", "3", "1")])

    def test_pending_override_replaces_stored(self):
        lb = _launch(ctx=32768, override=up.ModelOverride(key="k"))
        after = up.ModelOverride(key="k", context_length=32768)
        self.assertEqual(um._reload_drift(lb, after), [])

    def test_vision_needs_a_projector(self):
        with tempfile.TemporaryDirectory() as snap:
            _touch(os.path.join(snap, "M-Q4_K_M.gguf"))
            plain = _model(snap, "gguf")
            ov = up.ModelOverride(key="k", disable_vision=True)
            self.assertEqual(um._reload_drift(_launch(model=plain, override=ov)), [])

            _touch(os.path.join(snap, "mmproj-F16.gguf"))
            # Unset vision attaches the projector, and a reload that disables
            # it is drift -- the case that once went unseen for hours.
            self.assertEqual(um._reload_drift(_launch(model=plain, override=ov)),
                             [("vision", "on (Unsloth default)", "off")])
            self.assertEqual(
                um._reload_drift(_launch(model=plain, vision=False,
                                         override=up.ModelOverride(key="k"))),
                [("vision", "off", "on (mmproj reattached)")])


# =============================================================================
# Profile save routing
# =============================================================================

def _args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace(**{a: None for a in um._EDIT_ARGS.values()})
    ns.load_profile = "auto"
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


class ProfileSavePlan(unittest.TestCase):
    PRESET_RAW = {"name": "P", "params": {"temperature": 0.6},
                  "loadConfig": {"customContextLength": 8192}}

    def setUp(self):
        _studio_db(idle=300, presets=[self.PRESET_RAW])
        self.preset = up.parse_preset(self.PRESET_RAW)

    def _preset_launch(self, **kw):
        kw.setdefault("sampling", {"temperature": 0.6})
        kw.setdefault("ctx", 8192)
        return _launch(variant="Q4_K_M", preset=self.preset,
                       sampling_source="preset", load_source="preset P", **kw)

    def test_sampling_edit_goes_to_the_preset(self):
        lb = self._preset_launch(sampling={"temperature": 1.0})
        plan = um._profile_save_plan(lb, _args(temperature=1.0))
        self.assertEqual(plan.needs, {"temperature": "preset"})
        self.assertEqual(plan.preset, "P")
        self.assertEqual(plan.preset_values, {"temperature": 1.0})
        self.assertEqual(plan.preset_old, {"temperature": 0.6})
        self.assertEqual(plan.preset_raw, self.PRESET_RAW)

    def test_value_already_held_writes_nothing(self):
        # An override that already reproduces the launch, so the sync that a
        # save always brings along has nothing to add either.
        _studio_db(idle=300, presets=[self.PRESET_RAW], overrides={
            "org/Test-GGUF:Q4_K_M": {"custom_context_length": 8192}})
        stored = up.model_override(um.STUDIO_DB, "org/Test-GGUF", "Q4_K_M")
        self.assertTrue(stored)
        lb = self._preset_launch(override=stored)
        plan = um._profile_save_plan(lb, _args(temperature=0.6))
        self.assertEqual(plan.needs, {"temperature": "preset"})
        self.assertEqual(plan.preset_values, {})
        self.assertEqual(plan.override_values, {})
        self.assertFalse(plan.writes)

    def test_preset_save_syncs_the_override(self):
        """An edit saved only to the preset would be undone by the first idle
        reload, which rebuilds from the override -- so the override is made to
        reproduce this launch in full."""
        lb = self._preset_launch(ctx=32768)
        plan = um._profile_save_plan(lb, _args(ctx=32768))
        self.assertEqual(plan.preset_values, {"ctx": 32768})
        self.assertEqual(plan.override_key, "org/Test-GGUF:Q4_K_M")
        self.assertEqual(plan.override_values, {"ctx": 32768})
        self.assertTrue(plan.notes)          # auto will now read the override

    def test_load_edit_without_preset_goes_to_the_override(self):
        _studio_db(idle=300, overrides={"org/Test-GGUF:Q4_K_M": {"n_parallel": 2}})
        stored = up.ModelOverride(key="org/Test-GGUF:Q4_K_M", n_parallel=2)
        lb = _launch(variant="Q4_K_M", override=stored, parallel=4,
                     load_source="override")
        plan = um._profile_save_plan(lb, _args(parallel=4))
        self.assertEqual(plan.needs, {"parallel": "override"})
        self.assertEqual(plan.override_key, "org/Test-GGUF:Q4_K_M")
        self.assertEqual(plan.override_raw, {"n_parallel": 2})
        self.assertEqual(plan.override_values, {"parallel": 4})
        self.assertEqual(plan.override_old["parallel"], 2)
        self.assertEqual(plan.preset, "")

    def test_override_key_keeps_stored_spelling(self):
        stored = up.ModelOverride(key="Org/test-gguf:q4_k_m")
        lb = _launch(variant="Q4_K_M", override=stored)
        self.assertEqual(um._override_target_key(lb), "Org/test-gguf:q4_k_m")
        self.assertEqual(um._override_target_key(_launch(variant="Q8_0")),
                         "org/Test-GGUF:Q8_0")

    def test_no_profile_means_instance_only(self):
        lb = _launch(variant="Q4_K_M", ctx=4096, sampling={"temperature": 0.2},
                     sampling_source="none")
        plan = um._profile_save_plan(lb, _args(ctx=4096, temperature=0.2,
                                               load_profile="none"))
        self.assertEqual(plan.instance_only, {"ctx": 4096, "temperature": 0.2})
        self.assertEqual(set(plan.why_instance), {"ctx", "temperature"})
        self.assertFalse(plan.writes)
        self.assertEqual(plan.override_key, "")

    def test_draft_depth_without_a_drafter_is_not_a_change(self):
        stored = up.ModelOverride(key="org/Test-GGUF:Q4_K_M")
        lb = _launch(variant="Q4_K_M", override=stored, spec_mode="ngram",
                     spec_draft_n_max=8, load_source="override")
        plan = um._profile_save_plan(lb, _args(spec="ngram"))
        self.assertNotIn("spec_draft_n_max", plan.override_values)
        self.assertEqual(plan.override_values, {"spec_mode": "ngram"})


# =============================================================================
# Throughput
# =============================================================================

class SteadyTps(unittest.TestCase):
    def test_too_few_or_a_burst_is_no_rate(self):
        self.assertEqual(um._steady_tps([0.0, 1.0], 5.0), 0.0)
        burst = [i * 0.001 for i in range(50)]           # 50 chunks in 49ms
        self.assertEqual(um._steady_tps(burst, 5.0), 0.0)

    def test_shorter_than_a_window_is_the_overall_rate(self):
        times = [i * 0.05 for i in range(21)]            # 20 tok/s for 1s
        self.assertAlmostEqual(um._steady_tps(times, 5.0), 20.0)

    def test_a_stall_does_not_hide_the_sustained_rate(self):
        first = [i * 0.1 for i in range(101)]            # 10 tok/s, 10s
        second = [15.0 + i * 0.1 for i in range(101)]    # after a 5s stall
        times = first + second
        mean = (len(times) - 1) / (times[-1] - times[0])
        best = um._steady_tps(times, 5.0)
        self.assertAlmostEqual(best, 10.0, places=6)
        self.assertLess(mean, 9.0)

    def test_an_opening_burst_is_not_the_rate(self):
        """Only near-full windows count: 20 chunks landing 1ms apart at the
        start must not read as 1000 tok/s just because the first, still-short
        window holds nothing else. Later windows always stretch back to full
        width, so the start is where this guard earns its keep."""
        burst = [i * 0.001 for i in range(20)]
        steady = [0.02 + (i + 1) * 0.1 for i in range(100)]   # 10 tok/s, 10s
        self.assertLess(um._steady_tps(burst + steady, 5.0), 20.0)


# =============================================================================
# Token history
# =============================================================================

def _doc() -> dict:
    return {"models": {}, "hours": {}, "days": {}}


class TokenHistory(unittest.TestCase):
    NOW = 1_800_000_000.0                                # a fixed "now"

    def test_record_lands_in_the_hour(self):
        doc = _doc()
        um._tokens_record(doc, 10, 5, now=self.NOW)
        um._tokens_record(doc, 1, 2, now=self.NOW + 60)
        um._tokens_record(doc, 0, 0, now=self.NOW)       # no-op
        self.assertEqual(doc["hours"], {str(int(self.NOW // HOUR)): [11, 7]})

    def test_windows(self):
        doc = _doc()
        um._tokens_record(doc, 100, 10, now=self.NOW - 2 * HOUR)
        um._tokens_record(doc, 200, 20, now=self.NOW - 2 * DAY)
        um._tokens_record(doc, 400, 40, now=self.NOW - 20 * DAY)
        got = dict((label, (i, o)) for label, i, o
                   in um._tokens_windows(doc, now=self.NOW))
        self.assertEqual(list(got), ["24h", "7d", "30d", "365d"])
        self.assertEqual(got["24h"], (100, 10))
        self.assertEqual(got["7d"], (300, 30))
        self.assertEqual(got["30d"], (700, 70))
        self.assertEqual(got["365d"], (700, 70))

    def test_straddling_bucket_counts_whole(self):
        doc = _doc()
        start = (int(self.NOW // HOUR) - 24) * HOUR      # ends inside the window
        doc["hours"][str(start // HOUR)] = [5, 1]
        self.assertEqual(um._tokens_window(doc, DAY, now=start + HOUR + DAY - 1),
                         (5, 1))

    def test_rollup_folds_and_prunes_without_double_counting(self):
        doc = _doc()
        old_hour = self.NOW - 31 * DAY
        um._tokens_record(doc, 7, 3, now=old_hour)       # rolled up only later
        day_key = str(int(old_hour // HOUR) * HOUR // DAY)
        doc["days"][day_key] = [1, 1]
        doc["days"][str(int((self.NOW - 400 * DAY) // DAY))] = [99, 99]
        doc["hours"]["junk"] = [1, 1]
        doc["days"]["junk"] = [1, 1]
        recent = str(int(self.NOW // HOUR))
        doc["hours"][recent] = [2, 2]

        um._tokens_rollup(doc, self.NOW)
        self.assertEqual(doc["hours"], {recent: [2, 2]})
        self.assertEqual(doc["days"], {day_key: [8, 4]})
        self.assertEqual(um._tokens_window(doc, 365 * DAY, now=self.NOW), (10, 6))

    def test_bucket_tolerance(self):
        for bad in (None, ["x", 1], [1], {}):
            with self.subTest(bad=bad):
                self.assertEqual(um._tok_bucket(bad), (0, 0))
        self.assertEqual(um._tok_bucket([3, None]), (3, 0))

    def test_format(self):
        for n, want in ((812, "812"), (1000, "1K"), (9900, "9.9K"),
                        (820_000, "820K"), (18_900_000, "18.9M"),
                        (2_500_000_000_000, "2500B")):
            with self.subTest(n=n):
                self.assertEqual(um._fmt_tokens(n), want)


class Isolation(unittest.TestCase):
    def test_nothing_points_at_the_real_box(self):
        for path in (um.STATE_DIR, um.LOG_DIR, um.STUDIO_DB, um.TOKENS_FILE):
            self.assertTrue(path.startswith(_ROOT), path)


if __name__ == "__main__":
    unittest.main()
