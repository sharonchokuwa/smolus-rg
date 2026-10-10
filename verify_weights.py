"""
weight verification for all the methods
"""

from __future__ import annotations
import collections
import logging
import torch

logger = logging.getLogger(__name__)

_PREFIXES = ("base_model.model.", "base_model.", "model.")

def _norm(name: str):
    """Normalise a parameter name so trained-vs-base names line up.
    Strips PEFT wrapper prefixes and the ModulesToSaveWrapper suffixes so a
    wrapped module's parameter matches the same parameter on the plain base.
    """
    for p in _PREFIXES:
        if name.startswith(p):
            name = name[len(p):]
            break

    name = name.replace("modules_to_save.default.", "")
    name = name.replace("original_module.", "")
    # LoRA base_layer wrapping (if a merge was skipped)
    name = name.replace("base_layer.", "")
    return name

def _component_of(name: str, component_paths: dict):
    """Return which component a parameter name belongs to, or None."""
    # language can match both 'text_model' and 'lm_head'
    for part, subs in component_paths.items():
        if any(s in name for s in subs):
            return part
    return None

def verify_trained_weights(
    trained_model,
    cfg: dict,
    *,
    base_model=None,
    frozen_atol: float = 1e-6,
    raise_on_fail: bool = True,
    verbose: bool = True,
):
    from models.loader import (
        _resolve_components,
        _COMPONENT_PATHS,
        load_base_only,
    )

    components = _resolve_components(cfg["peft"])
    init_strategy = cfg["model"].get("init_strategy", "?")

    built_here = False
    if base_model is None:
        base_model = load_base_only(cfg)[0]      # (model, processor, lora_config)
        built_here = True

    # Index params by normalised name.
    base_params = {_norm(n): p for n, p in base_model.named_parameters()}
    trained_params = {_norm(n): p for n, p in trained_model.named_parameters()}

    agg = collections.defaultdict(
        lambda: {"n": 0, "max": 0.0, "sum_mean": 0.0, "matched": 0, "missing": 0}
    )

    with torch.no_grad():
        for name, bp in base_params.items():
            part = _component_of(name, _COMPONENT_PATHS)
            if part is None:
                continue
            tp = trained_params.get(name)
            a = agg[part]
            if tp is None or tp.shape != bp.shape:
                a["missing"] += 1
                continue
            d = (tp.detach().float().cpu() - bp.detach().float().cpu()).abs()
            a["n"] += 1
            a["matched"] += 1
            a["max"] = max(a["max"], float(d.max()))
            a["sum_mean"] += float(d.mean())

    report = {}
    failures = []
    for part in ("vision", "connector", "language"):
        if part not in components:
            continue
        strat = components[part]
        a = agg.get(part, None)
        if not a or a["matched"] == 0:
            report[part] = {"strategy": strat, "matched": 0, "ok": False,
                            "reason": "no parameters matched (name mismatch?)"}
            failures.append(part)
            continue
        max_diff = a["max"]
        mean_diff = a["sum_mean"] / max(a["n"], 1)
        changed = max_diff > frozen_atol
        expect_changed = strat in ("full", "lora")
        ok = (changed == expect_changed)
        report[part] = {
            "strategy": strat, "matched": a["matched"], "missing": a["missing"],
            "max_abs_diff": max_diff, "mean_abs_diff": mean_diff,
            "changed": changed, "expect_changed": expect_changed, "ok": ok,
        }
        if not ok:
            failures.append(part)

    if verbose:
        print("\n" + "=" * 74)
        print(f"  WEIGHT ROUND-TRIP CHECK   (method={cfg['peft'].get('method')}, "
              f"init={init_strategy})")
        print("=" * 74)
        print(f"  {'component':10} {'strategy':7} {'matched':>7} {'max|Δ|':>12} "
              f"{'changed':>8} {'expect':>7}  result")
        for part, r in report.items():
            if r.get("matched", 0) == 0:
                print(f"  {part:10} {r['strategy']:7} {0:>7}  "
                      f"{'—':>12} {'—':>8} {'—':>7}  ✗ {r.get('reason','')}")
                continue
            print(f"  {part:10} {r['strategy']:7} {r['matched']:>7} "
                  f"{r['max_abs_diff']:>12.3e} {str(r['changed']):>8} "
                  f"{str(r['expect_changed']):>7}  {'✓' if r['ok'] else '✗ MISMATCH'}")
        print("=" * 74)

    if built_here:
        del base_model
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if failures and raise_on_fail:
        details = "; ".join(
            f"{p}({report[p]['strategy']}): "
            f"{'unchanged but should be trained' if report[p].get('expect_changed') else 'changed but should be frozen'}"
            if report[p].get('matched') else f"{p}: no params matched"
            for p in failures
        )
        raise AssertionError(
            f"Weight round-trip verification FAILED for {failures}. {details}. "
            "Trained weights did not survive save->reload->merge, or a frozen "
            "component moved."
        )

    return report
