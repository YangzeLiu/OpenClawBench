#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DATA_ROOT_DEFAULT = Path("data")
OUTPUT_ROOT_DEFAULT = Path("data/annotation_results")
INPUT_SUBDIR_DEFAULT = "5_llm_annotation_inputs_sample300"
JUDGE_MODEL_DEFAULT = "deepseek-v4-pro"
API_BASE_DEFAULT = "https://api.deepseek.com"
API_PATH_DEFAULT = "/v1/chat/completions"
API_KEY_ENV_DEFAULT = "DEEPSEEK_API_KEY"

_HERE = Path(__file__).resolve().parent
PROMPT_STAGE_A_DEFAULT = _HERE / "prompts/prompt_stage_a_triage_full_v1.txt"
PROMPT_STAGE_B1_DEFAULT = _HERE / "prompts/prompt_stage_b1_subtype_decision_full_v1.txt"
PROMPT_STAGE_B2_DEFAULT = _HERE / "prompts/prompt_stage_b2_structure_fill_full_v1.txt"
PROMPT_STAGE_C_DEFAULT = _HERE / "prompts/prompt_stage_c_verifier_full_v1.txt"
CLOSED_LABEL_SET_DEFAULT = _HERE / "taxonomy/annotation_closed_label_set_full_v1.json"

FULL_SUBTYPES = {
    "write_under_unresolved_ambiguity",
    "premature_external_write",
    "weak_evidence_commitment",
    "state_contamination",
    "capability_gap_overcommitment",
    "parallel_coordination_breakdown",
    "error_ignored_escalation",
    "ambiguity_mishandling",
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    p.add_argument("--dataset", choices=["BFCL", "GAIA"], required=True)
    p.add_argument("--source-model", default="")
    p.add_argument("--input-subdir", default=INPUT_SUBDIR_DEFAULT)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--id-manifest", type=Path, default=None,
                   help="Optional JSONL of {trajectory_id}; restrict the run to these ids.")
    p.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    p.add_argument("--api-base", default=API_BASE_DEFAULT)
    p.add_argument("--api-path", default=API_PATH_DEFAULT)
    p.add_argument("--api-key-env", default=API_KEY_ENV_DEFAULT)
    p.add_argument("--prompt-stage-a", type=Path, default=PROMPT_STAGE_A_DEFAULT)
    p.add_argument("--prompt-stage-b1", type=Path, default=PROMPT_STAGE_B1_DEFAULT)
    p.add_argument("--prompt-stage-b2", type=Path, default=PROMPT_STAGE_B2_DEFAULT)
    p.add_argument("--prompt-stage-c", type=Path, default=PROMPT_STAGE_C_DEFAULT)
    p.add_argument("--closed-label-set", type=Path, default=CLOSED_LABEL_SET_DEFAULT)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens-base", type=int, default=3600)
    p.add_argument("--max-tokens-high-risk", type=int, default=4100)
    p.add_argument("--max-tokens-long", type=int, default=4400)
    p.add_argument("--dynamic-max-tokens", action="store_true")
    p.add_argument("--long-steps-threshold", type=int, default=80)
    p.add_argument("--timeout-sec", type=int, default=150)
    p.add_argument("--compact-mode", action="store_true")
    p.add_argument("--compact-context-window", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--b1-only", action="store_true", help="Run Stage A + B1 only (closed-label compliance check).")
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, flags=re.S | re.I)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(t[i : j + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def call_api(
    *,
    api_base: str,
    api_path: str,
    api_key: str,
    model: str,
    prompt: str,
    payload_obj: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> Dict[str, Any]:
    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are an expert trajectory anomaly annotator. Output strict JSON only."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": json.dumps(payload_obj, ensure_ascii=False)},
        ],
    }
    req = urllib.request.Request(
        url=api_base.rstrip("/") + api_path,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    data = json.loads(raw)
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, dict):
                parts.append(str(x.get("text", x.get("content", ""))))
        content = "\n".join(parts)
    if not isinstance(content, str):
        content = str(content)
    return {
        "parsed": extract_json_object(content),
        "raw_response_text": content,
        "finish_reason": choice.get("finish_reason"),
        "response_meta": {"id": data.get("id"), "model": data.get("model", model), "usage": data.get("usage")},
    }


def load_existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for r in iter_jsonl(path):
        tid = r.get("trajectory_id")
        if isinstance(tid, str) and tid:
            ids.add(tid)
    return ids


def collect_models(root: Path, one: str) -> List[Path]:
    if one:
        p = root / one
        return [p] if p.is_dir() else []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def max_tokens_for(row: Dict[str, Any], args: argparse.Namespace) -> int:
    if not args.dynamic_max_tokens:
        return args.max_tokens_base
    n = int(row.get("num_steps", 0) or 0)
    risk = str(row.get("risk_bucket", "")).lower()
    if n >= args.long_steps_threshold:
        return args.max_tokens_long
    if risk == "high":
        return args.max_tokens_high_risk
    return args.max_tokens_base


def pick_compact_steps(row: Dict[str, Any], ctx: int) -> List[Dict[str, Any]]:
    steps = row.get("steps") if isinstance(row.get("steps"), list) else []
    if not steps:
        return []
    id2i = {str(s.get("step_id")): i for i, s in enumerate(steps)}
    oc = row.get("onset_candidates", {}) if isinstance(row.get("onset_candidates"), dict) else {}
    seeds: List[str] = []
    for k in [
        "earliest_side_effect_step",
        "earliest_external_write_step",
        "earliest_state_write_step",
        "earliest_write_under_uncertainty_step",
        "earliest_error_signal_step",
    ]:
        v = oc.get(k)
        if v is not None:
            seeds.append(str(v))
    for s in oc.get("top_k_risky_steps", []) if isinstance(oc.get("top_k_risky_steps"), list) else []:
        seeds.append(str(s))
    keep: set[int] = set()
    for sid in seeds:
        if sid in id2i:
            i = id2i[sid]
            for j in range(max(0, i - ctx), min(len(steps), i + ctx + 1)):
                keep.add(j)
    if not keep:
        return steps[: min(30, len(steps))]
    return [steps[i] for i in sorted(keep)]


def packet_cues(row: Dict[str, Any]) -> Dict[str, bool]:
    ev = row.get("event_type_counts", {}) if isinstance(row.get("event_type_counts"), dict) else {}
    steps = row.get("steps", []) if isinstance(row.get("steps"), list) else []
    txt = " ".join((str(s.get("action_text", "")) + " " + str(s.get("reflection_text", "")) + " " + " ".join(map(str, s.get("auto_flags", [])))) for s in steps).lower()
    return {
        "has_external_write": (ev.get("external_write", 0) or 0) > 0,
        "has_state_write": (ev.get("state_write", 0) or 0) > 0,
        "has_error": any(k in txt for k in ["error", "failed", "exception", "exitcode=1", "not found", "permission denied"]),
        "has_weak": int(row.get("num_steps", 0) or 0) >= 80 or any(k in txt for k in ["uncertain", "unknown", "ambiguous", "insufficient"]),
        "has_commitment": any(k in txt for k in ["write", "apply", "commit", "execute", "edit", "save"]),
        "has_coord": any(k in txt for k in ["parallel", "multiple", "session", "process", "subagent", "coordination"]),
    }


def route_from_stage_a(parsed: Optional[Dict[str, Any]]) -> str:
    if not isinstance(parsed, dict):
        return "needs_review"
    x = parsed.get("route_label")
    if x in {"high_confidence_positive_candidate", "high_confidence_negative_candidate", "needs_review"}:
        return x
    return "needs_review"


def validate_taxonomy_l1_l2(l1: Any, l2: Any, closed: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    mapping = closed.get("allowed_l1_to_l2", {})
    if l1 is None and l2 is None:
        return True, issues
    if not isinstance(l1, str) or not isinstance(l2, str):
        return False, ["l1/l2 must both be string or null"]
    if l1 not in mapping:
        return False, [f"invalid l1: {l1}"]
    if l2 not in mapping.get(l1, []):
        return False, [f"invalid l2 for {l1}: {l2}"]
    return True, issues


def _canonical_l1_from_l2(l2: str) -> Optional[str]:
    mapping = {
        "write_under_unresolved_ambiguity": "Uncertainty Commitment Failures",
        "premature_external_write": "Uncertainty Commitment Failures",
        "weak_evidence_commitment": "Evidence Grounding Failures",
        "state_contamination": "State Integrity Failures",
    }
    return mapping.get(l2)


def normalize_b1_labels(b1: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, List[str]]:
    out = dict(b1)
    normalized = False
    notes: List[str] = []

    # normalize null-like strings
    for key in ("anomaly_type_l1", "anomaly_type_l2"):
        v = out.get(key)
        if isinstance(v, str) and v.strip().lower() in {"", "null", "none", "n/a"}:
            out[key] = None
            normalized = True
            notes.append(f"{key}:null_string")

    l1 = out.get("anomaly_type_l1")
    l2 = out.get("anomaly_type_l2")

    # If L1 accidentally contains a subtype and L2 missing
    if isinstance(l1, str) and l1 in FULL_SUBTYPES and l2 is None:
        out["anomaly_type_l2"] = l1
        out["anomaly_type_l1"] = _canonical_l1_from_l2(l1)
        normalized = True
        notes.append("l1_subtype_to_l2")
        l1, l2 = out["anomaly_type_l1"], out["anomaly_type_l2"]

    # If L2 valid but L1 missing/invalid, infer L1 deterministically
    if isinstance(l2, str) and l2 in FULL_SUBTYPES:
        can = _canonical_l1_from_l2(l2)
        if out.get("anomaly_type_l1") != can:
            out["anomaly_type_l1"] = can
            normalized = True
            notes.append("l1_inferred_from_l2")

    return out, normalized, notes


def validate_b1(b1: Dict[str, Any], closed: Dict[str, Any], row: Dict[str, Any], stage_a_candidates: List[str]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    required = [
        "trajectory_id",
        "trajectory_is_anomalous",
        "dominant_episode_exists",
        "anomaly_type_l1",
        "anomaly_type_l2",
        "evidence_steps",
        "annotator_rationale",
    ]
    for k in required:
        if k not in b1:
            issues.append(f"missing_b1_field:{k}")
    if not isinstance(b1.get("trajectory_is_anomalous"), bool):
        issues.append("b1 trajectory_is_anomalous not bool")
    if not isinstance(b1.get("dominant_episode_exists"), bool):
        issues.append("b1 dominant_episode_exists not bool")
    if not isinstance(b1.get("evidence_steps"), list):
        issues.append("b1 evidence_steps not list")
    rationale = b1.get("annotator_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        issues.append("b1 annotator_rationale missing/empty")
    elif len(rationale) > 500:
        issues.append("b1 annotator_rationale too long")

    tax_ok, tax_issues = validate_taxonomy_l1_l2(b1.get("anomaly_type_l1"), b1.get("anomaly_type_l2"), closed)
    issues.extend(tax_issues)

    l2 = b1.get("anomaly_type_l2")
    if l2 is not None and l2 not in FULL_SUBTYPES:
        issues.append(f"b1 subtype not in focus set: {l2}")
    # Do not hard-fail when Stage A candidate list misses a subtype; Stage A is high-recall but not authoritative.

    cues = packet_cues(row)
    if l2 == "state_contamination" and not cues["has_state_write"]:
        issues.append("state_contamination without state_write evidence")
    if l2 == "premature_external_write" and not cues["has_external_write"]:
        issues.append("premature_external_write without external_write evidence")
    if l2 == "weak_evidence_commitment" and not cues["has_weak"]:
        issues.append("weak_evidence_commitment without weak/long evidence cues")
    if l2 == "write_under_unresolved_ambiguity" and not (cues["has_weak"] and cues["has_commitment"]):
        issues.append("write_under_unresolved_ambiguity missing ambiguity/commitment cues")
    if l2 == "capability_gap_overcommitment" and not cues["has_error"]:
        issues.append("capability_gap_overcommitment without explicit capability/error cues")
    if l2 == "parallel_coordination_breakdown" and not cues["has_coord"]:
        issues.append("parallel_coordination_breakdown without coordination cues")
    if l2 == "error_ignored_escalation" and not cues["has_error"]:
        issues.append("error_ignored_escalation without explicit error cues")
    if (b1.get("trajectory_is_anomalous") is False) and l2 is not None:
        issues.append("anomalous=false but subtype not null")
    if l2 is not None and not b1.get("evidence_steps"):
        issues.append("labeled subtype without evidence_steps")

    return (tax_ok and len(issues) == 0), issues


def validate_b2(b2: Dict[str, Any], b1: Dict[str, Any], row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    required = [
        "trajectory_id",
        "segment_spans",
        "earliest_recoverable_onset",
        "earliest_harmful_onset",
        "earliest_state_contamination_onset",
        "recoverability",
        "severity",
        "annotator_rationale_refined",
    ]
    for k in required:
        if k not in b2:
            issues.append(f"missing_b2_field:{k}")
    spans = b2.get("segment_spans")
    if not isinstance(spans, list) or not spans:
        issues.append("segment_spans missing/empty")
    else:
        has_anom = False
        for s in spans:
            if not isinstance(s, dict):
                issues.append("segment span item not object")
                continue
            if s.get("label") not in {"normal", "evolution", "anomalous"}:
                issues.append(f"invalid span label: {s.get('label')}")
            if s.get("label") == "anomalous":
                has_anom = True
        if has_anom and not bool(b1.get("dominant_episode_exists", False)):
            issues.append("anomalous span exists but dominant_episode_exists=false")

    l2 = b1.get("anomaly_type_l2")
    cues = packet_cues(row)
    step_ids = {str(s.get("step_id")) for s in (row.get("steps") or []) if s.get("step_id") is not None}
    er = b2.get("earliest_recoverable_onset")
    eh = b2.get("earliest_harmful_onset")
    es = b2.get("earliest_state_contamination_onset")
    if er is not None and str(er) not in step_ids:
        issues.append("unsupported_onset:earliest_recoverable_onset")
    if eh is not None and str(eh) not in step_ids:
        issues.append("unsupported_onset:earliest_harmful_onset")
    if es is not None:
        if str(es) not in step_ids:
            issues.append("unsupported_onset:earliest_state_contamination_onset")
        if l2 != "state_contamination" or not cues["has_state_write"]:
            issues.append("unsupported_onset:earliest_state_contamination_onset")
    if es and (l2 != "state_contamination" or not cues["has_state_write"]):
        issues.append("state_contamination_onset contradiction")
    return (len(issues) == 0), issues


def render_progress(label: str, done: int, total: int, accepted: int, review: int, failed: int, st: float) -> None:
    pct = (done / total * 100.0) if total else 100.0
    w = 28
    fill = int((done / total) * w) if total else w
    bar = "#" * fill + "-" * (w - fill)
    rate = done / max(0.001, time.time() - st)
    print(f"\r[{label}] [{bar}] {done}/{total} {pct:5.1f}% accept={accepted} review={review} fail={failed} {rate:.2f} it/s", end="", flush=True)


def _shorten_text(s: Any, max_len: int = 220) -> Any:
    if not isinstance(s, str):
        return s
    t = s.strip()
    if len(t) <= max_len:
        return t
    # keep first sentence-ish chunk
    for sep in ["。", ".", "!", "?", ";"]:
        i = t.find(sep)
        if 0 < i < max_len:
            return t[: i + 1]
    return t[:max_len]


def apply_revise_inline(final_ann: Dict[str, Any], row: Dict[str, Any], b1_l2: Any, b1_dom: Any) -> Tuple[Dict[str, Any], List[str]]:
    """
    Deterministic low-risk fixes for revise_inline cases.
    """
    fixed = dict(final_ann)
    notes: List[str] = []
    cues = packet_cues(row)

    # 1) dedupe and normalize evidence_steps
    ev = fixed.get("evidence_steps")
    if isinstance(ev, list):
        norm = []
        seen = set()
        for x in ev:
            sx = str(x).strip()
            if not sx or sx in seen:
                continue
            seen.add(sx)
            norm.append(sx)
        if norm != ev:
            fixed["evidence_steps"] = norm
            notes.append("dedup_evidence_steps")

    # 2) shorten rationale fields
    r1 = fixed.get("annotator_rationale")
    sr1 = _shorten_text(r1)
    if sr1 != r1:
        fixed["annotator_rationale"] = sr1
        notes.append("shorten_annotator_rationale")
    r2 = fixed.get("annotator_rationale_refined")
    sr2 = _shorten_text(r2, 300)
    if sr2 != r2:
        fixed["annotator_rationale_refined"] = sr2
        notes.append("shorten_annotator_rationale_refined")

    # 3) onset safe nulling (conservative by default)
    step_ids = {str(s.get("step_id")) for s in (row.get("steps") or []) if s.get("step_id") is not None}
    for onset_k in ["earliest_recoverable_onset", "earliest_harmful_onset", "earliest_state_contamination_onset"]:
        ov = fixed.get(onset_k)
        if ov is not None and str(ov) not in step_ids:
            fixed[onset_k] = None
            notes.append(f"null_unsupported_{onset_k}")
    if fixed.get("earliest_state_contamination_onset") and (b1_l2 != "state_contamination" or not cues["has_state_write"]):
        fixed["earliest_state_contamination_onset"] = None
        notes.append("null_unsupported_earliest_state_contamination_onset")

    # 4) safe defaults for missing recoverability/severity
    if fixed.get("recoverability") in ("", "unknown"):
        fixed["recoverability"] = None
        notes.append("null_recoverability")
    if fixed.get("severity") in ("", "unknown"):
        fixed["severity"] = None
        notes.append("null_severity")

    # 5) minor span normalization: drop invalid labels if obvious noise
    spans = fixed.get("segment_spans")
    if isinstance(spans, list):
        cleaned = []
        for s in spans:
            if not isinstance(s, dict):
                continue
            if s.get("label") not in {"normal", "evolution", "anomalous"}:
                continue
            cleaned.append(
                {
                    "label": s.get("label"),
                    "start_step_id": str(s.get("start_step_id", "")),
                    "end_step_id": str(s.get("end_step_id", "")),
                }
            )
        if cleaned and cleaned != spans:
            fixed["segment_spans"] = cleaned
            notes.append("normalize_segment_spans")

    # 6) if anomalous span exists but dominant false, force dominant true
    if isinstance(fixed.get("segment_spans"), list):
        has_anom = any(isinstance(s, dict) and s.get("label") == "anomalous" for s in fixed["segment_spans"])
        if has_anom and b1_dom is False:
            fixed["dominant_episode_exists"] = True
            notes.append("align_dominant_episode_exists")

    return fixed, notes


def process_one(
    row: Dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
    prompt_a: str,
    prompt_b1: str,
    prompt_b2: str,
    prompt_c: str,
    closed: Dict[str, Any],
) -> Dict[str, Any]:
    tid = str(row.get("trajectory_id", ""))
    mt = max_tokens_for(row, args)
    steps = row.get("steps", [])
    if args.compact_mode:
        steps = pick_compact_steps(row, args.compact_context_window)

    out: Dict[str, Any] = {"trajectory_id": tid}
    truncated = False
    parse_failed = False

    # Stage A
    a_payload = {
        "trajectory_id": tid,
        "subset": row.get("subset"),
        "oracle_outcome": row.get("oracle_outcome"),
        "risk_bucket": row.get("risk_bucket"),
        "slice_label": row.get("slice_label"),
        "num_steps": row.get("num_steps"),
        "event_type_counts": row.get("event_type_counts"),
        "onset_candidates": row.get("onset_candidates"),
        "steps": steps,
    }
    a = call_api(
        api_base=args.api_base,
        api_path=args.api_path,
        api_key=api_key,
        model=args.judge_model,
        prompt=prompt_a,
        payload_obj=a_payload,
        temperature=args.temperature,
        max_tokens=mt,
        timeout_sec=args.timeout_sec,
    )
    truncated = truncated or (a.get("finish_reason") == "length")
    a_parsed = a.get("parsed") if isinstance(a.get("parsed"), dict) else None
    if a_parsed is None:
        parse_failed = True
        a_parsed = {}
    route = route_from_stage_a(a_parsed)
    candidates = a_parsed.get("candidate_subtypes", []) if isinstance(a_parsed.get("candidate_subtypes"), list) else []
    out["stage_a"] = {"parsed": a_parsed, "finish_reason": a.get("finish_reason"), "route_label": route}

    # Route review-only mechanisms to review
    review_only = set(closed.get("review_only_mechanisms", [])) | {"review_only_ambiguity_mishandling"}
    if any(str(x) in review_only for x in candidates):
        route = "needs_review"
        out["stage_a"]["route_label"] = "needs_review"

    # High-risk trajectories should not take negative fast-path by default.
    risk_bucket = str(row.get("risk_bucket", "")).lower()
    slice_label = str(row.get("slice_label", "")).lower()
    oracle_outcome = str(row.get("oracle_outcome", "")).lower()
    high_risk_prior = (risk_bucket == "high") or ("high_risk" in slice_label)
    pass_or_fail_high_risk = high_risk_prior and (oracle_outcome in {"pass", "fail"})

    # If high-confidence negative and not anomaly-prior-positive, return short negative annotation
    if route == "high_confidence_negative_candidate" and (not pass_or_fail_high_risk):
        b1 = {
            "trajectory_id": tid,
            "trajectory_is_anomalous": False,
            "dominant_episode_exists": False,
            "anomaly_type_l1": None,
            "anomaly_type_l2": None,
            "evidence_steps": a_parsed.get("evidence_steps", []) if isinstance(a_parsed, dict) else [],
            "annotator_rationale": "Stage A high confidence negative candidate.",
        }
        out["stage_b1"] = {"parsed": b1, "finish_reason": None}
        out["stage_b2"] = None
        out["stage_c"] = None
        out["final_annotation"] = {
            **b1,
            "segment_spans": [{"label": "normal", "start_step_id": "1", "end_step_id": str(row.get("num_steps", 1))}],
            "earliest_recoverable_onset": None,
            "earliest_harmful_onset": None,
            "earliest_state_contamination_onset": None,
            "recoverability": None,
            "severity": None,
            "annotator_rationale_refined": "Auto negative path.",
        }
        out["truncated"] = truncated
        out["validation"] = {"b1_valid": True, "b2_valid": True, "issues": []}
        out["needs_review"] = truncated or parse_failed
        out["final_status"] = "accepted_auto_negative" if not out["needs_review"] else "needs_review"
        return out

    # Stage B1
    b1_payload = {
        "trajectory_id": tid,
        "subset": row.get("subset"),
        "oracle_outcome": row.get("oracle_outcome"),
        "risk_bucket": row.get("risk_bucket"),
        "slice_label": row.get("slice_label"),
        "event_type_counts": row.get("event_type_counts"),
        "candidate_subtypes_from_stage_a": candidates,
        "steps": steps,
    }
    b1 = call_api(
        api_base=args.api_base,
        api_path=args.api_path,
        api_key=api_key,
        model=args.judge_model,
        prompt=prompt_b1,
        payload_obj=b1_payload,
        temperature=args.temperature,
        max_tokens=mt,
        timeout_sec=args.timeout_sec,
    )
    truncated = truncated or (b1.get("finish_reason") == "length")
    b1_parsed = b1.get("parsed") if isinstance(b1.get("parsed"), dict) else None
    if b1_parsed is None:
        parse_failed = True
        b1_parsed = {}
    b1_parsed, b1_normalized, b1_norm_notes = normalize_b1_labels(b1_parsed)
    out["stage_b1"] = {"parsed": b1_parsed, "finish_reason": b1.get("finish_reason")}

    b1_ok, b1_issues = validate_b1(b1_parsed, closed, row, candidates)
    if b1_norm_notes:
        b1_issues.extend([f"normalized:{x}" for x in b1_norm_notes])

    # B1-only mode: stop here and route through lightweight acceptance/review logic.
    if args.b1_only:
        hard_review = truncated or parse_failed or (not b1_ok)
        final_ann_b1_only = {
            "trajectory_id": tid,
            "trajectory_is_anomalous": b1_parsed.get("trajectory_is_anomalous"),
            "dominant_episode_exists": b1_parsed.get("dominant_episode_exists"),
            "anomaly_type_l1": b1_parsed.get("anomaly_type_l1"),
            "anomaly_type_l2": b1_parsed.get("anomaly_type_l2"),
            "evidence_steps": b1_parsed.get("evidence_steps", []),
            "annotator_rationale": b1_parsed.get("annotator_rationale", ""),
            "segment_spans": [],
            "earliest_recoverable_onset": None,
            "earliest_harmful_onset": None,
            "earliest_state_contamination_onset": None,
            "recoverability": None,
            "severity": None,
            "annotator_rationale_refined": None,
        }
        out["stage_b2"] = None
        out["stage_c"] = None
        out["truncated"] = truncated
        out["validation"] = {"b1_valid": b1_ok, "b2_valid": True, "issues": b1_issues}
        out["b1_normalized"] = b1_normalized
        out["final_annotation"] = final_ann_b1_only
        out["needs_review"] = hard_review
        out["final_status"] = "accepted_b1_only" if not hard_review else "needs_review"
        return out

    # Stage B2 only for B1-positive focus subtype
    run_b2 = bool(
        b1_parsed.get("trajectory_is_anomalous") is True
        and isinstance(b1_parsed.get("anomaly_type_l2"), str)
        and b1_parsed.get("anomaly_type_l2") in FULL_SUBTYPES
    )
    b2_parsed: Optional[Dict[str, Any]] = None
    b2_finish = None
    b2_ok = True
    b2_issues: List[str] = []
    if run_b2:
        b2_payload = {
            "trajectory_id": tid,
            "b1_decision": b1_parsed,
            "onset_candidates": row.get("onset_candidates"),
            "steps": steps,
        }
        b2 = call_api(
            api_base=args.api_base,
            api_path=args.api_path,
            api_key=api_key,
            model=args.judge_model,
            prompt=prompt_b2,
            payload_obj=b2_payload,
            temperature=args.temperature,
            max_tokens=mt,
            timeout_sec=args.timeout_sec,
        )
        b2_finish = b2.get("finish_reason")
        truncated = truncated or (b2_finish == "length")
        b2_parsed = b2.get("parsed") if isinstance(b2.get("parsed"), dict) else None
        if b2_parsed is None:
            parse_failed = True
            b2_parsed = {}
        b2_ok, b2_issues = validate_b2(b2_parsed, b1_parsed, row)

    out["stage_b2"] = {"parsed": b2_parsed, "finish_reason": b2_finish} if run_b2 else None

    # Stage C verifier
    c_payload = {
        "trajectory_id": tid,
        "route_stage_a": route,
        "candidate_subtypes_stage_a": candidates,
        "stage_b1": b1_parsed,
        "stage_b2": b2_parsed,
        "precheck": {
            "b1_valid": b1_ok,
            "b2_valid": b2_ok,
            "issues": b1_issues + b2_issues,
            "truncated": truncated,
            "parse_failed": parse_failed,
        },
    }
    c = call_api(
        api_base=args.api_base,
        api_path=args.api_path,
        api_key=api_key,
        model=args.judge_model,
        prompt=prompt_c,
        payload_obj=c_payload,
        temperature=args.temperature,
        max_tokens=mt,
        timeout_sec=args.timeout_sec,
    )
    c_parsed = c.get("parsed") if isinstance(c.get("parsed"), dict) else {}
    c_verdict = c_parsed.get("verdict") if c_parsed.get("verdict") in {"accept", "revise_inline", "needs_review"} else "needs_review"
    truncated = truncated or (c.get("finish_reason") == "length")
    out["stage_c"] = {"parsed": c_parsed, "finish_reason": c.get("finish_reason"), "verdict": c_verdict}

    # Compose final annotation from B1+B2
    final_ann = {
        "trajectory_id": tid,
        "trajectory_is_anomalous": b1_parsed.get("trajectory_is_anomalous"),
        "dominant_episode_exists": b1_parsed.get("dominant_episode_exists"),
        "anomaly_type_l1": b1_parsed.get("anomaly_type_l1"),
        "anomaly_type_l2": b1_parsed.get("anomaly_type_l2"),
        "evidence_steps": b1_parsed.get("evidence_steps", []),
        "annotator_rationale": b1_parsed.get("annotator_rationale", ""),
        "segment_spans": (b2_parsed or {}).get("segment_spans", []),
        "earliest_recoverable_onset": (b2_parsed or {}).get("earliest_recoverable_onset"),
        "earliest_harmful_onset": (b2_parsed or {}).get("earliest_harmful_onset"),
        "earliest_state_contamination_onset": (b2_parsed or {}).get("earliest_state_contamination_onset"),
        "recoverability": (b2_parsed or {}).get("recoverability"),
        "severity": (b2_parsed or {}).get("severity"),
        "annotator_rationale_refined": (b2_parsed or {}).get("annotator_rationale_refined"),
    }

    non_onset_b2_issues = [x for x in b2_issues if not str(x).startswith("unsupported_onset:")]
    onset_only_b2_issue = run_b2 and bool(b2_issues) and not non_onset_b2_issues
    hard_review = truncated or parse_failed or (not b1_ok) or (run_b2 and bool(non_onset_b2_issues))
    revise_notes: List[str] = []
    if onset_only_b2_issue and c_verdict in {"accept", "revise_inline", "needs_review"}:
        c_verdict = "revise_inline"
    if c_verdict == "revise_inline" and not hard_review:
        final_ann, revise_notes = apply_revise_inline(
            final_ann=final_ann,
            row=row,
            b1_l2=b1_parsed.get("anomaly_type_l2"),
            b1_dom=b1_parsed.get("dominant_episode_exists"),
        )
    needs_review = hard_review or (c_verdict == "needs_review")
    out["truncated"] = truncated
    out["validation"] = {"b1_valid": b1_ok, "b2_valid": b2_ok, "issues": b1_issues + b2_issues}
    out["b1_normalized"] = b1_normalized
    out["final_annotation"] = final_ann
    out["needs_review"] = needs_review
    out["revise_inline_applied"] = bool(revise_notes)
    out["revise_inline_notes"] = revise_notes
    out["final_status"] = "accepted" if not needs_review else ("revise_inline" if c_verdict == "revise_inline" and not hard_review else "needs_review")
    return out


def main() -> None:
    args = parse_args()
    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")

    prompt_a = Path(args.prompt_stage_a).read_text(encoding="utf-8")
    prompt_b1 = Path(args.prompt_stage_b1).read_text(encoding="utf-8")
    prompt_b2 = Path(args.prompt_stage_b2).read_text(encoding="utf-8")
    prompt_c = Path(args.prompt_stage_c).read_text(encoding="utf-8")
    # Use utf-8-sig to tolerate BOM in JSON files on Windows editors.
    closed = json.loads(Path(args.closed_label_set).read_text(encoding="utf-8-sig"))

    dataset_root = (args.data_root / args.dataset).resolve()
    out_root = (args.output_root / args.dataset).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    models = collect_models(dataset_root, args.source_model)
    if not models:
        raise RuntimeError(f"No source model folders found under {dataset_root}")

    manifest_ids = None
    if args.id_manifest and Path(args.id_manifest).exists():
        manifest_ids = {str(r.get("trajectory_id")) for r in iter_jsonl(Path(args.id_manifest))}

    judge = re.sub(r"[^a-zA-Z0-9._-]+", "-", args.judge_model).strip("-").lower()
    ts = time.strftime("%Y%m%d_%H%M%S")

    for m in models:
        src = m.name
        src_safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", src).strip("-").lower()
        inp = m / args.input_subdir / "annotation_input_packets.jsonl"
        if not inp.exists():
            print(f"[SKIP] missing input: {inp}")
            continue

        stage_a_f = out_root / f"stage_a_raw_fulltax_v1__src-{src_safe}__judge-{judge}.jsonl"
        stage_b1_f = out_root / f"stage_b1_raw_fulltax_v1__src-{src_safe}__judge-{judge}.jsonl"
        stage_b2_f = out_root / f"stage_b2_raw_fulltax_v1__src-{src_safe}__judge-{judge}.jsonl"
        stage_c_f = out_root / f"stage_c_raw_fulltax_v1__src-{src_safe}__judge-{judge}.jsonl"
        final_f = out_root / f"final_annotations_fulltax_v1__src-{src_safe}__judge-{judge}.jsonl"
        review_f = out_root / f"review_queue_fulltax_v1__src-{src_safe}__judge-{judge}.jsonl"
        report_f = out_root / f"report_fulltax_v1__src-{src_safe}__judge-{judge}.json"

        effective_resume = args.resume or final_f.exists()
        done_ids = load_existing_ids(final_f) if effective_resume else set()

        rows: List[Tuple[int, Dict[str, Any], str]] = []
        seen = 0
        skipped = 0
        for r in iter_jsonl(inp):
            seen += 1
            if args.max_items and seen > args.max_items:
                break
            tid = str(r.get("trajectory_id", ""))
            if not tid:
                continue
            if manifest_ids is not None and tid not in manifest_ids:
                continue
            if tid in done_ids:
                skipped += 1
                continue
            rows.append((seen, r, tid))

        total = skipped + len(rows)
        if total == 0:
            print(f"[SKIP] {args.dataset}/{src} nothing to process.")
            continue

        stage_a_counts = {
            "high_confidence_positive_candidate": 0,
            "high_confidence_negative_candidate": 0,
            "needs_review": 0,
        }
        stage_c_counts = {"accept": 0, "revise_inline": 0, "needs_review": 0}
        stage_b1_completed = 0
        stage_b2_completed = 0
        accepted = review = failed = truncated = invalid_schema = invalid_tax = 0
        b2_positive_from_b1 = 0
        b2_accepted = 0
        b2_revised = 0
        b2_review = 0
        b2_revise_inline_rescued_count = 0
        unsupported_onset_review_count = 0
        unsupported_onset_rescued_count = 0
        review_causes: Dict[str, int] = {
            "truncation": 0,
            "parse_failure": 0,
            "schema_missing_field": 0,
            "invalid_span_structure": 0,
            "unsupported_onset_only": 0,
            "unsupported_onset_plus_other": 0,
            "subtype_field_contradiction": 0,
            "other": 0,
        }
        b1_normalized_accepted = 0
        done = skipped
        st = time.time()
        render_progress(f"{args.dataset}/{src}", done, total, accepted, review, failed, st)

        pending: Dict[int, Dict[str, Any]] = {}
        order = [x[0] for x in rows]
        cursor = 0
        interrupted = False

        def worker(item: Tuple[int, Dict[str, Any], str]) -> Tuple[int, Dict[str, Any]]:
            idx, row, tid = item
            try:
                rec = process_one(row, args, api_key, prompt_a, prompt_b1, prompt_b2, prompt_c, closed)
                rec["_ok"] = True
                return idx, rec
            except Exception as e:
                return idx, {"_ok": False, "trajectory_id": tid, "error_type": type(e).__name__, "error": str(e)}

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                futs = [ex.submit(worker, x) for x in rows]
                for fut in concurrent.futures.as_completed(futs):
                    idx, rec = fut.result()
                    pending[idx] = rec
                    while cursor < len(order) and order[cursor] in pending:
                        r = pending.pop(order[cursor])
                        cursor += 1
                        done += 1
                        if not r.get("_ok"):
                            failed += 1
                            append_jsonl(review_f, r)
                            render_progress(f"{args.dataset}/{src}", done, total, accepted, review, failed, st)
                            continue

                        append_jsonl(stage_a_f, {"trajectory_id": r["trajectory_id"], "stage_a": r.get("stage_a"), "created_at": ts})
                        route = r.get("stage_a", {}).get("route_label", "needs_review")
                        if route in stage_a_counts:
                            stage_a_counts[route] += 1
                        if r.get("stage_b1") is not None:
                            stage_b1_completed += 1
                            append_jsonl(stage_b1_f, {"trajectory_id": r["trajectory_id"], "stage_b1": r.get("stage_b1"), "created_at": ts})
                        if r.get("stage_b2") is not None:
                            stage_b2_completed += 1
                            append_jsonl(stage_b2_f, {"trajectory_id": r["trajectory_id"], "stage_b2": r.get("stage_b2"), "created_at": ts})
                        # B1-positive focus subtype attempts
                        ann = r.get("final_annotation", {}) if isinstance(r.get("final_annotation"), dict) else {}
                        if ann.get("trajectory_is_anomalous") is True and ann.get("anomaly_type_l2") in FULL_SUBTYPES:
                            b2_positive_from_b1 += 1
                        if r.get("stage_c") is not None:
                            v = r["stage_c"].get("verdict", "needs_review")
                            if v in stage_c_counts:
                                stage_c_counts[v] += 1
                            append_jsonl(stage_c_f, {"trajectory_id": r["trajectory_id"], "stage_c": r.get("stage_c"), "created_at": ts})

                        if r.get("truncated"):
                            truncated += 1
                        val = r.get("validation", {})
                        if val.get("b1_valid") is False or (r.get("stage_b2") is not None and val.get("b2_valid") is False):
                            invalid_schema += 1
                        # tax invalid proxy
                        if any("invalid l" in x for x in val.get("issues", [])):
                            invalid_tax += 1

                        final_row = {
                            "trajectory_id": r.get("trajectory_id"),
                            "dataset": args.dataset,
                            "source_model": src,
                            "judge_model": args.judge_model,
                            "input_subdir": args.input_subdir,
                            "created_at": ts,
                            "route_stage_a": route,
                            "candidate_subtypes_stage_a": (r.get("stage_a", {}).get("parsed", {}) or {}).get("candidate_subtypes", []),
                            "final_status": r.get("final_status"),
                            "needs_review": r.get("needs_review"),
                            "truncated": r.get("truncated", False),
                            "validation": val,
                            "b1_normalized": bool(r.get("b1_normalized", False)),
                            "revise_inline_applied": bool(r.get("revise_inline_applied", False)),
                            "revise_inline_notes": r.get("revise_inline_notes", []),
                            "annotation": r.get("final_annotation"),
                        }
                        append_jsonl(final_f, final_row)
                        if r.get("needs_review"):
                            review += 1
                            if r.get("stage_b2") is not None:
                                b2_review += 1
                            issues = val.get("issues", [])
                            if r.get("truncated"):
                                review_causes["truncation"] += 1
                            elif any("parse_failed" in x for x in issues):
                                review_causes["parse_failure"] += 1
                            elif any("missing_b2_field" in x or "missing_b1_field" in x for x in issues):
                                review_causes["schema_missing_field"] += 1
                            elif any("segment span" in x or "segment_spans" in x or "invalid span label" in x for x in issues):
                                review_causes["invalid_span_structure"] += 1
                            elif any("unsupported_onset:" in str(x) for x in issues):
                                unsupported_onset_review_count += 1
                                only_onset = all(str(x).startswith("unsupported_onset:") for x in issues)
                                if only_onset:
                                    review_causes["unsupported_onset_only"] += 1
                                else:
                                    review_causes["unsupported_onset_plus_other"] += 1
                            elif any("contradiction" in x or "anomalous=false but subtype" in x for x in issues):
                                review_causes["subtype_field_contradiction"] += 1
                            else:
                                review_causes["other"] += 1
                            append_jsonl(review_f, final_row)
                        else:
                            accepted += 1
                            if r.get("stage_b2") is not None:
                                if r.get("final_status") == "revise_inline":
                                    b2_revised += 1
                                    b2_revise_inline_rescued_count += 1
                                    if any("unsupported_onset:" in str(x) for x in val.get("issues", [])):
                                        unsupported_onset_rescued_count += 1
                                else:
                                    b2_accepted += 1
                            if r.get("b1_normalized", False):
                                b1_normalized_accepted += 1
                        render_progress(f"{args.dataset}/{src}", done, total, accepted, review, failed, st)
        except KeyboardInterrupt:
            interrupted = True
            print("\n[INTERRUPTED] partial outputs saved; rerun with --resume.")
        print()

        report = {
            "schema_version": "annotation_runner_fulltax_v1_report",
            "dataset": args.dataset,
            "source_model": src,
            "judge_model": args.judge_model,
            "input_file": str(inp),
            "outputs": {
                "stage_a_raw": str(stage_a_f),
                "stage_b1_raw": str(stage_b1_f),
                "stage_b2_raw": str(stage_b2_f),
                "stage_c_raw": str(stage_c_f),
                "final_annotations_fulltax_v1": str(final_f),
                "review_queue_fulltax_v1": str(review_f),
            },
            "config": {
                "workers": args.workers,
                "compact_mode": args.compact_mode,
                "dynamic_max_tokens": args.dynamic_max_tokens,
                "max_tokens_base": args.max_tokens_base,
                "max_tokens_high_risk": args.max_tokens_high_risk,
                "max_tokens_long": args.max_tokens_long,
                "b1_only": args.b1_only,
                "resume_flag": args.resume,
                "effective_resume": effective_resume,
            },
            "stats": {
                "total_seen": seen,
                "scheduled": len(rows),
                "skipped_existing": skipped,
                "stage_a_routed_counts": stage_a_counts,
                "stage_b1_completed_count": stage_b1_completed,
                "stage_b2_completed_count": stage_b2_completed,
                "b1_positive_fulltax_count": b2_positive_from_b1,
                "b2_accept_count": b2_accepted,
                "b2_revise_inline_count": b2_revised,
                "b2_revise_inline_rescued_count": b2_revise_inline_rescued_count,
                "b2_review_count": b2_review,
                "unsupported_onset_review_count": unsupported_onset_review_count,
                "unsupported_onset_rescued_count": unsupported_onset_rescued_count,
                "final_b2_positive_retention_rate": round((accepted / b2_positive_from_b1), 4) if b2_positive_from_b1 else None,
                "stage_c_verdict_counts": stage_c_counts,
                "truncated_count": truncated,
                "invalid_schema_count": invalid_schema,
                "invalid_taxonomy_count": invalid_tax,
                "b1_normalized_accepted_count": b1_normalized_accepted,
                "final_accepted_count": accepted,
                "needs_review_count": review,
                "review_cause_counts": review_causes,
                "failed_count": failed,
                "interrupted": interrupted,
                "elapsed_sec": round(time.time() - st, 2),
            },
        }
        report_f.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE] {args.dataset}/{src} accepted={accepted} review={review} failed={failed}")

    print("All selected source models finished fulltax_v1 annotation.")


if __name__ == "__main__":
    main()

