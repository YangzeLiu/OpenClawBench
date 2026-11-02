"""Detector input/output schema + label space.

Single source of truth — all loader / inference / metric code references this.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


PROMPT_VERSION = "v0.1"

# Closed L2 label set (8 subtypes), aligned to anomaly_taxonomy_full_v1.
ANOMALY_L2 = [
    "write_under_unresolved_ambiguity",
    "premature_external_write",
    "weak_evidence_commitment",
    "state_contamination",
    "capability_gap_overcommitment",
    "parallel_coordination_breakdown",
    "error_ignored_escalation",
    "ambiguity_mishandling",
]

ANOMALY_L1 = [
    "Uncertainty Commitment Failures",
    "Evidence Grounding Failures",
    "Capability Boundary Failures",
    "State Integrity Failures",
    "Coordination and Control Failures",
]

L2_TO_L1 = {
    "write_under_unresolved_ambiguity": "Uncertainty Commitment Failures",
    "premature_external_write": "Uncertainty Commitment Failures",
    "ambiguity_mishandling": "Uncertainty Commitment Failures",
    "weak_evidence_commitment": "Evidence Grounding Failures",
    "capability_gap_overcommitment": "Capability Boundary Failures",
    "state_contamination": "State Integrity Failures",
    "parallel_coordination_breakdown": "Coordination and Control Failures",
    "error_ignored_escalation": "Coordination and Control Failures",
}

# 6 base models in the gold test pool. Used for per-base bucket reporting.
GOLD_BASE_MODELS = [
    "deepseek-r1-bfcl",
    "glm-4-5-air",
    "gpt-oss-20b",
    "qwen3-5-9b",
    "qwen3-6-27b",
    "qwen3-6-35b-a3b",
]


@dataclass
class TrajectoryStep:
    step_id: str
    thinking: str = ""
    action_text: str = ""
    reflection_text: str = ""
    event_type: str = ""
    side_effect: str = ""
    artifact_target: str = ""
    state_write_target: str = ""
    auto_flags: list[str] = field(default_factory=list)


@dataclass
class TrajectoryPacket:
    """Detector input. Mirrors annotation_input_packets.jsonl schema."""

    trajectory_id: str
    subset: str
    oracle_outcome: str  # "pass" | "fail"
    risk_bucket: str  # "low" | "medium" | "high"
    risk_score: float
    slice_label: str
    num_steps: int
    event_type_counts: dict
    onset_candidates: dict
    steps: list[TrajectoryStep]
    # source_model is NOT in the packet; injected by loader from file path.
    source_model: str = ""

    @classmethod
    def from_jsonl_row(cls, row: dict, source_model: str = "") -> "TrajectoryPacket":
        return cls(
            trajectory_id=row["trajectory_id"],
            subset=row.get("subset", ""),
            oracle_outcome=row.get("oracle_outcome", ""),
            risk_bucket=row.get("risk_bucket", ""),
            risk_score=float(row.get("risk_score") or 0.0),
            slice_label=row.get("slice_label", ""),
            num_steps=int(row.get("num_steps") or 0),
            event_type_counts=row.get("event_type_counts") or {},
            onset_candidates=row.get("onset_candidates") or {},
            steps=[TrajectoryStep(**s) for s in (row.get("steps") or [])],
            source_model=source_model,
        )


@dataclass
class AnnotationLabel:
    """Detector output (and silver/gold target). Mirrors `annotation` block of
    final_annotations_*.jsonl."""

    trajectory_id: str
    trajectory_is_anomalous: Optional[bool]
    anomaly_type_l1: Optional[str]
    anomaly_type_l2: Optional[str]
    earliest_harmful_onset: Optional[str]
    earliest_recoverable_onset: Optional[str] = None
    earliest_state_contamination_onset: Optional[str] = None
    severity: Optional[str] = None
    recoverability: Optional[str] = None
    evidence_steps: list[str] = field(default_factory=list)
    annotator_rationale: str = ""

    @classmethod
    def from_final_annotation_row(cls, row: dict) -> "AnnotationLabel":
        ann = row.get("annotation") or {}
        return cls(
            trajectory_id=ann.get("trajectory_id") or row.get("trajectory_id", ""),
            trajectory_is_anomalous=ann.get("trajectory_is_anomalous"),
            anomaly_type_l1=ann.get("anomaly_type_l1"),
            anomaly_type_l2=ann.get("anomaly_type_l2"),
            earliest_harmful_onset=ann.get("earliest_harmful_onset"),
            earliest_recoverable_onset=ann.get("earliest_recoverable_onset"),
            earliest_state_contamination_onset=ann.get("earliest_state_contamination_onset"),
            severity=ann.get("severity"),
            recoverability=ann.get("recoverability"),
            evidence_steps=list(ann.get("evidence_steps") or []),
            annotator_rationale=str(ann.get("annotator_rationale") or ""),
        )

    def to_target_json(self) -> dict:
        """Compact JSON used as fine-tune target (and zero-shot output spec)."""
        return {
            "is_anomaly": bool(self.trajectory_is_anomalous),
            "anomaly_type_l2": self.anomaly_type_l2,
            "earliest_harmful_onset": self.earliest_harmful_onset,
        }


@dataclass
class DetectorPrediction:
    trajectory_id: str
    is_anomaly: Optional[bool]
    anomaly_type_l2: Optional[str]
    earliest_harmful_onset: Optional[str]
    raw_text: str = ""  # raw model output for debugging


def validate_l2(label: Optional[str]) -> Optional[str]:
    """Coerce free-text L2 to the closed set; return None if invalid."""
    if not label:
        return None
    label = label.strip()
    if label in ANOMALY_L2:
        return label
    return None
