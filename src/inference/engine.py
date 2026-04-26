import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_PROCESSED_DIR = _DATA_DIR / "processed"
_DEFAULT_INPUT = _PROCESSED_DIR / "fhir_lite.jsonl"

_MAX_OBS_ENTRIES: int  = 1100
_MAX_LAB_ENTRIES: int  = 150
_MAX_MED_ENTRIES: int  = 250
_MAX_PROC_ENTRIES: int = 20

_VITAL_METRICS: frozenset = frozenset({
    "Heart Rate",
    "Systolic blood pressure",
    "Diastolic blood pressure",
    "Mean blood pressure",
    "Central Venous Pressure",
    "Pulmonary Artery Pressure systolic",
    "Pulmonary Artery Pressure diastolic",
    "Pulmonary Artery Pressure mean",
    "Respiratory rate",
    "Oxygen saturation",
    "O2 Flow",
    "Temperature",
    "Intra Cranial Pressure",
    "Cerebral Perfusion Pressure",
    "Glucose",
    "Heart Rhythm",
    "Ectopy Type 1",
    "O2 Delivery Device(s)",
    "Respiratory Pattern",
    "Respiratory Effort",
    "Level of Consciousness",
    "Richmond-RAS Scale",
    "GCS - Eye Opening",
    "GCS - Verbal Response",
    "GCS - Motor Response",
    "Pain Present",
    "Pain Level",
})

_STABLE_DELTA: Dict[str, float] = {
    "Heart Rate":                          10.0,
    "Systolic blood pressure":             15.0,
    "Diastolic blood pressure":            10.0,
    "Mean blood pressure":                 10.0,
    "Respiratory rate":                     4.0,
    "Oxygen saturation":                    3.0,
    "Temperature":                          0.5,
    "Central Venous Pressure":              4.0,
    "Intra Cranial Pressure":               5.0,
    "Cerebral Perfusion Pressure":         10.0,
    "Pulmonary Artery Pressure systolic":  10.0,
    "Pulmonary Artery Pressure diastolic": 10.0,
    "Pulmonary Artery Pressure mean":      10.0,
    "Glucose":                             30.0,
    "O2 Flow":                              2.0,
}
_DEFAULT_STABLE_DELTA: float = 10.0

_MAX_MODEL_LEN: int = 32_768

_DEFAULT_TEMPERATURE: float = 0.3
_DEFAULT_TOP_P: float = 0.9
_DEFAULT_MAX_TOKENS: int = 2048
_DEFAULT_BATCH_SIZE: int = 32
_DEFAULT_REPETITION_PENALTY: float = 1.15

_LOG_INTERVAL: int = 50

_SYSTEM_PROMPT: str = (
    "You are an experienced MICU nurse writing an end-of-shift progress note. "
    "Use clinical shorthand. Do not invent medications, procedures, or diagnoses "
    "not present in the context. Do not include a document header."
)

_FEW_SHOT_EXAMPLE: str = """\
## Example note (for format reference only; do not copy content)

Impaired Skin Integrity
Assessment:
All skin surfaces as documented in flowsheets. Some sloughing on buttocks, \
pt stooling loose bilious green BM.
Action:
Buttocks cleansed, antifungal cream and adaptic applied. Flexiseal ordered \
and inserted.
Response:
Buttock areas free of fecal contamination overnoc, areas appear slightly \
less irritated.
Plan:
Cleanse and treat impaired skin areas, protect open areas from friction and \
fecal contamination.

Renal failure, acute (ARF)
Assessment:
CRRT as ordered, no difficulties with circuit. UO via foley 10-15 cc clear \
yellow.
Action:
Removed 1-200 cc/h. Ca and K gtts titrated per labs.
Response:
Attained fluid removal goal, neg 2700 cc yesterday. Hemodynamically stable.
Plan:
Fluid removal goal 2.5-3 L/day. Monitor lytes q6 and titrate repletion gtts \
as ordered.\
"""

_CLINICAL_RANGES: Dict[str, Tuple[float, float, str, str]] = {
    "Heart Rate":              (60.0,  100.0,  "bpm",   "heart rate"),
    "Systolic blood pressure": (90.0,  140.0,  "mmHg",  "systolic BP"),
    "Diastolic blood pressure":(50.0,   90.0,  "mmHg",  "diastolic BP"),
    "Mean blood pressure":     (65.0,  105.0,  "mmHg",  "MAP"),
    "Respiratory rate":        (12.0,   20.0,  "br/min","respiratory rate"),
    "Oxygen saturation":       (95.0,  100.0,  "%",     "SpO2"),
    "Temperature":             (97.7,   99.5,  "°F",    "temperature"),
    "Central Venous Pressure": ( 2.0,   12.0,  "mmHg",  "CVP"),
    "Glucose":                 (70.0,  180.0,  "mg/dL", "blood glucose"),
}


class InferenceRecord(BaseModel):
    """A single generated nursing note with provenance and audit metadata.

    Attributes:
        note_id: Source ``ROW_ID`` from NOTEEVENTS.
        hadm_id: Hospital admission identifier.
        subject_id: De-identified patient identifier.
        model_id: HuggingFace model identifier string.
        generated_text: Raw model output string.
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens generated.
        generation_time_s: Wall-clock seconds for this record's generation.
        truncated_obs: Observation entries dropped by the budget cap.
        truncated_labs: Lab entries dropped.
        truncated_meds: Medication entries dropped.
    """

    note_id: int
    hadm_id: int
    subject_id: int
    model_id: str
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    generation_time_s: float
    truncated_obs: int = 0
    truncated_labs: int = 0
    truncated_meds: int = 0

    @field_validator("generated_text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("generated_text must not be empty")
        return v


class PromptBuilder:
    """Renders a FHIRLiteRecord dict into a model-ready prompt string.

    Context sections are rendered in this order:
    1. Admission Context
    2. Active Problems
    3. Vital Signs & Nursing Assessments (capped at ``_MAX_OBS_ENTRIES``)
    4. Laboratory Results (capped at ``_MAX_LAB_ENTRIES``)
    5. Medications Administered (capped at ``_MAX_MED_ENTRIES``)
    6. Procedures
    7. Task instruction
    """

    def _build_note_sections(
        self, record: Dict[str, Any]
    ) -> Tuple[str, int, int, int]:
        """Assemble the user-turn content and truncation counts for Sub-task 1.

        Returns:
            A 4-tuple of (user_content, truncated_obs, truncated_labs,
            truncated_meds).
        """
        meta = record.get("metadata", {})
        observations = record.get("observations", [])
        labs = record.get("labs", [])
        medications = record.get("medications", [])
        procedures = record.get("procedures", [])
        active_problems = record.get("active_problems", [])

        labs_capped, truncated_labs = self._cap(labs, _MAX_LAB_ENTRIES)
        meds_capped, truncated_meds = self._cap(medications, _MAX_MED_ENTRIES)
        procs_capped, _ = self._cap(procedures, _MAX_PROC_ENTRIES)
        obs_section, truncated_obs = self._render_observations(observations)
        active_problems = self._filter_active_problems(
            active_problems, meta.get("admit_diagnosis", "")
        )

        user_sections: List[str] = [
            self._render_admission_context(meta),
            self._render_active_problems(active_problems),
            obs_section,
            self._render_labs(labs_capped),
            self._render_medications(meds_capped),
            self._render_procedures(procs_capped),
            _FEW_SHOT_EXAMPLE,
            "## Task\nWrite the nursing progress note. "
            "One AARP block per active problem. "
            "Synthesise the data; do not restate it verbatim.",
        ]
        return "\n".join(user_sections), truncated_obs, truncated_labs, truncated_meds

    def build_messages(self, record: Dict[str, Any]) -> Tuple[List[Dict[str, str]], int, int, int]:
        """Build chat-style messages for Sub-task 1 (nursing note generation).

        Args:
            record: A ``FHIRLiteRecord`` serialised as a plain dict.

        Returns:
            A 4-tuple of (messages, truncated_obs, truncated_labs, truncated_meds).
        """
        user_content, truncated_obs, truncated_labs, truncated_meds = (
            self._build_note_sections(record)
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return messages, truncated_obs, truncated_labs, truncated_meds

    def build_abnormal(
        self, record: Dict[str, Any]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """Build a Sub-task 2 prompt for abnormal value identification.

        Uses the most recent reading per metric within ``_CLINICAL_RANGES``.

        Args:
            record: A ``FHIRLiteRecord`` serialised as a plain dict.

        Returns:
            A 2-tuple of (messages, gt_flags). ``gt_flags`` is a list of
            ``{"metric", "value", "unit", "is_abnormal", "low", "high", "label"}``
            dicts for the Sub-task 2 evaluator.
        """
        meta = record.get("metadata", {})
        observations = record.get("observations", [])

        def _elapsed_minutes(t: str) -> float:
            m = re.match(r"T-(\d+)h\s*(\d+)m", t)
            if m:
                return int(m.group(1)) * 60 + int(m.group(2))
            return float("inf")

        latest: Dict[str, Dict[str, Any]] = {}
        for obs in observations:
            metric = obs.get("metric", "")
            if metric not in _CLINICAL_RANGES:
                continue
            if obs.get("val_type") != "numeric":
                continue
            try:
                float(obs["val"])
            except (TypeError, ValueError):
                continue
            existing = latest.get(metric)
            if existing is None:
                latest[metric] = obs
            else:
                if _elapsed_minutes(obs["time"]) < _elapsed_minutes(existing["time"]):
                    latest[metric] = obs

        if not latest:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "No vital signs data available for this record. "
                        "State that you cannot assess abnormal values."
                    ),
                },
            ]
            return messages, []

        gt_flags: List[Dict[str, Any]] = []
        vital_lines: List[str] = []
        for metric, obs in sorted(latest.items()):
            low, high, unit_hint, label = _CLINICAL_RANGES[metric]
            val = float(obs["val"])
            unit = obs.get("unit") or unit_hint
            is_abnormal = val < low or val > high
            gt_flags.append(
                {
                    "metric": metric,
                    "value": val,
                    "unit": unit,
                    "is_abnormal": is_abnormal,
                    "low": low,
                    "high": high,
                    "label": label,
                }
            )
            vital_lines.append(
                f"- {metric}: {val} {unit}  (recorded at {obs['time']})"
            )

        admit_dx = meta.get("admit_diagnosis", "not documented")
        vitals_block = "\n".join(vital_lines)

        system_prompt = (
            "You are an experienced MICU nurse reviewing a patient's vital signs. "
            "Identify which values are clinically abnormal for an ICU adult patient. "
            "For each abnormal value: (1) state whether it is high or low, "
            "(2) give the normal adult range with units, "
            "(3) name the clinical concern in one sentence. "
            "If a value is within normal limits, do not mention it. "
            "Be concise. Use clinical shorthand."
        )

        user_content = (
            f"## Patient Context\n"
            f"Admitting diagnosis: {admit_dx}\n"
            f"Care unit: {meta.get('careunit', 'MICU')}\n\n"
            f"## Most Recent Vital Signs\n"
            f"{vitals_block}\n\n"
            f"## Task\n"
            f"Identify and explain any clinically abnormal vital signs above."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return messages, gt_flags

    def build_summary(
        self, record: Dict[str, Any]
    ) -> Tuple[List[Dict[str, str]], int, int, int]:
        """Build a Sub-task 3 prompt for admission summarisation.

        Args:
            record: A ``FHIRLiteRecord`` serialised as a plain dict.

        Returns:
            A 4-tuple of (messages, truncated_obs, truncated_labs, truncated_meds).
        """
        meta         = record.get("metadata", {})
        observations = record.get("observations", [])
        labs         = record.get("labs", [])
        medications  = record.get("medications", [])
        procedures   = record.get("procedures", [])
        active_problems = record.get("active_problems", [])

        labs_capped, truncated_labs = self._cap(labs, _MAX_LAB_ENTRIES)
        meds_capped, truncated_meds = self._cap(medications, _MAX_MED_ENTRIES)
        procs_capped, _             = self._cap(procedures, _MAX_PROC_ENTRIES)
        obs_section, truncated_obs  = self._render_observations(observations)
        active_problems = self._filter_active_problems(
            active_problems, meta.get("admit_diagnosis", "")
        )

        summary_system = (
            "You are an experienced MICU attending physician writing the "
            "Hospital Course section of a discharge summary. "
            "Synthesise the structured ICU data provided into a concise "
            "problem-oriented narrative. For each active problem: describe "
            "the clinical findings, key investigations, interventions, and "
            "patient response. Use clinical language. Be specific — cite "
            "abnormal values, medications, and procedures by name. "
            "Do not use bullet points or AARP headings; write flowing prose."
        )

        sections = [
            self._render_admission_context(meta),
            self._render_active_problems(active_problems),
            obs_section,
            self._render_labs(labs_capped),
            self._render_medications(meds_capped),
            self._render_procedures(procs_capped),
            (
                "## Task\n"
                "Write the Hospital Course narrative for this 12-hour ICU window. "
                "One paragraph per active problem. "
                "Synthesise the data — do not restate it verbatim."
            ),
        ]

        user_content = "\n".join(sections)
        messages = [
            {"role": "system", "content": summary_system},
            {"role": "user",   "content": user_content},
        ]
        return messages, truncated_obs, truncated_labs, truncated_meds

    @staticmethod
    def _cap(entries: List[Any], limit: int) -> Tuple[List[Any], int]:
        if len(entries) <= limit:
            return entries, 0
        return entries[:limit], len(entries) - limit

    @staticmethod
    def _render_admission_context(meta: Dict[str, Any]) -> str:
        admission_type  = meta.get("admission_type") or "Not recorded"
        admit_diagnosis = meta.get("admit_diagnosis") or "Not recorded"
        insurance       = meta.get("insurance") or "Not recorded"
        careunit        = meta.get("careunit") or "MICU"
        return (
            "## Admission Context\n"
            f"Type: {admission_type}\n"
            f"Presenting complaint: {admit_diagnosis}\n"
            f"Insurance: {insurance}\n"
            f"Care unit: {careunit}"
        )

    @staticmethod
    def _filter_active_problems(
        problems: List[Dict[str, Any]],
        admit_diagnosis: str,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return up to *top_n* problems ranked by relevance to the admitting diagnosis.

        Args:
            problems: List of ``{"icd9_code": ..., "description": ...}`` dicts.
            admit_diagnosis: Free-text admitting diagnosis from record metadata.
            top_n: Maximum number of problems to return.

        Returns:
            Filtered and re-ranked list of problem dicts (length ≤ top_n).
        """
        if len(problems) <= top_n:
            return problems

        _STOP = {"with", "and", "the", "of", "in", "for", "due", "to", "nos"}
        admit_tokens: set = {
            tok
            for tok in re.split(r"[\W;,]+", admit_diagnosis.lower())
            if len(tok) > 2 and tok not in _STOP
        }

        scored: List[Tuple[int, int, Dict[str, Any]]] = []
        for i, p in enumerate(problems):
            desc_tokens = {
                tok
                for tok in re.split(r"\W+", p.get("description", "").lower())
                if len(tok) > 2 and tok not in _STOP
            }
            score = len(desc_tokens & admit_tokens)
            scored.append((score, i, p))

        scored.sort(key=lambda t: (-t[0], t[1]))
        return [p for _, _, p in scored[:top_n]]

    @staticmethod
    def _render_active_problems(problems: List[Dict[str, Any]]) -> str:
        header = "## Active Problems"
        if not problems:
            return f"{header}\n(none recorded)"
        lines = [
            f"{i + 1}. {p.get('description', 'Unknown')} (ICD-9: {p.get('icd9_code', '?')})"
            for i, p in enumerate(problems)
        ]
        return f"{header}\n" + "\n".join(lines)

    @staticmethod
    def _render_observations(
        observations: List[Dict[str, Any]],
    ) -> Tuple[str, int]:
        """Render the Vital Signs & Nursing Assessments section.

        Vital-sign metrics are summarised as trends; non-vital metrics are
        listed flat. The hard cap ``_MAX_OBS_ENTRIES`` is applied after
        summarisation; out-of-range lines are preserved unconditionally.

        Args:
            observations: Full list of ObservationEntry dicts, most-recent first.

        Returns:
            A 2-tuple of (section string, lines dropped by cap).
        """
        header = "## Vital Signs & Nursing Assessments  [last 12h]"
        if not observations:
            return f"{header}\n(none recorded)", 0

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for obs in observations:
            grouped.setdefault(obs.get("metric", ""), []).append(obs)

        lines: List[str] = []

        for metric, entries in grouped.items():
            unit_str = (entries[0].get("unit") or "").strip()
            unit_disp = f" {unit_str}" if unit_str else ""

            if metric in _VITAL_METRICS:
                numeric = [e for e in entries if e.get("val_type") == "numeric"]
                categorical = [e for e in entries if e.get("val_type") != "numeric"]

                if numeric:
                    vals_with_time = []
                    for e in numeric:
                        raw = (e.get("val") or "").strip().lstrip("><").strip()
                        try:
                            vals_with_time.append((e["time"], float(raw), e.get("out_of_range", False)))
                        except ValueError:
                            pass

                    if vals_with_time:
                        latest_t,  latest_v,  _ = vals_with_time[0]
                        oldest_t,  oldest_v,  _ = vals_with_time[-1]
                        all_vals = [v for _, v, _ in vals_with_time]
                        mn, mx = min(all_vals), max(all_vals)

                        delta = _STABLE_DELTA.get(metric, _DEFAULT_STABLE_DELTA)
                        diff  = latest_v - oldest_v
                        if abs(diff) < delta:
                            trend = "stable"
                        elif diff > 0:
                            trend = "trending up"
                        else:
                            trend = "trending down"

                        summary = (
                            f"{metric}: {mn:.4g}–{mx:.4g}{unit_disp}, {trend}"
                            f"  ({oldest_t}: {oldest_v:.4g} → {latest_t}: {latest_v:.4g})"
                        ) if mn != mx else (
                            f"{metric}: {mn:.4g}{unit_disp} (constant)"
                        )

                        oor_entries = [(t, v) for t, v, oor in vals_with_time if oor]
                        if oor_entries:
                            oor_parts = ", ".join(
                                f"{v:.4g}{unit_disp} at {t}" for t, v in oor_entries
                            )
                            summary += f"  ⚠ OUT_OF_RANGE: {oor_parts}"

                        lines.append(summary)

                if categorical:
                    seq: List[str] = []
                    for e in reversed(categorical):
                        v = (e.get("val") or "").strip()
                        if not seq or seq[-1] != v:
                            seq.append(v)
                    if len(seq) == 1:
                        lines.append(f"{metric}: {seq[0]} (unchanged)")
                    else:
                        lines.append(f"{metric}: {' → '.join(seq)}")

            else:
                for e in entries:
                    val     = e.get("val", "")
                    time_s  = e.get("time", "")
                    oor     = e.get("out_of_range", False)
                    oor_str = "  ⚠ OUT_OF_RANGE" if oor else ""
                    lines.append(f"{metric}: {val}{unit_disp}  [{time_s}]{oor_str}")

        oor_lines    = [l for l in lines if "⚠ OUT_OF_RANGE" in l]
        normal_lines = [l for l in lines if "⚠ OUT_OF_RANGE" not in l]
        budget    = max(0, _MAX_OBS_ENTRIES - len(oor_lines))
        truncated = max(0, len(normal_lines) - budget)
        kept_lines = oor_lines + normal_lines[:budget]
        return f"{header}\n" + "\n".join(kept_lines), truncated

    @staticmethod
    def _render_labs(labs: List[Dict[str, Any]]) -> str:
        header = "## Laboratory Results  [last 12h, most recent first]"
        if not labs:
            return f"{header}\n(none recorded)"
        lines: List[str] = []
        for lab in labs:
            metric   = lab.get("metric", "")
            val      = lab.get("val", "")
            unit     = lab.get("unit") or ""
            time_str = lab.get("time", "")
            oor      = lab.get("out_of_range", False)
            unit_str = f" {unit.strip()}" if unit.strip() else ""
            oor_str  = "  ⚠ OUT_OF_RANGE" if oor else ""
            lines.append(f"{metric}: {val}{unit_str}  [{time_str}]{oor_str}")
        return f"{header}\n" + "\n".join(lines)

    @staticmethod
    def _render_medications(medications: List[Dict[str, Any]]) -> str:
        header = "## Medications Administered  [last 12h, most recent first]"
        if not medications:
            return f"{header}\n(none recorded)"
        lines: List[str] = []
        for med in medications:
            drug     = med.get("drug", "")
            amount   = med.get("amount")
            unit     = med.get("unit") or ""
            route    = med.get("route") or ""
            time_str = med.get("time", "")
            dose_str = ""
            if amount is not None:
                unit_str = f" {unit.strip()}" if unit.strip() else ""
                dose_str = f"  {amount}{unit_str}"
            route_str = f"  via {route.strip()}" if route.strip() else ""
            lines.append(f"{drug}{dose_str}{route_str}  [{time_str}]")
        return f"{header}\n" + "\n".join(lines)

    @staticmethod
    def _render_procedures(procedures: List[Dict[str, Any]]) -> str:
        header = "## Procedures  [last 12h, most recent first]"
        if not procedures:
            return f"{header}\n(none recorded)"
        lines: List[str] = []
        for proc in procedures:
            procedure    = proc.get("procedure", "")
            duration_min = proc.get("duration_min")
            status       = proc.get("status") or ""
            time_str     = proc.get("time", "")
            dur_str      = f"  {duration_min} min" if duration_min is not None else ""
            status_str   = f"  [{status.strip()}]" if status.strip() else ""
            lines.append(f"{procedure}{dur_str}{status_str}  [{time_str}]")
        return f"{header}\n" + "\n".join(lines)


class InferenceEngine:
    """Thin wrapper around vLLM ``LLM`` for batched generation."""

    def __init__(self, model_id: str, max_num_seqs: int = _DEFAULT_BATCH_SIZE) -> None:
        """Initialise the vLLM engine on Device 0.

        Args:
            model_id: HuggingFace model identifier.
            max_num_seqs: Maximum concurrent sequences.
        """
        try:
            from vllm import LLM  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed.  Install it with: pip install vllm"
            ) from exc

        self.model_id: str = model_id
        logger.info(
            "Initialising vLLM engine | model=%s | max_num_seqs=%d | "
            "dtype=bfloat16 | tensor_parallel_size=1 | device=cuda:0",
            model_id,
            max_num_seqs,
        )
        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=_MAX_MODEL_LEN,
            max_num_seqs=max_num_seqs,
        )
        logger.info("vLLM engine ready.")


def _default_output_path(model_id: str, task: str = "notes") -> Path:
    slug = model_id.replace("/", "_").replace("-", "_").lower()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _PROCESSED_DIR / f"inference_{task}_{slug}_{ts}.jsonl"


def run_inference(
    model_id: str,
    input_path: Path,
    output_path: Optional[Path] = None,
    sample: Optional[int] = None,
    one_per_admission: bool = False,
    temperature: float = _DEFAULT_TEMPERATURE,
    top_p: float = _DEFAULT_TOP_P,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    task: str = "notes",
    repetition_penalty: float = _DEFAULT_REPETITION_PENALTY,
) -> None:
    """inference loop end-to-end.

    Args:
        model_id: HuggingFace model identifier.
        input_path: Path to the FHIR-lite JSON file.
        output_path: Destination JSONL path; auto-named if not provided.
        sample: Cap input to the first ``sample`` records.
        one_per_admission: Select one record per unique ``hadm_id``.
        temperature: vLLM sampling temperature.
        top_p: Nucleus sampling probability.
        max_tokens: Maximum new tokens per completion.
        batch_size: Prompts submitted to vLLM per call.
        task: One of ``"notes"``, ``"abnormal"``, or ``"summary"``.
        repetition_penalty: vLLM repetition penalty.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ValueError: If ``task`` is not recognised, or ``sample`` and
            ``one_per_admission`` are both set.
    """
    if task not in {"notes", "abnormal", "summary"}:
        raise ValueError(f"Unknown task '{task}'. Must be 'notes', 'abnormal', or 'summary'.")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = _default_output_path(model_id, task)

    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if sample is not None and one_per_admission:
        raise ValueError("--sample and --one-per-admission are mutually exclusive.")

    logger.info("Loading FHIR-lite records from %s …", input_path)
    records: List[Dict[str, Any]] = []
    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
            if sample is not None and len(records) >= sample:
                break

    if one_per_admission:
        seen_hadm: set = set()
        filtered: List[Dict[str, Any]] = []
        for rec in records:
            hid = rec.get("metadata", {}).get("hadm_id")
            if hid not in seen_hadm:
                seen_hadm.add(hid)
                filtered.append(rec)
        logger.info(
            "One-per-admission mode: %d records -> %d unique admissions.",
            len(records),
            len(filtered),
        )
        records = filtered

    total_available = len(records)
    if sample is not None:
        logger.info("Sample mode: using %d records.", len(records))
    elif one_per_admission:
        logger.info("One-per-admission mode: %d independent records.", total_available)
    else:
        logger.info("Processing all %d records.", total_available)

    n_records = len(records)

    n_with_meds     = sum(1 for r in records if r.get("medications"))
    n_with_problems = sum(1 for r in records if r.get("active_problems"))
    if n_with_meds == 0 and n_with_problems == 0:
        logger.warning(
            "All %d records have empty 'medications' and 'active_problems' lists. ",
            n_records,
        )
    else:
        logger.info(
            "Context coverage: records_with_medications=%d / %d | "
            "records_with_active_problems=%d / %d",
            n_with_meds, n_records,
            n_with_problems, n_records,
        )

    builder = PromptBuilder()
    engine  = InferenceEngine(model_id=model_id, max_num_seqs=batch_size)

    logger.info(
        "Starting generation | model=%s | records=%d | batch_size=%d | "
        "temperature=%.2f | max_tokens=%d → %s",
        model_id, n_records, batch_size, temperature, max_tokens, output_path,
    )

    wall_start             = time.monotonic()
    total_prompt_tokens    = 0
    total_completion_tokens = 0
    accepted               = 0
    rejected               = 0

    with open(output_path, "w", encoding="utf-8") as out_fh:
        for batch_start in range(0, n_records, batch_size):
            batch   = records[batch_start: batch_start + batch_size]
            batch_t0 = time.monotonic()

            prompts: List[str] = []
            truncation_meta: List[Tuple[int, int, int]] = []
            gt_flags_batch: List[List[Dict[str, Any]]] = []

            if task == "notes":
                for rec in batch:
                    msgs, t_obs, t_labs, t_meds = builder.build_messages(rec)
                    prompt_str = engine.llm.get_tokenizer().apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                    prompts.append(prompt_str)
                    truncation_meta.append((t_obs, t_labs, t_meds))
                    gt_flags_batch.append([])
            elif task == "summary":
                for rec in batch:
                    msgs, t_obs, t_labs, t_meds = builder.build_summary(rec)
                    prompt_str = engine.llm.get_tokenizer().apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                    prompts.append(prompt_str)
                    truncation_meta.append((t_obs, t_labs, t_meds))
                    gt_flags_batch.append([])
            else:
                for rec in batch:
                    msgs, gt_flags = builder.build_abnormal(rec)
                    prompt_str = engine.llm.get_tokenizer().apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                    prompts.append(prompt_str)
                    truncation_meta.append((0, 0, 0))
                    gt_flags_batch.append(gt_flags)

            try:
                from vllm import SamplingParams  # type: ignore[import]
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty,
                )
                raw_outputs = engine.llm.generate(prompts, sampling_params)
            except Exception as exc:
                logger.error(
                    "vLLM generation failed for batch starting at record %d: %s",
                    batch_start, exc,
                )
                rejected += len(batch)
                continue

            batch_elapsed = time.monotonic() - batch_t0

            for i, (rec, output) in enumerate(zip(batch, raw_outputs)):
                t_obs, t_labs, t_meds = truncation_meta[i]
                gen_text       = output.outputs[0].text
                prompt_tok     = len(output.prompt_token_ids)
                completion_tok = len(output.outputs[0].token_ids)
                gen_time       = batch_elapsed / len(batch)

                meta = rec.get("metadata", {})
                try:
                    rec_dict: Dict[str, Any] = dict(
                        note_id=int(meta.get("note_id", 0)),
                        hadm_id=int(meta.get("hadm_id", 0)),
                        subject_id=int(meta.get("subject_id", 0)),
                        model_id=model_id,
                        generated_text=gen_text,
                        prompt_tokens=prompt_tok,
                        completion_tokens=completion_tok,
                        generation_time_s=round(gen_time, 4),
                        truncated_obs=t_obs,
                        truncated_labs=t_labs,
                        truncated_meds=t_meds,
                    )
                    inference_rec = InferenceRecord(**rec_dict)
                    out_dict = inference_rec.model_dump()
                    if task == "abnormal":
                        out_dict["gt_flags"] = gt_flags_batch[i]
                    if task == "summary":
                        out_dict["hospital_course"] = rec.get("hospital_course") or ""
                    out_fh.write(json.dumps(out_dict) + "\n")
                    total_prompt_tokens     += prompt_tok
                    total_completion_tokens += completion_tok
                    accepted += 1
                except Exception as exc:
                    logger.warning(
                        "InferenceRecord validation failed | note_id=%s | %s",
                        meta.get("note_id"), exc,
                    )
                    rejected += 1

            records_done = batch_start + len(batch)
            if records_done % _LOG_INTERVAL < batch_size or records_done == n_records:
                elapsed    = time.monotonic() - wall_start
                throughput = records_done / elapsed if elapsed > 0 else 0
                eta_s      = (n_records - records_done) / throughput if throughput > 0 else 0
                logger.info(
                    "Progress: %d / %d (%.0f%%)  |  %.1f rec/s  |  ETA %.0fs",
                    records_done, n_records,
                    100.0 * records_done / n_records,
                    throughput,
                    eta_s,
                )

    total_elapsed = time.monotonic() - wall_start
    logger.info(
        "Inference complete | accepted=%d | rejected=%d | "
        "prompt_tokens=%d | completion_tokens=%d | "
        "wall_time=%.1fs | mean_throughput=%.1f rec/s | output=%s",
        accepted,
        rejected,
        total_prompt_tokens,
        total_completion_tokens,
        total_elapsed,
        accepted / total_elapsed if total_elapsed > 0 else 0,
        output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate MICU nursing notes from FHIR-lite JSON using vLLM."
    )
    parser.add_argument("--model", type=str, default="google/medgemma-4b-it", metavar="MODEL_ID",
        help="HuggingFace model identifier (default: google/medgemma-4b-it).")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT, metavar="PATH",
        help=f"Path to the FHIR-lite JSON (default: {_DEFAULT_INPUT}).")
    parser.add_argument("--output", type=Path, default=None, metavar="PATH",
        help="Destination JSONL path; auto-named if omitted.")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
        help="Process only the first N records. Mutually exclusive with --one-per-admission.")
    parser.add_argument("--one-per-admission", action="store_true", default=False,
        help="Select the first record per unique hadm_id. Mutually exclusive with --sample.")
    parser.add_argument("--temperature", type=float, default=_DEFAULT_TEMPERATURE, metavar="F",
        help=f"Sampling temperature (default: {_DEFAULT_TEMPERATURE}).")
    parser.add_argument("--max-tokens", type=int, default=_DEFAULT_MAX_TOKENS, metavar="N",
        help=f"Maximum new tokens per completion (default: {_DEFAULT_MAX_TOKENS}).")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE, metavar="N",
        help=f"Prompts per vLLM call (default: {_DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--task", type=str, default="notes",
        choices=["notes", "abnormal", "summary"],
        help="Inference task: 'notes' (Sub-task 1), 'abnormal' (Sub-task 2), 'summary' (Sub-task 3).")
    parser.add_argument("--repetition-penalty", type=float, default=_DEFAULT_REPETITION_PENALTY,
        metavar="F",
        help=f"vLLM repetition penalty (default: {_DEFAULT_REPETITION_PENALTY}). Set to 1.0 to disable.")

    args = parser.parse_args()
    run_inference(
        model_id=args.model,
        input_path=args.input,
        output_path=args.output,
        sample=args.sample,
        one_per_admission=args.one_per_admission,
        temperature=args.temperature,
        top_p=_DEFAULT_TOP_P,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        task=args.task,
        repetition_penalty=args.repetition_penalty,
    )
