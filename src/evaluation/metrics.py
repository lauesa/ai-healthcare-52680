"""
CLI usage::

    python -m src.evaluation.metrics score \\
        --inference data/processed/inference_notes_medgemma_val637opa.jsonl \\
        --reference data/processed/fhir_lite_val.jsonl

    python -m src.evaluation.metrics compare \\
        --eval-a  data/processed/eval_notes_medgemma_val637opa.jsonl \\
        --eval-b  data/processed/eval_notes_gemma3_val637opa.jsonl
"""

import json
import logging
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import spacy
from rouge_score import rouge_scorer as _rouge_scorer
from scipy.stats import wilcoxon as _scipy_wilcoxon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
_log = logging.getLogger(__name__)

_AARP_HEADERS = re.compile(
    r"\b(assessment|action|response|plan)\b",
    re.IGNORECASE,
)
_AARP_WINDOW_CHARS: int = 2400

_OOR_SUFFIX = "[OUT_OF_RANGE]"
_MED_MARKER = "administered via"


def _f1_from_sets(gen_set: set, ref_set: set) -> float:
    """Compute token-level F1 between two entity sets.

    Args:
        gen_set: Lowercased entity strings from the generated text.
        ref_set: Lowercased entity strings from the reference text.

    Returns:
        F1 score in [0.0, 1.0]; 0.0 if either set is empty.
    """
    if not gen_set or not ref_set:
        return 0.0
    overlap   = gen_set & ref_set
    precision = len(overlap) / len(gen_set)
    recall    = len(overlap) / len(ref_set)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def _load_nlp() -> spacy.language.Language:
    """Load the scispaCy NER model.

    Returns:
        Loaded ``en_core_sci_sm`` pipeline.
    """
    return spacy.load("en_core_sci_sm")


def score_record(
    generated: str,
    reference: str,
    nlp: spacy.language.Language,
    active_problems: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Compute ROUGE-L, Clinical Entity F1 (aggregate + per-class), and AARP score.

    Args:
        generated: The model-generated note text.
        reference: The human-written ground-truth note text.
        nlp: Loaded ``en_core_sci_sm`` pipeline.
        active_problems: Active problem dicts from the FHIRLiteRecord.

    Returns:
        Dict with keys ``rouge_l``, ``entity_precision``, ``entity_recall``,
        ``entity_f1``, ``aarp_score``, ``entity_f1_drugs``,
        ``entity_f1_diagnoses``, ``entity_f1_vitals``.
    """
    _scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = _scorer.score(reference, generated)["rougeL"].fmeasure

    gen_doc = nlp(generated)
    ref_doc = nlp(reference)

    gen_ents = {e.text.lower().strip() for e in gen_doc.ents}
    ref_ents = {e.text.lower().strip() for e in ref_doc.ents}

    if gen_ents and ref_ents:
        overlap   = gen_ents & ref_ents
        precision = len(overlap) / len(gen_ents)
        recall    = len(overlap) / len(ref_ents)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
    else:
        precision = recall = f1 = 0.0

    _DRUG_SUFFIXES = re.compile(
        r"(mab|nib|pril|olol|statin|cillin|mycin|cycline|azole|"
        r"prazole|lukast|sartan|dipine|fenac|oxacin|pam|lam)\b",
        re.IGNORECASE,
    )
    _VITAL_KEYWORDS = re.compile(
        r"\b(heart rate|blood pressure|spo2|o2 sat|respiratory rate|"
        r"temperature|pulse|cvp|map|hr\b|bp\b|rr\b|sats?)\b",
        re.IGNORECASE,
    )

    def _label_ents(ents: list, label: str) -> set:
        return {e.text.lower().strip() for e in ents if e.label_ == label}

    def _surface_ents(ents: list, category: str) -> set:
        out = set()
        for e in ents:
            txt = e.text.lower().strip()
            if category == "drugs":
                if _DRUG_SUFFIXES.search(txt) or any(ch.isdigit() for ch in txt):
                    out.add(txt)
            elif category == "vitals":
                if _VITAL_KEYWORDS.search(txt):
                    out.add(txt)
        return out

    gen_ents_list = list(gen_doc.ents)
    ref_ents_list = list(ref_doc.ents)

    all_labels     = {e.label_ for e in gen_ents_list} | {e.label_ for e in ref_ents_list}
    has_typed_labels = bool(all_labels - {"ENTITY", ""})

    if has_typed_labels:
        f1_drugs     = _f1_from_sets(_label_ents(gen_ents_list, "CHEMICAL"),  _label_ents(ref_ents_list, "CHEMICAL"))
        f1_diagnoses = _f1_from_sets(_label_ents(gen_ents_list, "DISEASE"),   _label_ents(ref_ents_list, "DISEASE"))
        f1_vitals    = _f1_from_sets(_label_ents(gen_ents_list, "QUANTITY"),  _label_ents(ref_ents_list, "QUANTITY"))
    else:
        f1_drugs     = _f1_from_sets(_surface_ents(gen_ents_list, "drugs"),  _surface_ents(ref_ents_list, "drugs"))
        f1_diagnoses = 0.0
        f1_vitals    = _f1_from_sets(_surface_ents(gen_ents_list, "vitals"), _surface_ents(ref_ents_list, "vitals"))

    aarp_score = aarp_section_score(generated, active_problems)

    return {
        "rouge_l":             round(rouge_l, 6),
        "entity_precision":    round(precision, 6),
        "entity_recall":       round(recall, 6),
        "entity_f1":           round(f1, 6),
        "aarp_score":          round(aarp_score, 6),
        "entity_f1_drugs":     round(f1_drugs, 6),
        "entity_f1_diagnoses": round(f1_diagnoses, 6),
        "entity_f1_vitals":    round(f1_vitals, 6),
    }


def aarp_section_score(
    generated: str,
    active_problems: list[dict[str, Any]] | None = None,
) -> float:
    """Compute the AARP section presence score for a generated note.

    Args:
        generated: The model-generated note text.
        active_problems: Active problem dicts from FHIRLiteRecord.

    Returns:
        Float in [0.0, 1.0].
    """
    n_problems = len(active_problems) if active_problems else 0

    header_positions: dict[str, list[int]] = {
        h: [] for h in ("assessment", "action", "response", "plan")
    }
    for m in _AARP_HEADERS.finditer(generated):
        header_positions[m.group(1).lower()].append(m.start())

    complete_blocks = 0
    for assess_pos in header_positions["assessment"]:
        window_end   = assess_pos + _AARP_WINDOW_CHARS
        has_action   = any(assess_pos <= p <= window_end for p in header_positions["action"])
        has_response = any(assess_pos <= p <= window_end for p in header_positions["response"])
        has_plan     = any(assess_pos <= p <= window_end for p in header_positions["plan"])
        if has_action and has_response and has_plan:
            complete_blocks += 1

    if n_problems == 0:
        return 1.0 if complete_blocks >= 1 else 0.0

    return min(1.0, complete_blocks / n_problems)


def score_abnormal_record(
    generated: str,
    gt_flags: list[dict[str, Any]],
) -> dict[str, float]:
    """Score a Sub-task 2 (abnormal value identification) generation.

    Args:
        generated: The model-generated response text.
        gt_flags: Ground-truth list from ``PromptBuilder.build_abnormal``.

    Returns:
        Dict with keys ``abnormal_precision``, ``abnormal_recall``,
        ``abnormal_f1``, ``n_truly_abnormal``, ``n_flagged_by_model``.
    """
    if not gt_flags:
        return {
            "abnormal_precision": 0.0,
            "abnormal_recall":    0.0,
            "abnormal_f1":        0.0,
            "n_truly_abnormal":   0,
            "n_flagged_by_model": 0,
        }

    gen_lower        = generated.lower()
    tp = fp = fn     = 0
    n_truly_abnormal = sum(1 for f in gt_flags if f["is_abnormal"])
    n_flagged        = 0

    for flag in gt_flags:
        label    = flag["label"].lower()
        mentioned = bool(re.search(re.escape(label), gen_lower))
        if not mentioned:
            mentioned = bool(re.search(re.escape(flag["metric"].lower()), gen_lower))

        if mentioned and flag["is_abnormal"]:
            tp += 1
            n_flagged += 1
        elif mentioned and not flag["is_abnormal"]:
            fp += 1
            n_flagged += 1
        elif not mentioned and flag["is_abnormal"]:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "abnormal_precision": round(precision, 6),
        "abnormal_recall":    round(recall, 6),
        "abnormal_f1":        round(f1, 6),
        "n_truly_abnormal":   n_truly_abnormal,
        "n_flagged_by_model": n_flagged,
    }


def score_summary_record(
    generated: str,
    hospital_course: str,
    source_facts: list[str],
    active_problems: list[dict[str, Any]] | None = None,
    labs: list[dict[str, Any]] | None = None,
    medications: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Score a Sub-task 3 (admission summary) generation.

    Args:
        generated: The model-generated summary text.
        hospital_course: Discharge summary HOSPITAL COURSE reference text.
        source_facts: ``source_facts`` list from the FHIRLiteRecord.
        active_problems: Active problem dicts (for ``problem_coverage``).
        labs: Lab result dicts (for ``key_lab_coverage``).
        medications: Medication dicts (for ``medication_accuracy``).

    Returns:
        Dict with keys ``rouge_l``, ``bertscore_f1``,
        ``problem_coverage``, ``key_lab_coverage``, ``medication_accuracy``.
    """
    facts_blob = " ".join(source_facts).lower()

    def _sentence_split(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    hc_sentences = _sentence_split(hospital_course) if hospital_course else []

    def _is_relevant(sentence: str) -> bool:
        for word in re.findall(r"\b[a-zA-Z]{4,}\b", sentence.lower()):
            if word in facts_blob:
                return True
        return False

    relevant_sentences = [s for s in hc_sentences if _is_relevant(s)]
    filtered_reference = (
        " ".join(relevant_sentences)
        if len(relevant_sentences) >= 2
        else hospital_course
    )

    if filtered_reference and generated:
        _scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l = _scorer.score(filtered_reference, generated)["rougeL"].fmeasure
    else:
        rouge_l = 0.0

    bertscore_f1 = 0.0
    if filtered_reference and generated:
        try:
            from bert_score import score as _bert_score

            P, R, F = _bert_score(
                [generated],
                [filtered_reference],
                model_type="distilbert-base-uncased",
                lang="en",
                verbose=False,
                device="cpu",
            )
            bertscore_f1 = float(F[0].item())
        except Exception as exc:
            _log.warning("BERTScore failed: %s — setting bertscore_f1=0.0", exc)

    gen_lower = generated.lower()

    if active_problems:
        problem_names = []
        for prob in active_problems:
            name = prob.get("description", "") or prob.get("problem", "") or prob.get("name", "")
            if name:
                name = re.sub(r"\s*\(ICD-9:[^)]+\)", "", name).strip().lower()
                problem_names.append(name)
        if problem_names:
            hits             = sum(1 for n in problem_names if n and n in gen_lower)
            problem_coverage = hits / len(problem_names)
        else:
            problem_coverage = 0.0
    else:
        problem_coverage = 0.0

    oor_lab_keys: list[str] = []
    for fact in source_facts:
        if _OOR_SUFFIX in fact:
            parts = fact.split(" was ", 1)
            if parts:
                oor_lab_keys.append(parts[0].strip().lower())

    if oor_lab_keys:
        hits             = sum(1 for k in oor_lab_keys if k in gen_lower)
        key_lab_coverage = hits / len(oor_lab_keys)
    else:
        key_lab_coverage = 1.0

    med_names: list[str] = []
    for fact in source_facts:
        if _MED_MARKER in fact:
            tokens = fact.split()
            if tokens:
                med_names.append(tokens[0].lower())

    if med_names:
        hits                = sum(1 for m in med_names if m in gen_lower)
        medication_accuracy = hits / len(med_names)
    else:
        medication_accuracy = 1.0

    return {
        "rouge_l":             round(rouge_l, 6),
        "bertscore_f1":        round(bertscore_f1, 6),
        "problem_coverage":    round(problem_coverage, 6),
        "key_lab_coverage":    round(key_lab_coverage, 6),
        "medication_accuracy": round(medication_accuracy, 6),
    }


def compute_corpus_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-record evaluation scores to corpus-level mean ± std.

    Args:
        records: List of per-record dicts from :func:`score_record` or
            :func:`score_abnormal_record`.

    Returns:
        Summary dict with ``model_id``, ``n_records``, ``n_truncated``, and
        per-metric ``{"mean": float, "std": float}`` sub-dicts.
    """
    if not records:
        return {}

    model_id    = records[0].get("model_id", "unknown")
    n_truncated = sum(
        1 for r in records
        if r.get("truncated_obs", 0) > 0
        or r.get("truncated_labs", 0) > 0
        or r.get("truncated_meds", 0) > 0
    )

    summary: dict[str, Any] = {
        "model_id":    model_id,
        "n_records":   len(records),
        "n_truncated": n_truncated,
    }

    _ALL_METRICS = (
        "rouge_l",
        "entity_precision", "entity_recall", "entity_f1",
        "entity_f1_drugs", "entity_f1_diagnoses", "entity_f1_vitals",
        "aarp_score",
        "abnormal_precision", "abnormal_recall", "abnormal_f1",
        "bertscore_f1", "problem_coverage", "key_lab_coverage", "medication_accuracy",
    )

    for metric in _ALL_METRICS:
        vals = [r[metric] for r in records if metric in r]
        if not vals:
            continue
        summary[metric] = {
            "mean": round(statistics.mean(vals), 6),
            "std":  round(statistics.pstdev(vals), 6),
        }

    return summary


def wilcoxon_compare(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Paired Wilcoxon signed-rank test for continuous metrics.

    Records are matched by ``note_id``.

    Args:
        records_a: Per-record eval dicts for model A.
        records_b: Per-record eval dicts for model B.

    Returns:
        List of result dicts with keys ``metric``, ``model_a_mean``,
        ``model_b_mean``, ``delta``, ``n_pairs``, ``wilcoxon_p``,
        ``effect_r``, ``significant``.
    """
    a_by_id    = {r["note_id"]: r for r in records_a}
    b_by_id    = {r["note_id"]: r for r in records_b}
    common_ids = sorted(set(a_by_id) & set(b_by_id))

    if not common_ids:
        _log.warning("wilcoxon_compare: no common note_ids between the two eval files.")
        return []

    results = []
    _WILCOXON_METRICS = (
        "rouge_l", "entity_f1", "aarp_score",
        "entity_f1_drugs", "entity_f1_diagnoses", "entity_f1_vitals",
        "abnormal_f1", "abnormal_precision", "abnormal_recall",
        "bertscore_f1", "problem_coverage", "key_lab_coverage", "medication_accuracy",
    )
    for metric in _WILCOXON_METRICS:
        a_vals = [a_by_id[i][metric] for i in common_ids if metric in a_by_id[i]]
        b_vals = [b_by_id[i][metric] for i in common_ids if metric in b_by_id[i]]

        if len(a_vals) < 10:
            _log.warning("wilcoxon_compare: too few pairs for %s (%d).", metric, len(a_vals))
            continue

        diffs      = [a - b for a, b in zip(a_vals, b_vals)]
        stat, p    = _scipy_wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        n          = len(diffs)
        effect_r   = 1.0 - (2 * stat) / (n * (n + 1))
        mean_a     = statistics.mean(a_vals)
        mean_b     = statistics.mean(b_vals)
        results.append({
            "metric":       metric,
            "model_a_mean": round(mean_a, 6),
            "model_b_mean": round(mean_b, 6),
            "delta":        round(mean_a - mean_b, 6),
            "n_pairs":      n,
            "wilcoxon_p":   round(float(p), 6),
            "effect_r":     round(float(effect_r), 4),
            "significant":  bool(p < 0.05),
        })

    return results


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _print_summary_table(summary: dict[str, Any]) -> None:
    print()
    print(f"  Model   : {summary.get('model_id', 'unknown')}")
    print(f"  Records : {summary.get('n_records', 0)}  "
          f"(truncated: {summary.get('n_truncated', 0)})")
    print()
    print(f"  {'Metric':<26}  {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*26}  {'-'*8}  {'-'*8}")
    _ALL_DISPLAY = [
        ("rouge_l",              "ROUGE-L"),
        ("entity_precision",     "Entity Precision"),
        ("entity_recall",        "Entity Recall"),
        ("entity_f1",            "Entity F1"),
        ("entity_f1_drugs",      "Entity F1 (drugs)"),
        ("entity_f1_diagnoses",  "Entity F1 (diagnoses)"),
        ("entity_f1_vitals",     "Entity F1 (vitals)"),
        ("aarp_score",           "AARP Section Score"),
        ("abnormal_precision",   "Abnormal Precision"),
        ("abnormal_recall",      "Abnormal Recall"),
        ("abnormal_f1",          "Abnormal F1"),
        ("bertscore_f1",         "BERTScore F1"),
        ("problem_coverage",     "Problem Coverage"),
        ("key_lab_coverage",     "Key Lab Coverage"),
        ("medication_accuracy",  "Medication Accuracy"),
    ]
    for key, label in _ALL_DISPLAY:
        v = summary.get(key)
        if v is None:
            continue
        mean_s = f"{v['mean']:.4f}" if v.get("mean") is not None else "   N/A  "
        std_s  = f"{v['std']:.4f}"  if v.get("std")  is not None else "   N/A  "
        print(f"  {label:<26}  {mean_s:>8}  {std_s:>8}")
    print()


def _print_comparison_table(comparison: dict[str, Any]) -> None:
    a = comparison["model_a"]
    b = comparison["model_b"]
    print()
    print(f"  Comparison: {a}  vs  {b}")
    print(f"  Pairs: {comparison.get('n_pairs', '?')}")
    print()
    print(f"  {'Metric':<22}  {'A mean':>8}  {'B mean':>8}  {'Δ':>8}  {'p':>8}  {'r':>6}  {'Sig':>4}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*4}")
    for m in comparison.get("metrics", []):
        sig = "Yes" if m["significant"] else "No"
        print(
            f"  {m['metric']:<22}  {m['model_a_mean']:>8.4f}  {m['model_b_mean']:>8.4f}"
            f"  {m['delta']:>+8.4f}  {m['wilcoxon_p']:>8.4f}  {m['effect_r']:>6.3f}  {sig:>4}"
        )
    print()


def _main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=""
    )
    subparsers = parser.add_subparsers(dest="mode")

    score_p = subparsers.add_parser("score", help="Score one inference file.")
    score_p.add_argument("--inference", required=True, type=Path)
    score_p.add_argument("--reference", required=True, type=Path)

    cmp_p = subparsers.add_parser("compare", help="Compare two eval files.")
    cmp_p.add_argument("--eval-a",  required=True, type=Path)
    cmp_p.add_argument("--eval-b",  required=True, type=Path)

    args, _ = parser.parse_known_args(argv)
    if args.mode is None:
        score_p.set_defaults(mode="score")
        args = score_p.parse_args(argv)
        args.mode = "score"

    if args.mode == "compare":
        _run_compare(args)
    else:
        _run_score(args)


def _run_score(args: Any) -> None:
    inf_path: Path = args.inference
    ref_path: Path = args.reference

    stem    = inf_path.stem
    slug    = stem[len("inference_"):] if stem.startswith("inference_") else stem
    out_dir = inf_path.parent

    eval_path    = out_dir / f"eval_{slug}.jsonl"
    summary_path = out_dir / f"summary_{slug}.json"

    _log.info("Loading inference records from %s …", inf_path)
    inf_records = _load_jsonl(inf_path)
    _log.info("Loading reference records from %s …", ref_path)
    ref_records = _load_jsonl(ref_path)

    gt_by_id: dict[int, str] = {
        r["metadata"]["note_id"]: r["ground_truth"]
        for r in ref_records
        if "ground_truth" in r and "metadata" in r
    }
    ap_by_id: dict[int, list] = {
        r["metadata"]["note_id"]: r.get("active_problems", [])
        for r in ref_records
        if "metadata" in r
    }

    is_abnormal_task = bool(inf_records and "gt_flags" in inf_records[0])
    is_summary_task  = bool(inf_records and "hospital_course" in inf_records[0] and not is_abnormal_task)
    if is_abnormal_task:
        _log.info("Detected Sub-task 2 (abnormal) inference file.")
    elif is_summary_task:
        _log.info("Detected Sub-task 3 (summary) inference file.")

    src_by_id: dict[int, dict] = {
        r["metadata"]["note_id"]: r for r in ref_records if "metadata" in r
    }

    nlp = None
    if not is_abnormal_task and not is_summary_task:
        _log.info(
            "Loading scispaCy model (CUDA_VISIBLE_DEVICES=%s) …",
            os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        )
        nlp = _load_nlp()

    _log.info("Scoring %d records …", len(inf_records))
    eval_records: list[dict] = []

    for i, inf_rec in enumerate(inf_records, 1):
        note_id  = inf_rec["note_id"]
        gen_text = inf_rec.get("generated_text", "")

        if is_abnormal_task:
            gt_flags = inf_rec.get("gt_flags", [])
            scores   = score_abnormal_record(gen_text, gt_flags)
            eval_records.append({
                "note_id":             note_id,
                "hadm_id":             inf_rec.get("hadm_id"),
                "subject_id":          inf_rec.get("subject_id"),
                "model_id":            inf_rec.get("model_id", ""),
                "abnormal_precision":  scores["abnormal_precision"],
                "abnormal_recall":     scores["abnormal_recall"],
                "abnormal_f1":         scores["abnormal_f1"],
                "n_truly_abnormal":    scores["n_truly_abnormal"],
                "n_flagged_by_model":  scores["n_flagged_by_model"],
                "truncated_obs":       inf_rec.get("truncated_obs", 0),
                "truncated_labs":      inf_rec.get("truncated_labs", 0),
                "truncated_meds":      inf_rec.get("truncated_meds", 0),
            })
        elif is_summary_task:
            hospital_course = inf_rec.get("hospital_course", "")
            src_rec         = src_by_id.get(note_id, {})
            source_facts    = src_rec.get("source_facts", [])
            active_problems = src_rec.get("active_problems", [])
            scores = score_summary_record(
                generated=gen_text,
                hospital_course=hospital_course,
                source_facts=source_facts,
                active_problems=active_problems,
            )
            eval_records.append({
                "note_id":              note_id,
                "hadm_id":             inf_rec.get("hadm_id"),
                "subject_id":          inf_rec.get("subject_id"),
                "model_id":            inf_rec.get("model_id", ""),
                "rouge_l":             scores["rouge_l"],
                "bertscore_f1":        scores["bertscore_f1"],
                "problem_coverage":    scores["problem_coverage"],
                "key_lab_coverage":    scores["key_lab_coverage"],
                "medication_accuracy": scores["medication_accuracy"],
                "truncated_obs":       inf_rec.get("truncated_obs", 0),
                "truncated_labs":      inf_rec.get("truncated_labs", 0),
                "truncated_meds":      inf_rec.get("truncated_meds", 0),
            })
        else:
            gt_text = gt_by_id.get(note_id, "")
            if not gt_text:
                _log.warning("No ground truth for note_id=%s — skipping.", note_id)
                continue

            scores = score_record(gen_text, gt_text, nlp, ap_by_id.get(note_id))
            eval_records.append({
                "note_id":              note_id,
                "hadm_id":             inf_rec.get("hadm_id"),
                "subject_id":          inf_rec.get("subject_id"),
                "model_id":            inf_rec.get("model_id", ""),
                "rouge_l":             scores["rouge_l"],
                "entity_precision":    scores["entity_precision"],
                "entity_recall":       scores["entity_recall"],
                "entity_f1":           scores["entity_f1"],
                "entity_f1_drugs":     scores["entity_f1_drugs"],
                "entity_f1_diagnoses": scores["entity_f1_diagnoses"],
                "entity_f1_vitals":    scores["entity_f1_vitals"],
                "aarp_score":          scores["aarp_score"],
                "truncated_obs":       inf_rec.get("truncated_obs", 0),
                "truncated_labs":      inf_rec.get("truncated_labs", 0),
                "truncated_meds":      inf_rec.get("truncated_meds", 0),
            })

        if i % 50 == 0 or i == len(inf_records):
            _log.info("Progress: %d / %d", i, len(inf_records))

    _log.info("Writing per-record scores to %s …", eval_path)
    _write_jsonl(eval_records, eval_path)

    summary = compute_corpus_summary(eval_records)
    _log.info("Writing summary to %s …", summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    _print_summary_table(summary)

    _log.info("Done.")


def _run_compare(args: Any) -> None:
    eval_a  = _load_jsonl(args.eval_a)
    eval_b  = _load_jsonl(args.eval_b)

    model_a = eval_a[0].get("model_id", str(args.eval_a)) if eval_a else str(args.eval_a)
    model_b = eval_b[0].get("model_id", str(args.eval_b)) if eval_b else str(args.eval_b)

    _log.info("Running Wilcoxon comparison …")
    metric_results = wilcoxon_compare(eval_a, eval_b)

    stem_a   = args.eval_a.stem[len("eval_"):] if args.eval_a.stem.startswith("eval_") else args.eval_a.stem
    stem_b   = args.eval_b.stem[len("eval_"):] if args.eval_b.stem.startswith("eval_") else args.eval_b.stem
    out_path = args.eval_a.parent / f"comparison_{stem_a}_vs_{stem_b}.json"

    n_pairs = len({r["note_id"] for r in eval_a} & {r["note_id"] for r in eval_b})

    comparison = {
        "model_a": model_a,
        "model_b": model_b,
        "n_pairs": n_pairs,
        "metrics": metric_results,
    }

    out_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    _log.info("Comparison → %s", out_path)
    _print_comparison_table(comparison)


if __name__ == "__main__":
    _main()
