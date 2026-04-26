"""Phase 1 ETL pipeline: MIMIC-III → FHIR-lite JSONL.

Two-pass architecture with Parquet intermediaries:

Pass 1 (``--extract``):
    Scan each raw CSV.gz once, filter to MICU admissions, and write a
    compact Parquet file to ``data/intermediate/``.  Peak memory is bounded
    by the largest single table (~19 M CHARTEVENTS rows ≈ 3 GB), which is
    fully freed before the next table is processed.

Pass 2 (``--process``):
    Read exclusively from Parquet (seekable, ~10× faster than gz re-scan).
    Process ``hadm_id`` batches from columnar Parquet with predicate
    pushdown; resolve item labels and harmonisation from small in-memory
    DataFrames (12 k rows); stream validated FHIR-lite records to JSONL.

Pass 3 (``--split``):
    Patient-level train/val/test split of the JSONL output.  Two-pass
    streaming — no full file held in RAM.

Design principles:
- No single ``collect()`` spans more than one source table at a time.
- Label/harmonisation CSVs (tiny) are collected once and reused in-memory.
- ``--resume`` skips already-processed admissions on crash/restart.
- ``--sample N`` caps ``--process`` to the first N admissions (CI/smoke).
- All CPU cores used by Polars default multi-threaded scheduler.
- No GPU resources consumed; VRAM preserved for Phase 2 (vLLM).
"""

import argparse
import csv
import gzip
import json
import logging
import multiprocessing
import re
import random
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import polars as pl
from pydantic import ValidationError

from src.etl.schemas import (
    ActiveProblem,
    FHIRLiteRecord,
    MedicationEvent,
    NoteMetadata,
    ObservationEntry,
    ProcedureEvent,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_MIMIC_DIR = _DATA_DIR / "MIMICIII"
_INT_DIR   = _DATA_DIR / "intermediate"
_OUT_DIR   = _DATA_DIR / "processed"

# Raw source files
NOTEEVENTS_PATH         = _MIMIC_DIR / "NOTEEVENTS.csv.gz"
CHARTEVENTS_PATH        = _MIMIC_DIR / "CHARTEVENTS.csv.gz"
LABEVENTS_PATH          = _MIMIC_DIR / "LABEVENTS.csv.gz"
ICUSTAYS_PATH           = _MIMIC_DIR / "ICUSTAYS.csv.gz"
D_ITEMS_PATH            = _MIMIC_DIR / "D_ITEMS.csv.gz"
D_LABITEMS_PATH         = _MIMIC_DIR / "D_LABITEMS.csv.gz"
ADMISSIONS_PATH         = _MIMIC_DIR / "ADMISSIONS.csv.gz"
DIAGNOSES_ICD_PATH      = _MIMIC_DIR / "DIAGNOSES_ICD.csv.gz"
D_ICD_DIAGNOSES_PATH    = _MIMIC_DIR / "D_ICD_DIAGNOSES.csv.gz"
INPUTEVENTS_CV_PATH     = _MIMIC_DIR / "INPUTEVENTS_CV.csv.gz"
INPUTEVENTS_MV_PATH     = _MIMIC_DIR / "INPUTEVENTS_MV.csv.gz"
PROCEDUREEVENTS_MV_PATH = _MIMIC_DIR / "PROCEDUREEVENTS_MV.csv.gz"
ITEM_MAP_PATH           = _MIMIC_DIR / "itemid_to_variable_map.csv"
VARIABLE_RANGES_PATH    = _MIMIC_DIR / "variable_ranges.csv"

# Parquet intermediaries (Pass 1 output / Pass 2 input)
COHORT_PARQUET     = _INT_DIR / "cohort.parquet"
CHART_PARQUET      = _INT_DIR / "chart_micu.parquet"
LAB_PARQUET        = _INT_DIR / "lab_micu.parquet"
MED_PARQUET        = _INT_DIR / "med_micu.parquet"
PROC_PARQUET       = _INT_DIR / "proc_micu.parquet"
ADMISSIONS_PARQUET = _INT_DIR / "admissions.parquet"
DIAGNOSES_PARQUET  = _INT_DIR / "diagnoses.parquet"
DISCHARGE_PARQUET  = _INT_DIR / "discharge_summaries.parquet"

# Final output
OUTPUT_PATH = _OUT_DIR / "fhir_lite.jsonl"

# ---------------------------------------------------------------------------
# MIMIC-Extract resource URLs (Wang et al., CHIL 2020 — arXiv:1907.08322)
# ---------------------------------------------------------------------------
_MIMIC_EXTRACT_BASE = (
    "https://raw.githubusercontent.com/MLforHealth/MIMIC_Extract/master/resources"
)
_MIMIC_EXTRACT_RESOURCES = {
    ITEM_MAP_PATH:        f"{_MIMIC_EXTRACT_BASE}/itemid_to_variable_map.csv",
    VARIABLE_RANGES_PATH: f"{_MIMIC_EXTRACT_BASE}/variable_ranges.csv",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WINDOW_HOURS: int   = 12
_MIN_WORD_COUNT: int = 50
_MIN_DENSITY: int    = 5
_TARGET_CAREUNIT: str = "MICU"

# Regex matching monitor alarm/alert threshold entries — not physiological
# observations; excluded from the events surfaced to the model.
_METRIC_DENYLIST_RE: str = r"(?i)\balarm\b|\balert\b"

# Group keys used when aggregating per-note event lists.
# NOTE: "text" is intentionally excluded to avoid duplicating multi-KB note
# strings across every intermediate join row (~48M rows × 2 KB = 96 GB OOM).
# Text is re-joined onto the aggregated result after group_by().
_GROUP_KEYS: List[str] = ["note_id", "subject_id", "hadm_id", "careunit"]


# ---------------------------------------------------------------------------
# Step 0: MIMIC-Extract resource download
# ---------------------------------------------------------------------------


def download_mimic_extract_resources(force: bool = False) -> None:
    """Download MIMIC-Extract reference files from GitHub if not already present.

    Fetches two files from the MLforHealth/MIMIC_Extract repository
    (Wang et al., CHIL 2020, arXiv:1907.08322):

    - ``itemid_to_variable_map.csv`` — canonical ITEMID → variable name
      mapping used to harmonise CareVue and MetaVision ITEMIDs.
    - ``variable_ranges.csv`` — per-variable physiological plausible range
      bounds used to flag out-of-range values.

    Files are written to ``data/MIMICIII/``.

    Args:
        force: Re-download even if files already exist on disk.
    """
    _MIMIC_DIR.mkdir(parents=True, exist_ok=True)
    for dest_path, url in _MIMIC_EXTRACT_RESOURCES.items():
        if dest_path.exists() and not force:
            logger.info("Resource already present, skipping: %s", dest_path.name)
            continue
        logger.info("Downloading %s → %s", url, dest_path)
        try:
            urllib.request.urlretrieve(url, dest_path)
            logger.info("Downloaded %s (%d bytes)", dest_path.name, dest_path.stat().st_size)
        except Exception as exc:
            logger.error("Failed to download %s: %s", url, exc)
            raise


# ---------------------------------------------------------------------------
# Label / harmonisation / range-bounds loaders (small CSVs → in-memory)
# ---------------------------------------------------------------------------


def load_item_labels() -> pl.LazyFrame:
    """Load D_ITEMS: ITEMID → human-readable label for CHARTEVENTS/INPUTEVENTS.

    Returns:
        LazyFrame with columns ``item_id`` (Int64) and ``metric`` (Utf8).
    """
    return pl.scan_csv(D_ITEMS_PATH, infer_schema_length=5_000).select(
        [
            pl.col("ITEMID").alias("item_id"),
            pl.col("LABEL").alias("metric"),
        ]
    )


def load_lab_item_labels() -> pl.LazyFrame:
    """Load D_LABITEMS: ITEMID → human-readable label for LABEVENTS.

    Returns:
        LazyFrame with columns ``item_id`` (Int64) and ``metric`` (Utf8).
    """
    return pl.scan_csv(D_LABITEMS_PATH, infer_schema_length=1_000).select(
        [
            pl.col("ITEMID").alias("item_id"),
            pl.col("LABEL").alias("metric"),
        ]
    )


def load_item_harmonisation_map() -> Optional[pl.LazyFrame]:
    """Load the MIMIC-Extract ITEMID → canonical variable name map.

    Returns:
        LazyFrame with columns ``item_id`` (Int64) and
        ``canonical_metric`` (Utf8), or ``None`` if the file is absent.
    """
    if not ITEM_MAP_PATH.exists():
        logger.warning(
            "itemid_to_variable_map.csv not found — ITEMID harmonisation disabled. "
            "Run with --download-resources to fetch it."
        )
        return None
    return (
        pl.scan_csv(ITEM_MAP_PATH, infer_schema_length=5_000)
        .filter(pl.col("STATUS") == "ready")
        .select(
            [
                pl.col("ITEMID").alias("item_id"),
                pl.col("LEVEL2").alias("canonical_metric"),
            ]
        )
    )


def load_range_bounds() -> Optional[pl.LazyFrame]:
    """Load MIMIC-Extract per-variable physiological plausible range bounds.

    Returns:
        LazyFrame with columns ``range_metric`` (Utf8, lowercased),
        ``range_low`` (Float64), ``range_high`` (Float64), or ``None``.
    """
    if not VARIABLE_RANGES_PATH.exists():
        logger.warning(
            "variable_ranges.csv not found — range filtering disabled. "
            "Run with --download-resources to fetch it."
        )
        return None
    return (
        pl.scan_csv(VARIABLE_RANGES_PATH, infer_schema_length=100)
        .select(
            [
                pl.col("LEVEL2").str.to_lowercase().alias("range_metric"),
                pl.col("OUTLIER LOW").cast(pl.Float64, strict=False).alias("range_low"),
                pl.col("OUTLIER HIGH").cast(pl.Float64, strict=False).alias("range_high"),
            ]
        )
        .filter(pl.col("range_low").is_not_null() | pl.col("range_high").is_not_null())
    )


# ---------------------------------------------------------------------------
# ISO 8601 relative offset expression
# ---------------------------------------------------------------------------


def iso8601_relative_expr(
    note_time_col: str = "note_time",
    event_time_col: str = "event_time",
) -> pl.Expr:
    """Build a Polars expression formatting a duration as ``T-{h}h {m}m``.

    Args:
        note_time_col: Name of the note timestamp column.
        event_time_col: Name of the event timestamp column.

    Returns:
        A Polars ``Expr`` that evaluates to a ``Utf8`` column named ``time``.
    """
    total_minutes: pl.Expr = (
        (pl.col(note_time_col) - pl.col(event_time_col))
        .dt.total_minutes()
        .cast(pl.Int64)
    )
    hours: pl.Expr   = (total_minutes // 60).cast(pl.Int64)
    minutes: pl.Expr = (total_minutes % 60).cast(pl.Int64)
    return pl.concat_str(
        [pl.lit("T-"), hours.cast(pl.Utf8), pl.lit("h "), minutes.cast(pl.Utf8), pl.lit("m")]
    ).alias("time")


# ---------------------------------------------------------------------------
# Temporal window join helpers
# ---------------------------------------------------------------------------


def apply_temporal_window(
    cohort_lf: pl.LazyFrame,
    events_lf: pl.LazyFrame,
    item_labels_lf: pl.LazyFrame,
    harmonisation_lf: Optional[pl.LazyFrame] = None,
    range_bounds_lf: Optional[pl.LazyFrame] = None,
) -> pl.LazyFrame:
    """Inner-join events to nursing notes within the 12-hour look-back window.

    Retains event rows where ``note_time - 12h <= event_time <= note_time``.
    Applies ITEMID harmonisation and physiological range flagging when the
    corresponding reference frames are provided.

    ``events_lf`` is expected to already be pre-filtered to the batch's
    ``hadm_id`` set (i.e. read from Parquet with a filter predicate).

    Args:
        cohort_lf: Nursing cohort LazyFrame for the current batch.
        events_lf: CHARTEVENTS or LABEVENTS LazyFrame (pre-filtered).
        item_labels_lf: D_ITEMS or D_LABITEMS label LazyFrame.
        harmonisation_lf: Optional ITEMID → canonical name map.
        range_bounds_lf: Optional per-variable plausible range bounds.

    Returns:
        LazyFrame with columns: ``note_id``, ``subject_id``, ``hadm_id``,
        ``careunit``, ``note_time``, ``text``, ``event_time``, ``metric``,
        ``val``, ``val_type``, ``unit``, ``time``, ``out_of_range``.
    """
    window_td = pl.duration(hours=_WINDOW_HOURS)

    # Drop text before the join to avoid carrying multi-KB strings through
    # every intermediate row (join expansion can reach 48M+ rows, causing OOM).
    cohort_slim_lf = cohort_lf.drop("text")

    events_parsed = events_lf.with_columns(
        pl.col("event_time_raw")
        .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
        .alias("event_time")
    ).drop("event_time_raw")

    events_labelled = events_parsed.join(
        item_labels_lf, on="item_id", how="left"
    ).with_columns(
        pl.col("metric").fill_null(pl.col("item_id").cast(pl.Utf8))
    )

    if harmonisation_lf is not None:
        events_labelled = (
            events_labelled
            .join(harmonisation_lf, on="item_id", how="left")
            .with_columns(pl.coalesce(["canonical_metric", "metric"]).alias("metric"))
            .drop("canonical_metric")
        )

    events_labelled = events_labelled.drop("item_id")

    joined = (
        cohort_slim_lf.join(events_labelled, on="hadm_id", how="inner")
        .filter(
            (pl.col("event_time") >= pl.col("note_time") - window_td)
            & (pl.col("event_time") <= pl.col("note_time"))
        )
        .filter(pl.col("val").is_not_null())
        .filter(~pl.col("metric").str.contains(_METRIC_DENYLIST_RE))
        .with_columns(iso8601_relative_expr("note_time", "event_time"))
        .with_columns(
            pl.when(
                pl.col("val")
                .str.replace(r"^[><]?\s*", "")
                .cast(pl.Float64, strict=False)
                .is_not_null()
            )
            .then(pl.lit("numeric"))
            .otherwise(pl.lit("categorical"))
            .alias("val_type")
        )
    )

    if range_bounds_lf is not None:
        val_numeric_expr = (
            pl.col("val").str.replace(r"^[><]?\s*", "").cast(pl.Float64, strict=False)
        )
        joined = (
            joined
            .with_columns(
                pl.col("metric").str.to_lowercase().alias("_metric_lower"),
                val_numeric_expr.alias("_val_numeric"),
            )
            .join(range_bounds_lf, left_on="_metric_lower", right_on="range_metric", how="left")
            .with_columns(
                pl.when(
                    (pl.col("val_type") == "numeric")
                    & (
                        (pl.col("range_low").is_not_null() & (pl.col("_val_numeric") < pl.col("range_low")))
                        | (pl.col("range_high").is_not_null() & (pl.col("_val_numeric") > pl.col("range_high")))
                    )
                )
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
                .alias("out_of_range")
            )
            .drop(["_metric_lower", "_val_numeric", "range_low", "range_high"])
        )
    else:
        joined = joined.with_columns(pl.lit(False).alias("out_of_range"))

    return joined.select(
        ["note_id", "subject_id", "hadm_id", "careunit", "note_time",
         "event_time", "metric", "val", "val_type", "unit", "time", "out_of_range"]
    )


def apply_temporal_window_medications(
    cohort_lf: pl.LazyFrame,
    inputevents_lf: pl.LazyFrame,
    item_labels_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Inner-join INPUTEVENTS to nursing notes within the 12-hour window.

    ``inputevents_lf`` is expected to already be pre-filtered to the batch's
    ``hadm_id`` set.

    Args:
        cohort_lf: Nursing cohort LazyFrame for the current batch.
        inputevents_lf: INPUTEVENTS LazyFrame (pre-filtered to batch hadm_ids).
        item_labels_lf: D_ITEMS label LazyFrame.

    Returns:
        LazyFrame with columns: ``note_id``, ``subject_id``, ``hadm_id``,
        ``careunit``, ``note_time``, ``text``, ``event_time``, ``drug``,
        ``amount``, ``unit``, ``route``, ``time``.
    """
    window_td = pl.duration(hours=_WINDOW_HOURS)

    # Drop text before join to prevent OOM from text duplication.
    cohort_slim_lf = cohort_lf.drop("text")

    events_parsed = inputevents_lf.with_columns(
        pl.col("event_time_raw")
        .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
        .alias("event_time")
    ).drop("event_time_raw")

    events_labelled = (
        events_parsed
        .join(item_labels_lf, on="item_id", how="left")
        .with_columns(
            pl.col("metric").fill_null(pl.col("item_id").cast(pl.Utf8)).alias("drug")
        )
        .drop(["item_id", "metric"])
        .filter(pl.col("amount").is_not_null())
    )

    return (
        cohort_slim_lf
        .join(events_labelled, on="hadm_id", how="inner")
        .filter(
            (pl.col("event_time") >= pl.col("note_time") - window_td)
            & (pl.col("event_time") <= pl.col("note_time"))
        )
        .with_columns(iso8601_relative_expr("note_time", "event_time"))
        .select(["note_id", "subject_id", "hadm_id", "careunit", "note_time",
                 "event_time", "drug", "amount", "unit", "route", "time"])
    )


def apply_temporal_window_procedures(
    cohort_lf: pl.LazyFrame,
    procedureevents_lf: pl.LazyFrame,
    item_labels_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Inner-join PROCEDUREEVENTS_MV to nursing notes within the 12-hour window.

    ``procedureevents_lf`` is expected to already be pre-filtered to the
    batch's ``hadm_id`` set.

    Args:
        cohort_lf: Nursing cohort LazyFrame for the current batch.
        procedureevents_lf: PROCEDUREEVENTS_MV LazyFrame (pre-filtered).
        item_labels_lf: D_ITEMS label LazyFrame.

    Returns:
        LazyFrame with columns: ``note_id``, ``subject_id``, ``hadm_id``,
        ``careunit``, ``note_time``, ``text``, ``event_time``,
        ``procedure``, ``duration_min``, ``status``, ``time``.
    """
    window_td = pl.duration(hours=_WINDOW_HOURS)

    # Drop text before join to prevent OOM from text duplication.
    cohort_slim_lf = cohort_lf.drop("text")

    events_parsed = procedureevents_lf.with_columns(
        pl.col("event_time_raw")
        .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
        .alias("event_time")
    ).drop("event_time_raw")

    events_labelled = (
        events_parsed
        .join(item_labels_lf, on="item_id", how="left")
        .with_columns(
            pl.col("metric").fill_null(pl.col("item_id").cast(pl.Utf8)).alias("procedure")
        )
        .drop(["item_id", "metric"])
    )

    return (
        cohort_slim_lf
        .join(events_labelled, on="hadm_id", how="inner")
        .filter(
            (pl.col("event_time") >= pl.col("note_time") - window_td)
            & (pl.col("event_time") <= pl.col("note_time"))
        )
        .with_columns(iso8601_relative_expr("note_time", "event_time"))
        .select(["note_id", "subject_id", "hadm_id", "careunit", "note_time",
                 "event_time", "procedure", "duration_min", "status", "time"])
    )


# ---------------------------------------------------------------------------
# Pydantic helper builders
# ---------------------------------------------------------------------------


def _build_observation_list(raw: List[Dict]) -> List[ObservationEntry]:
    """Convert raw struct dicts to validated ObservationEntry objects."""
    return [ObservationEntry(**e) for e in raw]


def _build_active_problem_list(raw: List[Dict]) -> List[ActiveProblem]:
    """Convert raw struct dicts to validated ActiveProblem objects, skipping bad rows."""
    result: List[ActiveProblem] = []
    for e in raw:
        try:
            result.append(ActiveProblem(**e))
        except Exception:
            pass
    return result


def _build_medication_list(raw: List[Dict]) -> List[MedicationEvent]:
    """Convert raw struct dicts to validated MedicationEvent objects."""
    return [MedicationEvent(**e) for e in raw]


def _build_procedure_list(raw: List[Dict]) -> List[ProcedureEvent]:
    """Convert raw struct dicts to validated ProcedureEvent objects."""
    return [ProcedureEvent(**e) for e in raw]


def _build_source_facts(
    observations: List[ObservationEntry],
    labs: List[ObservationEntry],
    medications: List[MedicationEvent],
    procedures: List[ProcedureEvent],
    active_problems: List[ActiveProblem],
) -> List[str]:
    """Generate flat natural-language grounding statements for hallucination auditing.

    Args:
        observations: Chartevents observation entries.
        labs: Lab result entries.
        medications: Medication/fluid administration entries.
        procedures: Procedure event entries.
        active_problems: ICD-9 diagnosis entries.

    Returns:
        List of natural-language fact strings.
    """
    facts: List[str] = []
    for entry in observations + labs:
        unit_str = f" {entry.unit}" if entry.unit and entry.unit.strip() else ""
        fact = f"{entry.metric} was {entry.val}{unit_str} at {entry.time}"
        if entry.out_of_range:
            fact += " [OUT_OF_RANGE]"
        facts.append(fact)
    for med in medications:
        amount_str = ""
        if med.amount is not None:
            unit_str = f" {med.unit}" if med.unit and med.unit.strip() else ""
            amount_str = f" {med.amount}{unit_str}"
        route_str = f" via {med.route}" if med.route and med.route.strip() else ""
        facts.append(f"{med.drug}{amount_str} administered{route_str} at {med.time}")
    for proc in procedures:
        dur_str    = f" (duration: {proc.duration_min} min)" if proc.duration_min is not None else ""
        status_str = f" [{proc.status}]" if proc.status and proc.status.strip() else ""
        facts.append(f"{proc.procedure} performed at {proc.time}{dur_str}{status_str}")
    for prob in active_problems:
        facts.append(f"Patient diagnosed with {prob.description} (ICD-9: {prob.icd9_code})")
    return facts


# ---------------------------------------------------------------------------
# Pass 1 — extract_intermediaries
# ---------------------------------------------------------------------------


def extract_intermediaries(force: bool = False) -> None:
    """Extract MICU-relevant subsets of all source tables to Parquet.

    Processes each source table in isolation (one scan, one write, memory
    freed before the next step).  If a Parquet file already exists and
    ``force`` is False, that step is skipped.

    Intermediate files are written to ``data/intermediate/``.

    Gate counts (nursing_raw, after_wc, after_icu) are logged for the ACM
    methodology section.

    Args:
        force: Re-write Parquet files even if they already exist on disk.
    """
    _INT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: ICUSTAYS → collect MICU hadm_ids (tiny, ~9 k rows)
    # ------------------------------------------------------------------
    logger.info("Extract Step 1/8: scanning ICUSTAYS for MICU hadm_ids …")
    micu_hadm_ids: List[int] = (
        pl.scan_csv(ICUSTAYS_PATH, infer_schema_length=5_000)
        .filter(pl.col("FIRST_CAREUNIT") == _TARGET_CAREUNIT)
        .select(pl.col("HADM_ID").alias("hadm_id"))
        .unique()
        .collect()["hadm_id"]
        .to_list()
    )
    logger.info("  MICU admissions found: %d", len(micu_hadm_ids))

    # Helper: a tiny LazyFrame of just the hadm_id list for semi-joins.
    def _hadm_lf() -> pl.LazyFrame:
        return pl.LazyFrame({"hadm_id": micu_hadm_ids})

    # ------------------------------------------------------------------
    # Step 2: NOTEEVENTS → cohort.parquet
    # ------------------------------------------------------------------
    if COHORT_PARQUET.exists() and not force:
        logger.info("Extract Step 2/8: cohort.parquet exists, skipping.")
    else:
        logger.info("Extract Step 2/8: building cohort from NOTEEVENTS …")
        notes_lf = pl.scan_csv(NOTEEVENTS_PATH, infer_schema_length=10_000).select(
            [
                pl.col("ROW_ID").alias("note_id"),
                pl.col("SUBJECT_ID").alias("subject_id"),
                pl.col("HADM_ID").alias("hadm_id"),
                pl.col("CHARTTIME").alias("note_time_raw"),
                pl.col("CATEGORY").alias("category"),
                pl.col("TEXT").alias("text"),
            ]
        )

        # Gate counts in one scan.
        gate_df = (
            notes_lf.filter(pl.col("category") == "Nursing")
            .with_columns(pl.col("text").str.split(" ").list.len().alias("wc"))
            .select(
                pl.len().alias("nursing_raw"),
                (pl.col("wc") >= _MIN_WORD_COUNT).sum().alias("after_wc"),
            )
            .collect()
        )
        nursing_raw: int = gate_df["nursing_raw"][0]
        after_wc: int    = gate_df["after_wc"][0]

        # ICU stays for temporal bounds check.
        icu_lf = (
            pl.scan_csv(ICUSTAYS_PATH, infer_schema_length=5_000)
            .filter(pl.col("FIRST_CAREUNIT") == _TARGET_CAREUNIT)
            .select(
                [
                    pl.col("HADM_ID").alias("hadm_id"),
                    pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("icu_intime"),
                    pl.col("OUTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("icu_outtime"),
                    pl.col("FIRST_CAREUNIT").alias("careunit"),
                ]
            )
        )

        cohort_df = (
            notes_lf
            .filter(pl.col("category") == "Nursing")
            .with_columns(
                pl.col("note_time_raw").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("note_time")
            )
            .filter(pl.col("text").str.split(" ").list.len() >= _MIN_WORD_COUNT)
            .drop(["note_time_raw", "category"])
            .join(_hadm_lf(), on="hadm_id", how="semi")
            .join(icu_lf, on="hadm_id", how="inner")
            .filter(
                (pl.col("note_time") >= pl.col("icu_intime"))
                & (pl.col("note_time") <= pl.col("icu_outtime"))
            )
            .drop(["icu_intime", "icu_outtime"])
            .collect()
        )
        after_icu: int = len(cohort_df)

        logger.info(
            "  Cohort gates | Nursing notes: %d | after word_count>=%d: %d | after MICU stay bounds: %d",
            nursing_raw, _MIN_WORD_COUNT, after_wc, after_icu,
        )
        cohort_df.write_parquet(COHORT_PARQUET, compression="zstd")
        logger.info("  → %s (%s)", COHORT_PARQUET.name, _fmt_size(COHORT_PARQUET))
        del cohort_df

    # ------------------------------------------------------------------
    # Step 3: CHARTEVENTS → chart_micu.parquet
    # ------------------------------------------------------------------
    if CHART_PARQUET.exists() and not force:
        logger.info("Extract Step 3/8: chart_micu.parquet exists, skipping.")
    else:
        logger.info("Extract Step 3/8: filtering CHARTEVENTS to MICU admissions …")
        df = (
            pl.scan_csv(
                CHARTEVENTS_PATH,
                infer_schema_length=10_000,
                schema_overrides={"VALUE": pl.Utf8, "VALUENUM": pl.Utf8},
            )
            .select(
                [
                    pl.col("HADM_ID").alias("hadm_id"),
                    pl.col("CHARTTIME").alias("event_time_raw"),
                    pl.col("ITEMID").alias("item_id"),
                    pl.col("VALUE").alias("val"),
                    pl.col("VALUEUOM").alias("unit"),
                ]
            )
            .join(_hadm_lf(), on="hadm_id", how="semi")
            .collect()
        )
        df.write_parquet(CHART_PARQUET, compression="zstd")
        logger.info("  → %s (%s, %d rows)", CHART_PARQUET.name, _fmt_size(CHART_PARQUET), len(df))
        del df

    # ------------------------------------------------------------------
    # Step 4: LABEVENTS → lab_micu.parquet
    # ------------------------------------------------------------------
    if LAB_PARQUET.exists() and not force:
        logger.info("Extract Step 4/8: lab_micu.parquet exists, skipping.")
    else:
        logger.info("Extract Step 4/8: filtering LABEVENTS to MICU admissions …")
        df = (
            pl.scan_csv(
                LABEVENTS_PATH,
                infer_schema_length=10_000,
                schema_overrides={"VALUE": pl.Utf8, "VALUENUM": pl.Utf8},
            )
            .select(
                [
                    pl.col("HADM_ID").alias("hadm_id"),
                    pl.col("CHARTTIME").alias("event_time_raw"),
                    pl.col("ITEMID").alias("item_id"),
                    pl.col("VALUE").alias("val"),
                    pl.col("VALUEUOM").alias("unit"),
                ]
            )
            .join(_hadm_lf(), on="hadm_id", how="semi")
            .collect()
        )
        df.write_parquet(LAB_PARQUET, compression="zstd")
        logger.info("  → %s (%s, %d rows)", LAB_PARQUET.name, _fmt_size(LAB_PARQUET), len(df))
        del df

    # ------------------------------------------------------------------
    # Step 5: INPUTEVENTS_CV + INPUTEVENTS_MV → med_micu.parquet
    # ------------------------------------------------------------------
    if MED_PARQUET.exists() and not force:
        logger.info("Extract Step 5/8: med_micu.parquet exists, skipping.")
    else:
        logger.info("Extract Step 5/8: filtering INPUTEVENTS to MICU admissions …")
        cv_df = (
            pl.scan_csv(
                INPUTEVENTS_CV_PATH,
                infer_schema_length=10_000,
                schema_overrides={"AMOUNT": pl.Utf8},
            )
            .select(
                [
                    pl.col("HADM_ID").alias("hadm_id"),
                    pl.col("CHARTTIME").alias("event_time_raw"),
                    pl.col("ITEMID").alias("item_id"),
                    pl.col("AMOUNT").alias("amount"),
                    pl.col("AMOUNTUOM").alias("unit"),
                    pl.lit(None).cast(pl.Utf8).alias("route"),
                ]
            )
            .join(_hadm_lf(), on="hadm_id", how="semi")
            .collect()
        )
        mv_df = (
            pl.scan_csv(
                INPUTEVENTS_MV_PATH,
                infer_schema_length=10_000,
                schema_overrides={"AMOUNT": pl.Utf8},
            )
            .select(
                [
                    pl.col("HADM_ID").alias("hadm_id"),
                    pl.col("STARTTIME").alias("event_time_raw"),
                    pl.col("ITEMID").alias("item_id"),
                    pl.col("AMOUNT").alias("amount"),
                    pl.col("AMOUNTUOM").alias("unit"),
                    pl.col("ORDERCATEGORYNAME").alias("route"),
                ]
            )
            .join(_hadm_lf(), on="hadm_id", how="semi")
            .collect()
        )
        df = pl.concat([cv_df, mv_df], how="vertical")
        df.write_parquet(MED_PARQUET, compression="zstd")
        logger.info("  → %s (%s, %d rows)", MED_PARQUET.name, _fmt_size(MED_PARQUET), len(df))
        del cv_df, mv_df, df

    # ------------------------------------------------------------------
    # Step 6: PROCEDUREEVENTS_MV → proc_micu.parquet
    # ------------------------------------------------------------------
    if PROC_PARQUET.exists() and not force:
        logger.info("Extract Step 6/8: proc_micu.parquet exists, skipping.")
    else:
        logger.info("Extract Step 6/8: filtering PROCEDUREEVENTS_MV to MICU admissions …")
        df = (
            pl.scan_csv(PROCEDUREEVENTS_MV_PATH, infer_schema_length=10_000)
            .select(
                [
                    pl.col("HADM_ID").alias("hadm_id"),
                    pl.col("STARTTIME").alias("event_time_raw"),
                    pl.col("ITEMID").alias("item_id"),
                    pl.col("ENDTIME").alias("end_time_raw"),
                    pl.col("STATUSDESCRIPTION").alias("status"),
                ]
            )
            .join(_hadm_lf(), on="hadm_id", how="semi")
            .with_columns(
                [
                    pl.col("event_time_raw").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("_s"),
                    pl.col("end_time_raw").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("_e"),
                ]
            )
            .with_columns(
                pl.when(pl.col("_e").is_not_null())
                .then((pl.col("_e") - pl.col("_s")).dt.total_minutes().cast(pl.Int64))
                .otherwise(None)
                .alias("duration_min")
            )
            .drop(["_s", "_e", "end_time_raw"])
            .collect()
        )
        df.write_parquet(PROC_PARQUET, compression="zstd")
        logger.info("  → %s (%s, %d rows)", PROC_PARQUET.name, _fmt_size(PROC_PARQUET), len(df))
        del df

    # ------------------------------------------------------------------
    # Step 7: ADMISSIONS → admissions.parquet
    # ------------------------------------------------------------------
    if ADMISSIONS_PARQUET.exists() and not force:
        logger.info("Extract Step 7/8: admissions.parquet exists, skipping.")
    else:
        logger.info("Extract Step 7/8: writing ADMISSIONS …")
        df = (
            pl.scan_csv(ADMISSIONS_PATH, infer_schema_length=5_000)
            .select(
                [
                    pl.col("HADM_ID").alias("hadm_id"),
                    pl.col("DIAGNOSIS").alias("admit_diagnosis"),
                    pl.col("INSURANCE").alias("insurance"),
                    pl.col("ADMISSION_TYPE").alias("admission_type"),
                ]
            )
            .collect()
        )
        df.write_parquet(ADMISSIONS_PARQUET, compression="zstd")
        logger.info("  → %s (%s, %d rows)", ADMISSIONS_PARQUET.name, _fmt_size(ADMISSIONS_PARQUET), len(df))
        del df

    # ------------------------------------------------------------------
    # Step 8: DIAGNOSES_ICD + D_ICD_DIAGNOSES → diagnoses.parquet
    # ------------------------------------------------------------------
    if DIAGNOSES_PARQUET.exists() and not force:
        logger.info("Extract Step 8/8: diagnoses.parquet exists, skipping.")
    else:
        logger.info("Extract Step 8/8: building diagnoses lookup …")
        diag_lf = pl.scan_csv(
            DIAGNOSES_ICD_PATH,
            infer_schema_length=5_000,
            schema_overrides={"ICD9_CODE": pl.Utf8},
        ).select(
            [pl.col("HADM_ID").alias("hadm_id"), pl.col("ICD9_CODE").alias("icd9_code")]
        )
        labels_lf = pl.scan_csv(
            D_ICD_DIAGNOSES_PATH,
            infer_schema_length=5_000,
            schema_overrides={"ICD9_CODE": pl.Utf8},
        ).select(
            [pl.col("ICD9_CODE").alias("icd9_code"), pl.col("SHORT_TITLE").alias("description")]
        )
        df = (
            diag_lf
            .join(labels_lf, on="icd9_code", how="left")
            .with_columns(pl.col("description").fill_null(pl.col("icd9_code")))
            .filter(pl.col("icd9_code").is_not_null())
            .join(_hadm_lf(), on="hadm_id", how="semi")
            .group_by("hadm_id")
            .agg(pl.struct(["icd9_code", "description"]).alias("active_problems"))
            .collect()
        )
        df.write_parquet(DIAGNOSES_PARQUET, compression="zstd")
        logger.info("  → %s (%s, %d rows)", DIAGNOSES_PARQUET.name, _fmt_size(DIAGNOSES_PARQUET), len(df))
        del df

    # ------------------------------------------------------------------
    # Step 9: NOTEEVENTS (Discharge summary) → discharge_summaries.parquet
    # ------------------------------------------------------------------
    if DISCHARGE_PARQUET.exists() and not force:
        logger.info("Extract Step 9/9: discharge_summaries.parquet exists, skipping.")
    else:
        logger.info("Extract Step 9/9: extracting discharge summaries …")

        def _extract_hospital_course(text: str) -> str:
            """Return the HOSPITAL COURSE section or full text as fallback."""
            m = re.search(
                r"(brief\s+hospital\s+course|hospital\s+course)\s*:?\s*(.*?)(?=\n\s*\n[A-Z]|\Z)",
                text, re.DOTALL | re.IGNORECASE,
            )
            if m:
                return m.group(2).strip()
            return text.strip()

        rows = []
        with gzip.open(NOTEEVENTS_PATH, "rt", encoding="utf-8", errors="replace") as _f:
            reader = csv.DictReader(_f)
            for row in reader:
                if row.get("CATEGORY", "").strip() != "Discharge summary":
                    continue
                if not row.get("TEXT", "").strip():
                    continue
                rows.append({
                    "hadm_id":        int(row["HADM_ID"]),
                    "charttime":      row.get("CHARTTIME", ""),
                    "hospital_course": _extract_hospital_course(row["TEXT"]),
                })

        # Keep one summary per admission — latest CHARTTIME
        ds_df = (
            pl.DataFrame(rows)
            .sort("charttime", descending=True)
            .unique(subset=["hadm_id"], keep="first")
            .select(["hadm_id", "hospital_course"])
        )
        ds_df.write_parquet(DISCHARGE_PARQUET, compression="zstd")
        logger.info(
            "  → %s (%s, %d admissions with discharge summary)",
            DISCHARGE_PARQUET.name, _fmt_size(DISCHARGE_PARQUET), len(ds_df),
        )
        del ds_df, rows

    logger.info(
        "Extract complete. Intermediate files written to %s", _INT_DIR
    )


# ---------------------------------------------------------------------------
# Pass 2 — per-batch worker (module-level so multiprocessing can pickle it)
# ---------------------------------------------------------------------------


def _batch_worker(
    result_path: Path,
    batch_hadm: List[int],
    batch_idx: int,
    n_batches: int,
    out_path: Path,
    item_labels_df: "pl.DataFrame",
    lab_labels_df: "pl.DataFrame",
    harm_df: "Optional[pl.DataFrame]",
    rang_df: "Optional[pl.DataFrame]",
    admissions_df: "pl.DataFrame",
    diagnoses_df: "pl.DataFrame",
    discharge_df: "pl.DataFrame",
) -> None:
    """Process a single batch and write QC counters to ``result_path`` as JSON.

    Runs in an isolated subprocess (spawn context) so the OS reclaims all
    Polars/jemalloc allocator pages when the process exits.  Results are
    communicated back to the parent by writing a small JSON file.

    Args:
        result_path: Temp file path to write the result dict to.
        batch_hadm: List of ``hadm_id`` values for this batch.
        batch_idx: Zero-based batch index (for logging only).
        n_batches: Total number of batches (for logging only).
        out_path: JSONL file to append accepted records to.
        item_labels_df: D_ITEMS label DataFrame (pre-collected).
        lab_labels_df: D_LABITEMS label DataFrame (pre-collected).
        harm_df: Optional harmonisation map DataFrame.
        rang_df: Optional range bounds DataFrame.
        admissions_df: ADMISSIONS reference DataFrame.
        diagnoses_df: DIAGNOSES lookup DataFrame.
        discharge_df: Discharge summary hospital_course DataFrame (hadm_id →
            hospital_course).
    """
    # Re-import inside subprocess to ensure a clean module state.
    import json as _json
    import polars as _pl
    from pydantic import ValidationError as _ValidationError

    _log = logging.getLogger(__name__)
    _log.info(
        "Process: batch %d/%d | %d admissions …",
        batch_idx + 1, n_batches, len(batch_hadm),
    )

    cohort_batch_df = (
        _pl.scan_parquet(COHORT_PARQUET)
        .filter(_pl.col("hadm_id").is_in(batch_hadm))
        .collect()
    )
    cohort_batch_lf = cohort_batch_df.lazy()

    def _scan(path: Path) -> _pl.LazyFrame:
        return _pl.scan_parquet(path).filter(_pl.col("hadm_id").is_in(batch_hadm))

    def _maybe_lazy(df: Optional[_pl.DataFrame]) -> Optional[_pl.LazyFrame]:
        return df.lazy() if df is not None else None

    def _agg(lf: _pl.LazyFrame, list_col: str, struct_cols: List[str]) -> _pl.DataFrame:
        return (
            lf.sort("event_time", descending=True)
            .group_by(_GROUP_KEYS)
            .agg(_pl.struct(struct_cols).alias(list_col))
            .collect()
        )

    chart_df = _agg(
        apply_temporal_window(
            cohort_batch_lf, _scan(CHART_PARQUET),
            item_labels_df.lazy(), _maybe_lazy(harm_df), _maybe_lazy(rang_df),
        ),
        "observations", ["time", "metric", "val", "val_type", "unit", "out_of_range"],
    )
    lab_df = _agg(
        apply_temporal_window(
            cohort_batch_lf, _scan(LAB_PARQUET),
            lab_labels_df.lazy(), _maybe_lazy(harm_df), _maybe_lazy(rang_df),
        ),
        "labs", ["time", "metric", "val", "val_type", "unit", "out_of_range"],
    )
    med_df = _agg(
        apply_temporal_window_medications(
            cohort_batch_lf, _scan(MED_PARQUET), item_labels_df.lazy(),
        ),
        "medications", ["time", "drug", "amount", "unit", "route"],
    )
    proc_df = _agg(
        apply_temporal_window_procedures(
            cohort_batch_lf, _scan(PROC_PARQUET), item_labels_df.lazy(),
        ),
        "procedures", ["time", "procedure", "duration_min", "status"],
    )

    batch_admissions = admissions_df.filter(_pl.col("hadm_id").is_in(batch_hadm))
    batch_diagnoses  = diagnoses_df.filter(_pl.col("hadm_id").is_in(batch_hadm))
    batch_discharge  = discharge_df.filter(_pl.col("hadm_id").is_in(batch_hadm))
    text_df          = cohort_batch_df.select(["note_id", "text"])

    df_pre: _pl.DataFrame = (
        chart_df
        .join(lab_df,  on=_GROUP_KEYS, how="full", coalesce=True)
        .join(med_df,  on=_GROUP_KEYS, how="full", coalesce=True)
        .join(proc_df, on=_GROUP_KEYS, how="full", coalesce=True)
        .with_columns([
            _pl.col("observations").fill_null([]),
            _pl.col("labs").fill_null([]),
            _pl.col("medications").fill_null([]),
            _pl.col("procedures").fill_null([]),
        ])
        .join(text_df,          on="note_id",  how="left")
        .join(batch_admissions, on="hadm_id",  how="left")
        .join(batch_diagnoses,  on="hadm_id",  how="left")
        .join(batch_discharge,  on="hadm_id",  how="left")
        .with_columns(_pl.col("active_problems").fill_null([]))
    )
    pre = len(df_pre)

    df: _pl.DataFrame = df_pre.filter(
        (
            df_pre["observations"].list.len()
            + df_pre["labs"].list.len()
            + df_pre["medications"].list.len()
        ) >= _MIN_DENSITY
    )
    post = len(df)

    accepted = rejected = oor_chart = oor_lab = 0
    n_meds = n_procs = n_probs = n_adx = 0

    with open(out_path, "a", encoding="utf-8") as fh:
        for row in df.iter_rows(named=True):
            try:
                obs      = _build_observation_list(row.get("observations") or [])
                labs_lst = _build_observation_list(row.get("labs") or [])
                meds     = _build_medication_list(row.get("medications") or [])
                procs    = _build_procedure_list(row.get("procedures") or [])
                problems = _build_active_problem_list(row.get("active_problems") or [])
                record   = FHIRLiteRecord(
                    metadata=NoteMetadata(
                        subject_id=int(row["subject_id"]),
                        hadm_id=int(row["hadm_id"]),
                        note_id=int(row["note_id"]),
                        careunit=row["careunit"],
                        admit_diagnosis=row.get("admit_diagnosis"),
                        insurance=row.get("insurance"),
                        admission_type=row.get("admission_type"),
                    ),
                    observations=obs,
                    labs=labs_lst,
                    active_problems=problems,
                    medications=meds,
                    procedures=procs,
                    ground_truth=row["text"],
                    source_facts=_build_source_facts(obs, labs_lst, meds, procs, problems),
                    hospital_course=row.get("hospital_course") or None,
                )
                serialised = record.model_dump(mode="json")
            except (_ValidationError, Exception) as exc:
                _log.debug(
                    "Pydantic rejection | hadm_id=%s note_id=%s | %s",
                    row.get("hadm_id"), row.get("note_id"), exc,
                )
                rejected += 1
                continue

            fh.write(_json.dumps(serialised, ensure_ascii=False, default=str) + "\n")
            accepted += 1
            oor_chart += sum(1 for o in obs      if o.out_of_range)
            oor_lab   += sum(1 for l in labs_lst if l.out_of_range)
            if meds:     n_meds  += 1
            if procs:    n_procs += 1
            if problems: n_probs += 1
            if record.metadata.admit_diagnosis:
                n_adx += 1

    _log.info(
        "  batch %d/%d done | accepted this batch=%d",
        batch_idx + 1, n_batches, accepted,
    )
    with open(result_path, "w") as _rf:
        _json.dump(dict(
            accepted=accepted, rejected=rejected,
            pre=pre, post=post,
            oor_chart=oor_chart, oor_lab=oor_lab,
            n_meds=n_meds, n_procs=n_procs, n_probs=n_probs, n_adx=n_adx,
        ), _rf)


# ---------------------------------------------------------------------------
# Pass 2 — process_batches
# ---------------------------------------------------------------------------


def process_batches(
    batch_size: int = 50,
    resume: bool = False,
    sample: Optional[int] = None,
) -> None:
    """Process Parquet intermediaries into a validated FHIR-lite JSONL file.

    Reads exclusively from ``data/intermediate/`` Parquet files.  Parquet
    column-predicate pushdown ensures only the rows for each batch's
    ``hadm_id`` set are decoded, capping peak memory to a single batch
    (~1 M CHARTEVENTS rows for ``batch_size=500``).

    Item labels and harmonisation maps are tiny CSVs collected once into
    DataFrames and reused across all batches without re-scanning.

    Args:
        batch_size: Number of HADM_IDs to process per batch.
        resume: When True, read existing output JSONL and skip any
            ``hadm_id`` already present in it.
        sample: When set, process only the first ``sample`` admissions
            (useful for smoke tests).
    """
    for p in [COHORT_PARQUET, CHART_PARQUET, LAB_PARQUET, MED_PARQUET,
              PROC_PARQUET, ADMISSIONS_PARQUET, DIAGNOSES_PARQUET, DISCHARGE_PARQUET]:
        if not p.exists():
            raise FileNotFoundError(
                f"Intermediate file missing: {p}\n"
                "Run with --extract first."
            )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect small reference tables once.
    # ------------------------------------------------------------------
    logger.info("Process: loading label / harmonisation / range tables …")
    item_labels_df: pl.DataFrame    = load_item_labels().collect()
    lab_labels_df: pl.DataFrame     = load_lab_item_labels().collect()
    harm_lf: Optional[pl.LazyFrame] = load_item_harmonisation_map()
    rang_lf: Optional[pl.LazyFrame] = load_range_bounds()
    harm_df: Optional[pl.DataFrame] = harm_lf.collect() if harm_lf is not None else None
    rang_df: Optional[pl.DataFrame] = rang_lf.collect() if rang_lf is not None else None

    admissions_df: pl.DataFrame = pl.read_parquet(ADMISSIONS_PARQUET)
    diagnoses_df:  pl.DataFrame = pl.read_parquet(DIAGNOSES_PARQUET)
    discharge_df:  pl.DataFrame = pl.read_parquet(DISCHARGE_PARQUET)

    # ------------------------------------------------------------------
    # Build ordered hadm_id work list.
    # ------------------------------------------------------------------
    all_hadm_ids: List[int] = (
        pl.scan_parquet(COHORT_PARQUET)
        .select(pl.col("hadm_id"))
        .unique()
        .collect()["hadm_id"]
        .to_list()
    )
    all_hadm_ids.sort()   # deterministic ordering

    if sample is not None:
        all_hadm_ids = all_hadm_ids[:sample]
        logger.info("Process: --sample %d → limiting to first %d admissions.", sample, len(all_hadm_ids))

    # ------------------------------------------------------------------
    # Resume: skip hadm_ids already written.
    # ------------------------------------------------------------------
    done_hadm_ids: Set[int] = set()
    if resume and OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        with open(OUTPUT_PATH, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done_hadm_ids.add(json.loads(line)["metadata"]["hadm_id"])
                except Exception:
                    pass
        remaining = [h for h in all_hadm_ids if h not in done_hadm_ids]
        logger.info(
            "Process: --resume | %d admissions already done, %d remaining.",
            len(done_hadm_ids), len(remaining),
        )
        all_hadm_ids = remaining
    elif OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0 and not resume:
        logger.warning(
            "Output file %s already exists and --resume was not set. "
            "It will be overwritten.", OUTPUT_PATH
        )

    if not all_hadm_ids:
        logger.info("Process: nothing to do — all admissions already processed.")
        return

    n_batches = max(1, (len(all_hadm_ids) + batch_size - 1) // batch_size)
    logger.info(
        "Process: %d admissions → %d batches of ≤%d",
        len(all_hadm_ids), n_batches, batch_size,
    )

    # ------------------------------------------------------------------
    total_pre    = 0
    total_post   = 0
    accepted     = 0
    rejected     = 0
    oor_chart    = 0
    oor_lab      = 0
    n_with_meds  = 0
    n_with_procs = 0
    n_with_probs = 0
    n_with_adx   = 0

    # Truncate / touch output file once before the subprocess loop.
    open_mode = "a" if resume else "w"
    with open(OUTPUT_PATH, open_mode, encoding="utf-8"):
        pass  # create or truncate; batches will append

    ctx = multiprocessing.get_context("spawn")

    # Run each batch in a fresh subprocess so the OS reclaims all
    # jemalloc/mimalloc allocator pages when it exits.
    # Results are communicated via a temp JSON file (no IPC semaphores).
    import tempfile
    result_file = Path(tempfile.mktemp(suffix=".json"))

    for batch_idx in range(n_batches):
        batch_hadm = all_hadm_ids[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        p = ctx.Process(
            target=_batch_worker,
            args=(
                result_file, batch_hadm, batch_idx, n_batches, OUTPUT_PATH,
                item_labels_df, lab_labels_df, harm_df, rang_df,
                admissions_df, diagnoses_df, discharge_df,
            ),
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"Batch {batch_idx + 1} subprocess exited with code {p.exitcode}"
            )
        with open(result_file) as rf:
            result = json.load(rf)

        accepted     += result["accepted"]
        rejected     += result["rejected"]
        total_pre    += result["pre"]
        total_post   += result["post"]
        oor_chart    += result["oor_chart"]
        oor_lab      += result["oor_lab"]
        n_with_meds  += result["n_meds"]
        n_with_procs += result["n_procs"]
        n_with_probs += result["n_probs"]
        n_with_adx   += result["n_adx"]

        logger.info(
            "  batch %d/%d done | accepted this batch=%d | running total=%d",
            batch_idx + 1, n_batches, result["accepted"], accepted,
        )

    result_file.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------------
    logger.info(
        "Density gate (obs+labs+meds >= %d): before=%d | after=%d | discarded=%d",
        _MIN_DENSITY, total_pre, total_post, total_pre - total_post,
    )
    logger.info(
        "Pydantic validation complete: accepted=%d | rejected=%d", accepted, rejected,
    )
    logger.info(
        "Out-of-range entries flagged: chart=%d | lab=%d | total=%d",
        oor_chart, oor_lab, oor_chart + oor_lab,
    )
    logger.info(
        "Source coverage | meds=%d | procs=%d | active_problems=%d | admit_dx=%d",
        n_with_meds, n_with_procs, n_with_probs, n_with_adx,
    )
    logger.info("Process complete. %d records written to %s", accepted, OUTPUT_PATH)


# ---------------------------------------------------------------------------
# Pass 3 — split_dataset
# ---------------------------------------------------------------------------


def split_dataset(
    input_path: Path,
    output_dir: Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> None:
    """Partition a FHIR-lite JSONL dataset into patient-level train/val/test splits.

    Splitting is performed at the ``subject_id`` level to prevent a patient's
    notes from appearing in more than one partition.

    Two-pass streaming — no full file held in RAM simultaneously.

    Args:
        input_path: Path to the source FHIR-lite JSONL file.
        output_dir: Directory to write split files.
        train_frac: Fraction of patients assigned to training split.
        val_frac: Fraction of patients assigned to validation split.
        seed: Random seed for reproducible patient shuffling.

    Raises:
        ValueError: If ``train_frac + val_frac >= 1.0``.
        FileNotFoundError: If ``input_path`` does not exist.
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac ({train_frac}) + val_frac ({val_frac}) must be < 1.0"
        )

    jsonl_path = input_path.with_suffix(".jsonl") if input_path.suffix == ".json" else input_path
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input file not found: {jsonl_path}")

    test_frac = 1.0 - train_frac - val_frac
    logger.info("Split: loading %s (seed=%d) …", jsonl_path, seed)

    # Pass 1: build subject_id index without keeping records in RAM.
    patient_to_lines: Dict[int, List[int]] = {}
    with open(jsonl_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh):
            sid = json.loads(line)["metadata"]["subject_id"]
            patient_to_lines.setdefault(sid, []).append(lineno)

    subject_ids = list(patient_to_lines.keys())
    rng = random.Random(seed)
    rng.shuffle(subject_ids)

    n_patients = len(subject_ids)
    train_end  = int(n_patients * train_frac)
    val_end    = train_end + int(n_patients * val_frac)

    split_subjects: Dict[str, Set[int]] = {
        "train": set(subject_ids[:train_end]),
        "val":   set(subject_ids[train_end:val_end]),
        "test":  set(subject_ids[val_end:]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass 2: stream records into split files.
    split_counts: Dict[str, int] = {k: 0 for k in split_subjects}
    handles = {
        name: open(output_dir / f"fhir_lite_{name}.jsonl", "w", encoding="utf-8")
        for name in split_subjects
    }
    try:
        with open(jsonl_path, encoding="utf-8") as src:
            for line in src:
                sid = json.loads(line)["metadata"]["subject_id"]
                for name, sids in split_subjects.items():
                    if sid in sids:
                        handles[name].write(line)
                        split_counts[name] += 1
                        break
    finally:
        for fh in handles.values():
            fh.close()

    for name, count in split_counts.items():
        n_sids   = len(split_subjects[name])
        out_path = output_dir / f"fhir_lite_{name}.jsonl"
        logger.info(
            "Split %-5s | patients=%d (%.0f%%) | notes=%d → %s",
            name, n_sids, 100.0 * n_sids / n_patients, count, out_path,
        )
    logger.info(
        "Split complete | total patients=%d | train=%.0f%% val=%.0f%% test=%.0f%%",
        n_patients, 100.0 * train_frac, 100.0 * val_frac, 100.0 * test_frac,
    )


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _fmt_size(path: Path) -> str:
    """Return a human-readable file size string."""
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1 ETL: MIMIC-III → FHIR-lite JSONL (MICU nursing cohort)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full end-to-end run:
  python -m src.etl.pipeline --extract --process --split

  # Separate steps (recommended for large datasets):
  python -m src.etl.pipeline --extract
  python -m src.etl.pipeline --process --batch-size 500
  python -m src.etl.pipeline --split

  # Smoke test (first 20 admissions):
  python -m src.etl.pipeline --extract --process --sample 20 --split

  # Resume after a crash:
  python -m src.etl.pipeline --process --resume

  # Re-extract from scratch:
  python -m src.etl.pipeline --extract --force-extract
""",
    )

    # --- Pass 1 ---
    parser.add_argument(
        "--extract",
        action="store_true",
        default=False,
        help="Pass 1: scan raw CSV.gz files and write Parquet intermediaries to data/intermediate/.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        default=False,
        help="Re-write Parquet intermediaries even if they already exist (requires --extract).",
    )

    # --- Pass 2 ---
    parser.add_argument(
        "--process",
        action="store_true",
        default=False,
        help="Pass 2: process Parquet intermediaries into data/processed/fhir_lite.jsonl.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        metavar="N",
        help="HADM_IDs per batch during --process (default: 50).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip admissions already present in the output JSONL (requires --process).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Limit --process to the first N admissions (for smoke tests; requires --process).",
    )

    # --- Pass 3 ---
    parser.add_argument(
        "--split",
        action="store_true",
        default=False,
        help="Pass 3: partition fhir_lite.jsonl into patient-level train/val/test splits.",
    )
    parser.add_argument(
        "--split-input",
        type=Path,
        default=OUTPUT_PATH,
        metavar="PATH",
        help="JSONL file to split (default: data/processed/fhir_lite.jsonl).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        metavar="F",
        help="Fraction of patients in training split (default: 0.70).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        metavar="F",
        help="Fraction of patients in validation split (default: 0.15).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed for patient shuffling (default: 42).",
    )

    # --- Utilities ---
    parser.add_argument(
        "--download-resources",
        action="store_true",
        default=False,
        help="Download itemid_to_variable_map.csv and variable_ranges.csv from MIMIC-Extract.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        default=False,
        help="Re-download MIMIC-Extract resources even if they already exist.",
    )

    args = parser.parse_args()

    # Validation
    if args.force_extract and not args.extract:
        parser.error("--force-extract requires --extract")
    if args.resume and not args.process:
        parser.error("--resume requires --process")
    if args.sample is not None and not args.process:
        parser.error("--sample requires --process")
    if not any([args.download_resources, args.extract, args.process, args.split]):
        parser.error("At least one of --download-resources, --extract, --process, --split is required.")

    if args.download_resources:
        download_mimic_extract_resources(force=args.force_download)

    if args.extract:
        extract_intermediaries(force=args.force_extract)

    if args.process:
        process_batches(
            batch_size=args.batch_size,
            resume=args.resume,
            sample=args.sample,
        )

    if args.split:
        split_dataset(
            input_path=args.split_input,
            output_dir=_OUT_DIR,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.split_seed,
        )
