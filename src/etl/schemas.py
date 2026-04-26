"""Pydantic v2 schema definitions for the FHIR-lite ETL output.

This module defines the data models used to validate and serialize records
produced by the Phase 1 ETL pipeline before they are written to disk.

All models enforce structural and semantic constraints described in the
study's FHIR-lite JSON specification.
"""

import re
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Regex for ISO 8601 relative time offset, e.g. "T-4h 15m" or "T-0h 0m"
# ---------------------------------------------------------------------------
_TIME_OFFSET_RE = re.compile(r"^T-\d+h \d+m$")

# Type discriminator literals
ValType = Literal["numeric", "categorical"]


class ObservationEntry(BaseModel):
    """A single time-stamped clinical measurement.

    Attributes:
        time: ISO 8601 relative offset from the note's charttime, e.g. "T-2h 10m".
        metric: Human-readable label for the measurement (e.g. "Heart Rate").
        val: Measured value; may be numeric or a string token such as ">200" or "Trace".
        val_type: Discriminator indicating whether ``val`` is a numeric measurement
            (``"numeric"``) or a categorical/textual flag (``"categorical"``).
            Enables downstream models and evaluation code to distinguish
            quantitative physiological signals from qualitative state labels.
        unit: Unit of measure from VALUEUOM (e.g. "bpm", "mmHg", "mg/dL").
            ``None`` when the source record carries no unit information.
    """

    time: str
    metric: str
    val: Union[float, int, str]
    val_type: ValType
    unit: Optional[str] = None
    out_of_range: bool = False

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """Ensure the time offset string conforms to T-{h}h {m}m format.

        Args:
            v: Raw time string from the pipeline.

        Returns:
            The validated string, unchanged.

        Raises:
            ValueError: If the string does not match the expected pattern.
        """
        if not _TIME_OFFSET_RE.match(v):
            raise ValueError(
                f"time offset '{v}' does not match required format 'T-<h>h <m>m'"
            )
        return v

    @field_validator("metric")
    @classmethod
    def validate_metric_non_empty(cls, v: str) -> str:
        """Ensure metric label is a non-empty, stripped string.

        Args:
            v: Raw metric string.

        Returns:
            Stripped metric string.

        Raises:
            ValueError: If the metric is blank after stripping.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("metric must be a non-empty string")
        return stripped


class ActiveProblem(BaseModel):
    """A single active diagnosis from DIAGNOSES_ICD for the admission.

    Derived by joining DIAGNOSES_ICD to D_ICD_DIAGNOSES on ICD9_CODE.
    Provides the structured problem list that grounds the SBAR problem
    header in ``ground_truth`` nursing notes (e.g. "Respiratory failure,
    acute").

    Attributes:
        icd9_code: ICD-9-CM code string (e.g. ``"51881"``).
        description: Human-readable diagnosis label from D_ICD_DIAGNOSES
            (e.g. ``"Acute respiratory failure"``).
    """

    icd9_code: str
    description: str

    @field_validator("icd9_code", "description")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Ensure neither field is blank after stripping.

        Args:
            v: Raw string value.

        Returns:
            Stripped string.

        Raises:
            ValueError: If blank after stripping.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("icd9_code and description must be non-empty strings")
        return stripped


class MedicationEvent(BaseModel):
    """A single medication or fluid administration event within the 12-hour window.

    Sourced from INPUTEVENTS_CV (CareVue) and INPUTEVENTS_MV (MetaVision),
    unified into a common schema.  These events are the primary source for
    the ``Action`` section of SBAR notes (e.g. "Lasix 40mg IVP given",
    "Heparin gtt titrated to 850 units/hr").

    Attributes:
        time: ISO 8601 relative offset from the note's charttime.
        drug: Human-readable item label from D_ITEMS (e.g. "Furosemide (Lasix)").
        amount: Dose or volume administered; ``None`` when not recorded.
        unit: Unit for ``amount`` (e.g. "mg", "mL", "units/hr"); ``None``
            when not recorded.
        route: Administration route from INPUTEVENTS_MV ORDERCATEGORYNAME
            (e.g. "IV Push", "Continuous IV").  Always ``None`` for
            CareVue-sourced events which carry no route field.
    """

    time: str
    drug: str
    amount: Optional[Union[float, str]] = None
    unit: Optional[str] = None
    route: Optional[str] = None

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """Ensure the time offset string conforms to T-{h}h {m}m format.

        Args:
            v: Raw time string from the pipeline.

        Returns:
            The validated string, unchanged.

        Raises:
            ValueError: If the string does not match the expected pattern.
        """
        if not _TIME_OFFSET_RE.match(v):
            raise ValueError(
                f"time offset '{v}' does not match required format 'T-<h>h <m>m'"
            )
        return v

    @field_validator("drug")
    @classmethod
    def validate_drug_non_empty(cls, v: str) -> str:
        """Ensure drug label is a non-empty, stripped string.

        Args:
            v: Raw drug label string.

        Returns:
            Stripped drug label.

        Raises:
            ValueError: If blank after stripping.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("drug must be a non-empty string")
        return stripped


class ProcedureEvent(BaseModel):
    """A single bedside procedure event within the 12-hour window.

    Sourced from PROCEDUREEVENTS_MV (MetaVision only; CareVue has no
    equivalent structured procedure table).  Captures intubation, line
    placement, trach/PEG, dialysis initiation, etc. — events that
    nursing notes consistently reference in the ``Action`` and
    ``Response`` SBAR sections.

    Attributes:
        time: ISO 8601 relative offset from the note's charttime (based on
            PROCEDUREEVENTS_MV.STARTTIME).
        procedure: Human-readable item label from D_ITEMS.
        duration_min: Procedure duration in whole minutes derived from
            PROCEDUREEVENTS_MV STARTTIME / ENDTIME; ``None`` when ENDTIME
            is absent.
        status: Final status string from PROCEDUREEVENTS_MV.STATUSDESCRIPTION
            (e.g. ``"Finalized"``, ``"In Progress"``); ``None`` when absent.
    """

    time: str
    procedure: str
    duration_min: Optional[int] = None
    status: Optional[str] = None

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """Ensure the time offset string conforms to T-{h}h {m}m format.

        Args:
            v: Raw time string from the pipeline.

        Returns:
            The validated string, unchanged.

        Raises:
            ValueError: If the string does not match the expected pattern.
        """
        if not _TIME_OFFSET_RE.match(v):
            raise ValueError(
                f"time offset '{v}' does not match required format 'T-<h>h <m>m'"
            )
        return v

    @field_validator("procedure")
    @classmethod
    def validate_procedure_non_empty(cls, v: str) -> str:
        """Ensure procedure label is a non-empty, stripped string.

        Args:
            v: Raw procedure label string.

        Returns:
            Stripped procedure label.

        Raises:
            ValueError: If blank after stripping.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("procedure must be a non-empty string")
        return stripped


class NoteMetadata(BaseModel):
    """Identifiers linking the FHIR-lite record back to the source MIMIC-III rows.

    Attributes:
        subject_id: MIMIC-III SUBJECT_ID (de-identified patient identifier).
        hadm_id: Hospital admission identifier (HADM_ID).
        note_id: Source row identifier from NOTEEVENTS (ROW_ID).
        careunit: ICU care unit in which the note was written, sourced from
            ICUSTAYS.FIRST_CAREUNIT (e.g. ``"MICU"``).  Preserved for
            post-hoc stratification in Phase 3 without requiring a separate
            join at evaluation time.
        admit_diagnosis: Free-text admission diagnosis string from
            ADMISSIONS.DIAGNOSIS (e.g. ``"RESPIRATORY FAILURE"``).  Provides
            the presenting complaint for SBAR ``Situation`` generation.
            ``None`` when absent from ADMISSIONS.
        insurance: Patient insurance category from ADMISSIONS.INSURANCE
            (e.g. ``"Medicare"``, ``"Private"``).  ``None`` when absent.
        admission_type: Admission type from ADMISSIONS.ADMISSION_TYPE
            (e.g. ``"EMERGENCY"``, ``"ELECTIVE"``).  ``None`` when absent.
    """

    subject_id: int
    hadm_id: int
    note_id: int
    careunit: str
    admit_diagnosis: Optional[str] = None
    insurance: Optional[str] = None
    admission_type: Optional[str] = None


class FHIRLiteRecord(BaseModel):
    """Top-level FHIR-lite record combining metadata, observations, labs, and the source note.

    Observations, labs, medications, and procedures are ordered by recency
    (most recent first) so that downstream models encounter the most
    clinically proximate data at the start of each list, mitigating
    "Lost in the Middle" degradation.

    Attributes:
        metadata: Patient and admission identifiers, including admission
            context fields sourced from ADMISSIONS.
        observations: Chartevents measurements within the 12-hour window,
            sorted most-recent-first.
        labs: Lab results within the 12-hour window, sorted most-recent-first.
        active_problems: Structured diagnosis list for the admission from
            DIAGNOSES_ICD joined to D_ICD_DIAGNOSES.  Grounds the SBAR
            problem header.
        medications: Medication and fluid administration events within the
            12-hour window from INPUTEVENTS_CV and INPUTEVENTS_MV, sorted
            most-recent-first.  Grounds the SBAR ``Action`` section.
        procedures: Bedside procedure events within the 12-hour window from
            PROCEDUREEVENTS_MV, sorted most-recent-first.  Grounds the SBAR
            ``Action`` and ``Response`` sections.
        ground_truth: Full text of the original nursing note, preserved for
            Phase 3 evaluation.
        source_facts: Flat list of natural-language grounding statements
            derived from all structured sources (observations, labs,
            medications, procedures, active problems), e.g.
            ``"Heart Rate was 88 bpm at T-2h 10m"``.  Used by the Phase 3
            hallucination audit to distinguish intrinsic errors (model
            misrepresents a fact present in source_facts) from extrinsic
            errors (model fabricates a fact absent from source_facts).
        hospital_course: Extracted ``HOSPITAL COURSE`` / ``BRIEF HOSPITAL
            COURSE`` section from the discharge summary for this admission
            (``NOTEEVENTS`` category ``"Discharge summary"``).  Used as the
            reference target for Sub-task 3 (admission summarisation).
            ``None`` when no discharge summary is available for the admission.
    """

    metadata: NoteMetadata
    observations: List[ObservationEntry]
    labs: List[ObservationEntry]
    active_problems: List[ActiveProblem] = []
    medications: List[MedicationEvent] = []
    procedures: List[ProcedureEvent] = []
    ground_truth: str
    source_facts: List[str]
    hospital_course: Optional[str] = None

    @field_validator("ground_truth")
    @classmethod
    def validate_ground_truth_non_empty(cls, v: str) -> str:
        """Ensure the nursing note text is non-empty.

        Args:
            v: Raw note text.

        Returns:
            The original string if valid.

        Raises:
            ValueError: If the string is blank after stripping.
        """
        if not v.strip():
            raise ValueError("ground_truth must be a non-empty string")
        return v

    @model_validator(mode="after")
    def validate_density_gate(self) -> "FHIRLiteRecord":
        """Enforce the combined observation density hard gate.

        Rejects records whose total observation + lab + medication entry
        count falls below 5.  Medications are included in the gate because
        INPUTEVENTS is an equally strong signal source for notes that have
        fewer charted vitals (e.g. post-extubation floor-level monitoring).

        Returns:
            The validated FHIRLiteRecord instance.

        Raises:
            ValueError: If combined obs + labs + medications < 5.
        """
        total = len(self.observations) + len(self.labs) + len(self.medications)
        if total < 5:
            raise ValueError(
                f"density gate failed: combined observations + labs + medications"
                f" = {total} (minimum 5 required)"
            )
        return self
