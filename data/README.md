# Data sources

This directory contains de-identified derived tables only. Raw consultations, audio,
model weights, and reviewer worksheets are not distributed here.

## Source corpora

- ACI-Bench: doctor-patient dialogues with clinician notes (Yim et al., Sci Data 2023).
  https://github.com/wyim/aci-bench
- PriMock57: mock primary-care consultations with audio, human transcripts, and
  clinician notes (Korfiatis et al., ACL 2022).
  https://github.com/Babylonhealth/priMock57
- NoteChat: synthetic doctor-patient conversations conditioned on clinical notes.
  https://huggingface.co/datasets/akemiH/NoteChat

## Derived files

- `case_level.csv`: one row per encounter-model pair; dialogue-positive (`d_*`) and
  note-present (`g_*`) indicators for each liver-critical element.
- `audio_case_level.csv`: one row per drug instance; recognition flags for four pipelines.
- `stress_results_fuzzy.csv`, `stress_results_large.csv`: per-stimulus recognition and
  string similarity under clean, fast, and 5 dB SNR noise.
- `element_gold_model.csv`: de-identified per-encounter presence of each element in the
  clinician note and model note.
- `element_counts.csv`: element counts shared with the clinician note, model-only, and
  clinician-only, aggregated per model (generated from `element_gold_model.csv`).
- `reviewer_agreement.csv`, `severity_counts.csv`, `reviewer_concordance.csv`: aggregate
  adjudication counts and reviewer judgments used for displays.
- `plan_examples.csv`: plan-to-note examples shown in the figure.
- `asr_gee_ORs.csv`: odds ratios for non-recognition from the stress test.
- `audio_recall.csv`: per-pipeline recall with Wilson intervals.
- `element_f1.csv`: per-model element-level F1 with bootstrap intervals (generated).
- `table*.csv`, `stress_summary.csv`: table-ready summaries (generated).

Case identifiers are dataset identifiers from public mock corpora; they contain no
patient information.
