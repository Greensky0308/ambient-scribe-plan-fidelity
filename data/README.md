# Data sources and derived tables

## Public corpora

- **ACI-Bench** — Yim et al., Scientific Data 2023.
  https://github.com/wyim/aci-bench
  207 clinician-authored encounters; the release also contains transcript variants of the
  same encounters.
- **PriMock57** — Korfiatis et al., ACL 2022.
  https://github.com/Babylonhealth/priMock57
  57 mock primary-care consultations with audio, transcripts, and clinician notes.
- **NoteChat** — synthetic external corpus.
  https://huggingface.co/datasets/akemiH/NoteChat

No raw dialogues, audio, model weights, or clinician worksheets are distributed here.
Where the source corpora are public, download them from the links above.

## Derived tables

| File | Contents |
|---|---|
| `case_level.csv` | Per-encounter text-layer capture counts |
| `element_counts.csv` | Element-level positive / captured / omitted / over-documented counts |
| `element_f1.csv` | Element-level F1 by model |
| `element_gold_model.csv` | Per-element gold and model counts |
| `audio_case_level.csv` | Per-consultation audio-layer drug instances and misses |
| `audio_recall.csv` | Audio-layer recall by pipeline |
| `asr_gee_ORs.csv` | GEE odds ratios for the acoustic stress test |
| `stress_results_fuzzy.csv` | Stress-test recognition under string-similarity scoring |
| `stress_results_large.csv` | Stress-test recognition for the large ASR model |
| `plan_flow.csv` | Encounter flow and screening counts (264 independent encounters) |
| `severity_counts.csv` | Confirmed item counts by severity |
| `severity_by_nature.csv` | Confirmed high/medium items by nature and severity |
| `accuracy.csv` | Sensitivity, specificity, PPV, NPV with Wilson intervals |
| `reviewer_agreement.csv` | A/B agreement matrices and Cohen kappa by set |

The 57 ACI-Bench transcript variants are retained as a separate transcription set; the
plan-fidelity audit uses the 264 independent encounters (207 ACI-Bench + 57 PriMock57).
