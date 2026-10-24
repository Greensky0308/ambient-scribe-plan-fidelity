# Plan-fidelity analysis code

Reproducible analysis for a clinician-adjudicated audit of plan-level fidelity in
ambient documentation for hepatotoxic medications. The code runs from de-identified
derived tables; raw consultations, audio, and model weights are not included.

## Layout

```
data/derived/     de-identified inputs
src/              analysis and figure scripts
outputs/          generated figures and table summaries (git-ignored)
```

## Environment

Python 3.10 or later.

```
pip install -r requirements.txt
```

## Run order

```
python src/make_element_counts.py    # element counts for the clinician-note comparison
python src/compute_element_f1.py     # element-level F1 and bootstrap intervals
python src/compute_audio_stats.py    # audio recall and GEE odds ratios
python src/make_tables.py            # table-ready CSV summaries
python src/figure1.py
python src/figure2.py
python src/figure3.py
python src/figure4.py
```

Figures are written to `outputs/` as PDF and PNG. Table summaries are written to
`outputs/` as CSV.

## What each script does

- `make_element_counts.py`: aggregates element counts shared with the clinician note,
  model-only, and clinician-only from de-identified presence tables.
- `compute_element_f1.py`: computes per-model precision, recall, F1, and 2,000-resample
  bootstrap intervals for each liver-critical element.
- `compute_audio_stats.py`: computes per-pipeline Wilson intervals for drug-name recall
  and fits a generalized estimating equation for recognition in the stress test.
- `make_tables.py`: builds table-ready summaries for corpus composition, text-layer
  fidelity, audio-layer fidelity, and stress-test recognition.
- `figure1.py`: element-level F1, confidence intervals, encounter completeness, element
  counts, and false-negative/false-positive counts.
- `figure2.py`: audio-layer recall, miss matrix, odds ratios, per-drug misses, case-level
  misses, and paired recall.
- `figure3.py`: attrition, confirmed discrepancies, reviewer agreement, representative
  plan-to-note changes, and reviewer concordance.
- `figure4.py`: stress-test recognition by condition and model, similarity, and
  brand/generic comparisons.

## Notes

- The stress test uses 5 dB SNR noise.
- Element names follow the extraction schema; the zero-count family-history domain is
  omitted from displays where F1 is undefined.
- No credentials are stored in the repository.
- Case identifiers in the derived tables and scripts are study codes already shown in
  the published figures; they contain no patient information.
