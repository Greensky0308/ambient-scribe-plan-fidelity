# Plan-fidelity audit of ambient AI scribe notes

Analysis code for a study of plan-level errors in notes generated from doctor-patient
conversations. The scripts produce Figures 1-4 and Tables 2-3 from aggregate derived
tables in `data/derived/`.

## Data

The study uses three public corpora:

- **ACI-Bench**: 207 clinician-authored encounters. The public release also contains
  transcript variants of the same encounters; 57 additional dialogue texts are retained
  as a separate transcription set.
- **PriMock57**: 57 mock primary-care consultations with audio, human transcripts, and
  clinician notes.
- **NoteChat**: synthetic external text corpus used for a drug-name check (100 encounters).

Only aggregate derived tables are included. Raw dialogues, audio, model weights, and
clinician adjudication worksheets are not distributed. Source links are in `data/README.md`.

## Analysis layers

- **Text layer**: 321 dialogue texts from 264 independent encounters (207 ACI-Bench +
  57 transcript variants + 57 PriMock57), two note-generation models.
- **Plan-fidelity audit**: restricted to the 264 independent encounters.
- **Audio layer**: 57 PriMock57 consultations, 29 with at least one hepatotoxic medication.

## Layout

```
src/            analysis and figure scripts
data/derived/   aggregate tables used by the scripts
outputs/        generated figures and tables (not versioned)
```

## Environment

Python 3.11+.

```
pip install -r requirements.txt
```

## Run

From the repository root:

```
python src/figure1.py
python src/figure2.py
python src/figure3.py
python src/figure4.py
```

`figure3.py` reads `plan_flow.csv`, `severity_by_nature.csv`, `accuracy.csv`, and
`reviewer_agreement.csv`; it contains no result values.

Table scripts:

```
python src/make_tables.py
```

The element- and audio-level summary scripts (`compute_element_f1.py`,
`compute_audio_stats.py`, `make_element_counts.py`) expect the public corpora and are
included for provenance; the derived tables they produce are already in `data/derived/`.

## Outputs

`outputs/figure1.pdf`, `figure2.pdf`, `figure3.pdf`, `figure4.pdf` and their PNG versions.
`outputs/` is excluded from version control; run the scripts to recreate the figures.

## License

MIT. See `LICENSE`.
