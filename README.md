# Eris-ML

A collection of self-contained ML challenge packages, each with a generated dataset, held-out answer key, automated grader, and reference solution.

## Packages

- [`accent-g2p/`](accent-g2p/) — The Accent Translator: convert a phonetic transcription from one English accent into another.
- [`prompt-delta-upload/`](prompt-delta-upload/) — The Edit That Moved the Answer: given two prompts differing by one edit, identify what changed and how it shifted the output.
- [`pipeline-attribution-upload/`](pipeline-attribution-upload/) — Recovering OCR Batch Origin from Character Noise: cluster unlabeled text snippets by shared origin from noise patterns alone.
- [`FINAL-experimental-narrative-reconstruction/`](FINAL-experimental-narrative-reconstruction/) — The Order of Discovery: reconstruct the correct experimental sequence of a paper's Results section.

Each package directory contains `problem-description.md`, `dataset/`, `grade.py`, and a reference solution.
