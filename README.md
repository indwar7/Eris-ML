# Eris-ML

A collection of self-contained ML challenge packages, each with a generated dataset, held-out answer key, automated grader, and reference solution.

## Packages

- [`accent-g2p/`](accent-g2p/) — The Accent Translator: convert a phonetic transcription from one English accent into another.
- [`prompt-delta-upload/`](prompt-delta-upload/) — The Edit That Moved the Answer: given two prompts differing by one edit, identify what changed and how it shifted the output.
- [`pipeline-attribution-upload/`](pipeline-attribution-upload/) — Recovering OCR Batch Origin from Character Noise: cluster unlabeled text snippets by shared origin from noise patterns alone.
- [`FINAL-experimental-narrative-reconstruction/`](FINAL-experimental-narrative-reconstruction/) — The Order of Discovery: reconstruct the correct experimental sequence of a paper's Results section.

Each package directory contains `problem-description.md`, `dataset/`, `grade.py`, and a reference solution.


## Packages

### accent-g2p
Grapheme-to-phoneme (G2P) conversion challenge across accented speech data — tests a model's ability to map written text to correct phonetic transcription accounting for accent variation.

### pipeline-attribution-upload
Attribution/tracing challenge for multi-step ML pipelines — tests a model's ability to identify which stage or component in a pipeline is responsible for a given output or error.

### prompt-delta-upload
Prompt-sensitivity challenge — tests a model's ability to detect and reason about how small changes ("deltas") in a prompt affect output behavior.

### FINAL-experimental-narrative-reconstruction
Narrative reconstruction challenge — tests a model's ability to reconstruct coherent narrative structure from fragmented, shuffled, or partial text input.

## Structure (per package)
Each package includes:
- **Dataset** — generated input data for the challenge
- **Held-out answer key** — separated from the working dataset, used only for grading
- **Automated grader** — scores model output against the answer key
- **Reference solution** — a baseline solution used for calibration

## Usage
1. Load the dataset from the package folder
2. Run inference/solution against the dataset
3. Score output using the automated grader
4. Compare against the reference solution for calibration

## Contact
Abhaydeep Indwar — abhayindwar7@gmail.com — [linkedin.com/in/abhay-indwar7](https://linkedin.com/in/abhay-indwar7)
