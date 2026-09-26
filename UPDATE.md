# Update — EEG model layer and live analysis

**Branch:** `Phumi_Update` (from `farrel-update/eeg-pipeline`)

This adds the missing piece between acquisition and control: a model layer that
turns preprocessed EEG into one of four motor-imagery classes, and a browser UI
for watching it work.

---

## Run it

Everything needed is in the repository — the two BCI IV 2a recordings are already
tracked, so there is nothing to download.

```bash
git clone -b Phumi_Update https://github.com/BEARUBC/nerv-bionics.git
cd nerv-bionics
python -m pip install -r requirements.txt

./run_ui.sh
```

Then open:

| URL | What it is |
|---|---|
| <http://127.0.0.1:8000> | Test bench — pick a recording, class, window and model, press Run |
| <http://127.0.0.1:8000/live> | Live stream — trials play past the detector in real time |
| <http://127.0.0.1:8000/identify> | Pick any signal from a list and watch the model identify it |

No model file needs to be trained first: each page trains what it needs on the
spot and caches it. Ctrl+C in the terminal stops the server.

> `run_ui.sh` exists because a machine can easily have several Python
> installations with only some carrying numpy and scikit-learn. Running the
> server with the wrong one dies during import, nothing binds a port, and the
> browser reports ERR_CONNECTION_REFUSED with the real error sitting in the
> terminal. The script finds an interpreter that works. Every entry point also
> checks its own interpreter and names a working one instead of raising a bare
> `ModuleNotFoundError`.

Command-line use, if you prefer no browser:

```bash
python -m classification.train  --subjects A01T --tune-window            # train and save
python -m classification.detect --model models/a01t.joblib --subject A04T  # run detection
python -m classification.benchmark                                        # score every model
python -m pytest tests/ -v                                                # 21 checks
```

---

## What was chosen, and why

Eight pipelines were benchmarked. Three were selected:

| Rank | Model | Accuracy | Calibrate | Why |
|---|---|---|---|---|
| 1 | Riemannian TS + LR | **0.691** | 0.19 s | Best accuracy and κ of everything tested; nothing to tune |
| 2 | Riemannian MDM | 0.614 | **0.18 s** | Best cross-subject transfer (0.441); fitting is four averages |
| 3 | CSP + LDA | 0.646 | 1.05 s | The reference every BCI paper reports; within two points of rank 1 |

Chance is 0.250 across four classes. `CSP + SVM` and `EEGNet` stay in the
registry but were not selected.

### The biggest finding was not a model

The existing notebook benchmark used pre-cut **0.8 s** epochs. Motor imagery
ERD/ERS builds over roughly 0.5–3 s after the cue, so every model there was
reading the wrong slice of time. Re-cutting the same trials from the raw `.mat`
files at **0.5–3.5 s**, with no change to any model, roughly doubles accuracy:

| Model | 0.8 s window | 0.5–3.5 s window |
|---|---|---|
| Riemannian TS + LR | 0.389 | **0.664** |
| Riemannian MDM | 0.377 | **0.614** |
| CSP + LDA | 0.364 | **0.646** |

### EEGNet came last, and that is a data problem

EEGNet was expected to take rank 2 on cross-subject transfer. It finished last of
five: 0.543 accuracy, 117 s to calibrate, and it lost the cross-subject axis to
Riemannian MDM (0.361 vs 0.441) — the one axis it was picked for.

Read that as a **data-volume result, not a verdict on the architecture**. Each
fold trained on roughly 214 trials from one subject, far below what a CNN needs.
The code stays in place; re-run the benchmark once the other seven BCI IV 2a
subjects are added before concluding anything.

### Two tuning wins, already applied

- **Regularisation.** The tangent space of a 22×22 covariance has 253 dimensions
  and a subject gives a few hundred trials, so scikit-learn's default `C=1.0`
  overfits. `C=0.01` lifts the mean from 0.664 to 0.691 — and almost all of the
  gain lands on the harder subject (0.504 → 0.565), which is the right trade.
- **Per-subject epoch window.** A01T peaks at 0.5–3.5 s, A04T at 1.0–4.0 s.
  `--tune-window` searches per user and is worth about 3 points on A04T.

---

## The live view's guarantee

`/live` and `/identify` show you the true label while the model guesses. That
only means something if the label cannot reach the classifier, so two things
enforce it:

1. **The model never trains on what you watch.** `LiveSession` splits the
   recording, fits the detector on 70% and streams only the held-out 30%.
2. **The label is not on the model's path.** `LiveSession._frames_for()` takes
   the signal as its only argument — the label is not even a parameter. It is
   attached to the JSON response after the probabilities exist, purely for
   display.

Both are covered by tests in `tests/test_live.py`, including a signature check
that fails if anyone later passes the label in "just for logging".

You can confirm it from outside the code:

```bash
curl -s -X POST http://127.0.0.1:8000/api/live/trial \
  -H 'Content-Type: application/json' -d '{"subject":"A01T","index":0}' |
  python -c "import json,sys; d=json.load(sys.stdin); \
    print('truth at top level:', 'truth' in d); \
    print('truth inside any frame:', any('truth' in f for f in d['frames']))"
```

---

## Numbers worth trusting, and one worth ignoring

| What you run | Result | What it means |
|---|---|---|
| `train --subjects A01T` (held out) | **85.5%** | Reading a person the model is calibrated on |
| `/identify` or `/live` on A01T | **~73–78%** | Same, through the streaming path, on unseen trials |
| `detect --subject A01T` with the A01T model | 100% | **Nothing** — those trials were its training data |
| `detect --subject A04T` with the A01T model | **33.2%** | Reading a stranger: above the 25% chance level, not usable |

The tools warn you when you ask for the third one.

Cross-subject is worse than the single number suggests. Per class, trained on
A01T and tested on A04T: `right` 86.6%, `left` 32.3%, `foot` 13.4%,
`tongue` 0.0%. The model has not degraded evenly — it has collapsed toward
`right` on an unfamiliar brain, which is a domain-shift problem rather than a
weak-classifier one.

---

## What matters most next

Measured, not guessed:

1. **Per-user calibration beats every algorithmic fix.** Eight of a new user's own
   trials — about one minute of recording — takes them from 33% to 56%. Twenty-four
   trials reaches 67%. Nothing tried on the zero-calibration case came close.
2. **Do not pool subjects.** Adding another person's trials to a target user's own
   data *hurts* once the target has 24 trials or more (0.798 alone vs 0.692
   pooled at 160 trials). Concatenation is not transfer learning.
3. **Add the other seven subjects.** Every cross-subject number here rests on
   training with exactly one person.
4. **Finish `signal_analysis/ica.py`.** Still a skeleton; eye-blink removal would
   also let the 100 µV rejection threshold be loosened.

Things already tested and **not** worth your time: the frequency band (8–30 Hz is
already optimal), the covariance estimator (OAS already best), and a soft-vote
ensemble (0.672 vs 0.664 for triple the compute). Riemannian re-centring fixes
the cross-subject *collapse* — `tongue` recall goes 0% → 26% — but moves overall
accuracy only 33.2% → 35.5%.

---

## What is in this change

```
classification/          the model layer
├── data.py              cut labelled epochs from raw .mat; NaN interpolation; artifact flags
├── preprocessing.py     one chain shared by training and inference
├── models.py            the three selected models plus baselines, behind build_model(name)
├── detector.py          train, save, load, detect; carries its own preprocessing settings
├── eegnet.py            EEGNet as a scikit-learn estimator, torch imported lazily
├── realtime.py          rolling-buffer online decoder with confidence gating
├── live.py              live session: trains on one split, streams the other
├── benchmark.py         measures six axes, prints the scorecard, writes a CSV
├── train.py             CLI: train, report honestly, save to models/
├── detect.py            CLI: run a saved detector over a recording
├── server.py            the web UI: test bench, live stream, identify
├── app.py / ui.py       matplotlib equivalents, for use without a browser
├── live_page.py         HTML for the live stream
├── identify_page.py     HTML for the signal picker
└── README.md            full detail: selection, scores, how to improve accuracy

tests/                   21 checks that run against the real recordings
run_ui.sh                launcher that finds a Python with the dependencies
requirements.txt         added — the root README referenced it but it did not exist
```

Trained models (`models/*.joblib`) and benchmark output are build artifacts and
are gitignored; regenerate them with `classification.train` and
`classification.benchmark`.

Full detail, including the scoring method and every measurement, is in
[classification/README.md](classification/README.md).
