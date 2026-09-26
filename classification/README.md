# Model selection for EEG motor-imagery recognition

This package holds the model layer of the NERV pipeline: the classifiers that turn
preprocessed EEG epochs into one of four motor-imagery classes (`left`, `right`,
`foot`, `tongue`).

Eight candidate pipelines were benchmarked before picking three. This document
records what was measured, which three won, and why.

---

## 1. What was compared

The starting point was the eight-model benchmark in
[`trydataset2a/main.ipynb`](../trydataset2a/main.ipynb): CSP+LDA, CSP+SVM, a
voting ensemble, Riemannian MDM, Riemannian tangent space + LR, FBCSP+LR, XGBoost
on CSP features, and EEGNet.

That notebook ran on pre-cut **0.8 s** epochs, which is the wrong window for motor
imagery: ERD/ERS builds over roughly 0.5–3 s after the cue. Every model there
landed between 0.35 and 0.39 accuracy against a 0.25 chance level.

This package re-runs the survivors on a **0.5–3.5 s** window cut straight from the
raw `.mat` recordings in `acquisition/data/`. Same models, same protocol, correct
window — and accuracy roughly doubles. The window mattered far more than the model
choice.

The re-run also changed the shortlist. EEGNet was expected to take rank 2 on
cross-subject transfer and instead finished last, so the three selected models are
now Riemannian TS + LR, Riemannian MDM, and CSP + LDA.

## 2. Axes that were scored

Accuracy alone does not pick a model for a prosthetic that has to respond in real
time on a wearable computer, so `benchmark.py` measures six axes:

| Axis | What it means | Why it matters here |
|---|---|---|
| `acc` | Mean accuracy, 5-fold CV within each subject | The calibrated case: the user trains the decoder on their own data first |
| `kappa` | Cohen's κ | Chance-corrected, so 4-class chance maps to 0 rather than 0.25 |
| `macro_f1` | Macro-averaged F1 | Catches a model that quietly ignores one class |
| `fit_s` | Seconds to calibrate one subject | Calibration happens at the start of every session; a minute is the practical limit |
| `infer_ms` | Milliseconds to classify one window | 50 ms supports a 20 Hz control loop, faster than a hand can act |
| `loso_acc` | Leave-one-subject-out accuracy | The zero-calibration case: can a new user just put the headset on? |

Scores are normalised to 0–100. Quality axes run from a fixed floor (chance level
for accuracy, zero for κ and F1) up to the best model measured, so a model a few
points behind the leader scores a few points behind it. Cost axes are scored
against a fixed budget rather than against each other, because what matters is
whether a model fits the device's timing budget, not whether it beat a rival by a
microsecond.

## 3. The three selected models

### Rank 1 — Riemannian tangent space + logistic regression (`riemann_ts_lr`)

Each trial becomes a 22×22 covariance matrix. Those live on a curved manifold of
symmetric positive-definite matrices, so the pipeline projects them into the
tangent space at the geometric mean of the training set, which flattens the
manifold, and then fits multinomial logistic regression.

- **Best accuracy and best κ** of every pipeline tested, in both this run and the
  original notebook.
- Uses the **full covariance**. CSP throws away everything outside its six spatial
  filters; this keeps all 253 unique covariance entries.
- Calibrates in a fifth of a second and classifies a window in well under a
  millisecond.
- No hyperparameters worth tuning, which matters when every user needs their own
  calibration.

### Rank 2 — Riemannian MDM (`riemann_mdm`)

Minimum distance to mean: compute one geometric mean covariance per class, then
assign each trial to the nearest class mean under the affine-invariant metric.

- **Cheapest model to calibrate** — fitting is four averages, with no decision
  boundary to optimise. Attractive for recalibrating mid-session as electrodes
  drift.
- **Best cross-subject accuracy of anything tested** (0.441), so it degrades most
  gracefully when the training data is not the current user's. That includes
  beating EEGNet, which was the candidate picked specifically for this axis.
- **Shares its covariance front end with rank 1**, so supporting both costs almost
  no extra code and they can run side by side as a cross-check.
- Gives up five points of within-subject accuracy for that.

### Rank 3 — CSP + LDA (`csp_lda`)

Common Spatial Patterns finds a small set of electrode mixes that maximally
separate the classes, and LDA classifies their log-variances. This is the
reference pipeline in the BCI literature.

- Lands **within two points of rank 1** on within-subject accuracy, which is worth
  knowing before adopting anything more complicated.
- **Fastest inference** of all five (0.08 ms).
- Costs roughly **6× the calibration time** of either Riemannian model, and
  transfers worst but one across subjects (0.323).
- Keeping it means results stay directly comparable to published work.

### What was measured and set aside

| Model | Verdict |
|---|---|
| EEGNet | **Last on the composite score** (65.4). Lowest accuracy and κ, 117 s to calibrate, and it lost the cross-subject axis to Riemannian MDM. A data-volume result — see the note in section 4 — not a judgement on the architecture. Stays in the registry. |
| CSP + SVM | Tracks CSP + LDA within noise but adds a kernel and two hyperparameters to tune per user. |
| FBCSP + LR | Slowest model in the notebook by two orders of magnitude (1251 s) for accuracy no better than CSP + LDA. The cost/benefit is indefensible. |
| XGBoost on CSP | Lowest accuracy and κ of all eight in the notebook. Tree ensembles add nothing on six log-variance features. |
| Voting ensemble | Roughly matched plain CSP + LDA while costing three models' worth of fit time and complexity. |

## 4. Measured results

Two subjects (`A01T`, `A04T`) are present in `acquisition/data/`, 0.5–3.5 s window,
5-fold CV within subject, 4 classes, chance = 0.250.

| Model | Accuracy | Cohen's κ | Macro F1 | Calibrate | Latency | Cross-subject | Composite |
|---|---|---|---|---|---|---|---|
| **Riemannian TS + LR** | **0.691 ± 0.126** | **0.588** | **0.693** | 0.19 s | 0.44 ms | 0.416 | **97.9** |
| **Riemannian MDM** | 0.614 ± 0.133 | 0.486 | 0.615 | **0.18 s** | 0.60 ms | **0.441** | 89.2 |
| **CSP + LDA** | 0.646 ± 0.135 | 0.528 | 0.647 | 1.05 s | **0.08 ms** | 0.323 | 84.2 |
| CSP + SVM | 0.644 ± 0.133 | 0.526 | 0.648 | 1.02 s | 0.14 ms | 0.293 | 81.7 |
| EEGNet | 0.543 ± 0.135 | 0.390 | 0.536 | 117.2 s | 1.51 ms | 0.361 | 65.4 |

EEGNet's row predates the regularisation change in section 11; the others are
current.

Per subject, best model:

| Subject | Accuracy | Cohen's κ |
|---|---|---|
| A01T | 0.817 | 0.756 |
| A04T | 0.565 | 0.420 |

### EEGNet did not make the shortlist

It was picked as a candidate for rank 2 on the strength of its cross-subject
transfer, and the measurement did not support that. It came last on the composite
score: lowest accuracy (0.543), lowest κ (0.390), 117 s to calibrate against 0.21 s
for the winner, and even on the cross-subject axis it lost to Riemannian MDM
(0.361 vs 0.441) — the one axis it was supposed to own.

Read this as a **data-volume result, not a verdict on the architecture**. Each fold
trained on roughly 214 trials from a single subject. That is far below what a CNN
needs, and it is the regime where the Riemannian methods' strong built-in
assumptions pay off most. EEGNet stays in the registry; re-run the benchmark once
the other seven BCI IV 2a subjects are present before drawing a conclusion.

### The window mattered more than the model

Same models, same protocol, only the epoch window changed:

| Model | 0.8 s window (old notebook) | 0.5–3.5 s window |
|---|---|---|
| Riemannian TS + LR | 0.389 | **0.664** |
| Riemannian MDM | 0.377 | **0.614** |
| CSP + LDA | 0.364 | **0.646** |

Against a chance level of 0.250 that is roughly a threefold increase in
chance-corrected skill, from re-cutting the epochs rather than from any change to
the models.

## 5. Using the winner: train, save, detect

Two commands. The first trains the winning pipeline on your recordings and writes
it to `models/`; the second loads it and runs detection.

### Step 1 — train

```bash
python -m classification.train --subjects A01T
```

It holds back 25% of the trials, trains on the rest, and prints an honest score on
the held-back part before saving:

```
  HELD-OUT RESULT  (trials the model never saw during training)
  Accuracy : 85.5%   (random guessing would be 25%)
  Kappa    : 0.807   (0 = guessing, 1 = perfect)

  Confusion matrix   (rows = actual, columns = predicted)
                foot    left   right  tongue
  foot            12       0       0       5
  left             1      14       3       0
  right            0       0      17       0
  tongue           1       0       0      16
```

The confusion matrix is the useful part. Here `foot` is the weak class — 5 of its
17 trials were called `tongue` — while `right` was never missed. That tells you
where to spend effort far better than a single accuracy number does.

After reporting, it retrains on all the trials and saves to
`models/a01t.joblib`. The saved file carries the sampling rate, window length and
filter settings alongside the weights, so it cannot be fed the wrong kind of input
by accident.

### Step 2 — detect

```bash
python -m classification.detect --model models/a01t.joblib --subject A04T
```

One line per trial:

```
   #  actual   detected   conf                    result
   1  tongue   tongue      94%  ###############-  ok
   2  left     left        88%  ##############--  ok
   3  foot     tongue      41%  #######---------  MISS
```

Useful flags:

| Flag | Effect |
|---|---|
| `--limit 20` | Only the first 20 trials, for a quick look |
| `--delay 0.3` | Pause between lines so it scrolls like a live session |
| `--min-confidence 0.6` | Decline to answer below 60% instead of guessing |
| `--mode stream` | Replay sample by sample through the rolling-buffer decoder |

### Reading the numbers honestly

Running the A01T detector back over A01T gives **100%**, and that number is
worthless — those trials were in its training set. The command prints a warning
when you do this. The three numbers that mean something:

| What you run | Result | What it tells you |
|---|---|---|
| `train --subjects A01T` (held-out) | **85.5%** | How well it reads a person it has been calibrated on |
| `detect --subject A01T` (trained on A01T) | 100% | Nothing — it memorised these trials |
| `detect --subject A04T` (trained on A01T) | **33.2%** | How well it reads a stranger: above the 25% chance level, but far from usable |

That last row is the honest state of cross-subject decoding, and it is why
per-user calibration is currently the assumption.

### Streaming mode

```bash
python -m classification.detect --model models/a01t.joblib --subject A01T --mode stream
```

This runs the real-time path: samples go into a rolling buffer one at a time, and
every 0.5 s the decoder classifies the most recent 3-second window and averages
the last three results.

Stream mode deliberately reads 2 seconds further past the cue than the detector's
window, so the window has room to slide — otherwise the buffer only fills on the
very last sample and you get exactly one decision. You should see about 5
decisions per trial at ~2 ms each:

```
   #  actual   final call    conf  decisions  result
   1  tongue   tongue        45%          5  ok
   2  foot     foot          99%          5  ok
   5  left     tongue        68%          5  MISS
```

Confidence drops compared to the cue-aligned mode because the later windows run
past the imagery period into the rest phase. That is the honest picture of live
decoding, and it is the argument for `--min-confidence`.

### From Python

```python
from classification.detector import MotorImageryDetector

detector = MotorImageryDetector()
detector.fit_from_subjects(['A01T'])
detector.save('models/a01t.joblib')

# later, anywhere
detector = MotorImageryDetector.load('models/a01t.joblib')
result = detector.detect(trial)              # trial: (22, 750)
print(result.label, result.confidence)       # 'left' 0.88
print(result.probabilities)                  # {'foot': 0.04, 'left': 0.88, ...}
```

Against a live BrainFlow board:

```python
from classification.realtime import OnlineDecoder

decoder = OnlineDecoder.from_detector(detector, step_s=0.25, smoothing=3)

while streaming:
    chunk = board.get_board_data()[eeg_channels, :]    # (22, n_new_samples)
    for decision in decoder.push_chunk(chunk):
        if decision.label is not None:
            send_command(decision.label, decision.confidence)
```

## 6. The test bench in your browser

```bash
./run_ui.sh
```

It opens <http://127.0.0.1:8000> in your browser. Real form controls, a Run
button, and a table of every run so you can compare settings. Ctrl+C in the
terminal stops it.

Nothing leaves your machine — the server binds to loopback only and the page
talks to it over plain JSON.

| Control | What it changes |
|---|---|
| **Recording** | Which subject's `.mat` to test |
| **Type of data** | All four classes, or just `left` / `right` / `foot` / `tongue` |
| **Model** | Riemannian TS + LR, Riemannian MDM, or CSP + LDA |
| **start / end after cue** | The window, dragged in quarter-second steps |

Pressing Run cuts fresh epochs at your window, cleans them, and runs a real
5-fold cross-validation, so the accuracy is measured on trials the model did not
train on. The first run parses the recording and takes a few seconds; after that
each run is about a second because the signal is cached.

The window readout turns red and disables Run below 1 second, because covariance
estimation needs more samples than that.

### Settings in the URL

A configuration can be put in the address, which makes it shareable and
bookmarkable. `run=1` presses the button on load:

```
http://127.0.0.1:8000/?subject=A04T&tmin=1.0&tmax=4.0&class=foot&run=1
```

### Which Python

`run_ui.sh` exists because this machine has several Python installations and only
some have numpy, scikit-learn and pyriemann:

| Command | Interpreter | Works? |
|---|---|---|
| `python` | `~/opt/anaconda3/bin/python` | yes |
| `python3` | `/opt/homebrew/bin/python3` | no |

Running the server with the wrong one dies during import, so nothing binds a port
and the browser reports **ERR_CONNECTION_REFUSED** — the failure appears in the
terminal, not the browser. The launcher finds an interpreter that works and uses
it. Every entry point also checks its own interpreter first and prints which one
to use instead, rather than a bare `ModuleNotFoundError`.

To run a module directly, use `python`, not `python3`:

```bash
python -m classification.server
```

Or point the launcher at a specific interpreter:

```bash
NERV_PYTHON=/path/to/python ./run_ui.sh
```

### If port 8000 is taken

```bash
./run_ui.sh --port 8001
```

Any `classification.server` flag is passed straight through; `--no-browser` skips
opening a browser.

### Things worth trying

| Try this | What you should see |
|---|---|
| A01T at 0.5–3.5 s | ~82% — the easy subject at its best window |
| A04T at 0.5–3.5 s, then 1.0–4.0 s | ~57% → ~60%: same subject, better window |
| Either subject at 0.5–1.5 s | Run is disabled; the window is too short |
| Type of data → `tongue` | Scores that class alone |
| Model → CSP + LDA | Around two points behind, and slower to fit |

## 7. The live view — you see the answer, the model does not

At <http://127.0.0.1:8000/live>, or from the link at the top of the test bench.

Trials stream past a trained detector one window at a time. You watch the traces
go by, watch the model's four probabilities move as it makes up its mind, and the
true label sits in a separate panel, blurred until the trial ends so you can call
it yourself first.

### Why the model genuinely cannot cheat

Two things enforce it, not one:

1. **It never trained on what you watch.** `LiveSession` splits the recording,
   fits the detector on 70% and streams only the held-out 30%. Every trial on
   screen is one the model has never met — the header reports the split
   (for A01T: trained on 191 trials, streaming 82 unseen).
2. **The label is not on the model's path.** `LiveSession._frames_for()` takes the
   signal as its only argument — the label is not even a parameter, so it cannot
   reach the classifier by accident. It is attached to the JSON response after the
   probabilities exist, purely for display.

You can confirm the second point from outside: the response has `truth` at the top
level, and no frame contains it.

```bash
curl -s -X POST http://127.0.0.1:8000/api/live/trial \
  -H 'Content-Type: application/json' -d '{"subject":"A01T","index":0}' |
  python -c "import json,sys; d=json.load(sys.stdin); \
    print('truth at top level:', 'truth' in d); \
    print('truth inside any frame:', any('truth' in f for f in d['frames']))"
```

### Controls

| Control | What it does |
|---|---|
| Recording | Which subject to stream |
| Speed | Real time, 2×, 4×, or as fast as the server can go |
| Start / Stop | Runs through the held-out trials |
| Skip trial | Jump to the next one |
| **show the answer straight away** | Unblurs the label immediately, if you would rather not guess |

A running tally sits top right and a log of the last 15 trials at the bottom.
Expect around **73%** on A01T — these are unseen trials, so it is an honest number
rather than the 100% you get by re-running a detector over its own training data.

Settings work in the URL too:

```
http://127.0.0.1:8000/live?subject=A01T&speed=4&auto=1
```

### The first window is the decision point

The display keeps updating as the window slides, but the score is taken from the
**first complete window**, and that is deliberate. Accuracy per window on A01T's
82 held-out trials:

| Window | Ends at | Accuracy |
|---|---|---|
| 1 | 3.0 s | **78%** |
| 2 | 3.5 s | 77% |
| 3 | 4.0 s | 76% |
| 4 | 4.5 s | 72% |
| 5 | 5.0 s | 68% |

The later windows slide past the imagery into the rest period, so they get
steadily worse. Watching the confidence sag in real time is the clearest
demonstration of why the epoch window mattered so much (section 2).

## 8. Identify a signal you choose

At <http://127.0.0.1:8000/identify>.

The live stream plays trials in order. This one hands you the list: every signal
the model has never seen, each with a thumbnail of its waveform and its real
label. Click one and the model identifies it while you watch, with a confidence
ring.

| Part | What it is |
|---|---|
| Signal list | All held-out trials, with a preview trace and the true label |
| Class chips | Narrow the list to `left` / `right` / `foot` / `tongue` |
| Scope | The chosen signal playing, window by window |
| Model identifies | Its call, with a confidence ring |
| Really was | The true label, and whether it got it |
| Confidence bars | All four classes, updating as it watches |

A tally at the top keeps score across everything you have tried, and rows you
have already run are marked `identified` or `missed`, so you can work through the
list and see the pattern for yourself.

Direct links work here too — handy for showing someone a specific case:

```
http://127.0.0.1:8000/identify?subject=A01T&signal=1
```

The `signal` number matches the row label.

### Reading the confidence

The ring, the bars and the verdict all describe the **first complete window**,
which is the model's best read (section 7). During playback the bars keep moving
as the window slides; when the signal ends they settle back to that decision
frame, so everything on screen describes the same moment.

Confidence is worth watching rather than trusting. On a correct identification it
is typically 70–90%; on a miss it is often 35–55%, which is what makes
`--min-confidence` useful in the detector. But it is not reliable — section 14
shows the model being 100% confident and wrong on a subject it was not trained
on.

## 9. The same thing without a browser

```bash
python -m classification.app
```

A matplotlib version of the panel above: the same controls as radio buttons and
sliders in a plot window. Useful if you would rather not run a server, and it can
render straight to an image for a report:

```bash
python -m classification.app --subject A04T --tmin 1.0 --tmax 4.0 --snapshot result.png
```

## 10. Watching it work — the replay UI

`ui.py` opens a window that replays a recording through the detector and draws
what it is doing: the EEG going in, the four class probabilities updating as the
window slides, and the call it is currently making.

```bash
python -m classification.ui --model models/a01t.joblib --subject A01T
```

Three panels:

- **top** — the C3 / Cz / C4 traces inside the current 3-second window
- **bottom left** — a bar per class, with the true class marked and the 25% chance
  line drawn in red
- **bottom right** — the current call, its confidence, and a running score

### One class at a time

This is the useful way to see *which* movement the detector struggles with:

```bash
python -m classification.ui --model models/a01t.joblib --subject A01T --only left
python -m classification.ui --model models/a01t.joblib --subject A01T --only right
python -m classification.ui --model models/a01t.joblib --subject A01T --only foot
python -m classification.ui --model models/a01t.joblib --subject A01T --only tongue
```

The same flag works on the terminal tool, if you want numbers rather than a window:

```bash
python -m classification.detect --model models/a01t.joblib --subject A04T --only foot
```

### Options

| Flag | Effect |
|---|---|
| `--only tongue` | Replay one class only |
| `--limit 8` | How many trials to replay (default 8) |
| `--speed 2` | Play twice as fast |
| `--step 0.5` | Seconds between decisions |
| `--min-confidence 0.6` | Show "no call" instead of guessing below 60% |
| `--snapshot look.png` | Save one still frame and exit, instead of opening a window |

`--snapshot` is there for running over SSH or in CI, where no window can open.

### What per-class runs reveal

Running each class separately against a subject the model was **not** trained on
exposes something a single accuracy number hides:

| Class | Cross-subject accuracy (A01T model → A04T) |
|---|---|
| right | 86.6% |
| left | 32.3% |
| foot | 13.4% |
| tongue | 0.0% |

The overall 33% is not "uniformly mediocre" — the model has collapsed toward
`right` on an unfamiliar brain. That is a different problem from being generally
weak, and it points at class-balance and domain-adaptation work rather than at a
better classifier.

## 11. Running the tests

```bash
python -m pytest tests/ -v
```

21 tests, about 7 seconds. They train a real model on a real recording rather than
mocking it, because the failure worth catching is "the chain from recording to
label stopped working", not "does this function return a float". They skip rather
than fail if no `.mat` files are present.

What they cover:

| Test | Guards against |
|---|---|
| epoch shape, no NaNs, all four classes | A loader change silently dropping channels or corrupting trials |
| held-out accuracy well above chance | Any pipeline regression |
| `detect` vs `detect_many` agree | The batch and single-trial paths drifting apart |
| save/load predicts identically | A persistence bug that quietly degrades a shipped model |
| wrong window length / channel count raises | The worst failure mode: confident answers on malformed input |
| confidence floor makes it decline | The safety gate actually gating |
| all three selected models still train | A dependency upgrade breaking one of them |
| online decoder matches offline | Offline/online preprocessing mismatch — the classic reason a decoder cross-validates well and fails live |
| live view streams a held-out split | The live view quietly starting to show trials the model trained on |
| no frame carries the true label | The answer leaking onto the model's path |
| `_frames_for()` takes only the signal | A future edit passing the label in "just for logging" |
| live accuracy beats chance on unseen trials | The blind pipeline silently breaking |

## 12. Reproducing the benchmark

```bash
pip install -r ../requirements.txt

# Score every model on every axis and write acquisition/data/model_scorecard.csv
python -m classification.benchmark

# Just the two Riemannian models, and a different window
python -m classification.benchmark --models riemann_ts_lr riemann_mdm --tmin 0.5 --tmax 4.0

# Skip the cross-subject pass (much faster for EEGNet)
python -m classification.benchmark --models eegnet --no-loso
```

EEGNet is skipped with a message if PyTorch is not installed; the Riemannian models
do not need it.

## 13. Files

| File | Role |
|---|---|
| `data.py` | Cuts labelled epochs from the raw `.mat` recordings; interpolates dropped samples; honours the dataset's own artifact flags |
| `preprocessing.py` | Baseline correction, 8–30 Hz bandpass, peak-to-peak artifact mask. One chain, shared by training and inference |
| `models.py` | The three selected models plus the CSP baselines, behind `build_model(name)` |
| `eegnet.py` | EEGNet as a scikit-learn estimator, so it drops into the same benchmark and decoder |
| `benchmark.py` | Measures all six axes, prints the scorecard, writes `acquisition/data/model_scorecard.csv` |
| `realtime.py` | Rolling-buffer online decoder, plus `simulate()` to check online matches offline |
| `detector.py` | The shipping detector: train, save, load, classify. Carries its own preprocessing settings |
| `train.py` | CLI — train on your recordings, print an honest held-out report, save to `models/` |
| `detect.py` | CLI — load a saved detector and run it over a recording, per trial or streaming |
| `ui.py` | CLI — a live window showing the traces, the class probabilities and the current call |
| `app.py` | CLI — the matplotlib test bench: choose recording, class, window and model, press Run |
| `server.py` | CLI — the test bench served to your browser, plus the live view, with a run-history table |
| `live.py` | The live session: trains on one split, streams the other, keeps the label off the model's path |
| `live_page.py` | The live stream's HTML, kept out of `server.py` so both stay readable |
| `identify_page.py` | The identify view's HTML: the signal picker and confidence ring |
| `../tests/test_detector.py` | End-to-end checks over the real recordings |
| `../tests/test_live.py` | Checks the live view's promise that the label never reaches the model |

## 14. How to improve accuracy

Everything below was measured on the two recordings in this repository, not
guessed. Ordered by what it actually returned.

### Already applied

**Stronger regularisation — worth +2.7 points overall, +6 on the hard subject.**
The tangent space of a 22×22 covariance has 253 dimensions, and a subject gives a
few hundred trials, so scikit-learn's default `C=1.0` overfits. Sweeping it found
a plateau between 0.003 and 0.02:

| `C` | A01T | A04T | mean |
|---|---|---|---|
| 1.0 (default) | 0.824 | 0.504 | 0.664 |
| **0.01** | 0.817 | **0.565** | **0.691** |

`TANGENT_SPACE_C` in `models.py` is now 0.01. Note where the gain landed: it cost
the easy subject 0.7 points and rescued the hard one by 6. That is the right
trade — the weak subject is the one limiting the system.

**Per-subject window tuning — worth +3 points on A04T.** The best window is not
the same for everyone:

| Subject | best window | accuracy |
|---|---|---|
| A01T | 0.5–3.5 s | 0.817 |
| A04T | 1.0–4.0 s | 0.599 |

Use `--tune-window` and it searches per user:

```bash
python -m classification.train --subjects A04T --tune-window
```

Together these take the two-subject mean from **0.664 to 0.708**.

### Do this next

**Per-user calibration beats everything else, and needs about a minute.** This is
the measured learning curve on A01T — accuracy against how many of that person's
own trials the model has seen:

| Calibration trials | ≈ minutes | Accuracy |
|---|---|---|
| 0 (cross-subject) | 0 | 0.332 |
| 8 | 1 | 0.563 |
| 24 | 3 | 0.673 |
| 80 | 10 | 0.763 |
| 160 | 20 | 0.798 |

Eight trials — one minute of recording — takes a new user from 33% to 56%. No
amount of algorithmic work on the zero-calibration case came close to that. If
you only do one thing, make the app record a short calibration block on first use.

**Do not pool subjects.** Adding the other subject's trials to a target user's own
data *hurts* once the target has 24 trials or more:

| Target trials | Target only | + other subject |
|---|---|---|
| 8 | 0.563 | 0.486 |
| 40 | 0.696 | 0.584 |
| 160 | 0.798 | 0.692 |

Below ~24 target trials, pooling helps slightly on the harder subject. Above it,
pooling is consistently worse. Naive concatenation is not transfer learning.

### Tried and not worth it

| Change | Result |
|---|---|
| Frequency band | 8–30 Hz already optimal. 4–40, 8–35, 6–32, 4–30, 8–26 and 1–40 all scored lower. Do not spend time here. |
| Covariance estimator | OAS (0.664) beat Ledoit-Wolf (0.657) and the plain sample covariance (0.651). Already the default. |
| Soft-vote ensemble of all three models | 0.672 vs 0.664 for TS+LR alone — inside the noise, for triple the compute. |
| Riemannian re-centring | Fixes the *collapse* — cross-subject `tongue` recall goes 0.00 → 0.26 and predictions spread across all four classes — but overall accuracy only moves 0.332 → 0.355. Worth it only in the zero-calibration case, and even then calibration beats it. |

### Still untested, in order of likely payoff

1. **Add the other seven BCI IV 2a subjects.** Every cross-subject number here
   rests on training with exactly one person. This also gates a fair re-test of
   EEGNet, which is starved rather than beaten (section 4).
2. **Finish `signal_analysis/ica.py`.** Still a skeleton. Eye-blink removal is
   typically worth a few points and would let the 100 µV rejection threshold be
   loosened, recovering the 15–26 trials per subject currently discarded.
3. **Trial-level rejection by confidence during calibration.** Drop the trials the
   user plainly was not concentrating on, rather than trusting every cue.
4. **Proper transfer learning** — `pyriemann.transfer.MDWM` weights source and
   target domains rather than concatenating them, which is what failed above.

## 15. References

- Barachant, A. et al. (2012). *Multiclass Brain–Computer Interface Classification by Riemannian Geometry*. IEEE TNSRE 20(3).
- Lawhern, V. J. et al. (2018). *EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces*. J. Neural Eng. 15(5).
- Lotte, F. et al. (2018). *A review of classification algorithms for EEG-based BCIs: a 10 year update*. J. Neural Eng. 15(3).
- Brunner, C. et al. (2008). *BCI Competition 2008 – Graz Data Set A*. TU Graz.
