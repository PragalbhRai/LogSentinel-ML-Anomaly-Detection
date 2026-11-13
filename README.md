# LogSentinel — HDFS Log Anomaly Detection with HMMs

University Machine Learning Unit 4 project: a three-state categorical HMM with a
custom **scaled Forward algorithm**, class-weighted Random Forest and SMOTE
Random Forest, with frequency and first-order Markov reference baselines.
This is historical block-sequence anomaly detection, not a validated failure
forecasting or live monitoring system.

Colab experimentation is complete. The reusable inference backend is integrated;
the Streamlit dashboard presents three simple faculty-review pages using verified
original artifacts and saved Random Forest experiment results.
No model training or threshold selection runs in the dashboard. Tests use only
small isolated safety fixtures; they never retrain the saved models.

**Team ZIP:** If you received `LogSentinel_Team_Handoff.zip`, follow [SETUP.md](SETUP.md).
It includes the original notebook, trained model and all dashboard artifacts.

## Run locally in VS Code

Verified on Windows x64 with Python **3.12.10**. Open this directory as the VS Code
workspace and select `.venv/Scripts/python.exe` as the interpreter.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/verify_artifacts.py
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The complete suite recomputes all validation and test scores and can take around
one to two minutes on this machine. For fast checks use `-m "not integration"`.
For a new saved verification report use
`python scripts/verify_artifacts.py --output outputs/verification.json`.
The verifier refuses to overwrite an existing report or write into the original
artifact directory. Launch commands assume an editable repository installation.

`requirements.txt` installs the local package and runtime pins from
`requirements-runtime.txt`. `requirements-dev.txt` adds tests and optional SMOTE
checks. `requirements.lock.txt` is the historical Windows environment record;
it is not the Linux Cloud installation file. The
original Colab versions remain in `logsentinel_artifacts/dependency_versions.json`.
This local environment verifies inference, not rerunning the notebook training.

## Original experiment and artifact placement

Keep the original **`logsentinel.ipynb`** in the repository root. It is the Colab
notebook supplied for integration; it has not been moved or edited. Keep the 22
original exported files under `logsentinel_artifacts/`, including `figures/`.
The original artifact bundle and complete final Random Forest run are included
in the Git upload allowlist. Raw datasets, other outputs, virtual environments
and old handoff archives stay local. Artifact checksums must match
`manifests/artifact_checksums.json`. Git attributes preserve their original bytes.

The source dataset is [HDFS_v1 from Loghub](https://github.com/logpai/loghub/tree/master/HDFS).
The notebook records 11,175,629 log lines and 575,061 labeled blocks. Strict
chronological time purging excludes 81,201 boundary-crossing blocks, retaining
493,860. Membership is loaded from the frozen manifest, not recreated from nominal
70/15/15 proportions. Raw HDFS logs and their hashes are not present locally.

| Partition | Normal | Anomalous | Total |
| --- | ---: | ---: | ---: |
| Training | 345,462 | 11,674 | 357,136 |
| Validation | 49,485 | 971 | 50,456 |
| Test | 85,152 | 1,116 | 86,268 |

## Frozen inference contract

- Input observations are component-aware integer event IDs: 52 templates plus
  UNKNOWN at zero. No labels, block IDs, timestamps or severity become features.
- Parser substitutions and their order come from notebook cell 10. Sequence
  construction retains input line order and the original first-block matching
  policy. The notebook reports zero multi-block lines in the audited HDFS data.
- The HMM was trained only on normal training blocks. Candidate selection ranks
  validation average precision first, then F1; the selected pilot was followed by
  final normal-training fitting. Both stages remain documented separately.
- Scores are `-log P(events | model) / event_count` using the custom scaled Forward
  recurrence. Empty, fractional and out-of-vocabulary input is rejected.
- The HMM threshold is exactly **2.991257009736873**. A score **>= threshold** is
  anomalous. There is no threshold slider, optimizer or retraining API.
- Baselines load their own frozen normal-training probabilities and validation
  thresholds. Markov scores include the initial event probability and divide by
  event count, including one-event traces.
- Ground truth is used only for evaluation and historical comparisons. Metrics
  are block-level; false-alarm rate is FP/(FP+TN).

## Preserved final test findings

The **frequency baseline outperformed the HMM**. This is a result to explain,
not hide. The model comparison and dashboard show it prominently.

| Model | Precision | Recall | F1 | False-alarm rate | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frequency | 0.808696 | 1.000000 | **0.894231** | 0.003100 | **0.887492** |
| Markov | 0.798064 | 0.517025 | 0.627515 | 0.001715 | 0.645380 |
| HMM, 3 states | 0.695449 | 0.534050 | 0.604156 | 0.003065 | 0.692690 |

HMM test confusion counts: TP=596, FP=261, TN=84,891, FN=520. The original files
are the authority; the table rounds for display. Tests require score agreement
at absolute tolerance 1e-12 and exact prediction equality. In the verified local
environment all three methods reproduced every saved validation/test score with
maximum absolute difference 0.0 and no prediction mismatches.

## Backend usage

```python
from logsentinel import InferenceEngine, load_models, load_replay

engine = InferenceEngine(load_models())
replay = load_replay()
index = int(replay.indices("test")[0])
result = engine.predict(replay.sequence(index))
frequency = engine.predict(replay.sequence(index), method="frequency")
# Raw text: engine.predict_logs(lines), using the recovered Colab parser.
```

Only trusted, checksum-verified original joblib models are loaded. There is no
user-uploaded model path in the dashboard. `load_models()` loads small inference
artifacts; `load_replay()` additionally checks offsets, event counts, metadata
identity and frozen split boundaries. Labels are not passed to the engine.

## Project map

- `src/logsentinel/`: preprocessing, artifact loading, custom Forward, inference,
  input validation and evaluation verification.
- `app/`: the three-page faculty dashboard, shared styling, verified data
  adapters, and chart helpers retained for regression tests.
- `tests/`: original notebook parity, independent numerical checks, hmmlearn
  comparisons, negative artifact tests, full frozen regression and AppTest.
- `scripts/verify_artifacts.py`: read-only full-export inference verification.
- `manifests/`: original artifact/notebook hashes and integration environment.
- `configs/experiment.yaml`: actual frozen experiment settings.
- `docs/notebook_provenance.md`: cell-to-module mapping and methodology caveats.
- `docs/backend_architecture.md`, `docs/dashboard_design.md`: boundaries and next
  dashboard phase.
- `outputs/`: ignored generated verification reports.

## Remaining limits

Optional supervised Random Forest experiments for class imbalance are documented
in [docs/imbalance_experiments.md](docs/imbalance_experiments.md). They reuse the
frozen chronological split, save separate outputs, and leave dashboard behavior
unchanged. Their results are separate from the preserved Colab findings above.

Saved events have positions, not individual timestamps; no event timing is
fabricated. Block timestamps have no recorded timezone. New supplied traces may
be partial; the complete-block threshold is not validated for early-warning use.
The original parser keeps its HDFS-specific assumptions and first-match policy.

The raw dataset is absent, so a complete raw-log-to-export replay cannot run
locally. Parser tests use all exported templates and five real raw lines retained
in notebook output; encoded inference is tested on every holdout block. UNKNOWN
was not observed in retained experimental data. Full-training strict convergence
was not reached after 35 iterations, as the original report states.

Parser diagnostics inspected a full-timeline sample in Colab. Vocabulary fitting
was training-only, but parser-design independence from holdout content cannot be
proven. This exposure is documented rather than retroactively changing results.
Assistive-technology QA, the academic report and presentation remain future work.

## Dashboard workflows

The redesigned Streamlit app has exactly three pages:

| Page | What it shows |
| --- | --- |
| Overview | Simple introduction, all 575,061 labeled blocks, normal/anomaly counts, one class-imbalance chart |
| Model Comparison | Actual saved test metrics for HMM, class-weighted RF and SMOTE RF; one F1 chart; SMOTE training-only explanation and pattern-overlap limitation |
| Anomaly Detection | Search/select any saved test block, click Predict, compare each model's verified saved decision with the actual label |

The dashboard checks original artifact hashes and the completed RF run manifest,
then verifies test block IDs, labels, thresholds, predictions and metrics. It uses
saved predictions; it does not retrain or select thresholds. Missing or corrupt
files show an error without invented results. The original backend and shared helpers required by regression tests remain
available; the unused six-page view modules have been removed.

Keep `outputs/imbalance_experiments/run_20261113_01/` with this workspace. It is
required for the new comparison and prediction pages. Set
`LOGSENTINEL_EXPERIMENT_DIR` to another complete copy of that verified run if it
lives elsewhere. `LOGSENTINEL_ARTIFACT_DIR` can point to a copy of the original
bundle. The original handoff ZIP has not been rebuilt and contains the earlier
interface; it does not include the new RF outputs.

For a faculty walkthrough and verification details, see
[docs/dashboard_design.md](docs/dashboard_design.md).

Optional rendered desktop/mobile QA uses the installed Edge browser:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-qa.txt
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.port 8503
# In another terminal:
.\.venv\Scripts\python.exe scripts/browser_qa.py
```

New screenshots and the QA summary go to `outputs/dashboard_review_20261113/`.

## Streamlit Community Cloud

Use repository `PragalbhRai/LogSentinel-ML-Anomaly-Detection`, branch `main`,
entry point `app/streamlit_app.py`, and select **Python 3.12** in Advanced settings.
The root `requirements.txt` installs dependencies and the local `src/logsentinel`
package. No raw logs, secrets, model retraining or browser automation are needed
for hosting. See [docs/deployment.md](docs/deployment.md) for validation limits.
