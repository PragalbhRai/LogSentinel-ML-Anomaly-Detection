# GitHub and Streamlit Community Cloud

Deployment settings: repository `PragalbhRai/LogSentinel-ML-Anomaly-Detection`,
branch `main`, entry point `app/streamlit_app.py`, Python **3.12** selected in
Advanced settings. `.python-version` records the same choice for local tools;
it does not replace the Cloud version selector.

The root `requirements.txt` installs runtime pins and the editable local package.
Package metadata reads `requirements-runtime.txt`, so installation does not
recursively include itself. `requirements-dev.txt` adds pytest, YAML contract
checks and SMOTE safety fixtures. `requirements-experiments.txt` adds SMOTE for
optional research reruns. `requirements-qa.txt` is local browser QA only. The
original Windows `requirements.lock.txt` remains an archival record and is not
used to build the Cloud environment.

The runtime needs every original export in `logsentinel_artifacts/` and the entire
saved final run in `outputs/imbalance_experiments/run_20261113_01/`. They are
explicitly allowed by `.gitignore`; other outputs stay local. Preserve their
paths and manifests. `.gitattributes` disables text conversion for both bundles
and the notebook because checksums apply to exact bytes. No LFS pointer files
should replace local artifact contents without a functioning LFS checkout.

The dashboard uses verified saved predictions for all three models; it performs
no training or threshold tuning. Raw HDFS logs and secrets are not required.
HMM and Forward inference and both RF model bundles remain available locally.

For a new checkout:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts/verify_imbalance_run.py outputs/imbalance_experiments/run_20261113_01
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The existing handoff ZIP and old reports remain local and are excluded from Git.
Their original contents are unchanged. The packaging script now includes the
final RF payload when creating a new archive, and never overwrites old archives.

Pre-upload checks cover ignored files, artifact integrity, model/prediction
parity, package import, Linux wheel dependency resolution and desktop/mobile
browser behavior. Linux dependency resolution is not a live Community Cloud
runtime test: hosted startup and resource use must still be verified after an
authorized deployment. No deployment or push is performed during preparation.

Official deployment instructions:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
