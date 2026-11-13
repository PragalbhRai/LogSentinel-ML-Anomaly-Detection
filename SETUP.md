# LogSentinel teammate setup — Windows PowerShell

This team ZIP includes the complete source project, original Colab notebook and
all 22 original experiment exports. The dashboard runs from those exports without
the raw HDFS dataset, retraining, or a threshold change. Keep the extracted folder
structure intact. Start here rather than restoring artifacts separately.

## 1. Install Python and extract the project

Use **64-bit CPython 3.12** on Windows. The verified interpreter is **3.12.10**;
the project requires `>=3.12,<3.13`. Download Python from the
[official Python 3.12.10 release](https://www.python.org/downloads/release/python-31210/)
and enable the Windows Python launcher during installation. VS Code is optional.
Python 3.13/3.14 are not the supported environment for this handoff.

Save `LogSentinel_Team_Handoff.zip` in your Downloads folder. Run the following in
PowerShell. The destination should be a new folder; no existing project is
overwritten. Change only the first two paths if you saved the ZIP elsewhere.

```powershell
$handoffZip = Join-Path $env:USERPROFILE 'Downloads\LogSentinel_Team_Handoff.zip'
$handoffDestination = Join-Path $env:USERPROFILE 'Documents\LogSentinelTeam'
Expand-Archive -LiteralPath $handoffZip -DestinationPath $handoffDestination
Set-Location -LiteralPath (Join-Path $handoffDestination 'LogSentinel')
py -3.12 --version
```

You should now see `app`, `src`, `tests`, `logsentinel_artifacts`, `SETUP.md`, and
`logsentinel.ipynb` directly in the current folder. Run every command below from
this folder. Avoid extracting into OneDrive or another actively synchronized
directory while installing or testing.

## 2. Create an isolated environment and install the locked dependencies

An Internet connection to PyPI is required; dependencies are not bundled in the
ZIP. The runtime lock is the verified Windows x64/Python 3.12 environment, distinct
from the original Colab dependency record, which remains unchanged.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pip check
```

The editable install is required: it connects `logsentinel` to this extracted
checkout and preserves the repository-relative artifact paths. Do not copy an
existing `.venv` from someone else's computer. No PowerShell activation script
or execution-policy change is needed because every command names the interpreter.
The editable package build also downloads its setuptools build dependency.

## 3. Verify the package, inference and tests

The ZIP contains `HANDOFF_MANIFEST.json` with a SHA-256 and byte count for every
payload file. Its own generated contents are intentionally outside that list.
The separate original-experiment manifest verifies the notebook and frozen
exports. Run these checks before editing source files:

```powershell
.\.venv\Scripts\python.exe scripts/package_handoff.py --verify-tree .
.\.venv\Scripts\python.exe scripts/verify_artifacts.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Full inference verification checks all **50,456 validation** and **86,268 test**
blocks for HMM, frequency and Markov inference. It does not fit any model or
select thresholds. The full test suite may take a few minutes. A faster test run
is available with `-m "not integration"`; it does not replace full verification.

The HMM threshold remains **2.991257009736873**, with inclusive `score >= threshold`
classification. The frequency baseline's test F1 **0.894231** and HMM test F1
**0.604156** are preserved and displayed honestly.

## 4. Launch the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** in your browser. Keep this PowerShell window open;
press **Ctrl+C** to stop the server. The original handoff sidebar pages use the included saved
validation/test data. Startup verifies the model and data bundle before display.

If port 8501 is occupied:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.port 8502
```

Then open http://localhost:8502. In VS Code, open the extracted `LogSentinel`
folder and select `.venv\Scripts\python.exe` as the Python interpreter.

## 5. Optional: original HDFS_v1 dataset

**Skip this section to run the dashboard.** The original raw dataset is excluded
from this ZIP. Loghub lists HDFS_v1 as 11,175,629 lines and 1.47 GiB raw size;
the compressed download size differs. Do not substitute its 2,000-line sample,
HDFS_v2, or HDFS_v3 for the original experiment.

Official source: [Loghub HDFS documentation](https://github.com/logpai/loghub/tree/master/HDFS)
and [Loghub download table](https://github.com/logpai/loghub#logs-currently-available).
The table links to the following Zenodo-hosted HDFS_v1 archive:

```powershell
New-Item -ItemType Directory -Path .\data\raw -Force | Out-Null
Invoke-WebRequest -Uri 'https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1' -OutFile .\data\raw\HDFS_v1.zip -UseBasicParsing
Get-FileHash -LiteralPath .\data\raw\HDFS_v1.zip -Algorithm SHA256
Expand-Archive -LiteralPath .\data\raw\HDFS_v1.zip -DestinationPath .\data\raw\HDFS_v1
```

Allow disk space for the compressed archive and extracted logs/preprocessed
files. Record the downloaded hash; this project does not supply an original raw
dataset hash. The raw logs and labels are for academic research; follow Loghub's
dataset citation and use guidance. Keep downloads out of Git.

`logsentinel.ipynb` is the **unchanged original Colab notebook**, with Colab paths,
download/training cells and saved outputs. It is included for inspection and
provenance, not as a dashboard startup step. Do not run all its cells merely to
start the application. Local notebook execution and a full raw-data round-trip
require additional notebook tooling/path adaptation and are not part of this
inference-only environment. The notebook can be opened in VS Code or Colab.

## 6. Troubleshooting and optional browser checks

- **`py` not found:** install the Windows Python launcher, reopen PowerShell, and
  check `py -3.12 --version`. Alternatively use the full path of a Python 3.12 x64
  executable for the initial `-m venv` command.
- **`No module named logsentinel`:** use this folder's `.venv` interpreter and
  rerun the editable install from step 2.
- **Artifact missing/checksum mismatch:** restore the original ZIP in a fresh
  destination. Do not regenerate models, change manifests or lower validation
  checks to bypass the error. Use the sidebar's bundle reverification after
  restoring files.
- **A different artifact directory is being used:** check whether an inherited
  `LOGSENTINEL_ARTIFACT_DIR` environment variable points elsewhere. To use the
  bundled artifacts in this shell, run
  `Remove-Item Env:LOGSENTINEL_ARTIFACT_DIR -ErrorAction SilentlyContinue`.
- **Installation cannot reach PyPI:** resolve the connection/proxy or ask for an
  approved mirror. The archive deliberately contains no platform-specific venv
  or offline dependency wheelhouse.
- **Pytest temporary-folder permissions:** rerun in a writable local location or
  pass a new explicit `--basetemp` folder under your own workspace. Do not delete
  another person's test directories.

Optional screenshot QA uses the existing Microsoft Edge browser and is separate
from the default test suite. Browser screenshots are not included in this ZIP.
Keep Streamlit running in one terminal; in a second PowerShell terminal in the
same project directory run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-qa.txt
.\.venv\Scripts\python.exe scripts/browser_qa.py
```

Generated verification reports, CSV downloads and screenshots belong under
ignored `outputs/`. The original results/figures under `logsentinel_artifacts/`
must remain unchanged. This project demonstrates historical anomaly detection;
it does not provide live monitoring or validated failure prediction. Event
timelines use event positions because per-event timestamps are unavailable.

The current workspace has a redesigned three-page dashboard and additionally needs
`outputs/imbalance_experiments/run_20261113_01/`. See the current README for setup.
The existing handoff ZIP retains its earlier interface and has not been rebuilt.
