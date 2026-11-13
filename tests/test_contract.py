from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_config_contains_chronological_split():
    config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    assert config["split"]["strategy"] == "strict_chronological_time_purged"
    assert sum(config["split"]["provisional_fractions"]) == 1.0
    assert config["scoring"]["threshold"] == 2.991257009736873
    assert config["scoring"]["comparison"] == ">="


def test_evaluation_and_custom_algorithm_requirements_documented():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    for term in ["scaled forward", "chronological", "false-alarm", "notebook", "streamlit"]:
        assert term in readme
