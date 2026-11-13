import pytest

from logsentinel.evaluation import verify_frozen_inference


@pytest.mark.integration
def test_all_colab_scores_predictions_and_metrics():
    report = verify_frozen_inference()
    assert report["splits"]["validation"]["blocks"] == 50456
    assert report["splits"]["test"]["blocks"] == 86268
    for split in report["splits"].values():
        for result in split["models"].values():
            assert result["prediction_mismatches"] == 0
            assert result["max_absolute_score_difference"] <= 1e-12
    assert report["splits"]["test"]["models"]["frequency"]["metrics"]["f1"] > report["splits"]["test"]["models"]["hmm"]["metrics"]["f1"]
