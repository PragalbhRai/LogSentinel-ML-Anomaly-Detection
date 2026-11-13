from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.data import load_foundation
from app.design import COLORS, MODEL_COLORS, model_comparison_chart

ROOT = Path(__file__).resolve().parents[1]


def app():
    return AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30)


def test_real_overview_and_all_prepared_destinations():
    view = app().run()
    assert not view.exception
    assert "575,061" in " ".join(m.value for m in view.markdown)
    for page in ["model-comparison", "anomaly-detection"]:
        view.radio(key="workspace").set_value(page).run()
        assert not view.exception


def test_unavailable_artifacts_show_error_without_fake_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGSENTINEL_ARTIFACT_DIR", str(tmp_path))
    view = app().run()
    assert not view.exception
    assert "could not be verified" in view.error[0].value
    assert any(button.label == "Retry verification" for button in view.button)
    assert "0.6042" not in " ".join(m.value for m in view.markdown)


def luminance(color):
    channels = [int(color[i:i+2], 16) / 255 for i in (1, 3, 5)]
    linear = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in channels]
    return sum(v*w for v, w in zip(linear, [.2126, .7152, .0722]))


@pytest.mark.parametrize("foreground", ["text", "muted", "cyan", "teal", "amber", "coral"])
@pytest.mark.parametrize("background", ["canvas", "surface", "raised"])
def test_text_contrast(foreground, background):
    light, dark = sorted([luminance(COLORS[foreground]), luminance(COLORS[background])], reverse=True)
    assert (light+.05)/(dark+.05) >= 4.5


def test_chart_preserves_actual_comparison_and_color_semantics():
    data = load_foundation()
    figure = model_comparison_chart(data.comparison)
    trace = figure.data[0]
    assert sorted(trace.x) == sorted(data.comparison.f1)
    assert trace.marker.color[-1] == MODEL_COLORS["frequency"]
