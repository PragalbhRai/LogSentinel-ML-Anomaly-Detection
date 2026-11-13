"""Interactions and integrity checks for the three-page faculty review."""

from pathlib import Path
import shutil

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.review_data import DEFAULT_RUN, MODEL_NAMES, load_review_data
from logsentinel.artifacts import ArtifactError

ROOT = Path(__file__).resolve().parents[1]


def open_page(name):
    view = AppTest.from_file(str(ROOT / 'app/streamlit_app.py'), default_timeout=60)
    view.query_params['view'] = name
    view.run()
    assert not view.exception
    return view


def test_only_three_pages_and_one_overview_chart():
    view = open_page('overview')
    assert view.radio(key='workspace').options == ['Overview', 'Model Comparison', 'Anomaly Detection']
    assert len(view.radio) == 1
    text = ' '.join(m.value for m in view.markdown)
    data = load_review_data()
    for count in data.counts.values():
        assert f'{count:,}' in text
    assert len(view.get('plotly_chart')) == 1
    assert not view.slider and not view.toggle
    for key in ['model-comparison', 'anomaly-detection', 'overview']:
        view.radio(key='workspace').set_value(key).run()
        assert not view.exception


def test_comparison_uses_actual_saved_results():
    view = open_page('model-comparison')
    table = view.dataframe[0].value
    assert table.Model.tolist() == list(MODEL_NAMES.values())
    assert table['F1-score'].tolist() == ['0.6042', '0.9969', '0.9987']
    assert table['False-alarm rate'].tolist() == ['0.3065%', '0.0000%', '0.0000%']
    assert 'SMOTE Random Forest' in view.success[0].value
    assert '0.998654' in view.success[0].value
    assert any('99.90%' in message.value for message in view.info)
    assert len(view.get('plotly_chart')) == 1


def test_predict_replays_every_model_and_hides_stale_results():
    view = open_page('anomaly-detection')
    assert not view.dataframe
    assert not view.get('plotly_chart')
    view.button(key='predict').click().run()
    assert not view.exception
    block = view.selectbox(key='test_block').value
    data = load_review_data()
    expected = data.test.loc[block]
    table = view.dataframe[0].value
    assert table.Model.tolist() == list(MODEL_NAMES.values())
    assert table['Actual label'].tolist() == [expected.Label] * 3
    assert table.Prediction.tolist() == ['Anomaly' if expected[m+'_prediction'] else 'Normal' for m in MODEL_NAMES]
    other = view.selectbox(key='test_block').options[1]
    view.selectbox(key='test_block').set_value(other).run()
    assert not view.dataframe
    # Search reaches blocks outside the first 200, including genuine anomalies.
    anomaly = data.test[data.test.Label == 'Anomaly'].iloc[-1].BlockId
    view.text_input(key='block_search').set_value(anomaly).run()
    assert view.selectbox(key='test_block').options == [anomaly]
    assert not view.dataframe
    view.button(key='predict').click().run()
    assert view.dataframe[0].value['Actual label'].tolist() == ['Anomaly'] * 3
    view.text_input(key='block_search').set_value('missing_block').run()
    assert not view.exception and not view.dataframe
    assert 'No saved test blocks' in view.info[0].value


def test_old_url_falls_back_to_overview():
    view = open_page('event-timeline')
    assert view.radio(key='workspace').value == 'overview'


def test_all_saved_prediction_identities_and_counts():
    data = load_review_data()
    assert data.counts['normal'] + data.counts['anomaly'] == data.counts['total'] == 575061
    assert len(data.test) == 86268 and data.test.BlockId.is_unique
    for method in ['rf_class_weight', 'rf_smote']:
        with np.load(DEFAULT_RUN / method / 'test_predictions.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(data.test[method+'_prediction'], saved['predictions'])
            np.testing.assert_array_equal(data.test.BlockId.to_numpy(dtype=str), saved['block_ids'])


def test_corrupt_experiment_invalidates_cache_and_fails_closed(tmp_path):
    target = tmp_path / 'run'
    shutil.copytree(DEFAULT_RUN, target)
    load_review_data(run=target)
    (target / 'rf_smote/test_predictions.npz').write_bytes(b'corrupt')
    with pytest.raises(ArtifactError, match='checksum'):
        load_review_data(run=target)


def test_missing_experiment_shows_error_without_predictions(tmp_path, monkeypatch):
    monkeypatch.setenv('LOGSENTINEL_EXPERIMENT_DIR', str(tmp_path))
    view = open_page('anomaly-detection')
    assert view.error and not view.dataframe
    assert any(button.label == 'Retry verification' and not button.disabled for button in view.button)
