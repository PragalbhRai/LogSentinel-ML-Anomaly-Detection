"""Rendered desktop/mobile checks of the three-page Streamlit dashboard."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8503')
    parser.add_argument('--output', type=Path, default=Path('outputs/dashboard_review_20261113'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pages = [('overview', 'Overview', 1), ('model-comparison', 'Model Comparison', 1),
             ('anomaly-detection', 'Anomaly Detection', 0)]
    errors, results = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1050}, reduced_motion='reduce')
        page.on('pageerror', lambda error: errors.append(str(error)))
        for width, height in [(1440, 1050), (390, 844)]:
            page.set_viewport_size({'width': width, 'height': height})
            for key, title, charts in pages:
                page.goto(args.url + '?view=' + key)
                page.get_by_role('heading', name=title, exact=True).wait_for(timeout=60000)
                page.get_by_text('LogSentinel · Historical HDFS data · Verified saved results', exact=True).wait_for(timeout=60000)
                expect(page.locator('[data-testid="stPlotlyChart"]')).to_have_count(charts)
                if key == 'model-comparison':
                    expect(page.get_by_text('SMOTE Random Forest has the highest test F1-score:', exact=False)).to_be_visible()
                    expect(page.get_by_text('Limitation: 99.90%', exact=False)).to_be_visible()
                if key == 'anomaly-detection':
                    page.get_by_role('button', name='Predict', exact=True).click()
                    page.get_by_role('heading', name='Prediction results', exact=True).wait_for(timeout=30000)
                    expect(page.locator('[data-testid="stDataFrame"]')).to_have_count(1)
                    expect(page.get_by_text('Actual label:', exact=False).first).to_be_visible()
                if page.locator('[data-testid="stException"]').count():
                    raise RuntimeError('Streamlit exception on ' + key)
                overflow = page.locator('[data-testid="stMain"]').evaluate('el => el.scrollWidth > el.clientWidth + 2')
                screenshot = f'{key}-{width}.png'
                page.screenshot(path=str(args.output / screenshot), full_page=True, animations='disabled')
                results.append({'page': key, 'width': width, 'horizontal_overflow': overflow, 'screenshot': screenshot})
        page.set_viewport_size({'width': 1440, 'height': 1050})
        page.locator('[data-testid="stSidebar"]').get_by_text('Overview', exact=True).click()
        page.get_by_role('heading', name='Overview', exact=True).wait_for(timeout=30000)
        browser.close()
    report = {'browser': 'Installed Microsoft Edge / Playwright', 'pages': results, 'page_errors': errors,
              'interactions': ['Three-page navigation', 'Verified saved prediction replay']}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if errors or any(row['horizontal_overflow'] for row in results):
        raise SystemExit('Browser errors or horizontal overflow detected.')


if __name__ == '__main__':
    main()
