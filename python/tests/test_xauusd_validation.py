import os
import pandas as pd
import numpy as np

from evaluation.backtrader_runner import BacktraderRunner


def _make_synth_csv(path: str, n: int = 300) -> None:
    ts = pd.date_range('2022-01-01', periods=n, freq='D')
    prices = np.cumprod(1.0 + np.random.normal(0.0005, 0.01, size=n)) * 1800.0
    df = pd.DataFrame({
        'timestamp': ts,
        'open': prices,
        'high': prices * (1.0 + 0.005),
        'low': prices * (1.0 - 0.005),
        'close': prices,
        'volume': np.random.randint(1000, 5000, size=n),
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def test_backtrader_runner_generates_reports(tmp_path):
    csv_path = tmp_path / 'xauusd_synth_1d.csv'
    _make_synth_csv(str(csv_path))
    runner = BacktraderRunner(initial_cash=10000.0, commission=0.001)
    res = runner.run_once(str(csv_path), strategy_name='sma', timeframe='1d', strategy_params={'period_short': 10, 'period_long': 30})
    assert isinstance(res.total_return_pct, float)
    img_path, img_uri = runner.plot_curves(res, out_prefix='test')
    assert os.path.exists(img_path)
    html = runner.generate_html_report([res], title='Teste', image_data_uris=[img_uri], out_path=str(tmp_path / 'report.html'))
    assert os.path.exists(html)
    pdf = runner.generate_pdf_report([res], title='Teste', image_paths=[img_path], out_path=str(tmp_path / 'report.pdf'))
    assert os.path.exists(pdf)