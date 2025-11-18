from config.config import EnvironmentConfig


def test_environment_config_defaults_and_overrides():
    cfg = EnvironmentConfig()

    # Defaults
    assert cfg.env_id == 'ohlcv_discrete'
    assert cfg.orderbook_levels == 5
    assert cfg.use_orderbook_synthetic is True

    # Overrides via atribuição
    cfg.env_id = 'orderbook_discrete'
    cfg.orderbook_levels = 9
    cfg.use_orderbook_synthetic = False

    assert cfg.env_id == 'orderbook_discrete'
    assert cfg.orderbook_levels == 9
    assert cfg.use_orderbook_synthetic is False

