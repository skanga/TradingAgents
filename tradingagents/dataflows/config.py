from contextvars import ContextVar
from copy import deepcopy
from typing import Dict

import tradingagents.default_config as default_config

_config_var: ContextVar[dict | None] = ContextVar("tradingagents_config", default=None)


def _merged_config(config: Dict) -> Dict:
    """Merge config into defaults, preserving sibling nested keys."""
    base = deepcopy(default_config.DEFAULT_CONFIG)
    incoming = deepcopy(config)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


def initialize_config():
    """Initialize the configuration with default values."""
    if _config_var.get() is None:
        _config_var.set(deepcopy(default_config.DEFAULT_CONFIG))


def set_config(config: Dict):
    """Set configuration for the current context."""
    _config_var.set(_merged_config(config))


def get_config() -> Dict:
    """Get the current configuration."""
    cfg = _config_var.get()
    if cfg is None:
        cfg = deepcopy(default_config.DEFAULT_CONFIG)
        _config_var.set(cfg)
    return deepcopy(cfg)


def use_config(config: Dict):
    """Apply configuration to the current context and return a reset token."""
    return _config_var.set(_merged_config(config))


def reset_config(token) -> None:
    """Restore the configuration context represented by ``token``."""
    _config_var.reset(token)


# Initialize with default config
initialize_config()
