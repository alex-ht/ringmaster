"""_flash_ops resolves the Dao low-level ops without going back to the Hub when transformers
already loaded a flash kernel for the model."""

import sys
import types

import pytest

fau = pytest.importorskip("transformers.modeling_flash_attention_utils")

from ringmaster.ring import p2p_attn  # noqa: E402

HUB_FA2 = "kernels-community/flash-attn2"


def _fake_interface(name):
    mod = types.ModuleType(name)

    def flash_attn_func(*args, **kwargs):
        raise AssertionError("not called")

    flash_attn_func.__module__ = name
    mod.flash_attn_func = flash_attn_func
    mod._flash_attn_forward = object()
    mod._flash_attn_backward = object()
    return mod


@pytest.fixture
def env(monkeypatch):
    p2p_attn._resolve_flash_ops.cache_clear()
    loaded = _fake_interface("_ringmaster_test_loaded_fa")
    hub = _fake_interface("_ringmaster_test_hub_fa")
    monkeypatch.setitem(sys.modules, loaded.__name__, loaded)
    monkeypatch.setitem(sys.modules, hub.__name__, hub)
    monkeypatch.setattr(
        fau, "FLASH_ATTN_KERNEL_FALLBACK", {"flash_attention_2": HUB_FA2}, raising=False
    )
    monkeypatch.setattr(fau, "_flash_fn", loaded.flash_attn_func, raising=False)
    yield monkeypatch, loaded, hub
    p2p_attn._resolve_flash_ops.cache_clear()


def _hub_down(*args, **kwargs):
    raise ValueError("429 Too Many Requests")


def test_reuses_loaded_hub_kernel_without_lookup(env):
    monkeypatch, loaded, _ = env
    monkeypatch.setattr(fau, "_loaded_implementation", HUB_FA2, raising=False)
    monkeypatch.setattr(fau, "_lazy_imports", _hub_down)
    assert p2p_attn._flash_ops() == (loaded._flash_attn_forward, loaded._flash_attn_backward)


def test_other_loaded_implementation_falls_back_to_lookup(env):
    monkeypatch, _, hub = env
    monkeypatch.setattr(fau, "_loaded_implementation", "flash_attention_3", raising=False)
    monkeypatch.setattr(fau, "_lazy_imports", lambda impl: (hub.flash_attn_func,))
    assert p2p_attn._flash_ops() == (hub._flash_attn_forward, hub._flash_attn_backward)


def test_failed_lookup_is_logged_and_retried_once_a_kernel_loads(env, caplog):
    monkeypatch, loaded, _ = env
    monkeypatch.setattr(fau, "_loaded_implementation", None, raising=False)
    monkeypatch.setattr(fau, "_lazy_imports", _hub_down)
    assert p2p_attn._flash_ops() is None
    assert "429" in caplog.text
    monkeypatch.setattr(fau, "_loaded_implementation", HUB_FA2, raising=False)
    assert p2p_attn._flash_ops() == (loaded._flash_attn_forward, loaded._flash_attn_backward)
