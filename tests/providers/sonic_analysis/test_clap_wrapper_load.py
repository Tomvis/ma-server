"""Tests for how the vendored CLAP wrapper reads its checkpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import torch

import music_assistant.providers.sonic_analysis.vendored_clap.clap_wrapper as cw_module

if TYPE_CHECKING:
    import pytest


def test_checkpoint_is_memory_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkpoint is mmapped, so tensors the model does not keep are never read into RAM."""
    load_kwargs: list[dict[str, Any]] = []

    def fake_load(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        load_kwargs.append(kwargs)
        return {"model": {}}

    monkeypatch.setattr(torch, "load", fake_load)
    monkeypatch.setattr(cw_module, "CLAP", MagicMock())

    cw_module.CLAPWrapper(model_fp="weights.pth", text_enabled=False)

    assert len(load_kwargs) == 1
    assert load_kwargs[0].get("mmap") is True
