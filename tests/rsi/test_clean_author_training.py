
import pytest
import torch

from agent.rsi.cases.clean_author_training import cached_embeddings, verify_trace


def test_cache_returns_fresh_tensors_and_preserves_uncached_load(tmp_path):
    cached, uncached = tmp_path / "cached.pt", tmp_path / "uncached.pt"
    torch.save(torch.tensor([1.0, 2.0]), cached)
    torch.save(torch.tensor([3.0, 4.0]), uncached)
    original = torch.load
    with cached_embeddings({str(cached.resolve()): torch.tensor([1.0, 2.0])}) as record:
        a = torch.load(cached)
        a[0] = -1
        assert torch.equal(torch.load(cached), torch.tensor([1.0, 2.0]))
        assert torch.equal(torch.load(uncached), torch.tensor([3.0, 4.0]))
        assert record["cache_hits"] == 2
    assert torch.load is original


def test_cache_restores_torch_after_failure(tmp_path):
    original = torch.load
    with pytest.raises(RuntimeError), cached_embeddings({}):
        raise RuntimeError("test failure")
    assert torch.load is original


@pytest.mark.parametrize("epochs", [[1], [1, 1], [2, 1], [1, 2, 3], [True, 2], [1, False]])
def test_incomplete_repeated_or_extra_epochs_fail(epochs):
    with pytest.raises(ValueError, match="epochs"):
        verify_trace([{"epoch": e, "loss": 0.5} for e in epochs], {"epochs": 2})


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -1.0])
def test_nonfinite_training_loss_fails(loss):
    with pytest.raises(ValueError, match="loss"):
        verify_trace([{"epoch": 1, "loss": loss}], {"epochs": 1})


def test_complete_training_trace_is_accepted():
    verify_trace([{"epoch": 1, "loss": 1.5}, {"epoch": 2, "loss": 0.7}], {"epochs": 2})
