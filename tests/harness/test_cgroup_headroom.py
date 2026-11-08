"""Cgroup accounting must distinguish resident memory from disposable cache."""
import pytest
from agent.harness.sandbox_resources import cgroup_headroom

@pytest.mark.parametrize("stat", [
    "total_cache 70\ntotal_inactive_file 60\ntotal_shmem 5\ntotal_dirty 3\ntotal_writeback 2\n",
    "file 70\ninactive_file 60\nshmem 5\nfile_dirty 3\nfile_writeback 2\n",
])
def test_clean_inactive_cache_can_be_reclaimed(tmp_path, stat):
    (tmp_path / "memory.stat").write_text(stat)
    assert cgroup_headroom(tmp_path, "100", "95") == 60

@pytest.mark.parametrize("stat", [
    "file 80\ninactive_file 80\nshmem 80\n",
    "file 80\ninactive_file 80\nfile_dirty 70\nfile_writeback 10\n",
    "file 0\ninactive_file 90\n",
    "file invalid\ninactive_file 80\n",
])
def test_shared_dirty_or_invalid_memory_stays_charged(tmp_path, stat):
    (tmp_path / "memory.stat").write_text(stat)
    assert cgroup_headroom(tmp_path, "100", "95") == 5

def test_missing_statistics_never_relax_the_limit(tmp_path):
    assert cgroup_headroom(tmp_path, "100", "95") == 5
    assert cgroup_headroom(tmp_path, "90", "95") == 0

def test_hierarchical_usage_uses_hierarchical_statistics(tmp_path):
    (tmp_path / "memory.stat").write_text("cache 80\ninactive_file 80\ntotal_cache 40\ntotal_inactive_file 30\n")
    assert cgroup_headroom(tmp_path, "100", "95") == 35


@pytest.mark.parametrize("stat", [
    "total_cache 90\ntotal_active_file 60\ntotal_inactive_file 25\ntotal_shmem 5\ntotal_dirty 3\ntotal_writeback 2\ntotal_rss 8\n",
    "file 90\nactive_file 60\ninactive_file 25\nshmem 5\nfile_dirty 3\nfile_writeback 2\nanon 8\n",
])
def test_clean_active_file_cache_does_not_starve_a_sandbox(tmp_path, stat):
    (tmp_path / "memory.stat").write_text(stat)
    assert cgroup_headroom(tmp_path, "100", "98") == 82


def test_unevictable_pages_are_not_reclaimable_file_lru(tmp_path):
    (tmp_path / "memory.stat").write_text(
        "file 90\nactive_file 20\ninactive_file 10\nunevictable 60\n"
    )
    assert cgroup_headroom(tmp_path, "100", "98") == 32


def test_hierarchy_does_not_use_child_active_pages(tmp_path):
    (tmp_path / "memory.stat").write_text(
        "cache 80\nactive_file 80\ntotal_cache 80\ntotal_inactive_file 10\ntotal_active_file 20\n"
    )
    assert cgroup_headroom(tmp_path, "100", "95") == 35
