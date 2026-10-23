"""CPU thread limit from the container's quota."""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'h_jepa'))

import cpu_threads  # noqa: E402


def quota(tmp_path, text):
    path = tmp_path / 'cpu.max'
    path.write_text(text)
    return str(path)


def test_allowed_cpus_reads_the_quota(tmp_path):
    assert cpu_threads.allowed_cpus(quota(tmp_path, '1360000 100000\n')) == 13
    assert cpu_threads.allowed_cpus(quota(tmp_path, '300000 100000')) == 3
    # less than one CPU still gives one thread
    assert cpu_threads.allowed_cpus(quota(tmp_path, '50000 100000')) == 1


def test_allowed_cpus_without_a_quota_is_the_cpu_count(tmp_path):
    cpus = os.cpu_count()
    assert cpu_threads.allowed_cpus(quota(tmp_path, 'max 100000')) == cpus
    assert cpu_threads.allowed_cpus(quota(tmp_path, 'garbage')) == cpus
    assert cpu_threads.allowed_cpus(str(tmp_path / 'missing')) == cpus


def test_limit_threads_follows_the_quota_and_the_limit(
    tmp_path, monkeypatch
):
    monkeypatch.delenv('OMP_NUM_THREADS', raising=False)
    before = torch.get_num_threads()
    try:
        small = quota(tmp_path, '300000 100000')
        assert cpu_threads.limit_threads(cpu_max=small) == 3
        assert torch.get_num_threads() == 3
        large = quota(tmp_path, '1360000 100000')
        assert cpu_threads.limit_threads(cpu_max=large) == 8
        assert cpu_threads.limit_threads(limit=4, cpu_max=large) == 4
    finally:
        torch.set_num_threads(before)


def test_omp_num_threads_decides_when_set(tmp_path, monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '2')
    before = torch.get_num_threads()
    try:
        small = quota(tmp_path, '300000 100000')
        assert cpu_threads.limit_threads(cpu_max=small) == before
        assert torch.get_num_threads() == before  # left alone
    finally:
        torch.set_num_threads(before)
