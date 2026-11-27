"""Ticket capacity protects both open episodes and results currently being scored."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from fjallstation import score, server


@pytest.fixture
def station(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MAX_TICKETS", 3)
    spec = {"msl": "4.1.0", "libraries": [], "hidden_files": [],
            "eval": {"start_time": 0, "stop_time": 10, "interval": 5, "outputs": ["T"]}}
    task = {"id": "room", "prompt": "Build a room.", "model_name": "Room", "spec": spec,
            "reference_outputs": {"T": [20, 21, 22]}}
    return server.Station({"room": task}, tmp_path / "rewards")


def result_for(station, ticket):
    return {"ticket": ticket, "image": station.image, "status": "simulated", "values": {"T": [20, 21, 22]}}


def test_cleanup_removes_the_oldest_completed_record_and_keeps_running_work(station):
    running, second, third = [station.sample()["ticket"] for _ in range(3)]
    station.result(result_for(station, third))
    station.result(result_for(station, second))
    new = station.sample()["ticket"]
    assert set(station.tickets) == {running, second, new}          # completion order, not creation order
    assert station.result(result_for(station, running))["reward"] == 1.0
    assert len((station.data / "rewards.jsonl").read_text().splitlines()) == 3


def test_a_result_being_scored_cannot_be_evicted(station, monkeypatch):
    running, finished, other = [station.sample()["ticket"] for _ in range(3)]
    station.result(result_for(station, finished))
    started, release = threading.Event(), threading.Event()
    real = score.score

    def slow_score(*args):
        started.set()
        assert release.wait(timeout=10)
        return real(*args)

    monkeypatch.setattr(score, "score", slow_score)
    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(station.result, result_for(station, running))
        try:
            assert started.wait(timeout=10)
            new = station.sample()["ticket"]
            assert station.tickets[running]["state"] == "scoring"
            assert set(station.tickets) == {running, other, new}
            with pytest.raises(server.StationFull):
                station.sample()
        finally:
            release.set()
        assert future.result(timeout=10)["reward"] == 1.0
    assert station.tickets[running]["state"] == "done"
    assert station.result(result_for(station, running))["reward"] == 1.0


def test_failed_scoring_keeps_the_ticket_active_until_a_successful_retry(station, monkeypatch):
    monkeypatch.setattr(server, "MAX_TICKETS", 1)
    ticket = station.sample()["ticket"]

    def failed_score(*args):
        raise OSError("scorer unavailable")

    with monkeypatch.context() as broken:
        broken.setattr(score, "score", failed_score)
        with pytest.raises(OSError):
            station.result(result_for(station, ticket))
    assert station.tickets[ticket]["state"] == "open"
    with pytest.raises(server.StationFull):
        station.sample()
    assert station.result(result_for(station, ticket))["reward"] == 1.0
    new = station.sample()["ticket"]
    assert list(station.tickets) == [new]


def test_duplicate_delivery_does_not_create_another_cleanup_entry(station, monkeypatch):
    monkeypatch.setattr(server, "MAX_TICKETS", 2)
    completed = station.sample()["ticket"]
    result = result_for(station, completed)
    first = station.result(result)
    assert station.result(result) == station.result(result) == first
    running = station.sample()["ticket"]
    new = station.sample()["ticket"]
    with pytest.raises(server.StationFull):
        station.sample()
    assert set(station.tickets) == {running, new}
    with pytest.raises(KeyError):
        station.result(result)                                  # cached reply was removed; reward log remains
    assert len((station.data / "rewards.jsonl").read_text().splitlines()) == 1


def test_parallel_requests_cannot_take_the_same_last_slot(station):
    existing = {station.sample()["ticket"] for _ in range(2)}
    start = threading.Barrier(8)

    def request():
        start.wait(timeout=10)
        try:
            return station.sample()["ticket"]
        except server.StationFull:
            return None

    with ThreadPoolExecutor(max_workers=8) as workers:
        answers = list(workers.map(lambda _: request(), range(8)))
    accepted = {answer for answer in answers if answer is not None}
    assert len(accepted) == 1 and answers.count(None) == 7
    assert set(station.tickets) == existing | accepted
