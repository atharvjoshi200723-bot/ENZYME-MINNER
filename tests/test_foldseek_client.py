"""Offline tests for src/foldseek_client.py (no network access)."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from src.foldseek_client import (
    FoldseekClient,
    FoldseekConfigError,
    FoldseekError,
    FoldseekSettings,
    FoldseekTimeout,
)

DATABASES_PAYLOAD = {
    "databases": [
        {"name": "AlphaFold/UniProt50", "path": "afdb50", "version": "v6"},
        {"name": "MGnify-ESM30", "path": "mgnify_esm30", "version": "v1"},
    ]
}


def hit(target: str, **overrides) -> dict:
    base = {
        "query": "job_A",
        "target": target,
        "seqId": 66.2,
        "alnLength": 261,
        "qStartPos": 3,
        "qEndPos": 257,
        "dbStartPos": 34,
        "dbEndPos": 294,
        "prob": 1.0,
        "eval": 3.4e-43,
        "score": 1649,
        "qLen": 263,
        "dbLen": 298,
        "qAln": "ACDE",
        "dbAln": "AC-E",
        "taxId": 413882,
        "taxName": "Caldimonas brevitalea",
    }
    base.update(overrides)
    return base


def result_payload(hits: list[dict]) -> dict:
    return {
        "type": "structuresearch",
        "queries": [{"header": "job_A SEED", "sequence": "ACDEFG"}],
        "mode": "3diaa",
        "results": [{"db": "afdb50", "alignments": [hits], "taxonomyreports": []}],
    }


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, content=b"", text="",
                 headers=None):
        self.status_code = status_code
        self._json = json_data
        self.content = content
        self.text = text or (json.dumps(json_data) if json_data is not None else "")
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeSession:
    """Replays queued responses and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method, url, timeout=None, **kwargs):
        self.calls.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError(f"unexpected request {method} {url}")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_settings(tmp_path: Path, **overrides) -> FoldseekSettings:
    defaults = dict(
        server_url="https://search.foldseek.test",
        submit_path="/api/ticket",
        ticket_path="/api/ticket/{ticket_id}",
        result_path="/api/result/{ticket_id}/{db_index}",
        databases_path="/api/databases",
        cache_dir=tmp_path / "cache",
        poll_interval_sec=0.0,
        max_poll_attempts=5,
        timeout_sec=5,
        retries=2,
        min_request_interval_sec=0.0,
    )
    defaults.update(overrides)
    return FoldseekSettings(**defaults)


def make_client(tmp_path: Path, responses, **overrides):
    session = FakeSession(responses)
    client = FoldseekClient(make_settings(tmp_path, **overrides), session=session,
                            sleep=lambda _s: None)
    return client, session


@pytest.fixture
def seed_file(tmp_path):
    path = tmp_path / "seed.pdb"
    path.write_text("ATOM ...\n")
    return path


def test_search_submits_polls_and_parses_json(tmp_path, seed_file):
    payload = result_payload([
        hit("AF-A0A0G3BI90-F1-model_v6 Triacylglycerol lipase"),
        hit("AF-A0A1F4JXW8-F1-model_v6 PET hydrolase", prob=0.9, eval=1e-20),
    ])
    client, session = make_client(tmp_path, [
        FakeResponse(json_data=DATABASES_PAYLOAD),
        FakeResponse(json_data={"id": "t1", "status": "PENDING"}),
        FakeResponse(json_data={"id": "t1", "status": "RUNNING"}),
        FakeResponse(json_data={"id": "t1", "status": "COMPLETE"}),
        FakeResponse(content=json.dumps(payload).encode()),
    ])

    df = client.search(seed_file, ["afdb50"], mode="3diaa")

    assert list(df.columns) == [
        "query", "target_id", "description", "database", "probability",
        "tm_score", "evalue", "score", "seq_identity", "aln_length",
        "qstart", "qend", "tstart", "tend", "qlen", "tlen", "qaln", "taln",
        "taxid", "taxname",
    ]
    # Sorted by probability descending.
    assert df.loc[0, "target_id"] == "AF-A0A0G3BI90-F1-model_v6"
    assert df.loc[0, "description"] == "Triacylglycerol lipase"
    assert df.loc[0, "taxname"] == "Caldimonas brevitalea"
    assert df.loc[0, "tstart"] == 34
    assert df.loc[0, "taln"] == "AC-E"

    urls = [call[1] for call in session.calls]
    assert urls[1] == "https://search.foldseek.test/api/ticket"
    assert urls[-1].endswith("/api/result/t1/0")
    # The seed file was posted as multipart field "q".
    assert "files" in session.calls[1][2]
    assert session.calls[1][2]["data"]["database[]"] == "afdb50"


def test_search_uses_disk_cache_on_second_run(tmp_path, seed_file):
    payload = result_payload([hit("AF-X-F1-model_v6 Something")])
    client, _ = make_client(tmp_path, [
        FakeResponse(json_data=DATABASES_PAYLOAD),
        FakeResponse(json_data={"id": "t1"}),
        FakeResponse(json_data={"status": "COMPLETE"}),
        FakeResponse(content=json.dumps(payload).encode()),
    ])
    first = client.search(seed_file, ["afdb50"])

    # Fresh client, fresh session, same cache dir: no HTTP call is allowed.
    idle = FakeSession([])
    second_client = FoldseekClient(make_settings(tmp_path), session=idle,
                                   sleep=lambda _s: None)
    second = second_client.search(seed_file, ["afdb50"])

    pd.testing.assert_frame_equal(first, second)
    assert idle.calls == []


def test_unknown_database_fails_with_current_names(tmp_path, seed_file):
    client, _ = make_client(tmp_path, [FakeResponse(json_data=DATABASES_PAYLOAD)])
    with pytest.raises(FoldseekConfigError) as err:
        client.search(seed_file, ["afdb_proteome_typo"])
    message = str(err.value)
    assert "afdb50" in message and "mgnify_esm30" in message
    # Validation happens before submitting anything.
    assert all(call[0] != "POST" for call in client.session.calls)


def test_unsupported_mode_is_rejected(tmp_path, seed_file):
    client, _ = make_client(tmp_path, [FakeResponse(json_data=DATABASES_PAYLOAD)])
    with pytest.raises(FoldseekConfigError, match="3diaa"):
        client.search(seed_file, ["afdb50"], mode="fast")


def test_job_error_state_raises(tmp_path):
    client, _ = make_client(tmp_path, [
        FakeResponse(json_data={"status": "ERROR", "message": "bad structure"}),
    ])
    with pytest.raises(FoldseekError, match="failed server-side"):
        client.wait("t1")


def test_polling_timeout_raises(tmp_path):
    client, _ = make_client(tmp_path, [
        FakeResponse(json_data={"status": "RUNNING"}),
    ] * 3, max_poll_attempts=3)
    with pytest.raises(FoldseekTimeout, match="did not finish"):
        client.wait("t1")


def test_server_error_is_retried(tmp_path):
    client, session = make_client(tmp_path, [
        FakeResponse(status_code=503, text="busy"),
        FakeResponse(json_data=DATABASES_PAYLOAD),
    ])
    databases = client.list_databases()
    assert databases[0]["path"] == "afdb50"
    assert len(session.calls) == 2


def test_tar_archive_m8_fallback(tmp_path):
    m8 = "\t".join(["job_A", "AF-P123-F1-model_v6 Lipase", "66.2", "261",
                    "82", "2", "3", "257", "34", "294", "3.4e-43", "1649"]) + "\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        data = m8.encode()
        info = tarfile.TarInfo("result.m8")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    client, _ = make_client(tmp_path, [])
    df = client._parse_result_payload(buffer.getvalue(), "afdb50")
    assert len(df) == 1
    assert df.loc[0, "target_id"] == "AF-P123-F1-model_v6"
    assert df.loc[0, "tstart"] == 34
    assert df.loc[0, "qaln"] is None


def test_unreadable_payload_raises(tmp_path):
    client, _ = make_client(tmp_path, [])
    with pytest.raises(FoldseekError, match="neither JSON"):
        client._parse_result_payload(b"not a result", "afdb50")


def test_max_hits_trims_after_sorting(tmp_path, seed_file):
    payload = result_payload([
        hit("AF-LOW-F1-model_v6 low", prob=0.5),
        hit("AF-HIGH-F1-model_v6 high", prob=0.99),
    ])
    client, _ = make_client(tmp_path, [
        FakeResponse(json_data=DATABASES_PAYLOAD),
        FakeResponse(json_data={"id": "t1"}),
        FakeResponse(json_data={"status": "COMPLETE"}),
        FakeResponse(content=json.dumps(payload).encode()),
    ])
    df = client.search(seed_file, ["afdb50"], max_hits=1)
    assert len(df) == 1
    assert df.loc[0, "target_id"] == "AF-HIGH-F1-model_v6"


def test_databases_cache_is_written(tmp_path):
    client, _ = make_client(tmp_path, [FakeResponse(json_data=DATABASES_PAYLOAD)])
    client.list_databases()
    cache = tmp_path / "cache" / "databases.json"
    assert cache.exists()
    assert json.loads(cache.read_text())[0]["path"] == "afdb50"
