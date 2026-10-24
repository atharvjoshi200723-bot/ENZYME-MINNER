"""
foldseek_client.py
------------------
Thin client for the public Foldseek web server (search.foldseek.com).

Verified against the live service on 2026-10-24
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* POST /api/ticket                      multipart: q=<file>, database[]=<name>, mode=<3diaa|tmalign>
* GET  /api/ticket/{id}                  -> {"id": ..., "status": "PENDING|RUNNING|COMPLETE|ERROR"}
* GET  /api/result/{id}/{db_index}       -> JSON structure-search payload (text/plain content type)
* GET  /api/databases                    -> {"databases": [{"name": ..., "path": ...}, ...]}

The result endpoint returns JSON for the current server, not the tar.gz
archive older docs describe.  `_parse_result_payload` still understands a
tar/gzip archive of m8 files, so an older deployment or a future switch back
does not break the pipeline.

Database names are data, not assumptions: they are passed in from
config.yaml and validated against /api/databases.  A wrong name fails with
the list of currently valid names instead of a silent empty result.

Honest limits
~~~~~~~~~~~~~
* The public server is shared and unauthenticated; searches can queue.
  We cache raw responses on disk so reruns and the demo do not re-query.
* This client does not upload more than one structure per request.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Bump when the parser output changes so old cache entries are ignored.
CACHE_FORMAT_VERSION = 1

# Extra columns present in the JSON payload that are worth keeping.
HIT_COLUMNS = [
    "query",
    "target_id",
    "description",
    "database",
    "probability",
    "tm_score",
    "evalue",
    "score",
    "seq_identity",
    "aln_length",
    "qstart",
    "qend",
    "tstart",
    "tend",
    "qlen",
    "tlen",
    "qaln",
    "taln",
    "taxid",
    "taxname",
]

# Job states reported by the server.
PENDING_STATES = {"PENDING", "RUNNING", "UNKNOWN"}
COMPLETE_STATE = "COMPLETE"
FAILED_STATE = "ERROR"


class FoldseekError(RuntimeError):
    """Any failure while talking to Foldseek."""


class FoldseekConfigError(FoldseekError):
    """A database name or mode in config.yaml is not accepted by the server."""


class FoldseekTimeout(FoldseekError):
    """The job did not finish within the configured poll budget."""


@dataclass
class FoldseekSettings:
    """Everything the client needs, normally built from config.yaml."""

    server_url: str
    submit_path: str
    ticket_path: str
    result_path: str
    databases_path: str
    cache_dir: Path
    poll_interval_sec: float = 10.0
    max_poll_attempts: int = 60
    timeout_sec: float = 60.0
    retries: int = 3
    min_request_interval_sec: float = 1.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FoldseekSettings":
        fs = config["foldseek"]
        ep = fs["endpoints"]
        return cls(
            server_url=fs["server_url"].rstrip("/"),
            submit_path=ep["submit"],
            ticket_path=ep["ticket"],
            result_path=ep["result"],
            databases_path=ep["databases"],
            cache_dir=Path(fs["cache_dir"]),
            poll_interval_sec=float(fs["poll_interval_sec"]),
            max_poll_attempts=int(fs["max_poll_attempts"]),
            timeout_sec=float(fs.get("request_timeout_sec", 60)),
            retries=int(fs.get("retries", 3)),
            min_request_interval_sec=float(fs.get("min_request_interval_sec", 1.0)),
        )


class FoldseekClient:
    """Submit a structure, poll the job, download and parse the hits."""

    def __init__(
        self,
        settings: FoldseekSettings,
        session: Optional[requests.Session] = None,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: Optional[float] = None
        self._databases_cache: Optional[list[dict[str, Any]]] = None
        settings.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- HTTP plumbing ----------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.settings.server_url}{path}"

    def _throttle(self) -> None:
        """Keep at least min_request_interval_sec between requests."""
        interval = self.settings.min_request_interval_sec
        if interval <= 0 or self._last_request_at is None:
            return
        wait = interval - (self._clock() - self._last_request_at)
        if wait > 0:
            self._sleep(wait)

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """HTTP request with throttle, retries and useful error messages."""
        url = self._url(path)
        last_error: Optional[Exception] = None

        for attempt in range(self.settings.retries + 1):
            self._throttle()
            try:
                response = self.session.request(
                    method, url, timeout=self.settings.timeout_sec, **kwargs
                )
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("Foldseek %s %s failed (%s); retrying", method, url, exc)
            else:
                self._last_request_at = self._clock()

                if response.status_code == 429 or response.status_code >= 500:
                    last_error = FoldseekError(
                        f"{method} {url} -> HTTP {response.status_code}"
                    )
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        pause = min(float(retry_after), 30.0)
                    else:
                        pause = min(2.0 ** attempt, 30.0)
                    if attempt < self.settings.retries:
                        logger.warning(
                            "%s; retrying in %.0fs (attempt %d/%d)",
                            last_error, pause, attempt + 1, self.settings.retries,
                        )
                        self._sleep(pause)
                    continue

                if response.status_code >= 400:
                    # Client error (bad ticket, bad index, bad form field):
                    # retrying will not help, so surface the server message.
                    raise FoldseekError(
                        f"{method} {url} -> HTTP {response.status_code}: "
                        f"{response.text[:300]}"
                    )
                return response

            if attempt < self.settings.retries:
                self._sleep(min(2.0 ** attempt, 30.0))

        raise FoldseekError(f"{method} {url} failed after retries: {last_error}")

    # -- API steps --------------------------------------------------------

    def list_databases(self, use_cache: bool = True) -> list[dict[str, Any]]:
        """
        Return the databases the server currently advertises.

        Cached on disk for the life of a session to avoid a request per run.
        """
        if self._databases_cache is not None:
            return self._databases_cache

        cache_file = self.settings.cache_dir / "databases.json"
        if use_cache and cache_file.exists():
            try:
                self._databases_cache = json.loads(cache_file.read_text())
                return self._databases_cache
            except (OSError, json.JSONDecodeError):
                logger.warning("Ignoring unreadable database cache %s", cache_file)

        response = self._request("GET", self.settings.databases_path)
        payload = response.json()
        databases = payload.get("databases")
        if not isinstance(databases, list) or not databases:
            raise FoldseekError(
                "Unexpected /api/databases response; cannot validate database names."
            )
        self._databases_cache = databases
        cache_file.write_text(json.dumps(databases, indent=2))
        return databases

    def validate_databases(self, names: Iterable[str]) -> None:
        """Fail clearly if config.yaml names a database the server does not have."""
        wanted = list(names)
        if not wanted:
            raise FoldseekConfigError("No Foldseek databases configured.")

        available = self.list_databases()
        valid = {db.get("path") for db in available} | {db.get("name") for db in available}
        unknown = [name for name in wanted if name not in valid]
        if unknown:
            listing = ", ".join(sorted(str(v) for v in valid if v))
            raise FoldseekConfigError(
                f"Unknown Foldseek database(s) {unknown}. "
                f"The server currently offers: {listing}. "
                "Update foldseek.databases in config.yaml."
            )

    def submit(self, structure_path: Path, database: str, mode: str = "3diaa") -> str:
        """POST the structure and return a ticket id."""
        if mode not in {"3diaa", "tmalign"}:
            raise FoldseekConfigError(
                f"Unsupported Foldseek mode {mode!r}; expected '3diaa' or 'tmalign'."
            )
        if not structure_path.exists():
            raise FileNotFoundError(f"Structure file not found: {structure_path}")

        with structure_path.open("rb") as handle:
            files = {"q": (structure_path.name, handle)}
            data = {"database[]": database, "mode": mode}
            response = self._request("POST", self.settings.submit_path,
                                     files=files, data=data)

        try:
            payload = response.json()
        except ValueError as exc:
            raise FoldseekError(
                "Foldseek submit returned non-JSON; the endpoint may have changed."
            ) from exc

        ticket = payload.get("id")
        if not ticket:
            raise FoldseekError(f"Foldseek submit response has no ticket id: {payload}")
        logger.info("Submitted %s to %s, ticket %s", structure_path.name, database, ticket)
        return str(ticket)

    def wait(self, ticket: str) -> None:
        """Poll until the job completes, raising on ERROR or timeout."""
        for attempt in range(self.settings.max_poll_attempts):
            response = self._request(
                "GET", self.settings.ticket_path.format(ticket_id=ticket)
            )
            try:
                status = response.json().get("status", "UNKNOWN")
            except ValueError as exc:
                raise FoldseekError("Foldseek ticket endpoint returned non-JSON.") from exc

            if status == COMPLETE_STATE:
                logger.info("Ticket %s complete after %d polls", ticket, attempt + 1)
                return
            if status == FAILED_STATE:
                raise FoldseekError(
                    f"Foldseek job {ticket} failed server-side: {response.text[:300]}"
                )

            logger.debug("Ticket %s status %s (%d/%d)",
                         ticket, status, attempt + 1, self.settings.max_poll_attempts)
            self._sleep(self.settings.poll_interval_sec)

        raise FoldseekTimeout(
            f"Foldseek job {ticket} did not finish after "
            f"{self.settings.max_poll_attempts} polls "
            f"({self.settings.max_poll_attempts * self.settings.poll_interval_sec:.0f}s)."
        )

    def fetch_result(self, ticket: str, db_index: int) -> bytes:
        """Download the raw result body (JSON in the current server)."""
        response = self._request(
            "GET", self.settings.result_path.format(ticket_id=ticket, db_index=db_index)
        )
        return response.content

    # -- caching ----------------------------------------------------------

    @staticmethod
    def _cache_key(structure_bytes: bytes, database: str, mode: str) -> str:
        digest = hashlib.sha256()
        digest.update(structure_bytes)
        digest.update(f"|{database}|{mode}|v{CACHE_FORMAT_VERSION}".encode())
        return digest.hexdigest()[:24]

    def _cache_file(self, key: str, database: str) -> Path:
        safe_db = "".join(c if c.isalnum() else "_" for c in database)
        return self.settings.cache_dir / f"{key}_{safe_db}.json"

    # -- parsing ----------------------------------------------------------

    @staticmethod
    def _split_target(target: str) -> tuple[str, str]:
        """'AF-P12345-F1-model_v6 Lipase' -> ('AF-P12345-F1-model_v6', 'Lipase')."""
        parts = target.split(" ", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    def _parse_json_payload(self, payload: dict[str, Any], database: str) -> pd.DataFrame:
        """Convert the structure-search JSON into one row per alignment."""
        queries = payload.get("queries", [])
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise FoldseekError("Unexpected Foldseek result JSON: 'results' is not a list.")

        rows: list[dict[str, Any]] = []
        for result in results:
            result_db = result.get("db", database)
            if database and result_db not in (database, None):
                continue
            per_query = result.get("alignments", [])
            for query_index, hits in enumerate(per_query):
                query_id = "query_0"
                if query_index < len(queries):
                    header = str(queries[query_index].get("header", "query_0"))
                    # "job_A CRYSTAL STRUCTURE ..." -> "job_A"
                    query_id = header.split(" ", 1)[0]
                for hit in hits:
                    target_id, description = self._split_target(
                        str(hit.get("target", ""))
                    )
                    rows.append({
                        "query": query_id,
                        "target_id": target_id,
                        "description": description,
                        "database": result_db,
                        "probability": hit.get("prob"),
                        "tm_score": hit.get("alntmscore"),
                        "evalue": hit.get("eval"),
                        "score": hit.get("score"),
                        "seq_identity": hit.get("seqId"),
                        "aln_length": hit.get("alnLength"),
                        "qstart": hit.get("qStartPos"),
                        "qend": hit.get("qEndPos"),
                        "tstart": hit.get("dbStartPos"),
                        "tend": hit.get("dbEndPos"),
                        "qlen": hit.get("qLen"),
                        "tlen": hit.get("dbLen"),
                        "qaln": hit.get("qAln"),
                        "taln": hit.get("dbAln"),
                        "taxid": hit.get("taxId"),
                        "taxname": hit.get("taxName"),
                    })

        return pd.DataFrame(rows, columns=HIT_COLUMNS)

    @staticmethod
    def _parse_m8_text(text: str, database: str) -> pd.DataFrame:
        """Parse Foldseek m8 text (12+ tab fields) as a fallback."""
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                raise FoldseekError(
                    "Result archive does not look like Foldseek m8 (fewer than "
                    f"12 tab-separated fields): {line[:120]!r}"
                )
            target_id, description = FoldseekClient._split_target(fields[1])
            rows.append({
                "query": fields[0],
                "target_id": target_id,
                "description": description,
                "database": database,
                "probability": None,
                "tm_score": None,
                "evalue": float(fields[10]),
                "score": float(fields[11]),
                "seq_identity": float(fields[2]),
                "aln_length": int(float(fields[3])),
                "qstart": int(float(fields[6])),
                "qend": int(float(fields[7])),
                "tstart": int(float(fields[8])),
                "tend": int(float(fields[9])),
                "qlen": None,
                "tlen": None,
                "qaln": None,
                "taln": None,
                "taxid": None,
                "taxname": None,
            })
        return pd.DataFrame(rows, columns=HIT_COLUMNS)

    def _parse_result_payload(self, raw: bytes, database: str) -> pd.DataFrame:
        """
        Parse whatever the result endpoint returned.

        Current server: JSON.  Older/alternate deployments: tar.gz of m8 files.
        Anything else raises instead of producing an empty shortlist.
        """
        stripped = raw.lstrip()

        if stripped.startswith(b"{"):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise FoldseekError("Foldseek returned malformed JSON.") from exc
            return self._parse_json_payload(payload, database)

        # Archive fallback: plain gzip or a tar archive of text files.
        try:
            if tarfile.is_tarfile(io.BytesIO(raw)):
                frames = []
                with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as archive:
                    for member in archive.getmembers():
                        if not member.isfile():
                            continue
                        handle = archive.extractfile(member)
                        if handle is None:
                            continue
                        text = handle.read().decode("utf-8", errors="replace")
                        frames.append(self._parse_m8_text(text, database))
                if not frames:
                    raise FoldseekError("Foldseek archive contained no result files.")
                return pd.concat(frames, ignore_index=True)

            text = gzip.decompress(raw).decode("utf-8", errors="replace")
            return self._parse_m8_text(text, database)
        except (OSError, tarfile.TarError) as exc:
            raise FoldseekError(
                "Foldseek result was neither JSON nor a readable tar/gzip archive."
            ) from exc

    # -- high level -------------------------------------------------------

    def search(
        self,
        structure_path: Path,
        databases: Iterable[str],
        mode: str = "3diaa",
        max_hits: Optional[int] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Run one Foldseek search per database and return the combined hits.

        Results are cached under foldseek.cache_dir keyed by the structure
        bytes, database and mode, so reruns cost no network calls.
        """
        databases = list(databases)
        self.validate_databases(databases)

        structure_path = Path(structure_path)
        structure_bytes = structure_path.read_bytes()
        frames: list[pd.DataFrame] = []

        for database in databases:
            key = self._cache_key(structure_bytes, database, mode)
            cache_file = self._cache_file(key, database)

            if use_cache and cache_file.exists():
                logger.info("Using cached Foldseek result %s", cache_file.name)
                raw = cache_file.read_bytes()
            else:
                ticket = self.submit(structure_path, database, mode)
                self.wait(ticket)
                raw = self.fetch_result(ticket, 0)
                cache_file.write_bytes(raw)
                logger.info("Cached Foldseek result at %s", cache_file)

            frame = self._parse_result_payload(raw, database)
            logger.info("%s: %d alignments", database, len(frame))
            frames.append(frame)

        if not frames:
            raise FoldseekError("No databases requested.")

        combined = pd.concat(frames, ignore_index=True)
        # Prefer probability where present, then e-value.
        combined = combined.sort_values(
            ["probability", "evalue"], ascending=[False, True], na_position="last"
        ).reset_index(drop=True)

        if max_hits is not None:
            combined = combined.head(max_hits).reset_index(drop=True)

        # `prob` is empty in tmalign mode but `tm_score` may be present.
        if combined["probability"].isna().all() and "tm_score" in combined:
            combined["probability"] = combined["tm_score"]

        return combined
