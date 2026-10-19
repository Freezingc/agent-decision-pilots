"""Local HTTP/SQLite service with controlled, reproducible fault injection.

SQLite writes, cache entries, operation-journal entries, observations, and pending
events are durable. A separate process serves loopback HTTP requests, each in a
SQLite transaction. Logical ticks advance on actions, rather than elapsed time:
this injects controlled visibility/write delays, not real network latency, and
provides no evidence of production tool reliability. Scenario configuration and
the audit interface belong to the runner; they must not be exposed to a policy.

The service is a local research harness, not an authentication boundary. Reading
the database or invoking audit would reveal hidden state. Opaque random IDs do
not encode scenario labels. Callers using multiprocessing spawn from a script
must use the usual ``if __name__ == '__main__'`` guard.
"""
from __future__ import annotations

from contextlib import closing
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import urllib.error
import urllib.request
import uuid

_ACTIONS = {"cache", "journal", "fields", "wait", "retry", "repair"}
_CONFIG_KEYS = {"template", "expected_fields", "outcome", "apply_delay", "cache_lag",
                "journal_lag", "journal_mode", "idempotent"}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
 episode_id TEXT PRIMARY KEY, operation_id TEXT NOT NULL UNIQUE,
 config_json TEXT NOT NULL, tick INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS records (
 record_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, operation_id TEXT NOT NULL,
 idempotency_key TEXT, fields_json TEXT NOT NULL,
 UNIQUE(episode_id, idempotency_key)
);
CREATE TABLE IF NOT EXISTS cache_entries (
 episode_id TEXT PRIMARY KEY, record_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS operation_journal (
 episode_id TEXT PRIMARY KEY, state TEXT NOT NULL, record_id TEXT
);
CREATE TABLE IF NOT EXISTS observed_records (
 episode_id TEXT NOT NULL, record_id TEXT NOT NULL,
 PRIMARY KEY(episode_id, record_id)
);
CREATE TABLE IF NOT EXISTS scheduled_events (
 event_id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
 due_tick INTEGER NOT NULL, kind TEXT NOT NULL, payload_json TEXT NOT NULL,
 done INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS event_log (
 event_id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
 tick INTEGER NOT NULL, kind TEXT NOT NULL, record_id TEXT, writer_pid INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS due_events ON scheduled_events(episode_id, done, due_tick, event_id);
"""


def _validate_config(config):
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise ValueError("config must contain exactly the documented fields")
    if not isinstance(config["template"], str):
        raise ValueError("template must be a string")
    fields = config["expected_fields"]
    if not isinstance(fields, dict) or len(fields) < 2 or any(not isinstance(k, str) for k in fields):
        raise ValueError("expected_fields needs at least two string keys")
    if config["outcome"] not in ("none", "full", "partial"):
        raise ValueError("invalid outcome")
    if config["journal_mode"] not in ("atomic", "lagged", "unavailable"):
        raise ValueError("invalid journal_mode")
    if not isinstance(config["idempotent"], bool):
        raise ValueError("idempotent must be boolean")
    for key in ("apply_delay", "cache_lag", "journal_lag"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(key + " must be a nonnegative integer")
    # JSON is the durable wire/storage format; reject non-finite JSON values.
    json.dumps(config, allow_nan=False)


def _episode(con, episode_id):
    row = con.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
    if row is None:
        raise ValueError("unknown episode")
    return row, json.loads(row["config_json"])


def _log(con, episode_id, tick, kind, record_id=None):
    con.execute("INSERT INTO event_log(episode_id,tick,kind,record_id,writer_pid) VALUES(?,?,?,?,?)",
                (episode_id, tick, kind, record_id, os.getpid()))


def _schedule(con, episode_id, tick, kind, record_id=None):
    con.execute("INSERT INTO scheduled_events(episode_id,due_tick,kind,payload_json) VALUES(?,?,?,?)",
                (episode_id, tick, kind, json.dumps({"record_id": record_id})))


def _is_complete(fields, expected):
    return all(key in fields and fields[key] == value for key, value in expected.items())


def _journal_write(con, episode_id, record_id, config):
    fields = json.loads(con.execute("SELECT fields_json FROM records WHERE record_id=? AND episode_id=?",
                                   (record_id, episode_id)).fetchone()[0])
    state = "complete" if _is_complete(fields, config["expected_fields"]) else "partial"
    con.execute("INSERT INTO operation_journal(episode_id,state,record_id) VALUES(?,?,?) "
                "ON CONFLICT(episode_id) DO UPDATE SET state=excluded.state,record_id=excluded.record_id",
                (episode_id, state, record_id))


def _publish_after_write(con, episode_id, record_id, tick, config):
    _schedule(con, episode_id, tick + config["cache_lag"], "publish_cache", record_id)
    if config["journal_mode"] == "atomic":
        _journal_write(con, episode_id, record_id, config)
        _log(con, episode_id, tick, "publish_journal", record_id)
    elif config["journal_mode"] == "lagged":
        _schedule(con, episode_id, tick + config["journal_lag"], "publish_journal", record_id)


def _write(con, episode, config, tick, fields, kind):
    eid, operation_id = episode["episode_id"], episode["operation_id"]
    key = operation_id if config["idempotent"] else None
    existing = None if key is None else con.execute(
        "SELECT record_id,fields_json FROM records WHERE episode_id=? AND idempotency_key=?", (eid, key)).fetchone()
    if existing:
        record_id = existing["record_id"]
        merged = json.loads(existing["fields_json"])
        merged.update(fields)
        con.execute("UPDATE records SET fields_json=? WHERE record_id=?", (json.dumps(merged), record_id))
    else:
        record_id = uuid.uuid4().hex
        con.execute("INSERT INTO records(record_id,episode_id,operation_id,idempotency_key,fields_json) VALUES(?,?,?,?,?)",
                    (record_id, eid, operation_id, key, json.dumps(fields)))
    _log(con, eid, tick, kind, record_id)
    _publish_after_write(con, eid, record_id, tick, config)
    return record_id


def _due(con, episode_id, tick):
    episode, config = _episode(con, episode_id)
    while True:
        event = con.execute("SELECT * FROM scheduled_events WHERE episode_id=? AND done=0 AND due_tick<=? "
                            "ORDER BY due_tick,event_id LIMIT 1", (episode_id, tick)).fetchone()
        if event is None:
            return
        due_tick, kind = event["due_tick"], event["kind"]
        record_id = json.loads(event["payload_json"])["record_id"]
        con.execute("UPDATE scheduled_events SET done=1 WHERE event_id=?", (event["event_id"],))
        if kind == "apply_original":
            fields = config["expected_fields"]
            if config["outcome"] == "partial":
                first = next(iter(fields))
                fields = {first: fields[first]}
            _write(con, episode, config, due_tick, fields, "apply_original")
        elif kind == "publish_cache":
            con.execute("INSERT INTO cache_entries(episode_id,record_id) VALUES(?,?) "
                        "ON CONFLICT(episode_id) DO UPDATE SET record_id=excluded.record_id", (episode_id, record_id))
            _log(con, episode_id, due_tick, kind, record_id)
        elif kind == "publish_journal":
            # Read the real row now: a repair/full idempotent retry must not be
            # downgraded by an older scheduled partial-journal publication.
            _journal_write(con, episode_id, record_id, config)
            _log(con, episode_id, due_tick, kind, record_id)
        else:
            raise RuntimeError("invalid persisted event")


def _observe(con, episode_id, record_id):
    con.execute("INSERT OR IGNORE INTO observed_records(episode_id,record_id) VALUES(?,?)", (episode_id, record_id))


def _start(con, config):
    _validate_config(config)
    episode_id, operation_id = uuid.uuid4().hex, uuid.uuid4().hex
    con.execute("INSERT INTO episodes(episode_id,operation_id,config_json) VALUES(?,?,?)",
                (episode_id, operation_id, json.dumps(config, allow_nan=False)))
    _log(con, episode_id, 0, "start")
    if config["journal_mode"] == "atomic":
        state = "not_found" if config["outcome"] == "none" else "pending"
        con.execute("INSERT INTO operation_journal(episode_id,state) VALUES(?,?)", (episode_id, state))
    if config["outcome"] != "none":
        _schedule(con, episode_id, config["apply_delay"], "apply_original")
    _due(con, episode_id, 0)
    return {"episode_id": episode_id, "operation_id": operation_id,
            "template": config["template"], "expected_fields": config["expected_fields"],
            "journal_mode": config["journal_mode"], "idempotent": config["idempotent"], "status": "timeout"}


def _action(con, episode_id, action, record_id):
    if action not in _ACTIONS:
        raise ValueError("unsupported action")
    episode, config = _episode(con, episode_id)
    tick = episode["tick"] + 1
    con.execute("UPDATE episodes SET tick=? WHERE episode_id=?", (tick, episode_id))
    _due(con, episode_id, tick)
    _log(con, episode_id, tick, "action_" + action)
    if action == "wait":
        return {"state": "waited"}
    if action == "cache":
        row = con.execute("SELECT record_id FROM cache_entries WHERE episode_id=?", (episode_id,)).fetchone()
        if row is None:
            return {"state": "absent"}
        _observe(con, episode_id, row["record_id"])
        return {"state": "present", "record_id": row["record_id"]}
    if action == "journal":
        if config["journal_mode"] == "unavailable":
            return {"state": "unavailable"}
        row = con.execute("SELECT state,record_id FROM operation_journal WHERE episode_id=?", (episode_id,)).fetchone()
        if row is None:
            return {"state": "unknown"}
        result = {"state": row["state"]}
        if row["record_id"] is not None:
            _observe(con, episode_id, row["record_id"])
            result["record_id"] = row["record_id"]
        return result
    if action == "fields":
        rows = con.execute("SELECT record_id,fields_json FROM records WHERE episode_id=? ORDER BY rowid", (episode_id,)).fetchall()
        for row in rows:
            _observe(con, episode_id, row["record_id"])
        return {"records": [{"record_id": row["record_id"], "fields": json.loads(row["fields_json"])} for row in rows]}
    if action == "retry":
        record_id = _write(con, episode, config, tick, config["expected_fields"], "retry")
    elif action == "repair":
        if not isinstance(record_id, str) or con.execute(
                "SELECT 1 FROM observed_records WHERE episode_id=? AND record_id=?", (episode_id, record_id)).fetchone() is None:
            raise ValueError("repair requires a record_id observed in this episode")
        row = con.execute("SELECT fields_json FROM records WHERE episode_id=? AND record_id=?", (episode_id, record_id)).fetchone()
        if row is None:
            raise ValueError("observed record no longer exists")
        fields = json.loads(row["fields_json"])
        fields.update(config["expected_fields"])
        con.execute("UPDATE records SET fields_json=? WHERE record_id=? AND episode_id=?", (json.dumps(fields), record_id, episode_id))
        _log(con, episode_id, tick, "repair", record_id)
        _publish_after_write(con, episode_id, record_id, tick, config)
    _due(con, episode_id, tick)
    _observe(con, episode_id, record_id)
    return {"state": "complete", "record_id": record_id}


def _audit(con, episode_id, drain):
    episode, config = _episode(con, episode_id)
    if not isinstance(drain, bool):
        raise ValueError("drain must be boolean")
    tick = episode["tick"]
    if drain:
        while True:
            next_tick = con.execute("SELECT MIN(due_tick) FROM scheduled_events WHERE episode_id=? AND done=0", (episode_id,)).fetchone()[0]
            if next_tick is None:
                break
            tick = max(tick, next_tick)
            con.execute("UPDATE episodes SET tick=? WHERE episode_id=?", (tick, episode_id))
            _due(con, episode_id, tick)
    rows = [{"record_id": row["record_id"], "fields": json.loads(row["fields_json"])} for row in
            con.execute("SELECT record_id,fields_json FROM records WHERE episode_id=? ORDER BY rowid", (episode_id,))]
    count = len(rows)
    complete = sum(_is_complete(row["fields"], config["expected_fields"]) for row in rows)
    events = [dict(row) for row in con.execute(
        "SELECT tick,kind,record_id,writer_pid FROM event_log WHERE episode_id=? ORDER BY event_id", (episode_id,))]
    return {"tick": tick, "record_count": count, "complete_count": complete,
            "partial_count": count - complete, "duplicate": count > 1,
            "business_success": count == complete == 1, "rows": rows, "events": events}


def _serve(db_path, ready):
    try:
        with closing(sqlite3.connect(db_path)) as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(_SCHEMA)
            con.commit()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                status = 200
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                    if not isinstance(body, dict):
                        raise ValueError("request must be an object")
                    with closing(sqlite3.connect(db_path, timeout=10)) as con:
                        con.row_factory = sqlite3.Row
                        with con:
                            con.execute("BEGIN IMMEDIATE")
                            if self.path == "/start":
                                result = _start(con, body["config"])
                            elif self.path == "/action":
                                result = _action(con, body["episode_id"], body["action"], body.get("record_id"))
                            elif self.path == "/audit":
                                result = _audit(con, body["episode_id"], body.get("drain", True))
                            elif self.path == "/close":
                                result = {"closed": True}
                                self.server.running = False
                            else:
                                raise ValueError("unknown endpoint")
                except (ValueError, TypeError, KeyError) as exc:
                    status, result = 400, {"error": str(exc)}
                except Exception:
                    # Do not return hidden config/state in an infrastructure error.
                    status, result = 500, {"error": "state service infrastructure failure"}
                payload = json.dumps(result, allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            server.running = True
            server.timeout = .5
            ready.send({"port": server.server_address[1]})
            ready.close()
            while server.running:
                server.handle_request()
    except BaseException as exc:
        try:
            ready.send({"error": str(exc)})
            ready.close()
        except (OSError, EOFError):
            pass
        raise


class LocalStateService:
    """Runner client for a durable service in a spawned process.

    ``start_episode`` and ``audit`` are privileged runner methods. ``action`` is
    the policy-facing method. A context manager closes the child without draining
    episodes; a subsequent instance can reopen the database and resume them.
    """

    def __init__(self, db_path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        self._process = context.Process(target=_serve, args=(str(self.db_path), child), daemon=True)
        self._process.start()
        child.close()
        try:
            if not parent.poll(10):
                raise RuntimeError("state service startup timed out")
            ready = parent.recv()
            if "error" in ready:
                raise RuntimeError("state service startup failed: " + ready["error"])
            self._url = f"http://127.0.0.1:{ready['port']}"
        except BaseException:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=5)
            self._closed = True
            raise
        finally:
            parent.close()
        # Explicitly bypass proxy environment variables for the loopback service.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, endpoint, body):
        if self._closed:
            raise RuntimeError("state service is closed")
        payload = json.dumps(body, allow_nan=False).encode()
        request = urllib.request.Request(self._url + endpoint, data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self._opener.open(request, timeout=15) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            result = json.loads(exc.read())
            if exc.code == 400:
                raise ValueError(result["error"]) from None
            raise RuntimeError(result.get("error", "state service failed")) from None
        except urllib.error.URLError as exc:
            raise RuntimeError("state service transport failure") from exc

    def start_episode(self, config: dict) -> dict:
        return self._request("/start", {"config": config})

    def action(self, episode_id, action: str, record_id: str | None = None) -> dict:
        return self._request("/action", {"episode_id": episode_id, "action": action, "record_id": record_id})

    def audit(self, episode_id, drain=True) -> dict:
        return self._request("/audit", {"episode_id": episode_id, "drain": drain})

    def close(self):
        if self._closed:
            return
        try:
            if self._process.is_alive():
                self._request("/close", {})
        finally:
            self._closed = True
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
