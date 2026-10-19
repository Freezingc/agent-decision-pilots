"""Read-only trajectory verification plus lossless DB archive; no new episodes."""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import sys

root = Path(__file__).resolve().parent
project = root.parents[2]
sys.path.insert(0, str(project / "artifacts/p5"))
import probe_identifiability as probe


def require(value, message):
    if not value:
        raise ValueError(message)


manifest = json.loads((root / "manifest.json").read_text())
for name, item in manifest.items():
    path = root / name
    require(path.stat().st_size == item["bytes"], "size " + name)
    require(hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], "hash " + name)
frozen = json.loads((root / "frozen.json").read_text())
for source, digest in frozen["source_hashes"].items():
    require(hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest, "source changed " + source)
require(hashlib.sha256((project / "docs/p5-identifiability-protocol.md").read_bytes()).hexdigest() == frozen["protocol_sha256"], "protocol hash")

with gzip.open(root / "trajectories.jsonl.gz", "rt") as handle:
    rows = [json.loads(line) for line in handle]
require(len(rows) == 4106, "total episode count")
require(len({r["public"]["start"]["episode_id"] for r in rows}) == 4106, "independent episodes")
main = [r for r in rows if r["role"] == "enumeration"]
require(len(main) == 4096, "main count")
by_key = {(r["sequence_index"], r["branch"], r["world"]): r for r in main}
require(len(by_key) == 4096, "duplicate main index")
compared = 0
for index, sequence in enumerate(itertools.product(("cache", "journal", "fields", "wait"), repeat=5)):
    for branch in ("short", "ambiguous"):
        pair = [by_key[index, branch, w] for w in ("A", "B")]
        for world, row in zip(("A", "B"), pair):
            require(row["sequence"] == list(sequence), "sequence content")
            require([x["action"] for x in row["public"]["steps"]] == list(sequence), "action order")
            require([x["tick"] for x in row["public"]["steps"]] == [1, 2, 3, 4, 5], "tick order")
            require(row["normalized_public"] == probe.public_episode(row), "normalization replay")
            expected_count = 0 if world == "A" and branch == "ambiguous" else 1
            require(row["terminal_audit"]["record_count"] == row["terminal_audit"]["complete_count"] == expected_count, "main final rows")
            require(not row["terminal_audit"]["duplicate"], "main duplicate")
        require(pair[0]["normalized_public"] == pair[1]["normalized_public"], "public pair differs")
        require(probe.branch_weights("A")[branch] == probe.branch_weights("B")[branch], "mixture weights")
        compared += 1
comparisons = [json.loads(line) for line in (root / "comparisons.jsonl").read_text().splitlines()]
require(len(comparisons) == 2048, "comparison count")
require(all(r["initial_equal"] and all(r["step_equal"]) and r["full_public_equal"] and not r["different_step_indices"] for r in comparisons), "comparison receipts")

extra = {(r["role"], r["branch"], r["world"]): r for r in rows if r["role"] != "enumeration"}
for world, expected_rows in (("A", 1), ("B", 2)):
    retry = extra["nonidempotent_retry", "ambiguous", world]
    require([s["action"] for s in retry["public"]["steps"]] == ["fields"] * 5 + ["retry"], "retry action path")
    require(retry["terminal_audit"]["record_count"] == expected_rows, "ambiguous retry count")
    require(extra["nonidempotent_retry", "short", world]["terminal_audit"]["duplicate"], "short retry duplicate")
    abstain = extra["abstain", "ambiguous", world]
    require(abstain["controller_terminal"] == "abstain" and not abstain["controller_success"], "controller abstention")
    require(abstain["terminal_audit"]["record_count"] == (0 if world == "A" else 1), "abstain audit difference")
    atomic = extra["atomic_journal", "ambiguous", world]
    require(atomic["public"]["steps"][0]["response"] == {"state": "not_found" if world == "A" else "pending"}, "atomic observation")
    idem = extra["idempotent_retry", "ambiguous", world]
    require(idem["terminal_audit"]["record_count"] == idem["terminal_audit"]["complete_count"] == 1 and not idem["terminal_audit"]["duplicate"], "idempotent rows")

http = [json.loads(line) for line in (root / "http-calls.jsonl").read_text().splitlines()]
require(len(http) == 28740 and all(r["ok"] for r in http), "HTTP receipts")
require(Counter(r["endpoint"] for r in http) == {"start_episode": 4106, "action": 20528, "audit": 4106}, "HTTP counts")
summary = json.loads((root / "summary.json").read_text())
require(summary["status"] == "completed" and all(summary["checks"].values()), "run status")
require(abs(sum(r["ms"] for r in http) - summary["http"]["total_measured_ms"]) < 1e-6, "HTTP times")
with sqlite3.connect(f"file:{root / 'service.sqlite'}?mode=ro", uri=True) as con:
    require(con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 4106, "database episodes")
    require(con.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "database integrity")
    require(con.execute("SELECT COUNT(*) FROM scheduled_events WHERE done=0").fetchone()[0] == 0, "undrained events")

db = root / "service.sqlite"
archived = root / "service.sqlite.gz"
with db.open("rb") as source, archived.open("wb") as stream:
    with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as target:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            target.write(block)
db_hash = hashlib.sha256(db.read_bytes()).hexdigest()
with gzip.open(archived, "rb") as stream:
    restored_hash = hashlib.sha256(stream.read()).hexdigest()
require(restored_hash == db_hash, "lossless database archive")

verification = {"schema": "p5-identifiability-verification-v1", "status": "matched", "verified_utc": datetime.now(timezone.utc).isoformat(),
                "main_episodes": 4096, "additional_episodes": 10, "independent_episode_ids": 4106,
                "paired_public_histories_equal": compared, "all_non_id_fields_and_list_order_preserved": True,
                "exact_declared_branch_weights": {"short": "4/5", "ambiguous": "1/5"},
                "mixture_interpretation": "equal weighted laws by exhaustive paired equality, not empirical 80/20 sample counts",
                "service_http_requests_measured": 28740, "http_close_requests_unmeasured": 1,
                "all_http_requests_including_close": 28741,
                "all_mechanism_controls_match": True, "source_and_protocol_hashes_match": True,
                "database_integrity": "ok", "database_uncompressed_sha256": db_hash,
                "database_uncompressed_bytes": db.stat().st_size, "database_archive": archived.name,
                "database_archive_sha256": hashlib.sha256(archived.read_bytes()).hexdigest(),
                "gzip_restore_sha256_matches": True,
                "actual_run_exit_code": 0, "hard_timeout_seconds": 300, "automatic_retries": 0,
                "scope": "post hoc explanatory controlled-service mechanism, no learning, natural-sampling or production reliability claim"}
(root / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
for name in ("service.sqlite.gz", "verification.json", Path(__file__).name):
    path = root / name
    manifest[name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
manifest["service.sqlite"]["archived_as"] = "service.sqlite.gz"
manifest["service.sqlite.gz"]["restores_sha256"] = db_hash
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(verification, indent=2))
