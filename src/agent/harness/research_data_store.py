"""Host-owned measurement tables and opaque evaluation references for Scientist."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path

from utils.scoped_files import read_bytes, write_bytes, parts, open_file, atomic_writer, chunks

SCHEMA = "venusfactory.research-evaluation.v1"
TABLE_SUFFIXES = {".csv", ".tsv", ".tab", ".json", ".jsonl", ".xlsx", ".xls", ".parquet"}
MAX_BYTES = 20 * 1024**2
MAX_PROVENANCE_BYTES = 65536
PROVENANCE_TEXT_FIELDS = {
    "dataset", "title", "description", "url", "doi", "revision", "version",
    "source_file", "sha256", "license", "license_from_dataset_card", "acquisition",
}


def provenance_only(raw):
    """Recognize a bounded source descriptor, never a table or evaluation JSON."""
    if len(raw) > MAX_PROVENANCE_BYTES:
        return False
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError):
        return False
    if not isinstance(value, dict) or not value:
        return False
    allowed = PROVENANCE_TEXT_FIELDS | {"rows", "fields", "original_validation_and_test_supplied"}
    if set(value) - allowed or not (value.get("dataset") or value.get("title")):
        return False
    if not (value.get("url") or value.get("doi")):
        return False
    for key, item in value.items():
        if key in PROVENANCE_TEXT_FIELDS:
            if not isinstance(item, str) or not item.strip() or len(item) > 4000:
                return False
        elif key == "rows":
            if type(item) is not int or item < 0:
                return False
        elif key == "fields":
            if not isinstance(item, list) or len(item) > 128 or not all(
                    isinstance(field, str) and 0 < len(field) <= 120 for field in item):
                return False
        elif type(item) is not bool:
            return False
    return True


class ResearchDataStore:
    def __init__(self, root, workspace):
        self.root = Path(root).resolve()
        self.workspace = Path(workspace).resolve()
        if self.root == self.workspace or self.workspace in self.root.parents:
            raise ValueError("Research data storage must be outside the agent workspace")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.manifest_path = self.root / "inputs.json"
        self.entries = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        if not isinstance(self.entries, dict):
            raise ValueError("Invalid protected input manifest")

    def _reload(self):
        self.entries=json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        if not isinstance(self.entries,dict):
            raise ValueError('Invalid protected input manifest')

    def _save(self):
        from .host import write_json
        write_json(self.manifest_path, self.entries)
        self.manifest_path.chmod(0o600)

    def _put(self, relative, raw):
        from .research_development import immutable_copy
        immutable_copy(self.root, relative, raw)
        (self.root / relative).chmod(0o600)

    def ingest(self, path, raw, expected_sha):
        from agent.rsi.engine.storage import file_lock
        with file_lock(self.root/'.inputs.lock'):
            self._reload()
            return self._ingest(path,raw,expected_sha)

    def ingest_file(self, path, source_root, source_name, expected_sha):
        """Stream an owner-verified upload into private storage before exposing metadata."""
        from agent.rsi.engine.storage import file_lock
        from contextlib import nullcontext
        parts(path)
        if Path(path).suffix.lower() not in TABLE_SUFFIXES:
            raise ValueError("Unsupported protected table type")
        if not isinstance(expected_sha, str) or not re.fullmatch("[a-f0-9]{64}", expected_sha):
            raise ValueError("Invalid upload hash")
        with file_lock(self.root / ".inputs.lock"):
            self._reload()
            previous = self.entries.get(path)
            if previous and previous.get("sha256") != expected_sha:
                raise ValueError("Protected input identity changed")
            relative = "tables/" + expected_sha
            exists = (self.root / relative).exists()
            if exists:
                self._verify_table(relative, expected_sha)
            destination = nullcontext(None) if exists else atomic_writer(self.root, relative)
            with destination as output, open_file(source_root, source_name) as source:
                sha = hashlib.sha256()
                size = 0
                for block in chunks(source):
                    sha.update(block)
                    size += len(block)
                    if output is not None:
                        output.write(block)
                if sha.hexdigest() != expected_sha:
                    raise ValueError("Protected input differs from its upload receipt")
            self.entries.setdefault(path, {"sha256": expected_sha, "bytes": size, "partition": None})
            self._save()
            self._restore_view(path)

    def _verify_table(self, relative, expected_sha):
        sha = hashlib.sha256()
        with open_file(self.root, relative) as stream:
            for block in chunks(stream):
                sha.update(block)
        if sha.hexdigest() != expected_sha:
            raise ValueError("Protected input hash changed")

    def verify_original(self, path):
        self._reload()
        parts(path)
        entry = self.entries.get(path)
        if not isinstance(entry, dict) or not re.fullmatch("[a-f0-9]{64}", str(entry.get("sha256", ""))):
            raise ValueError("Select an original protected input table")
        self._verify_table("tables/" + entry["sha256"], entry["sha256"])
        return entry

    def _ingest(self, path, raw, expected_sha):
        parts(path)
        if Path(path).suffix.lower() not in TABLE_SUFFIXES:
            raise ValueError("Unsupported protected table type")
        sha = hashlib.sha256(raw).hexdigest()
        if len(raw) > MAX_BYTES or sha != expected_sha:
            raise ValueError("Protected input is oversized or differs from its upload receipt")
        previous = self.entries.get(path)
        if previous and previous.get("sha256") != sha:
            raise ValueError("Protected input identity changed")
        self._put("tables/" + sha, raw)
        self.entries.setdefault(path, {"sha256": sha, "bytes": len(raw), "partition": None})
        self._save()
        self._restore_view(path)

    def restore_view(self, path):
        from agent.rsi.engine.storage import file_lock
        with file_lock(self.root/'.inputs.lock'):
            self._reload()
            return self._restore_view(path)

    def _restore_view(self, path):
        entry = self.entries[path]
        view = entry.get("view_sha256")
        if view is not None:
            if not re.fullmatch("[a-f0-9]{64}", str(view)):
                raise ValueError("Invalid training view reference")
            raw = read_bytes(self.root, "views/" + view, max_bytes=MAX_BYTES)
            if hashlib.sha256(raw).hexdigest() != view:
                raise ValueError("Training view changed")
        else:
            if path.lower().endswith(".json") and 0 < entry.get("bytes", MAX_BYTES) <= MAX_PROVENANCE_BYTES:
                original = read_bytes(self.root, "tables/" + entry["sha256"], max_bytes=MAX_PROVENANCE_BYTES)
                if hashlib.sha256(original).hexdigest() != entry["sha256"]:
                    raise ValueError("Protected input hash changed")
                if provenance_only(original):
                    write_bytes(self.workspace, path, original)
                    return
            columns = []
            if path.lower().endswith((".csv", ".tsv", ".tab")):
                with open_file(self.root, "tables/" + entry["sha256"]) as stream:
                    original = stream.read(65536)
                delimiter = "\t" if path.lower().endswith((".tsv", ".tab")) else ","
                reader = csv.reader(io.StringIO(original[:65536].decode("utf-8-sig", errors="replace")),
                                    delimiter=delimiter)
                columns = [str(value)[:120] for value in next(reader, [])[:128]]
            raw = json.dumps({"data_access": "protected_research_table", "source_file": path,
                "sha256": entry["sha256"], "columns": columns,
                "next_step": "Use prepare_research_dataset for CSV/TSV. Only training rows will be exposed."}).encode()
        write_bytes(self.workspace, path, raw)

    def protect_inputs(self, inputs, *, prior_agent_execution=False):
        self._reload()
        protected = []
        for item in inputs if isinstance(inputs, list) else []:
            if not isinstance(item, dict):
                continue
            path = item.get("path", "")
            parts(path)
            if Path(path).suffix.lower() not in TABLE_SUFFIXES:
                continue
            if path in self.entries:
                self.verify_original(path)
                self.restore_view(path)
            else:
                if prior_agent_execution:
                    raise ValueError("This table was available in earlier execution; start a fresh research goal for held-out evaluation")
                self.ingest_file(path, self.workspace, path, item.get("sha256"))
            protected.append(path)
        return protected

    def expose_training(self, path, data):
        from agent.rsi.engine.storage import file_lock
        with file_lock(self.root/'.inputs.lock'):
            self._reload()
            return self._expose_training(path,data)

    def _expose_training(self, path, data):
        original = self.original_bytes(path)
        preparation = data["preparation"]
        if preparation["source_file"] != path or preparation["source_sha256"] != hashlib.sha256(original).hexdigest():
            raise ValueError("Prepared data does not match its protected input")
        assignment = {key: sorted(row["id"] for row in rows) for key, rows in data["splits"].items()}
        allowed = set(assignment["train"])
        partition_data = assignment
        if "support" in data:
            supports = {key: sorted(row["id"] for row in rows) for key, rows in data["support"].items()}
            allowed.update(supports["train"])
            partition_data = {"queries": assignment, "support": supports}
        partition = hashlib.sha256(json.dumps(partition_data, sort_keys=True).encode()).hexdigest()
        plans = [(path, original, partition, allowed, preparation["id_column"], None)]
        if "unlabeled" in preparation:
            pool = preparation["unlabeled"]
            pool_raw = self.original_bytes(pool["source_file"])
            if hashlib.sha256(pool_raw).hexdigest() != pool["source_sha256"]:
                raise ValueError("Prepared unlabeled data differs from its protected input")
            columns = list(dict.fromkeys(([pool["id_column"]] if pool["id_column"] else [])+preparation["input_columns"]))
            pool_partition = hashlib.sha256(json.dumps({"role": "unlabeled", "columns": columns}, sort_keys=True).encode()).hexdigest()
            plans.append((pool["source_file"], pool_raw, pool_partition, None, pool["id_column"], columns))
        # Validate every source before exposing either view. A failed second
        # input must not publish the first input's training observations.
        for source, _, fixed_partition, _, _, _ in plans:
            entry = self.entries[source]
            previous = {value.get("partition") for value in self.entries.values()
                        if value.get("sha256") == entry["sha256"]}
            if previous - {None, fixed_partition}:
                raise ValueError("Keep the original held-out partition; changing the split would expose withheld observations")
        views = []
        for source, raw, fixed_partition, selected_ids, id_column, columns in plans:
            delimiter = "\t" if source.lower().endswith((".tsv", ".tab")) else ","
            reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter=delimiter)
            fields = reader.fieldnames if columns is None else columns
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=fields, delimiter=delimiter)
            writer.writeheader()
            for index, row in enumerate(reader, 1):
                identity = row[id_column].strip() if id_column else self.entries[source]["sha256"][:12]+":"+str(index)
                if selected_ids is None or identity in selected_ids:
                    writer.writerow({column: row[column] for column in fields})
            content = buffer.getvalue().encode()
            view = hashlib.sha256(content).hexdigest()
            views.append((source, fixed_partition, view, content))
        for source, fixed_partition, view, content in views:
            self._put("views/"+view, content)
            self.entries[source].update(partition=fixed_partition, view_sha256=view)
        self._save()
        for source, _, _, _ in views:
            self._restore_view(source)

    def original_bytes(self, path):
        self._reload()
        parts(path)
        entry = self.entries.get(path)
        if not isinstance(entry, dict) or not re.fullmatch("[a-f0-9]{64}", str(entry.get("sha256", ""))):
            raise ValueError("Select an original protected input table")
        if entry.get("bytes", 0) > MAX_BYTES:
            raise ValueError("Prepare a CSV/TSV cohort up to 20 MiB; the complete upload is retained privately")
        raw = read_bytes(self.root, "tables/" + entry["sha256"], max_bytes=MAX_BYTES)
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Protected input hash changed")
        return raw

    def input_bytes(self, path):
        return self.original_bytes(path) if path in self.entries else read_bytes(self.workspace, path, max_bytes=MAX_BYTES)

    def write_evaluation(self, path, raw):
        parts(path)
        data = json.loads(raw)
        if not isinstance(data, dict) or "splits" not in data:
            raise ValueError("Expected host evaluation data")
        identity = hashlib.sha256(raw).hexdigest()
        self._put("evaluations/" + identity + ".json", raw)
        from .research_development import immutable_copy
        handle = json.dumps({"schema": SCHEMA, "id": identity}, sort_keys=True).encode()
        immutable_copy(self.workspace, path, handle)
        return identity

    def load_evaluation(self, path):
        value = json.loads(read_bytes(self.workspace, path, max_bytes=2048))
        if not isinstance(value, dict) or set(value) != {"schema", "id"} or value.get("schema") != SCHEMA:
            raise ValueError("Use a host-prepared evaluation reference; labels cannot be supplied from the agent workspace")
        identity = value.get("id")
        if not isinstance(identity, str) or not re.fullmatch("[a-f0-9]{64}", identity):
            raise ValueError("Invalid evaluation reference")
        raw = read_bytes(self.root, "evaluations/" + identity + ".json", max_bytes=MAX_BYTES)
        if hashlib.sha256(raw).hexdigest() != identity:
            raise ValueError("Protected evaluation hash changed")
        return raw


def store_for_goal(folder, *, force=False):
    """Resolve storage from trusted goal state, never from tool arguments."""
    folder = Path(folder)
    if not (folder/"protocol.json").is_file():
        if force:
            raise ValueError("Autonomous research protocol is missing")
        return None
    protocol = json.loads((folder/"protocol.json").read_text())
    if protocol.get("kind") != "autonomous_research.v1":
        return None
    state = json.loads((folder/"goal.json").read_text())
    from .research_development import model_goal
    data_root = folder/"development"/"private_data"
    objective = state["definition"]["goal"]["objective"]
    if force or model_goal(objective) or state.get("progress", {}).get("plan", {}).get("deliverable") == "model" or (data_root/"inputs.json").exists():
        return ResearchDataStore(data_root, folder/"workspace")
    return None
