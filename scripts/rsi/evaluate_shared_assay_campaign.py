"""Score a terminal assay campaign using an explicitly frozen protocol."""

import argparse
import datetime
from pathlib import Path

from evaluate_shared_context_campaign import prepare, read, write

from agent.rsi import shared_assay_evaluation as evaluation
from agent.rsi.protein_model import digest
from agent.rsi.storage import sha256_file


def main(args):
    output = args.output.absolute()
    if output.exists() or output == args.experiment.absolute() or not output.is_relative_to(args.experiment.absolute()):
        raise ValueError("Evaluation requires a new directory inside the experiment")
    if any(p.is_symlink() for p in [output, *output.parents]):
        raise ValueError("Symlinked output paths unsupported")
    if sha256_file(args.frozen_protocol) != args.frozen_protocol_sha256:
        raise ValueError("Frozen scoring protocol changed")
    frozen = read(args.frozen_protocol)
    # Reuse the existing terminal process, publication audit and API-receipt barriers.
    experiment, store, state, reuse, bindings = prepare(args)
    tasks, published = [], {}
    for name, batch in store.specifications(state).items():
        tasks.extend(batch["tasks"])
        for attempt in state["batches"][name]["attempts"]:
            for row in (attempt.get("publication") or {}).get("outcomes", []):
                if row["task_id"] in published:
                    raise ValueError("Repeated task publication")
                published[row["task_id"]] = row["entry_id"]
    evaluation.verify_protocol(frozen, tasks)
    expected = {r["task_id"]: r for r in reuse["task_bindings"]}
    inference = read(args.reuse / "inference_results.json")["results"]
    predictions = {r["receipt"]["entry_id"]: r["predictions"] for r in inference}
    if (
        len(predictions) != len(inference)
        or len(expected) != len(reuse["task_bindings"])
        or not set(published) <= {t["id"] for t in tasks}
        or published != {k: r["entry_id"] for k, r in expected.items()}
        or set(published.values()) != set(predictions)
    ):
        raise ValueError("Independent inference must cover every published model exactly")
    output.mkdir()
    write(
        output / "protocol.json",
        {
            "schema": evaluation.SCHEMA,
            "utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "frozen_protocol": str(args.frozen_protocol.absolute()),
            "frozen_protocol_sha256": args.frozen_protocol_sha256,
            "terminal_and_reuse_bindings": bindings,
            "statistical_independence_assumed": False,
        },
    )
    rows = []
    for task in tasks:
        result = evaluation.evaluate_task(task, predictions.get(published.get(task["id"])))
        witness = expected.get(task["id"])
        if witness is not None and (
            result["query_sha256"] != witness["input_sha256"]
            or digest(result["scientist"]["predictions"]) != witness["prediction_sha256"]
        ):
            raise ValueError("Scored predictions differ from independently reused outputs")
        write(output / (task["id"] + ".json"), result)
        rows.append(result)
    evaluation.verify_protocol(frozen, tasks)
    if any(sha256_file(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("Acquisition or reuse evidence changed during scoring")
    report = {
        "schema": evaluation.SCHEMA,
        "status": "scored_pending_independent_replay",
        "protocol_sha256": sha256_file(output / "protocol.json"),
        "totals": evaluation.summarize(rows),
        "task_reports": {r["task_id"]: sha256_file(output / (r["task_id"] + ".json")) for r in rows},
        "final_labels_scored": True,
        "external_model_calls": 0,
        "new_autonomous_models": 0,
        "domain_baseline_comparison_complete": False,
        "statistical_independence_assumed": False,
    }
    write(output / "report.json", report)
    print(report["totals"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal-summary-sha256", required=True)
    parser.add_argument("--reuse-report-sha256", required=True)
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    parser.add_argument("--frozen-protocol-sha256", required=True)
    main(parser.parse_args())
