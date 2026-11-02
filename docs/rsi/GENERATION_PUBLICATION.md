# Publishing generations during a Scientist goal

An evolving Scientist goal can publish an accepted generation after its independent
fresh-session reuse, before starting the next generation. `run_science_evolution.py`
enables this when `--model-library` is set. Scientist model campaigns also record
these publications immediately. Agent inference and cross-task experience use the
existing model library interface.

The parent retains its original objective, number of generations, budget, status,
and full history. A publication snapshots the actual settled checkpoint into the
owner's `<library-name>_milestones` directory. Its manifest binds every evidence
file, the parent identity, completed prefix, and original requested generation
count. It does not rewrite an unfinished goal to look completed.

Each model still passes native Scientist execution verification, independent fit
and development-validation replay, the original strict improvement thresholds,
source lineage checks, and fresh-session prediction replay. The snapshot does not
score future labels. Model identity and deduplication are unchanged. Finishing the
whole goal later does not create another model count for the same key.

A later failed generation leaves the earlier published artifact available. When
counting that model for a campaign, the auditor also verifies that the actual
parent still contains the frozen episode prefix. Copies from unrelated goals do
not contribute to that campaign's new-model target. Publication failures retain
their reports and interrupt the goal before another model episode; resume retries
from saved progress without resetting its model-call budget.

The independent campaign evaluator accepts these generation origins only after
the existing campaign-wide development barrier. It scores the allocation's own
fitted artifact, even if its catalog key deduplicates another run. An incomplete
multi-generation allocation remains incomplete in the primary feasibility
denominator; any earlier published-model performance is conditional evidence.

Snapshot creation and audit are not new autonomous executions. Software fixtures
and re-audits of historical models must not be counted toward scientific targets.
