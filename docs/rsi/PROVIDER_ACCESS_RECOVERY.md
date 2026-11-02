# Provider access recovery for persistent Scientist goals

New generated-model goals and paper-reproduction goals detect terminal HTTP 401, 402 and 403 responses in the host's model-request log. Detection uses the last paired request/response and the failed executor record, not the model's prose or a tool's error text. A completed executor, a request that subsequently recovered, an unpaired response, or an unrelated HTTP status does not trigger this access gate.

On a detected access failure, the episode remains unsuccessful, its real model/tool usage remains charged, and the goal enters the existing `waiting_for_input` state. The outcome binds the request log and runtime report by SHA256. The scheduler does not immediately spend the remaining episodes repeating the rejected request. No alternate model, credential or provider is selected.

After the operator restores access, the ordinary `resume` action continues the same goal at the next episode. Prior work, registered capabilities, failed episodes, and all consumed budgets remain. A goal that already reached its declared budget is not reopened or given an unrecorded extension. Interrupted scientific work and failed review remain subject to the existing evidence gates.

This protection applies to newly created goals using the revised implementation. Frozen historical studies keep their original source and results. A new experiment must freeze the revised source before inference; modifying a running historical snapshot is not a recovery mechanism.

Validation includes synthetic request-pair tests, a real two-paper scheduler test (wait after the first access failure, resume, register both papers and recompute them), and a read-only classification of three existing GLM 403 records. The latter makes zero external model calls and leaves the old goal unchanged. Infrastructure-aware handling of 429/5xx responses and timeouts remains a separate policy decision; this change does not silently grant retries for them.
