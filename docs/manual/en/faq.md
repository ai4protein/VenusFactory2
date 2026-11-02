# Troubleshooting

## Which mode should I choose?

Agent answers questions and uses existing tools. Scientist develops new methods/models through experiments and review. Researcher investigates a target protein and writes an evidence-based report. A protein report does not automatically create a reusable model.

## Why is a model grey or missing?

It may be disabled by the administrator, unsupported online, missing weights/dependencies, or unavailable for the selected harness. Check the availability hint and current model selector. Protein predictors and the conversational Model selector are separate catalogs.

## My job is queued or the page was disconnected

Keep its job/session identifier. Reopen the same conversation or job view and check progress before submitting again. Switching pages does not intentionally stop accepted jobs. If the status does not update, refresh once and provide the identifier, time and displayed error to the administrator. A disconnected page alone is not proof that the computation failed.

## Why can Agent not read another directory?

Analysis programs run in a sandbox bound to the conversation. Upload the required files or choose an accessible Library item. Use registered retrieval tools for external data. A host path or another user's conversation ID does not grant access.

## Where is my trained model?

Check the training job's artifacts and configuration. Scientist outputs can be private drafts or pending review; execution success does not publish them. Ask for the review status and missing evaluation evidence. A published model must also be enabled before Agent can use it.

## Upload or checkpoint errors

Online files must fit within 10 MB each. Check the requested file type and required columns. Prediction/evaluation must use a checkpoint compatible with the original backbone and configuration. Do not rename a file extension to make its content fit a different format.

## How do I get more conversations?

Use the reset time shown by the site or apply for an extra-credit key as described in [Settings](manual:settings). Agent, Scientist and Researcher share the online daily allowance.

## Need help with the platform?

Ask Agent a specific question such as “How do I prepare my CSV for training?” It can read the current Manual, explain the steps and cite the relevant section. Include the page name and visible error, but no credentials.
