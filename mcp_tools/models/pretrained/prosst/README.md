# ProSST structure tokenizer

This package contains the shared structure-tokenization implementation used by
ProSST mutation scoring and factory training data preparation. It does not store
model weights in the source tree.

Set `prosst.structure_weights` in `configs/model_runtimes.json` to a directory
containing `AE.pt` and the selected vocabulary's `<size>.joblib`. The default is
`runtime/local/model-runtimes/prosst/structure-tokenizer/weights`. Provision these
upstream tokenizer assets before enabling ProSST on a fresh installation.
`weights-manifest.json` records the hashes of the existing validated assets moved
out of the old training package; it contains no model data or credentials.

Language-model checkpoints are separate: the ProSST runtime selects its model
ID or local HF-format weights for each vocabulary variant. Moving tokenizer
weights does not change checkpoint names or vocabulary sizes.
