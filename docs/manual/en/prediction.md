# Predict with a model

## Choose the right entry

For a platform predictor, open its card in [Model Plaza](manual:models). For your trained model, open custom model Prediction and select the matching checkpoint/configuration. For a Scientist capability, inspect its registered input contract and biological scope first.

## Prepare inputs

- **Single sequence**: paste an amino-acid sequence into the sequence field. Follow the form's FASTA/plain-sequence requirements.
- **Batch**: upload a table or sequence file and choose the sequence column when prompted. Keep an identifier per row so outputs can be joined back to samples.
- **Structure/design**: provide the required PDB/mmCIF or other supported structure and review chain, residue and sampling settings.
- **Small molecules**: use the molecular tool's stated SMILES or structure-file format; a protein sequence is not an ADMET input.

For a custom checkpoint, retain the backbone, training method, pooling and label mapping used during training. A similarly named backbone is not necessarily compatible.

## Submit and retrieve results

1. Select an enabled model and inspect all relevant parameters.
2. Submit once; keep the job number and follow queued/running/completed status.
3. Open the result table and download output files. For batch work, check that sample identifiers and the number of valid predictions match your inputs.
4. Note model version, units, score direction and limitations before selecting candidates for experiments.

Jobs belong to their submitting user/browser identity. If you cannot find a result, return to the original browser or conversation rather than guessing another user's job ID.

## Using Chat

Upload the input to Agent and name the desired property. It can search the current tool catalog, request missing information and run an enabled predictor. Use Researcher when you need predictions integrated with literature into a longer report.

Disabled models cannot be invoked through Chat. A successful prediction is a computational result, not experimental confirmation. To measure predictive performance, use an independent labeled set and [Evaluation](manual:evaluation).
