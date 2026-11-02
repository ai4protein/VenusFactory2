# Build a reusable workflow

## When to use a workflow

Use Workflows when the steps and data flow are already clear and you expect to repeat them. Use Agent for exploratory questions and Scientist when the method itself needs research or development.

## Create and connect steps

1. Open Workflows, name the workflow and choose tools/models from the categorized catalog.
2. Add the input files or data required by the first step.
3. Drag cards into place and connect compatible outputs to later inputs. The connection selects the data source; visual position alone does not create a dependency.
4. Open each card to set its model-specific parameters and required inputs. Remove unwanted cards with their delete control.
5. Save the workflow before running it. Reopen the saved workflow to edit or reuse the process.

For example, retrieve a sequence, calculate properties, then pass the appropriate sequence file to a compatible predictor. A PDB output cannot be connected to an arbitrary CSV input without conversion.

## Training and registered tools

The catalog can include registered tools and a custom model training component. Supply the training data/configuration, then connect its model output to a compatible prediction step. Training enters the shared queue; online ESM2-8M and tool-access limits still apply.

Only enabled components can execute. A saved graph does not grant access to a model that has since been disabled or removed.

## Run and inspect

Check required inputs, save, then run. Follow the job and individual step results. Open generated files and inspect errors at the failed step before resubmitting. Saved workflow definitions and job outputs are separate: saving records the process, not a successful execution.

## Input data and connections

An Input data card selects one uploaded dataset: CSV/TSV/JSON contains one sample per row; FASTA becomes `id` and `sequence` columns; PDB/PDBQT is a structure and SDF/MOL is a ligand file. Select the sequence, feature and target columns in the consuming step. Extra files, such as a docking ligand, are configured in that model's settings.

Connections pass actual upstream output. Steps run in dependency order, regardless of where cards are placed. A prediction step has separate data and trained-model inputs. A single input can have only one source; choosing another replaces that connection. Cycles, incompatible file kinds and invalid ports are rejected. Known columns are checked before submission; generated columns are checked again before the next step executes.

During execution, completed edges turn green and active edges animate. Select a step in the execution trace to inspect its input/output preview, record counts, elapsed time and processing log. Registered tools report record progress; model adapters without internal progress show their running state until completion. If a step fails, later steps are not executed. Saved job traces remain available after refresh.
