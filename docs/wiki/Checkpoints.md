<!-- Generated compatibility view; edit docs/manual/en/models.md -->

# Model Plaza

## Find a model by function

Open Model Plaza and browse the functional categories or search by model name, protein property or task. There is one catalog, without Quick/Advanced tiers. Models include platform predictors and published custom or Scientist capabilities that the current deployment makes available.

The catalog can include mutation scoring, protein/residue properties, sequence design, structure-related tools and molecular ADMET. Open a card to see its inputs and parameters. Availability depends on enabled tools, weights and dependencies; an entry in the catalog is not a promise that it can run on every deployment.

## Run a tool directly

1. Open a model matching the biological question.
2. Supply the required sequence, structure, table or small-molecule input.
3. Review adjustable parameters. Use advanced settings when you need chain selection, sampling or another model-specific option.
4. Submit once and follow the job. Inspect its result files and preserve the job number for follow-up.

Disabled cards remain visible online. Their availability hint explains why they cannot be used. Choosing another Chat mode does not bypass an administrator's tool switch.

## Custom models and workflows

Use the training, prediction and evaluation entries for your own models. A private model does not become public just because training finishes. Shared publication requires the applicable review and visibility settings.

For repeated multi-step analyses, open [Workflows](./Workflows.md) and connect compatible tools. For conversational use, ask Agent to find a suitable model and inspect its scope.

## Leaderboards

Model Plaza includes VenusX and a ProteinGym entry/link. Compare evaluation task and dataset before interpreting rankings; leaderboard positions do not guarantee performance for your protein or assay.

## Where models come from

- **Pretrained methods**: ProSST, ProtSSN, ESM-1v, ProteinMPNN and LigandMPNN use upstream methods and checkpoints.
- **Factory-trained models**: a name such as **DeepSol · ESM2-650M** identifies a dataset-specific predictor and its backbone. Different datasets can produce different models from the same backbone.
- **Scientist models**: RSI develops a new model or reusable capability. Read its scope, evaluation report and review status before use. Private research results stay private until publication is approved.

Use function and search to find a model; use its type and source to understand how it was produced. Agent can search available tools and read the same manual when you need help.

## Installing models

For complete local setup use `python3 install_full.py`; `--components external,factory,rsi,harnesses` selects optional families. Database/web search and public protein services are built in. `--rsi-library` attaches an existing reviewed model library.

`python3 initialize.py install` prepares a small Python environment. Add `--dependencies platform --stages environment,frontend` for the Web service; `--tools proteinmpnn,ligandmpnn`, `--harnesses deepseek-harness,codex`, or `--factory-preset demo` installs only that component. Use `doctor` with the same component flags to check it. Downloads and environments are separate from source.
