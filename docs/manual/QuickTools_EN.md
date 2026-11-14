<!-- Compatibility view; edit en/models.md -->

# Model Plaza

## Find a model by function

Open Model Plaza and browse the functional categories or search by model name, protein property or task. There is one catalog, without Quick/Advanced tiers. Models include available pretrained tools and Factory-trained predictors. Scientist projects have their own RSI Plaza.

The catalog can include mutation scoring, protein/residue properties, sequence design, structure-related tools and molecular ADMET. Open a card to see its inputs and parameters. Availability depends on enabled tools, weights and dependencies; an entry in the catalog is not a promise that it can run on every deployment.

## Run a tool directly

1. Open a model matching the biological question.
2. Supply the required sequence, structure, table or small-molecule input.
3. Review adjustable parameters. Use advanced settings when you need chain selection, sampling or another model-specific option.
4. Submit once and follow the job. Inspect its result files and preserve the job number for follow-up.

Disabled cards remain visible online. Their availability hint explains why they cannot be used. Choosing another Chat mode does not bypass an administrator's tool switch.

## Custom models and RSI cases

Use the training, prediction and evaluation entries for your own models. A private model does not become public just because training finishes. Public release requires operator approval after an email application; the model records the author email.

Ask Agent to find an enabled platform model and inspect its scope. Open [RSI Plaza](./en/rsi-plaza.md) to preview Scientist case repositories and download their project packages.

## Leaderboards

Model Plaza includes VenusX and a ProteinGym entry/link. Compare evaluation task and dataset before interpreting rankings; leaderboard positions do not guarantee performance for your protein or assay.

## Where models come from

- **Pretrained methods**: ProSST, ProtSSN, ESM-1v, ProteinMPNN and LigandMPNN use upstream methods and checkpoints.
- **Factory-trained models**: a name such as **DeepSol · ESM2-650M** identifies a dataset-specific predictor and its backbone. Different datasets can produce different models from the same backbone.
- **RSI cases**: Scientist-developed models and methods appear separately in RSI Plaza as repository previews and ZIP downloads. They are not executable Model Plaza entries or Agent/Researcher tools. Private results remain visible only to their owner.

Use function and search to find a model; use its type and source to understand how it was produced. Agent can search available tools and read the same manual when you need help.

## Installing models

For complete local setup use `python3 install_full.py`; `--components external,factory,rsi,harnesses` selects optional families. Database/web search and public protein services are built in. `--rsi-library` selects existing research artifacts for Scientist and case browsing; it does not register prediction tools in Agent or Researcher.

`python3 initialize.py install` prepares a small Python environment. Add `--dependencies platform --stages environment,frontend` for the Web service; `--tools proteinmpnn,ligandmpnn`, `--harnesses deepseek-harness,codex`, or `--factory-preset demo` installs only that component. Use `doctor` with the same component flags to check it. Downloads and environments are separate from source.
