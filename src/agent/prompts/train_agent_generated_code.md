# Train / Agent-generated code prompt

Generate a COMPLETE, executable Python script for this task.

**TASK:** {task_description}

**INPUT FILES (use the absolute `path` field for ALL file I/O):** {file_info}
**OUTPUT DIR (absolute):** {output_directory}
**MODEL REGISTRY:** {model_registry_dir}
**AVAILABLE TRAINED MODELS:** {available_models}

⚠️ **PATH RULE:** Always use the `"path"` field from INPUT FILES (it is an
absolute path) when opening any input file. The script runs under a sandbox
working directory that is NOT the repo root, so project-relative paths
(e.g. `temp_outputs/...`) will fail with `FileNotFoundError`. Do NOT copy
paths from TASK or other free-form text — use the structured INPUT FILES
entries verbatim. Same rule for OUTPUT DIR: open/save under the absolute
path provided, never under a project-relative prefix.

⚠️ **OUTPUT RULE:** OUTPUT_DIR is already an absolute path. Save files
DIRECTLY under it; do NOT prepend `temp_outputs/`, `web_v2/`, or any
other project-relative prefix. Examples:

  GOOD:  out = os.path.join(OUTPUT_DIR, "result.csv")
  GOOD:  out = f"{{output_directory}}/result.csv"
  BAD:   out = os.path.join("temp_outputs", "result.csv")  # cwd mismatch
  BAD:   out = f"temp_outputs/web_v2/.../result.csv"        # double prefix

Treat OUTPUT_DIR as you would `/tmp` — it's a real absolute directory,
not a path fragment to extend. Do NOT echo the OUTPUT_DIR string inside
another path concatenation; pass it as the base directory exactly once.

⚠️ **SUCCESS REPORTING:** Set `"success": false` (not true) and put the
real error message into `"summary"` whenever an exception is caught, a
file cannot be opened, or the intended output is not produced. Reporting
`"success": true` on a failed run misleads downstream verifiers.

⚠️ **INSPECT BEFORE ASSUMING (MANDATORY):** Never assume CSV column
names, JSON field names, or list-of-dict shapes. For every input file
you load:
1. **CSV:** load with `pd.read_csv`, then `print("Columns:",
   list(df.columns))` and `print(df.head(2).to_dict())` BEFORE any column
   access. If your task wants a "score" or "position" column that isn't
   literally named that, fuzzy-match by substring (any column containing
   "score" / "prob" / "pred" / "site" / "pos"), and report what you used
   in the summary.
2. **JSON:** after `json.load`, print `type(data)` and (if dict)
   `list(data.keys())[:20]`, OR (if list) `len(data)` and
   `list(data[0].keys())[:20]`. Pick the field by substring not by
   guessed name.

This eliminates the most common agent_generated_code failure mode:
script crashes with KeyError / "Could not identify columns" because the
upstream tool's schema didn't match the LLM's assumption.

⚠️ **VISUALIZATION IS MANDATORY WHEN ASKED:** If the task description
contains any of: `plot`, `chart`, `figure`, `visualize`, `bar`, `scatter`,
`heatmap`, `histogram`, `boxplot`, `图`, `可视化`, `绘`, then your script
**MUST** save at least one image file (`.png` recommended, 300+ dpi) to
OUTPUT_DIR and include that path in `output_files`. Use `matplotlib`
(`plt.savefig(path, dpi=300, bbox_inches='tight')`) or `seaborn`. A `.txt`
summary alone is NOT acceptable for a plot task — return `success: false`
with reason `"plot task but no image was produced"` if you cannot
generate an actual image (so the harness can auto-retry).

⚠️ **PUBLICATION-GRADE STYLE (when the task asks for a "publication" /
"Nature" / "出版" / "学术配图" / "manuscript figure"):** Use the
`nature_figure` skill's PALETTE and font/SVG rules. The CB plan should
have inserted a `read_skill nature_figure` step before this one — if
not, the minimal mandatory preamble is:

```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'   # keeps text editable in SVG
plt.rcParams['pdf.fonttype'] = 42       # embed TrueType for PDF
plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6
PALETTE = {  # subset; full list in nature_figure/references/api.md
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B", "red_strong": "#B64342",
    "teal": "#42949E", "violet": "#9A4D8E", "gold": "#FFD700",
    "neutral_mid": "#767676", "neutral_dark": "#4D4D4D",
}
```

Save as both `.png` (300 dpi) and `.pdf` (vector) when the task asks
for "publication" or "Nature". Default figure size: single-column
~3.4 inches wide, double-column ~7.0 inches.

⚠️ **EMPTY-FIELD FALLBACK (NOT EMPTY-SCRIPT FALLBACK):** When a SPECIFIC
field you wanted to plot is empty (e.g. HPA `RNA tissue specific nTPM`
is null because the gene has "Low tissue specificity") but the input
file itself is structurally valid:
1. **DO NOT just write a no_data.txt and exit.** That hides the rest of
   the data from the report.
2. Look for FALLBACK fields in the same input that ARE populated. For
   HPA: `RNA tissue distribution`, `RNA tissue specificity`, `Tissue
   expression cluster`, `RNA tissue cell type enrichment`. For BRENDA:
   if `KM` is empty try `kcat`, `optimum_temperature`, `optimum_ph`. For
   CSVs: if the requested column is empty, check sibling columns.
3. **Produce the plot using the fallback fields**, with a clear title
   like "TP53 tissue expression — nTPM unavailable (Low tissue
   specificity); showing tissue distribution categories instead". Save
   the PNG.
4. In your JSON summary, state which field was missing and which
   fallback you used.

Only write `<task>_no_data.txt` and skip the plot when EVERY
substantive field in the input is null/empty — i.e. the upstream tool
genuinely produced no usable data. In that rare case set
`"success": false` so the harness can flag it; do NOT report success
on a no-data placeholder if the task expected a chart.

⚠️ **WHEN THE SCRIPT TRULY CANNOT RUN:** Return `"success": false` only
for actual exceptions, file-not-found, or invalid format. A merely
sparse data field is NOT a script failure — use the fallback above.

**SECURITY (MANDATORY):** The code runs in a sandbox. You MUST NOT use: subprocess, os.system, os.popen, eval(), exec(), __import__(), compile(), input(), breakpoint(), socket, pty, shutil.rmtree, os.remove, os.unlink, os.rmdir, or __builtins__/__globals__. Use only standard data-processing and file I/O within the output directory.

**PATH ISOLATION (MANDATORY):** Do NOT write to `/tmp`, `~`, the user's home dir, the project root, or any path outside OUTPUT_DIR. OUTPUT_DIR is the session-scoped sandbox grant — use it as both your scratch and final-output location. The sandbox already gives full read+write there; any other write target is either denied (sandbox path validation) or globally shared across sessions (breaks isolation). If you need a tempfile, place it under OUTPUT_DIR (e.g. `os.path.join(OUTPUT_DIR, ".scratch_<uuid>.tmp")`), not in `/tmp`.

**CRITICAL REQUIREMENTS:**
1. Write COMPLETE code - DO NOT truncate or use placeholders like "# ... rest of code"
2. Include ALL imports at the top
3. Save all outputs to: {output_directory}
4. Use try-except for error handling
5. End with JSON output:
   print(json.dumps({{"success": True/False, "output_files": [...], "summary": "...", "model_info": {{...}}, "details": {{...}}}})))

**TASK-SPECIFIC GUIDELINES:**

📊 CSV DATA SPLITTING:
- Use train_test_split from sklearn.model_selection
- Split ratios: 70% train, 15% validation, 15% test
- Use stratify parameter for classification tasks
- Save as: train.csv, val.csv, test.csv

🤖 MODEL TRAINING (New Model):
- Auto-detect task type (classification/regression)
- Use models: LogisticRegression, RandomForestClassifier, RandomForestRegressor, XGBoost, LightGBM
- Create a timestamped folder in MODEL REGISTRY: {model_registry_dir}/model_YYYYMMDD_HHMMSS/
- Save model: joblib.dump(model, 'model.pkl')
- Save metadata: JSON file with task type, features, metrics, training date
- Save feature names and preprocessing info for later use
- Report metrics: accuracy/F1 for classification, RMSE/R2 for regression
- Return model_info with model path and name

🔮 MODEL PREDICTION (Using Existing Model):
- Check AVAILABLE TRAINED MODELS list
- Load model: model = joblib.load(model_path)
- Load metadata to understand feature requirements
- Apply same preprocessing as training
- Make predictions on new data
- Save predictions to CSV
- Report prediction statistics

🧬 SEQUENCE MUTATION:
- Use Bio.SeqIO for FASTA files
- Mutation format: A12R = position 12 (0-indexed: 11), Ala→Arg
- Save mutant as new FASTA file

**MULTI-TURN WORKFLOW EXAMPLE:**

Turn 1 - Training:
```python
import joblib, json, os
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save to registry
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = os.path.join("{model_registry_dir}", f"model_{{timestamp}}")
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "model.pkl"))

# Save metadata
metadata = {{
    "model_name": f"model_{{timestamp}}",
    "task_type": "classification",
    "features": list(X_train.columns),
    "accuracy": 0.95,
    "created_at": timestamp
}}
with open(os.path.join(model_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f)

print(json.dumps({{
    "success": True,
    "model_info": {{
        "name": f"model_{{timestamp}}",
        "path": model_dir
    }},
    "summary": "Model trained and saved"
}}))
```

Turn 2 - Prediction:
```python
import joblib, json, os

# Load latest model or specified model
model_dir = "{model_registry_dir}/model_20241203_140530"  # Use available model
model = joblib.load(os.path.join(model_dir, "model.pkl"))

# Load metadata
with open(os.path.join(model_dir, "metadata.json")) as f:
    metadata = json.load(f)

# Make predictions
predictions = model.predict(X_new)

print(json.dumps({{
    "success": True,
    "output_files": ["predictions.csv"],
    "summary": "Predictions completed",
    "model_info": metadata
}}))
```

**CODE STRUCTURE:**
```python
import json
import os
import joblib
from datetime import datetime
# ... other imports

def main():
    try:
        # Your implementation here
        
        # Final JSON output
        result = {{
            "success": True,
            "output_files": [],
            "summary": "Task completed",
            "model_info": {{}}  # Include if model training/prediction
        }}
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({{"success": False, "error": str(e)}}))

if __name__ == "__main__":
    main()
```

**IMPORTANT:**
- Return ONLY Python code (no markdown, no explanations)
- Code must be complete and runnable
- For training: ALWAYS save model to MODEL REGISTRY with metadata
- For prediction: ALWAYS load model from MODEL REGISTRY
- Include model_info in JSON output for tracking

## Language & Tool Execution Rules
- You MUST answer, reason, and output your final response in the **same language** that the user used in their query (e.g., if the user asks in Chinese, you must reply in Chinese).
- **CRITICAL**: When calling ANY tools (including search tools, predictors, database queries, etc.), all tool arguments, keywords, and technical parameters MUST be in **English**. Do not translate protein names, genes, or scientific terms into the user's language when passing them to tools.
