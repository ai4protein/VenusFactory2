# Train / Agent-generated code prompt

Generate a COMPLETE, executable Python script for this task.

**TASK:** {task_description}

**INPUT FILES:** {file_info}
**OUTPUT DIR:** {output_directory}
**MODEL REGISTRY:** {model_registry_dir}
**AVAILABLE TRAINED MODELS:** {available_models}

**SECURITY (MANDATORY):** The code runs in a sandbox. You MUST NOT use: subprocess, os.system, os.popen, eval(), exec(), __import__(), compile(), input(), breakpoint(), socket, pty, shutil.rmtree, os.remove, os.unlink, os.rmdir, or __builtins__/__globals__. Use only standard data-processing and file I/O within the output directory.

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
