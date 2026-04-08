# VenusFactory2 Advanced Tools User Guide

## 1\. Introduction to Advanced Tools

**Advanced Tools** gives you more control over your protein predictions. While Quick Tools uses pre-selected models for speed, Advanced Tools lets you choose exactly which models to use and compare results from different approaches.

**Use Advanced Tools when you**:
- Want to compare predictions from different models (ESM2, ProtBert, ProtT5, etc.)
- Need structure-based analysis in addition to sequence-based
- Are optimizing for a specific use case and want to test different model configurations
- Want to validate results by cross-checking multiple datasets

The tradeoff: more flexibility means more choices to make, but you get more customized and potentially more accurate results.

### 1.1 Directed Evolution

This module utilizes AI to **accurately predict mutation effects** to guide protein directed mutation experiments. Users can choose different model types based on their data and required precision.

**Core Advantage:** Users can select the underlying **PLM** (e.g., **ESM2**, **VenusPLM**) and decide whether to integrate **structural information**.

  * **Sequence-based:**
      * **Description:** Utilizes the user-selected PLM to directly predict the impact of mutations on protein function from the amino acid sequence.
      * **Application Scenario:** Suitable for scenarios where **structural information is unavailable** or for those requiring **large-scale, high-throughput screening**.
  * **Structure-based:**
      * **Description:** Combines an uploaded **PDB structure file** with structure-based models or algorithms for a more refined assessment of mutation effects.
      * **Application Scenario:** Suitable for rational design that requires high-precision analysis of mutation effects on **residue interactions** and **structural stability**.

### 1.2 Protein Function

This feature allows users to leverage different PLM models and multiple datasets to predict the overall properties of a protein, aiming for more reliable results.

**Core Advantage:** Users can freely select the **PLM** (e.g., **ESM2-650M**) and simultaneously select **multiple fine-tuned dataset models** (e.g., **DeepSol, ProtSolM**) for prediction.

  * **Description:** Users predict the overall function or properties (e.g., solubility, localization) of a given amino acid sequence by selecting different PLMs and datasets.
  * **Application Scenario:** **Multi-model cross-validation** is its key advantage. By comparing results from different models on the same task, it significantly **improves prediction reliability and robustness**.

### 1.3 Functional Residue 

This module allows users to select specific PLM models for high-precision localization and prediction of critical functional residues within the sequence.

**Core Advantage:** Users can freely select the underlying **PLM** to specifically **optimize residue-level prediction performance**.

  * **Description:** Focuses on identifying the functional role of individual amino acid residues in the protein sequence, such as **Activity Site** or **Binding Site**.
  * **Application Scenario:** By selecting the optimal PLM, users can **accurately label functional sites** and obtain **residue-level prediction probabilities**, aiding in the establishment of a precise structure-function map.

### 1.4 Protein Discovery

This module starts from a user-provided protein structure and searches large-scale structure and sequence databases for potentially homologous, structurally similar, or functionally related candidate proteins.

**Core Advantage:** Users can configure the key discovery parameters and run a full VenusMine pipeline for enzyme/protein discovery.

  * **Description:** VenusMine combines structural alignment, sequence similarity search, redundancy reduction, representation-based ranking, and phylogenetic tree construction to support multi-stage candidate screening.
  * **Application Scenario:** Suitable for enzyme function discovery, mining functionally similar proteins from unknown structures, and preparing candidate templates for protein engineering, directed evolution, or sequence design.

-----

## 2\. Advanced Tools Configuration and Results

### 2.1 Model Configuration and Data Input Area

This area is used to define the prediction task, select the underlying model and datasets, and provide the protein sequence for analysis.

**Model Configuration:**

  * **Select Sequence-based Model / Select Structure-based Model:** This is a key configuration for Advanced Tools. Users can **freely select the underlying Protein Language Model (PLM)** (e.g., **VenusPLM, ESM2-650M**) or a structure model for inference, instead of using a preset model.
  * **Select Task / Select Protein function / Select Properties of Protein** (Specific option determined by the tool's function): A drop-down menu used to select the type of prediction task:
      * directed mutation: **Select Protein function**
      * Protein function prediction: **Select Task**
      * Functional residual prediction: **Select Task**
  * **Select Datasets:** In certain tasks (e.g., Protein function prediction), users can simultaneously select **multiple fine-tuned dataset models** (e.g., **DeepSol, ProtSolM**) for prediction, enabling multi-model cross-validation to enhance result reliability.

**Data Input:** This area allows users to provide sequence data via file upload or pasting, and it displays the content of the uploaded sequence.

  * **Upload Protein File:** The selected tab for uploading sequence/structure files via a file (may support **.pdb** depending on the task).
  * **Paste Protein Content:** The alternative tab for directly pasting sequence content in **FASTA format** into a text box.
  * **Uploaded Protein Sequence:** This area displays the **raw amino acid sequence** read from the input file or pasted content.

**Configure AI Analysis (Optional):**

  * **AI Settings:** Used to select or configure specific parameters for AI analysis.
  * **Enable AI Summary:** Activating this option enables the system to generate a **textual summary and analysis** after the prediction is complete.
  * **Start Prediction:** Initiates the prediction task.

### 2.2 Results Display Functionality

This module displays the prediction results after task completion and provides result interpretation and download options.

**Status:** Provides **real-time feedback** on the prediction process. Due to flexible model configuration, this area may also display detailed diagnostic and warning information.

**Result Display Tabs:** Results are displayed in tabular form and are divided into the following sections:

  * **Raw Results:**
      * Since Advanced Tools allows for the selection of multiple datasets, the results table may include an additional **Dataset** column, showing the prediction results from different models on the same sequence for comparison and analysis.
      * Generates the corresponding initial prediction results based on the task and model configuration.
  * **Prediction Plots:** Provided for the **Protein function prediction** task. It visually displays the prediction results, for example, using a bar chart to show the model's prediction probabilities for different locations if subcellular localization is being predicted.
  * **AI Expert Analysis:** If **Enable AI Summary** was enabled, this tab displays a **textual analysis report** generated by the AI.
  * **Prediction heatmap:** Both **directed mutation** and **Functional residual prediction** tasks still provide this chart for visualizing residue-level or mutation effect prediction results.
  * **Download Results:** Allows the user to download all detailed prediction results from the table to their local computer.

-----

## 3\. Advanced Tools - Sequence-based Model

### 3.1 Directed Evolution

This module allows users to utilize a **freely selected Protein Language Model (PLM)** to accurately predict the impact of mutations on protein function, enabling customized guidance for directed mutation.

#### 3.1.1 Task Configuration and Data Input

**Model Configuration:**

  * **Select Sequence-based Model:** This is the core of the module. Users can freely select different underlying PLM engines as the prediction core from the drop-down menu. **Available models** include but are not limited to: **VenusPLM, ESM2-650M, ESM-1v, or ESM-1b**. Users can choose based on their performance requirements or model preference.

**Data Input:** This area allows users to provide sequence data.

  * **Upload Protein File:** The selected tab for uploading sequence via file, supporting formats like **.fasta, .fa, or .pdb**.
  * **Paste Protein Content:** The alternative tab for directly entering sequence content in **FASTA format** into the text box.
  * **Uploaded Protein Sequence:** This area displays the **raw amino acid sequence** read from the input file or pasted content.

**Configure AI Analysis (Optional):**

  * Users can choose to enable **Enable AI Summary** to receive a **textual evaluation** of the results from a professional **AI biology expert** after the prediction is complete.

#### 3.1.2 Execute Prediction

1.  **Ensure all model configuration parameters are set correctly.**
2.  Click the **"Start Prediction"** button to start the prediction process.
3.  The system will display **prediction progress and status information.**
4.  To abort the prediction, click the **"Abort"** button.

#### 3.1.3 Results Display Functionality

**Status:** Provides **real-time feedback** on prediction progress (e.g., "Prediction completed successfully\!").

**Raw Results:** This table clearly lists the **optimal mutants** predicted by the model and their performance metrics. The table typically includes **Mutant** name (e.g., P66T), **Prediction Rank**, and **Prediction Score** (usually a probability value between 0 and 1). Users can use this score to intuitively understand each mutant's potential to improve the target function and select high-ranking mutants for subsequent experimental validation.

**Prediction Heatmap:** This visualization displays the **potential impact** of a mutation to any other amino acid at every residue position in the sequence, presented as a **two-dimensional matrix**.

  * The **Y-axis** shows the **residue position** (sorted by impact), and the **X-axis** shows the **type of mutated amino acid**.
  * The **color intensity** in the heatmap represents the **Normalized Effect** of the mutation on the target function (e.g., activity or stability).
  * **Darker colors (High)** indicate a **stronger positive enhancement** effect on the protein function; **Lighter colors (Low)** indicate a weaker enhancement effect or a negative impact. The inference for this heatmap is based on the PLM model selected by the user, making it highly customizable, which helps users quickly identify potential **"hotspot" residues** to guide rational design.

**AI Expert Analysis:** A professional **AI biology expert** evaluates the prediction results, providing in-depth interpretation and experimental suggestions in a **textual format**.

**Download Results:** Users can click this button to **download all detailed prediction data** from the table to their local computer for further data processing and archival.

-----

## 4. Sequence Design

This module exposes **ProteinMPNN** with full inference controls for structure-conditioned sequence generation.

### 4.1 Input and core controls

- **Structure input:** Upload a `.pdb` file (or use the example PDB).
- **Model Family selector:** `Soluble` / `Vanilla` / `CA` (manual choice with recommendation).
  - **Soluble**: recommended for protein discovery and general sequence design.
  - **Vanilla**: recommended for membrane proteins.
  - **CA**: only for C-alpha-only coarse-grained coordinates.
- **Designed chains / fixed chains:** Configure redesign targets and fixed context chains.
- **Fixed residues (text syntax):** Pin selected positions using readable syntax like `A12,C13` or `A:12,13;B:5-8`.
- **Homomer mode:** Enable tied-position constraints for symmetric design.
- **Number of sequences / temperatures:** Configure sampling count and diversity.

### 4.2 Full ProteinMPNN inference options

Advanced Sequence Design provides explicit fields for core and advanced options, including:

- **Model & sampling:** `model_name`, `omit_aas` (backbone noise is auto-mapped from selected model)
- **Runtime controls:** `seed`, `batch_size`, `max_length`
- **Optional advanced rules (text):** tied positions, omit-AA rules, AA bias, bias-by-residue, and PSSM rules entered as text; backend auto-converts to JSONL
- **PSSM numeric controls:** `pssm_multi`, `pssm_threshold`, `pssm_log_odds_flag`, `pssm_bias_flag`

Model recommendation:
- **Default first choice:** `v_48_020` (auto uses 0.20A noise policy)
- **High-resolution native structures only:** `v_48_002` (auto uses 0.02A noise policy)

### 4.3 Results

- **Summary status:** Completion state and run summary.
- **Table:** FASTA preview rows (header, sequence, length, score fields when available).
- **Raw:** Full response payload for automation or downstream parsing.
- **Download Result:** Generated FASTA file from ProteinMPNN.

### 4.4 Recommended usage

Use this module when you need explicit control over ProteinMPNN inference behavior for design experiments, benchmarking, or reproducible protocol development.  
If you prefer default-safe settings with fewer decisions, use **Quick Tools / Sequence Design**.


## 5\. Advanced Tools - Protein Function

This module allows users to **freely select the underlying PLM model** and simultaneously select **multiple fine-tuned datasets** to perform multi-dimensional predictions of the protein's overall function and properties, aiming for more reliable prediction results.

### 5.1 Task Configuration and Sequence Input

**Model Configuration:**

  * **Select Model:** This is a key advanced configuration item where users can freely select the underlying PLM engine.
      * **Available Models:** **ESM2-650M, Ankh-large, ProtBert, ProtT5-xl-uniref50**.
  * **Select Task:** Users must select the specific goal for the protein function prediction.
      * **Available Tasks:** **Solubility, Localization, Metal ion binding, Stability, Sorting signal, Optimum temperature**.
  * **Select Datasets:** Users can simultaneously select multiple fine-tuned dataset models for prediction, used for cross-validation.
      * **Available Datasets:** **DeepSol, DeepSoluE, ProtSolM**.

**Data Input:**

  * **Upload FASTA File:** The selected tab for uploading sequence via file, supporting the **FASTA format**.
  * **Paste FASTA Content:** The alternative tab for directly entering the sequence into the text box.
  * **Uploaded Protein Sequence:** This area displays the **raw amino acid sequence** read from the input file or pasted content.

**Configure AI Analysis (Optional):**

  * Users can choose to enable **Enable AI Summary** to receive a **textual evaluation** of the results from a professional **AI biology expert** after the prediction is complete.

### 5.2 Execute Prediction

1.  **Ensure all model configuration parameters are set correctly.**
2.  Click the **"Start Prediction"** button to start the prediction process.
3.  The system will display **prediction progress and status information.**
4.  To abort the prediction, click the **"Abort"** button.

### 5.3 Results Display Functionality

**Status:** Provides **real-time feedback** on prediction progress (e.g., "All predictions completed\!").

**Raw Results:** This table shows the core output of the multi-model/multi-dataset prediction, used for cross-validation.

  * **Table Content:** **Dataset, Protein Name, Sequence, Predicted Class, Confidence Score**.

**Prediction Plots:** (New Feature) This tab provides **visualized charts** of the prediction results. For example, if subcellular localization is predicted, a bar chart might show the model's prediction probabilities for different locations.

**AI Expert Analysis:** A professional **AI biology expert** evaluates the prediction results, providing in-depth interpretation and experimental suggestions in a **textual format**.

**Download Results:** Users can click this button to **download all detailed prediction data** from the table to their local computer for further data processing and archival.

-----

## 6\. Protein Discovery (VenusMine)

VenusMine is used to start from a known protein structure and discover candidate proteins from large-scale structure and sequence databases. It is suitable for enzyme mining, functionally similar protein discovery, candidate template screening, and preparing starting sequence sets for downstream directed evolution or sequence design.

### 6.1 Input and Core Configuration

**Input Configuration:**

  * **Upload PDB Structure:** Upload the `.pdb` structure file to analyze. VenusMine uses the structure as the entry point, first running a FoldSeek structural similarity search and then expanding the candidate set through sequence-level search.
  * **Protected Region:** Define the structure or functional segment that should be retained or emphasized.
      * **Start Position:** Start residue position of the protected region. Default: `1`.
      * **End Position:** End residue position of the protected region. Default: `100`.
      * The system preferentially keeps FoldSeek hits that cover this region, making this useful for preserving active pockets, binding segments, or key structural domains.

**MMseqs2 Search Parameters:**

  * **Threads:** Number of threads for the MMseqs2 search. Default: `96`. Lower this value if your machine has fewer available CPU resources.
  * **Iterations:** Number of iterative search rounds. Default: `3`. More iterations may find more distant sequence relationships but will take longer.
  * **Max Sequences:** Maximum number of hit sequences retained per search round. Default: `100`; configurable from `100` to `5000`.
  * **Database:** The default database is **UniRef90**. If no database path is configured, VenusMine will try to auto-download it to `data/VenusMine/uniref90.fasta.gz`.


**Clustering Parameters:**

  * **Min Sequence Identity:** Minimum sequence identity threshold for MMseqs2 clustering. Default: `0.5`. Higher values make clustering stricter and remove redundancy more conservatively.
  * **Threads:** Number of threads used during clustering. Default: `96`.

**Tree Building Parameters:**

  * **Top N Results:** Number of candidate sequences used for final tree construction and display. Default: `10`. Increase this value to inspect a broader candidate set.
  * **E-value Threshold:** E-value cutoff for filtering sequence search results. Default: `1e-5`. Smaller values make the results stricter.

### 6.2 Execution Workflow

After clicking **Start VenusMine Pipeline**, the system runs the following workflow:

1. **Set up output directories:** Create the result directory for this VenusMine run and record the protected region, database path, and run parameters.
2. **FoldSeek structural search:** Submit the uploaded PDB to FoldSeek and search structure-related databases such as AFDB, Swiss-Prot, CATH, MGnify, and PDB.
3. **Parse FoldSeek hits:** Read the downloaded `.m8` alignment files and extract candidate sequences that cover the protected region.
4. **MMseqs2 sequence search:** Use the FoldSeek-derived candidate sequences as queries to search UniRef90 or a user-specified database for additional similar sequences.
5. **Convert TSV to FASTA:** Convert MMseqs2 search results to FASTA for downstream clustering and embedding calculation.
6. **MMseqs2 clustering and redundancy reduction:** Remove overly similar candidate sequences according to the configured minimum sequence identity threshold and keep representative sequences.
7. **ProstT5 representation calculation:** Compute protein representation vectors for candidate and reference sequences for distance-based ranking and tree construction.
8. **Candidate filtering and tree construction:** Filter representative sequences using representation distance, Top N, and E-value settings, then build a hierarchical clustering or phylogenetic tree.
9. **Package results:** Generate the tree image, sequence label table, complete result archive, and processing log.

### 6.3 Result Display

**Structure Visualization:** Displays the uploaded structure viewer area or placeholder to help confirm the current analysis target.

**Phylogenetic Tree:** Shows hierarchical relationships between candidate proteins and reference sequences. Use this plot to see whether candidates form clear branches, whether they are close to known functional references, and which sequences may be worth prioritizing for validation.

**Sequence Labels:** Lists discovered sequences, source types, labels, and cluster information in a table. This is useful for manual screening, candidate sequence organization, and exporting to downstream analysis workflows.

**Complete Results:** Provides a ZIP download containing the main files generated in the VenusMine run, such as the tree image, label table, intermediate FASTA files, search results, and logs.

**Processing Log:** Shows real-time status for each VenusMine step, including database path, hit counts, clustering results, and error messages. If a run fails, check this tab first for issues such as missing MMseqs2, missing UniRef90 database, FoldSeek download failure, or no sequences matching the protected region.

### 6.4 Recommended Usage

  * The first run may take a long time if UniRef90 needs to be auto-downloaded, and it requires sufficient disk space. For production use, prepare the database in advance and set `VENUSMINE_MMSEQS_DATABASE_PATH`.
  * If no sequences match the protected region, relax the **Start Position / End Position** range or choose a more conserved key segment.
  * If too few candidate sequences are found, increase **Max Sequences**, increase **Iterations**, or relax the **E-value Threshold**.
  * If the candidate set is too redundant, adjust **Min Sequence Identity** or lower the final **Top N Results**.
  * VenusMine provides candidate discovery and prioritization results. Use the output together with downstream function prediction, structural inspection, and experimental validation.

-----

## 7\. Functional Residue

This module allows users to **freely select the underlying PLM model** for high-precision localization and prediction of critical functional residues within the protein sequence.

### 7.1 Task Configuration and Sequence Input

**Model Configuration:**

  * **Select Model:** This is a key advanced configuration item where users can freely select the underlying PLM engine.
      * **Available Models:** **ESM2-650M, Ankh-large, ProtT5-xl-uniref50**.
  * **Select Task:** Users must select the specific goal for the protein functional residue prediction.
      * **Activity Site:** Predicts key residue positions in the protein sequence responsible for **catalytic or biological function**.
      * **Binding Site:** Predicts key residue positions where the protein **binds** with ligands, ions, or other molecules.
      * **Conserved Site:** Predicts residue positions that are **highly retained** during mutation, which are usually critical for protein structure and function.
      * **Motif:** Predicts short amino acid **patterns** in the sequence that may form a specific structural or functional feature.

**Data Input:**

  * **Upload FASTA File:** The selected tab for uploading sequence via file, supporting the **FASTA format**.
  * **Paste FASTA Content:** The alternative tab for directly entering the sequence into the text box.
  * **Uploaded Protein Sequence:** This area displays the **raw amino acid sequence** read from the input file or pasted content.

**Configure AI Analysis (Optional):**

  * Users can choose to enable **Enable AI Summary** to receive a **textual evaluation** of the results from a professional **AI biology expert** after the prediction is complete.

### 7.2 Execute Prediction

1.  **Ensure all model configuration parameters are set correctly.**
2.  Click the **"Start Prediction"** button to start the prediction process.
3.  The system will display **prediction progress and status information.**
4.  To abort the prediction, click the **"Abort"** button.

### 7.3 Results Display Functionality

**Status:** Provides **real-time feedback** on prediction progress.

**Raw Results:** This table displays the **residue-by-residue prediction results**. The table content treats each amino acid in the sequence as a separate prediction object:

  * **Position:** The index number of the residue in the sequence (starting from 0 or 1).
  * **Residue:** The amino acid letter corresponding to that position.
  * **Predicted Label:** The model's classification result, usually in binary form (e.g., **1** represents a target residue, **0** represents not a target residue).
  * **Probability:** The model's **confidence score** (between 0 and 1) that the residue belongs to the target category. A higher score indicates a greater likelihood that the residue is the target site.

**Prediction Heatmap:** This heatmap visualizes the **probability distribution** of the residue predictions across the entire sequence in the form of a linear bar chart.

  * **Chart Type:** This is a **one-dimensional bar chart or probability distribution graph** where the horizontal axis (**X-axis**) represents the **Residue Position**.
  * **Information Display:** The changes in **color or height** in the chart intuitively map the **Probability** or intensity for each residue position predicted by the model as a target site (e.g., an activity site). The inference is based on the PLM model selected by the user, which helps users quickly identify potential functional sites.

**AI Expert Analysis:** A professional **AI biology expert** evaluates the prediction results, providing in-depth interpretation and experimental suggestions in a **textual format**.

**Download Results:** Users can click this button to **download all detailed prediction data** from the table to their local computer for further data processing and archival.

-----

