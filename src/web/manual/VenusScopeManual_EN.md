# VenusScope Function Manual

### 1. Introduction

**VenusScope** is a comprehensive advanced analysis tool provided by the VenusFactory platform. It is designed to perform up to four key prediction and analysis tasks on a single protein sequence via **one-click operation**, integrating all results into a clear, comprehensive **"Comprehensive Protein Analysis Report"**. This tool significantly simplifies the user workflow, offering full insights from mutation optimization to physicochemical properties, making it the premier entry point for protein engineering and basic research.

---

### 2. Interface Overview

The VenusScope interface is designed to be intuitive, primarily divided into three sections: Data Input, Analysis Type Selection, and Report Output.


**Data Input**:
- **Upload Protein File (.fasta, .fa, .pdb) / Paste Protein Content**: Users can upload FASTA or PDB files, or directly paste the sequence, to provide the protein for analysis.
- **Uploaded Protein Sequence**: Displays the amino acid sequence that has been input or uploaded.

**Analysis Type Selection**:
- Provides checkboxes for four core prediction tasks, allowing users to freely select the analysis types to be integrated into the report based on their needs.
- After selection, clicking the **Generate Brief Report** button starts the analysis.

**Report Output**:
- After the analysis is complete, the interface displays a report summary and experimental recommendations, and provides the complete **"Comprehensive Protein Analysis Report"** for user review.

---

### 3. Supported Analysis Type

VenusScope allows users to customize the report by checking the boxes for the prediction modules they want to include, supporting the following four types:

- **Protein Mutation Prediction**: Predicts the effect of single point mutations on protein function (e.g., activity, stability) to guide directed evolution optimization.
- **Protein Function Prediction**: Predicts the overall function and properties of the sequence, such as solubility, subcellular localization, and stability.
- **Functional Residue Prediction**: Identifies key functional sites, such as active sites, binding sites, and conserved residues.
- **Physical & Chemical Properties**: Calculates the sequence-based fundamental attributes of the protein, such as molecular weight, theoretical pI, and instability index.

---

### 4. Report Content in Detail

The generated **"Comprehensive Protein Analysis Report"** is a structured document that presents all prediction results in detail:

- **Comprehensive Summary**:
    - Located at the top of the report, it provides basic protein information (e.g., molecular weight, theoretical pI) and a brief overview and assessment of the overall prediction results.

- **Mutation Prediction Analysis**:
    - **Top beneficial Mutations**: A table listing the highest-scoring mutants, including **Rank**, **Position**, **Mutation**, **Prediction Score**, and **Potential Function** description.
    - **Secondary Beneficial Mutations**: Lists lower-ranked beneficial mutations, providing additional optimization options.
    - **Key Site Optimization**: Focuses on potential optimization mutations at critical functional sites (e.g., active sites).

- **Protein Function Analysis**:
    - Displays overall function prediction results in a table format, including: **Functional/Property Assessment**, **Predicted Value/Class**, **Confidence**, and **Description**.
    - Report content covers: solubility, subcellular localization, metal ion binding, stability, sorting signal, and optimum temperature.

- **Functional Residue Prediction**:
    - Displays residue-level prediction results in a list or table format, including **Binding Site Prediction**, **Functional Residues Prediction**, and **Functional Motifs**.
    - The report provides sequence position and probability for key sites, helping users precisely locate functional regions.

- **Physical & Chemical Properties**:
    - **Biophysical Characterization**: Lists calculated sequence-based properties (e.g., molecular weight, pI, aromaticity).
    - **Stability Considerations**: Provides the **stability prediction result** based on the Instability Index (e.g., Predicted as unstable protein) and corresponding experimental suggestions.

- **Experimental Recommendations**:
    - Based on all prediction results, the system provides summary suggestions for **function/stability**, **technical considerations**, and **experimental protocol suggestions**, directly guiding the user's wet-lab work.

- **Conclusion**:
    - Provides a final summary of the report, reiterating the protein's key characteristics and the most important directions for experimental optimization.

---

### 5. User Guide

VenusScope is dedicated to providing the most concise and efficient comprehensive protein analysis experience.

- **Operation Flow**:
    1.  **Input Data**: Provide the protein sequence/structure by uploading a file or pasting the sequence.
    2.  **Select Tasks**: In the **Select Analysis Types** area, check the boxes for one or more analysis modules you wish to include in the report (checking all is recommended for the most comprehensive insight).
    3.  **Generate Report**: Click the **Generate Brief Report** button to start the analysis.
    4.  **Review Results**: The system will generate the "Comprehensive Protein Analysis Report," which you can scroll through to view detailed analyses and find key conclusions in the **Experimental Recommendations** section.

- **Usage Tips**:
    - **Prioritize PDB Files**: If the target protein has an resolved or high-quality predicted structure (e.g., AlphaFold structure), please upload a **PDB file** to ensure that structure-based analyses (like mutation prediction) achieve the highest accuracy.
    - **Cross-Functional Integration**: VenusScope's greatest value lies in its integrated analysis. For example, comparing key sites from **Functional Residue Prediction** with beneficial mutations from **Mutation Prediction Analysis** enables more precise rational design.