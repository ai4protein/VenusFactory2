"""
id: protein-mutation-tool
name: protein mutation tool
description: A comprehensive tool for protein mutation analysis using AI models. Supports both saturation mutagenesis (all possible mutations) and specific mutation scoring.

WHEN TO USE THIS TOOL:
- User asks for protein mutation analysis
- User mentions ESM-2, ESM-1v, or other supported models
- User uploads protein files (FASTA/PDB) and wants analysis
- User asks for saturation mutagenesis or specific mutation scoring

HOW TO USE THIS TOOL:

SIMPLE APPROACH (Recommended):
1. Call help_with_protein_analysis(user_request) to analyze user's request
2. If user confirms, call analyze_protein_mutations(model, task) for automatic file detection

ADVANCED APPROACH:
1. Call get_recent_uploaded_files() to see what files user uploaded
2. Call run_protein_analysis() with exact filenames from step 1

EXAMPLE WORKFLOW:
1. User asks "help me use ESM-2 for saturation analysis"
2. Call help_with_protein_analysis("help me use ESM-2 for saturation analysis")
3. If user says "yes", call analyze_protein_mutations("esm2", "saturation")

USAGE INSTRUCTIONS:
1. User uploads protein files (FASTA for sequence models, PDB for structure models)
2. AI calls run_protein_analysis() with EXACT filenames from user uploads
3. Tool automatically claims files and submits PBS jobs
4. Results are sent to user's email
5. Do not send the file path to the user, just send the result to the user's email

SUPPORTED MODELS:
- Sequence models (require FASTA files): esm2, esm1v
- Structure models (require PDB files): mifst, prosst, protssn

TASK TYPES:
- saturation: Analyze all possible mutations (1 input file)
- specific: Score specific mutations (2 input files: structure + CSV)

CRITICAL: AI must provide exact filenames that user uploaded (e.g., ["A0A0C5B5G6.fasta"])

Features:
- Automatic file claiming and user isolation
- Enhanced file search with flexible matching
- File type validation based on selected model
- Secure file handling and job submission
- Configurable PBS job parameters
- Auto-fill user email from profile
- Comprehensive error handling and debugging
- Support for OpenWebUI file upload format
version: 2.8.0
"""
import os
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field

# DATA_DIR: default DATA_DIR for OpenWebUI
DATA_DIR = Path(os.getenv("DATA_DIR", "~/openwebui_data"))
# PROJECT_ROOT: the absolute path of the project root
PROJECT_ROOT = "/home/tanyang/workspace/VenusFactory"
# UPLOAD_DIR: the public, temporary upload "inbox"
UPLOAD_DIR = DATA_DIR.expanduser() / "uploads"
# STORAGE_DIR: the root directory for all user private files
STORAGE_DIR = DATA_DIR.expanduser() / "storage"
# RESULT_DIR: the directory for analysis results
RESULT_DIR = DATA_DIR.expanduser() / "results"
# JOB_TRACKER_FILE: the JSON file for tracking PBS job status
JOB_TRACKER_FILE = DATA_DIR.expanduser() / "job_tracker.json"
# TEMP_SCRIPT_DIR: the directory for temporary generated PBS scripts
TEMP_SCRIPT_DIR = DATA_DIR.expanduser() / "temp_pbs_scripts"

class Tools:
    class Valves(BaseModel):
        """Global configuration valves for PBS job settings and user management"""
        # PBS Job Configuration
        PBS_QUEUE: str = Field(
            default="gpu", 
            description="PBS queue name (e.g., 'gpu', 'huge', 'ai')"
        )
        PBS_NCPUS: int = Field(
            default=4, 
            description="Number of CPU cores needed for the job"
        )
        PBS_NGPUS: int = Field(
            default=1, 
            description="Number of GPUs needed for the job"
        )
        PBS_MEM: str = Field(
            default="10gb", 
            description="Memory requirement (e.g., '8gb', '16gb', '32gb')"
        )
        PBS_WALLTIME: str = Field(
            default="24:00:00", 
            description="Maximum job runtime in format 'HH:MM:SS'"
        )
        # User Management
        AUTO_FILL_EMAIL: bool = Field(
            default=True,
            description="Whether to automatically fill email from user profile.",
        )

    def __init__(self):
        """Initialize the tool and ensure all required directories exist"""
        self.valves = self.Valves()
        
        # Create all required directories
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"Protein mutation tool loaded. Project root: {PROJECT_ROOT}")
        print(f"PBS configuration: queue={self.valves.PBS_QUEUE}, ncpus={self.valves.PBS_NCPUS}, ngpus={self.valves.PBS_NGPUS}, mem={self.valves.PBS_MEM}, walltime={self.valves.PBS_WALLTIME}")

    def get_user_directories(self, __user__: Optional[Dict] = None) -> tuple[str, str]:
        """
        Get user ID and conversation ID for file operations.
        If not available, use default values with timestamp.
        
        :param __user__: User information
        :return: Tuple of (user_id, conversation_id)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not __user__:
            user_id = f"default_user_{timestamp}"
            conversation_id = f"default_chat_{timestamp}"
        else:
            # Extract user_id with fallbacks
            user_id = (
                __user__.get("id") or 
                __user__.get("user_id") or 
                __user__.get("username") or 
                f"default_user_{timestamp}"
            )
            
            # Extract conversation_id with fallbacks
            conversation_id = (
                __user__.get("chat_id") or 
                __user__.get("conversation_id") or 
                __user__.get("session_id") or 
                f"default_chat_{timestamp}"
            )
        
        print(f"DEBUG: Using user_id: {user_id}, conversation_id: {conversation_id}")
        return user_id, conversation_id

    def get_user_email(self, __user__: Optional[Dict] = None) -> Optional[str]:
        """
        Get user email from user information.
        
        :param __user__: User information
        :return: Email address if available, None otherwise
        """
        if not __user__:
            return None
            
        return __user__.get("email")

    def claim_and_move_file(self, original_filename: str, __user__: Optional[Dict] = None) -> str:
        """
        Find the uploaded file in uploads directory and move it to user's private storage.
        
        :param original_filename: The original filename uploaded by the user (e.g., 'my_protein.pdb')
        :param __user__: User information for directory creation
        :return: Success message with secure path or error message
        """
        print(f"DEBUG: Claiming file: {original_filename}")
        print(f"DEBUG: User info: {__user__}")
        print(f"DEBUG: Upload directory: {UPLOAD_DIR}")
        print(f"DEBUG: Upload directory exists: {UPLOAD_DIR.exists()}")
        
        # Get user directories
        user_id, conversation_id = self.get_user_directories(__user__)

        # Validate filename
        if "/" in original_filename or "\\" in original_filename:
            return "❌ Error: Original filename cannot contain path separators."

        try:
            # Search for the file in uploads directory
            now = datetime.now()
            found_file = None
            all_files = []
            
            if not UPLOAD_DIR.exists():
                return f"❌ Error: Upload directory {UPLOAD_DIR} does not exist."
            
            for filepath in UPLOAD_DIR.iterdir():
                if filepath.is_file():
                    all_files.append(filepath.name)
                    file_age = now - datetime.fromtimestamp(filepath.stat().st_mtime)
                    print(f"DEBUG: Found file: {filepath.name}, age: {file_age}")
                    
                    # Check if filename ends with original filename and was uploaded recently
                    # Open WebUI format: UUID_original_filename (e.g., 8acb70ab-b146-4b1b-84ca-158a4e85a212_C3H.fasta)
                    if filepath.name.endswith(original_filename) and file_age < timedelta(minutes=10):
                        found_file = filepath
                        print(f"DEBUG: Matched file: {filepath.name}")
                        break
                    
                    # Also check for exact match (in case file was uploaded directly)
                    elif filepath.name == original_filename and file_age < timedelta(minutes=10):
                        found_file = filepath
                        print(f"DEBUG: Exact match file: {filepath.name}")
                        break

            print(f"DEBUG: All files in upload directory: {all_files}")
            
            if not found_file:
                # Try a more flexible search - look for files with similar names
                print(f"DEBUG: No exact match found, trying flexible search...")
                for filepath in UPLOAD_DIR.iterdir():
                    if filepath.is_file():
                        file_age = now - datetime.fromtimestamp(filepath.stat().st_mtime)
                        # Check if the original filename is contained in the uploaded filename
                        if original_filename in filepath.name and file_age < timedelta(minutes=10):
                            found_file = filepath
                            print(f"DEBUG: Flexible match found: {filepath.name}")
                            break
                
                if not found_file:
                    return f"❌ Error: No file found for '{original_filename}' in the last 10 minutes. Available files: {all_files[:5]}... Please confirm you have just successfully uploaded."

            # Create user's private directory
            user_storage_dir = STORAGE_DIR / user_id / conversation_id
            user_storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Move file to user's private directory, restoring original filename
            secure_file_path = user_storage_dir / original_filename
            shutil.move(found_file, secure_file_path)
            
            print(f"DEBUG: File moved successfully to: {secure_file_path}")
            print(f"DEBUG: File size: {secure_file_path.stat().st_size} bytes")

            return f"✅ File '{original_filename}' claimed and moved to: \n`{secure_file_path.absolute()}`"

        except Exception as e:
            print(f"DEBUG: Exception during file claiming: {e}")
            return f"❌ Error: Failed to claim file: {e}"

    def run_protein_analysis(
        self,
        task_type: Literal["saturation", "specific"],
        model_name: Literal["esm2", "esm1v", "mifst", "prosst", "protssn"],
        input_filenames: List[str],
        user_email: Optional[str] = None,
        __user__: Optional[Dict] = None,
        job_name: Optional[str] = None,
        queue: Optional[str] = None,
        ncpus: Optional[int] = None,
        ngpus: Optional[int] = None,
        mem: Optional[str] = None,
        walltime: Optional[str] = None,
    ) -> str:
        """
        Submit protein analysis job: claim files first, then submit to PBS.
        
        IMPORTANT: You MUST provide the exact filename(s) that the user uploaded.
        For example, if user uploaded "A0A0C5B5G6.fasta", you must pass ["A0A0C5B5G6.fasta"].
        
        :param task_type: The type of task to execute:
            - "saturation": Analyze all possible mutations (requires 1 input file)
            - "specific": Score specific mutations (requires 2 input files: structure + CSV)
        :param model_name: The model to use:
            - Sequence models (require FASTA): "esm2", "esm1v"
            - Structure models (require PDB): "mifst", "prosst", "protssn"
        :param input_filenames: List of EXACT filenames that user uploaded (e.g., ["A0A0C5B5G6.fasta"])
        :param user_email: Email for result notification (auto-filled if not provided)
        :param __user__: User information (automatically provided by system)
        :param job_name: Optional PBS job name
        :param queue: Optional PBS queue (uses default if not provided)
        :param ncpus: Optional CPU cores (uses default if not provided)
        :param ngpus: Optional GPUs (uses default if not provided)
        :param mem: Optional memory (uses default if not provided)
        :param walltime: Optional walltime (uses default if not provided)
        
        Examples:
        - For saturation analysis: input_filenames=["A0A0C5B5G6.fasta"]
        - For specific scoring: input_filenames=["protein.pdb", "mutations.csv"]
        """
        # Use configurable defaults
        queue = queue or self.valves.PBS_QUEUE
        ncpus = ncpus or self.valves.PBS_NCPUS
        ngpus = ngpus or self.valves.PBS_NGPUS
        mem = mem or self.valves.PBS_MEM
        walltime = walltime or self.valves.PBS_WALLTIME
        
        print(f"DEBUG: Starting protein analysis - task: {task_type}, model: {model_name}")
        print(f"DEBUG: PBS settings - queue: {queue}, ncpus: {ncpus}, ngpus: {ngpus}, mem: {mem}, walltime: {walltime}")
        
        # Get user directories
        user_id, conversation_id = self.get_user_directories(__user__)

        # Auto-fill email if not provided
        if not user_email and self.valves.AUTO_FILL_EMAIL:
            user_email = self.get_user_email(__user__)
            if user_email:
                print(f"DEBUG: Auto-filled email: {user_email}")
        
        if not user_email:
            return "❌ Error: user_email is required. Please provide an email address or ensure your user profile has an email configured."

        # Step 1: Claim all uploaded files
        claimed_files = []
        failed_files = []
        
        for filename in input_filenames:
            print(f"DEBUG: Attempting to claim file: {filename}")
            claim_result = self.claim_and_move_file(filename, __user__)
            print(f"DEBUG: Claim result: {claim_result}")
            
            if claim_result.startswith("✅"):
                # Extract secure path from success message
                secure_path = claim_result.split("`")[1]
                print(f"DEBUG: Extracted secure path: {secure_path}")
                claimed_files.append(secure_path)
            else:
                failed_files.append((filename, claim_result))
                print(f"DEBUG: Failed to claim file: {filename}")

        print(f"DEBUG: All claimed files: {claimed_files}")
        print(f"DEBUG: Failed files: {failed_files}")
        
        # If any files failed to claim, provide detailed error information
        if failed_files:
            error_msg = "❌ Error: Failed to claim the following files:\n"
            for filename, error in failed_files:
                error_msg += f"- {filename}: {error}\n"
            
            # Add helpful debugging information
            error_msg += "\n💡 Troubleshooting tips:\n"
            error_msg += "1. Make sure you have just uploaded the file(s)\n"
            error_msg += "2. Check that the filename matches exactly (case-sensitive)\n"
            error_msg += "3. Try uploading the file again\n"
            error_msg += "4. Use the list_upload_files() method to see available files\n"
            
            return error_msg

        if not claimed_files:
            return "❌ Error: No files were successfully claimed. Please check your file uploads."

        # Step 2: Validate and build command
        command = ""
        output_file_path_str = ""
        sequence_models = ["esm2", "esm1v"]
        structure_models = ["mifst", "prosst", "protssn"]
        
        if task_type == "saturation":
            if len(claimed_files) != 1:
                return "❌ Error: Saturation analysis needs exactly 1 input file (FASTA/PDB)."
            
            source_file = Path(claimed_files[0])
            print(f"DEBUG: Source file path: {source_file}")
            print(f"DEBUG: Source file exists: {source_file.exists()}")
            print(f"DEBUG: Source file is file: {source_file.is_file()}")
            
            if not source_file.is_file():
                return f"❌ Error: File not found: {source_file}"
            
            # Validate file type compatibility
            is_compatible, error_msg = self.validate_file_model_compatibility(str(source_file), model_name)
            if not is_compatible:
                return f"❌ Error: {error_msg}"
            
            # Create results directory
            output_dir = RESULT_DIR / user_id / conversation_id / "saturation"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{source_file.stem}_saturation.csv"
            output_file_path_str = str(output_file.absolute())
            print(f"DEBUG: Output file path: {output_file_path_str}")

            # Build command
            if model_name in sequence_models:
                command = (
                    f"python src/mutation/models/{model_name}.py "
                    f"--fasta_file {source_file.absolute()} "
                    f"--output_csv {output_file_path_str}"
                )
            else:
                command = (
                    f"python src/mutation/models/{model_name}.py "
                    f"--pdb_file {source_file.absolute()} "
                    f"--output_csv {output_file_path_str}"
                )
            print(f"DEBUG: Generated command: {command}")
            final_job_name = job_name or f"saturation_{source_file.stem}"

        elif task_type == "specific":
            if len(claimed_files) != 2:
                return "❌ Error: Specific scoring needs exactly 2 input files (1 structure file, 1 CSV file)."
            
            structure_file_str = next((f for f in claimed_files if Path(f).suffix.lower() in ['.pdb', '.fasta']), None)
            mutation_csv_str = next((f for f in claimed_files if Path(f).suffix.lower() == '.csv'), None)

            if not structure_file_str or not mutation_csv_str:
                return "❌ Error: Input files must contain a structure file (.pdb/.fasta) and a mutation list (.csv)."
            
            structure_file = Path(structure_file_str)
            mutation_csv = Path(mutation_csv_str)
            
            # Validate file type compatibility
            is_compatible, error_msg = self.validate_file_model_compatibility(str(structure_file), model_name)
            if not is_compatible:
                return f"❌ Error: {error_msg}"
            
            # Create results directory
            output_dir = RESULT_DIR / user_id / conversation_id / "specific_scoring"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{structure_file.stem}_specific_scores.csv"
            output_file_path_str = str(output_file.absolute())

            # Build command
            if model_name in sequence_models:
                command = (
                    f"python src/mutation/models/{model_name}.py "
                    f"--fasta_file {structure_file.absolute()} "
                    f"--mutations_csv {mutation_csv.absolute()} "
                    f"--output_csv {output_file_path_str}"
                )
            else:
                command = (
                    f"python src/mutation/models/{model_name}.py "
                    f"--pdb_file {structure_file.absolute()} "
                    f"--mutations_csv {mutation_csv.absolute()} "
                    f"--output_csv {output_file_path_str}"
                )
            final_job_name = job_name or f"scoring_{structure_file.stem}"
        
        else:
            return f"❌ Error: Unknown task type: {task_type}"
        
        # Step 3: Submit to PBS
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_base = DATA_DIR / "pbs_logs" / f"{final_job_name}_{timestamp}"
            log_file_base.parent.mkdir(parents=True, exist_ok=True)

            pbs_script_content = f"""
#!/bin/bash
#PBS -N {final_job_name}
#PBS -q {queue}
#PBS -l walltime={walltime}
#PBS -l select=1:ncpus={ncpus}:ngpus={ngpus}:mem={mem}
#PBS -o {log_file_base}.out
#PBS -e {log_file_base}.err

echo "Job started on $(hostname) at $(date)"
cd {PROJECT_ROOT} || exit 1

zsh
source ~/.zshrc
source /home/tanyang/miniconda3/bin/activate /home/tanyang/miniconda3/envs/agent

{command}

echo "Job finished at $(date) with exit code $?"
"""
            script_path = TEMP_SCRIPT_DIR / f"{final_job_name}_{timestamp}.pbs"
            script_path.write_text(pbs_script_content)

            result = subprocess.run(f"qsub {script_path}", shell=True, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip()
            
            self._record_job(job_id, user_email, task_type, output_file_path_str, user_id, conversation_id)
            
            return f"✅ '{task_type}' job submitted successfully!\n- Job ID: {job_id}\n- Result will be sent to: {user_email}\n- Output file: {output_file_path_str}\n- PBS settings: queue={queue}, ncpus={ncpus}, ngpus={ngpus}, mem={mem}, walltime={walltime}"

        except subprocess.CalledProcessError as e:
            return f"❌ Job submission failed: {e.stderr}"
        except Exception as e:
            return f"❌ Error: Unknown error when creating or submitting script: {e}"

    def analyze_protein_mutations(
        self,
        model: str,
        task: str = "saturation",
        __user__: Optional[Dict] = None,
    ) -> str:
        """
        Simplified wrapper for protein mutation analysis.
        This function automatically finds uploaded files and runs the analysis.
        
        :param model: Model name (esm2, esm1v, mifst, prosst, protssn)
        :param task: Task type (saturation or specific)
        :param __user__: User information (automatically provided)
        :return: Analysis result or error message
        """
        print(f"DEBUG: Simplified analysis called - model: {model}, task: {task}")
        
        # First, get recent uploaded files
        files_info = self.get_recent_uploaded_files(__user__, minutes=60)
        print(f"DEBUG: Files info: {files_info}")
        
        if "No files uploaded" in files_info:
            return f"❌ No files found. Please upload your protein file first.\n\n{files_info}"
        
        # Extract filenames from the files info
        lines = files_info.split('\n')
        filenames = []
        for line in lines:
            if line.startswith('• ') and '(' in line:
                # Extract filename from "• filename.ext (TYPE, size, age)"
                filename = line.split('• ')[1].split(' (')[0]
                filenames.append(filename)
        
        if not filenames:
            return f"❌ Could not extract filenames from uploaded files.\n\n{files_info}"
        
        print(f"DEBUG: Extracted filenames: {filenames}")
        
        # Determine task type and validate file count
        if task == "saturation":
            if len(filenames) >= 1:
                input_files = [filenames[0]]  # Use the most recent file
            else:
                return "❌ Saturation analysis requires at least 1 protein file."
        elif task == "specific":
            if len(filenames) >= 2:
                # Find structure file and CSV file
                structure_file = None
                csv_file = None
                for filename in filenames:
                    if filename.endswith(('.fasta', '.pdb')):
                        structure_file = filename
                    elif filename.endswith('.csv'):
                        csv_file = filename
                
                if structure_file and csv_file:
                    input_files = [structure_file, csv_file]
                else:
                    return f"❌ Specific scoring requires 1 structure file (.fasta/.pdb) and 1 CSV file. Found: {filenames}"
            else:
                return "❌ Specific scoring requires at least 2 files (structure + CSV)."
        else:
            return f"❌ Unknown task type: {task}. Use 'saturation' or 'specific'."
        
        print(f"DEBUG: Using input files: {input_files}")
        
        # Run the analysis
        return self.run_protein_analysis(
            task_type=task,
            model_name=model,
            input_filenames=input_files,
            __user__=__user__
        )

    def help_with_protein_analysis(self, user_request: str, __user__: Optional[Dict] = None) -> str:
        """
        AI helper function that analyzes user requests and provides guidance on how to proceed.
        This function should be called first when user asks for protein analysis.
        
        :param user_request: The user's original request/query
        :param __user__: User information (automatically provided)
        :return: Guidance on how to proceed with the analysis
        """
        print(f"DEBUG: Help function called with request: {user_request}")
        
        # Check for uploaded files first
        files_info = self.get_recent_uploaded_files(__user__, minutes=60)
        
        # Analyze the user request
        request_lower = user_request.lower()
        
        # Determine model from request
        model = None
        if "esm2" in request_lower:
            model = "esm2"
        elif "esm1v" in request_lower:
            model = "esm1v"
        elif "mifst" in request_lower:
            model = "mifst"
        elif "prosst" in request_lower:
            model = "prosst"
        elif "protssn" in request_lower:
            model = "protssn"
        else:
            # Default to esm2 for sequence analysis
            model = "esm2"
        
        # Determine task type
        task = "saturation"
        if "specific" in request_lower or "score" in request_lower:
            task = "specific"
        
        # Check if files are available
        if "No files uploaded" in files_info:
            return f"""❌ No protein files found. 

To proceed with {model.upper()} {task} analysis, please:

1. Upload your protein file:
   - For {model}: Upload a FASTA file (.fasta) with protein sequence
   - For structure models: Upload a PDB file (.pdb) with protein structure

2. After uploading, ask me again to run the analysis.

{self.get_supported_models_and_files()}"""
        
        # Files are available, provide guidance
        guidance = f"""✅ I found uploaded files and can help with {model.upper()} {task} analysis.

{files_info}

To proceed with the analysis, I will:
1. Use the most recent protein file for analysis
2. Submit a PBS job to run {model.upper()} {task} analysis
3. Send results to your email when complete

Would you like me to start the analysis now? Just say "yes" or "proceed" and I'll run it immediately.

Or if you want to use specific files, tell me which ones to use."""
        
        return guidance

    def _record_job(self, job_id: str, user_email: str, job_type: str, output_path: str, user_id: str, conversation_id: str):
        """Record job information to JSON tracking file"""
        try:
            if not JOB_TRACKER_FILE.exists():
                tracker = {"active_jobs": {}, "completed_jobs": {}}
            else:
                with open(JOB_TRACKER_FILE, 'r') as f:
                    tracker = json.load(f)

            tracker["active_jobs"][job_id] = {
                "user_email": user_email,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "job_type": job_type,
                "output_file_path": output_path,
                "submission_time": datetime.now().isoformat(),
                "status": "Queued"
            }

            with open(JOB_TRACKER_FILE, 'w') as f:
                json.dump(tracker, f, indent=4)
        except Exception as e:
            print(f"!!! Error: Failed to record job {job_id} to {JOB_TRACKER_FILE}: {e}")

    def get_recent_uploaded_files(self, __user__: Optional[Dict] = None, minutes: int = 30) -> str:
        """
        Get a list of recently uploaded files that can be used for analysis.
        This helps AI understand what files are available for processing.
        
        :param __user__: User information (automatically provided by system)
        :param minutes: How many minutes back to look for files (default: 30)
        :return: Formatted string with recent files information
        """
        if not UPLOAD_DIR.exists():
            return "📁 Upload directory does not exist."
        
        files = []
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=minutes)
        
        for filepath in UPLOAD_DIR.iterdir():
            if filepath.is_file():
                file_age = now - datetime.fromtimestamp(filepath.stat().st_mtime)
                if file_age < timedelta(minutes=minutes):
                    # Extract original filename from OpenWebUI format (UUID_original_filename)
                    original_name = filepath.name
                    if "_" in filepath.name:
                        original_name = "_".join(filepath.name.split("_")[1:])
                    
                    files.append({
                        'original_name': original_name,
                        'full_name': filepath.name,
                        'size': filepath.stat().st_size,
                        'age_minutes': file_age.total_seconds() / 60,
                        'extension': filepath.suffix.lower()
                    })
        
        if not files:
            return f"📁 No files uploaded in the last {minutes} minutes."
        
        # Sort by age (newest first)
        files.sort(key=lambda x: x['age_minutes'])
        
        result = f"📁 Files uploaded in the last {minutes} minutes:\n\n"
        
        for f in files:
            age_str = f"{f['age_minutes']:.1f} minutes ago"
            file_type = "FASTA" if f['extension'] == '.fasta' else "PDB" if f['extension'] == '.pdb' else "CSV" if f['extension'] == '.csv' else f['extension'].upper()
            result += f"• {f['original_name']} ({file_type}, {f['size']} bytes, {age_str})\n"
        
        result += f"\n💡 To use these files, pass the ORIGINAL filename to run_protein_analysis().\n"
        result += f"   Example: input_filenames=[\"{files[0]['original_name']}\"]\n"
        
        return result

    def get_supported_models_and_files(self) -> str:
        """
        Get information about supported models and their required file types.
        
        :return: Formatted string with model and file type information
        """
        info = "🔬 Supported Models and File Types:\n\n"
        
        info += "📄 Sequence-based Models (require FASTA files):\n"
        info += "- esm2: ESM-2 protein language model\n"
        info += "- esm1v: ESM-1v protein language model\n\n"
        
        info += "🏗️ Structure-based Models (require PDB files):\n"
        info += "- mifst: MIF-ST structure-based model\n"
        info += "- prosst: ProSST structure-based model\n"
        info += "- protssn: ProtSSN structure-based model\n\n"
        
        info += "📋 Task Types:\n"
        info += "- saturation: Analyze all possible mutations (requires 1 input file)\n"
        info += "- specific: Score specific mutations (requires 2 input files: structure + CSV)\n\n"
        
        info += "💡 Tips:\n"
        info += "- FASTA files should contain protein sequences\n"
        info += "- PDB files should contain protein structures\n"
        info += "- CSV files for specific scoring should have columns: position, wild_type, mutant\n"
        
        return info

    def validate_file_model_compatibility(self, file_path: str, model_name: str) -> tuple[bool, str]:
        """
        Validate if a file is compatible with the specified model.
        
        :param file_path: Path to the file to validate
        :param model_name: Name of the model to check compatibility with
        :return: Tuple of (is_compatible, error_message)
        """
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return False, f"File not found: {file_path}"
            
            file_extension = file_path_obj.suffix.lower()
            sequence_models = ["esm2", "esm1v"]
            structure_models = ["mifst", "prosst", "protssn"]
            
            if model_name in sequence_models:
                if file_extension != '.fasta':
                    return False, f"Model '{model_name}' requires a FASTA file (.fasta), but you provided a {file_extension} file."
            elif model_name in structure_models:
                if file_extension != '.pdb':
                    return False, f"Model '{model_name}' requires a PDB file (.pdb), but you provided a {file_extension} file."
            else:
                return False, f"Unknown model: {model_name}"
            
            return True, "File type is compatible with the model."
            
        except Exception as e:
            return False, f"Error validating file: {e}"

    def list_upload_files(self, __user__: Optional[Dict] = None) -> str:
        """List all files in the upload directory for debugging purposes"""
        if not UPLOAD_DIR.exists():
            return "📁 Upload directory does not exist."
        
        files = []
        now = datetime.now()
        
        for filepath in UPLOAD_DIR.iterdir():
            if filepath.is_file():
                file_age = now - datetime.fromtimestamp(filepath.stat().st_mtime)
                files.append({
                    'name': filepath.name,
                    'size': filepath.stat().st_size,
                    'age': file_age,
                    'age_minutes': file_age.total_seconds() / 60
                })
        
        if not files:
            return "📁 No files found in upload directory."
        
        # Sort by age (newest first)
        files.sort(key=lambda x: x['age'])
        
        file_list = []
        for f in files[:10]:  # Show only the 10 most recent files
            age_str = f"{f['age_minutes']:.1f} minutes ago"
            file_list.append(f"- {f['name']} ({f['size']} bytes, {age_str})")
        
        return f"📁 Recent files in upload directory:\n" + "\n".join(file_list)

    def get_user_files(self, __user__: Optional[Dict] = None) -> str:
        """List all files in the user's private directory"""
        user_id, conversation_id = self.get_user_directories(__user__)
        user_storage_dir = STORAGE_DIR / user_id / conversation_id

        if not user_storage_dir.exists():
            return "📁 No files found in your private directory."

        files = list(user_storage_dir.iterdir())
        if not files:
            return "📁 No files found in your private directory."

        file_list = "\n".join([f"- {f.name} ({f.stat().st_size} bytes)" for f in files])
        return f"📁 Files in your private directory:\n{file_list}"

    def get_job_status(self, job_id: str, __user__: Optional[Dict] = None) -> str:
        """Get the status of a specific job"""
        if not JOB_TRACKER_FILE.exists():
            return "❌ Error: No job tracker file found."

        try:
            with open(JOB_TRACKER_FILE, 'r') as f:
                tracker = json.load(f)

            if job_id in tracker["active_jobs"]:
                job_info = tracker["active_jobs"][job_id]
                return f"📊 Job {job_id} status: {job_info['status']}\n- Type: {job_info['job_type']}\n- Submitted: {job_info['submission_time']}"
            elif job_id in tracker["completed_jobs"]:
                job_info = tracker["completed_jobs"][job_id]
                return f"✅ Job {job_id} completed\n- Type: {job_info['job_type']}\n- Completed: {job_info.get('completion_time', 'unknown')}"
            else:
                return f"❌ Job {job_id} not found in tracker."

        except Exception as e:
            return f"❌ Error reading job tracker: {e}"
