"""
id: protein-mutation-tool
name: protein mutation tool
description: AI tool for protein mutation design. User MUST specify a model name. Returns CSV scoring results only.

AI INTERACTION GUIDE:
When user mentions protein mutation design with a specific model, follow this workflow:

1. DETECT MODEL: Extract model name from user's request
   - ESM-2: "esm2", "ESM-2", "esm-2"
   - ESM-IF1: "esmif1", "ESM-IF1", "esm-if1", "esm if1"
   - SaProt: "saprot", "SaProt", "sa-prot"
   - Others: "mifst", "prosst", "protssn", "esm1v"

2. DETERMINE TASK: Identify task type
   - "saturation" = all possible mutations (1 file)
   - "specific" = score specific mutations (2 files: structure + CSV)

3. CALL SEQUENCE:
   a) First: help_with_protein_analysis(user_request)
   b) If user confirms: analyze_protein_mutations(model, task)

SUPPORTED MODELS:
- Sequence models (FASTA files): esm2, esm1v
- Structure models (PDB files): mifst, prosst, protssn, esmif1, saprot

USER EXAMPLES:
- "Use ESM-2 for saturation mutagenesis"
- "Run ESM-IF1 on my protein"
- "Design mutations with SaProt"
- "Use ESM-1v for specific scoring"

OUTPUT: Only CSV scoring results sent to user's email

CRITICAL FOR AI:
- Always extract model name from user request
- Always call help_with_protein_analysis() first
- Only proceed if user confirms
- Never return file paths to user
version: 3.0.1
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
        model_name: Literal["esm2", "esm1v", "mifst", "prosst", "protssn", "esmif1", "saprot"],
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
        AI: ADVANCED function - use analyze_protein_mutations() instead for simplicity.
        
        This function:
        1. Claims uploaded files by exact filename
        2. Submits PBS job for protein mutation design
        3. Returns job submission result
        
        CRITICAL: Must provide EXACT filenames user uploaded.
        
        :param task_type: "saturation" (1 file) or "specific" (2 files)
        :param model_name: Model to use (esm2, esm1v, mifst, prosst, protssn, esmif1, saprot)
        :param input_filenames: EXACT filenames user uploaded (e.g., ["protein.fasta"])
        :param user_email: Email for results (auto-filled if not provided)
        :param __user__: User info (auto-provided)
        :return: Job submission result
        """
        # Use configurable defaults
        queue = queue or self.valves.PBS_QUEUE
        ncpus = ncpus or self.valves.PBS_NCPUS
        ngpus = ngpus or self.valves.PBS_NGPUS
        mem = mem or self.valves.PBS_MEM
        walltime = walltime or self.valves.PBS_WALLTIME
        
        print(f"DEBUG: Starting protein mutation design - task: {task_type}, model: {model_name}")
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
            
            # Add helpful debugging information for AI
            error_msg += "\n💡 AI Troubleshooting:\n"
            error_msg += "1. Check if user just uploaded files (use get_recent_uploaded_files())\n"
            error_msg += "2. Verify exact filename spelling (case-sensitive)\n"
            error_msg += "3. Ask user to upload file again if needed\n"
            error_msg += "4. Use list_upload_files() to see all available files\n"
            
            return error_msg

        if not claimed_files:
            return "❌ Error: No files were successfully claimed. Please check your file uploads."

        # Step 2: Validate and build command
        command = ""
        output_file_path_str = ""
        sequence_models = ["esm2", "esm1v"]
        structure_models = ["mifst", "prosst", "protssn", "esmif1", "saprot"]
        
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
            elif model_name in structure_models:
                command = (
                    f"python src/mutation/models/{model_name}.py "
                    f"--pdb_file {source_file.absolute()} "
                    f"--output_csv {output_file_path_str}"
                )
            else:
                return f"❌ Error: Unknown model: {model_name}"
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
            elif model_name in structure_models:
                command = (
                    f"python src/mutation/models/{model_name}.py "
                    f"--pdb_file {structure_file.absolute()} "
                    f"--mutations_csv {mutation_csv.absolute()} "
                    f"--output_csv {output_file_path_str}"
                )
            else:
                return f"❌ Error: Unknown model: {model_name}"
            final_job_name = job_name or f"scoring_{structure_file.stem}"
        
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
            
            return f"✅ '{task_type}' design job submitted successfully!\n- Job ID: {job_id}\n- Model scoring CSV results will be sent to: {user_email}\n- Output file: {output_file_path_str}\n- PBS settings: queue={queue}, ncpus={ncpus}, ngpus={ngpus}, mem={mem}, walltime={walltime}"

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
        AI: CALL THIS SECOND after user confirms in help_with_protein_analysis().
        
        This function:
        1. Automatically finds uploaded files
        2. Runs protein mutation design
        3. Returns job submission result
        
        :param model: Model name (esm2, esm1v, mifst, prosst, protssn, esmif1, saprot)
        :param task: Task type (saturation or specific)
        :param __user__: User info (auto-provided)
        :return: Job submission result or error message
        """
        print(f"DEBUG: Simplified design called - model: {model}, task: {task}")
        
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
                return "❌ Saturation design requires at least 1 protein file."
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
        
        # Run the design
        return self.run_protein_analysis(
            task_type=task,
            model_name=model,
            input_filenames=input_files,
            __user__=__user__
        )

    def help_with_protein_analysis(self, user_request: str, __user__: Optional[Dict] = None) -> str:
        """
        AI: CALL THIS FIRST when user mentions protein mutation design with a model.
        
        This function:
        1. Extracts model name from user request
        2. Checks for uploaded files
        3. Provides guidance to user
        4. Waits for user confirmation
        
        :param user_request: User's full request (e.g., "Use ESM-2 for saturation mutagenesis")
        :param __user__: User info (auto-provided)
        :return: Guidance message for user
        """
        print(f"DEBUG: Help function called with request: {user_request}")
        
        # Check for uploaded files first
        files_info = self.get_recent_uploaded_files(__user__, minutes=60)
        
        # Analyze the user request
        request_lower = user_request.lower()
        
        # Determine model from request
        model = None
        if "esm2" in request_lower or "esm-2" in request_lower:
            model = "esm2"
        elif "esm1v" in request_lower or "esm-1v" in request_lower:
            model = "esm1v"
        elif "mifst" in request_lower or "mif-st" in request_lower:
            model = "mifst"
        elif "prosst" in request_lower or 'sst' in request_lower:
            model = "prosst"
        elif "protssn" in request_lower or "ssn" in request_lower:
            model = "protssn"
        elif "esmif1" in request_lower or "esm-if1" in request_lower or "esm if1" in request_lower:
            model = "esmif1"
        elif "saprot" in request_lower or "sa-prot" in request_lower:
            model = "saprot"
        else:
            # If no specific model mentioned, ask user to specify
            return f"""❌ AI: No model detected in user request.

Ask user to specify which model to use for protein mutation design:

{self.get_supported_models_and_files()}

Example requests:
- "Use ESM-2 for saturation mutagenesis"
- "Run ESM-IF1 design on my protein"
- "Design mutations with SaProt"
- "Use ESM-1v for specific mutation scoring"

What model would you like to use?"""
        
        # Determine task type
        task = "saturation"
        if "specific" in request_lower or "score" in request_lower:
            task = "specific"
        
        # Check if files are available
        if "No files uploaded" in files_info:
            return f"""❌ AI: No protein files found for {model.upper()} {task} design.

Ask user to upload protein file:
- For sequence models (ESM-2, ESM-1v): FASTA file (.fasta) with protein sequence
- For structure models (ESM-IF1, SaProt, MIF-ST, ProSST, ProtSSN): PDB file (.pdb) with protein structure

After upload, ask again to run {model.upper()} design.

{self.get_supported_models_and_files()}"""
        
        # Files are available, provide guidance
        guidance = f"""✅ AI: Found uploaded files for {model.upper()} {task} design.

{files_info}

AI will:
1. Use most recent protein file for design
2. Submit PBS job for {model.upper()} {task} design
3. Send CSV results to user's email

Ask user: \"Would you like me to start the {model.upper()} design now? Just say 'yes' or 'proceed' and I'll run it immediately.\"

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
        AI: DEBUG function - shows what files user uploaded recently.
        
        Use this to understand what files are available for processing.
        
        :param __user__: User info (auto-provided)
        :param minutes: How many minutes back to look (default: 30)
        :return: List of recent uploaded files
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
        AI: INFO function - shows supported models and file requirements.
        
        Use this to explain to users what models are available.
        
        :return: Information about supported models and file types
        """
        info = "🔬 Supported Models and File Types:\n\n"
        
        info += "📄 Sequence-based Models (require FASTA files):\n"
        info += "- ESM-2: State-of-the-art protein language model for sequence analysis\n"
        info += "- ESM-1v: ESM-1v model for variant effect prediction\n\n"
        
        info += "🏗️ Structure-based Models (require PDB files):\n"
        info += "- ESM-IF1: ESM Inverse Folding model for structure-guided design\n"
        info += "- SaProt: Structure-aware protein language model\n"
        info += "- MIF-ST: Multi-task inverse folding with structure transformer\n"
        info += "- ProSST: Protein structure and sequence transformer (most recommended for most cases)\n"
        info += "- ProtSSN: Protein structure sequence network (highly recommended if a crystal structure is provided)\n\n"
        
        info += "📋 Task Types:\n"
        info += "- saturation: Design all possible mutations (requires 1 input file)\n"
        info += "- specific: Score specific mutations (requires 2 input files: structure + CSV)\n\n"
        
        info += "💡 Usage Examples:\n"
        info += "- \"Use ESM-2 for saturation mutagenesis\"\n"
        info += "- \"Run ESM-IF1 design on my protein structure\"\n"
        info += "- \"Design mutations with SaProt\"\n"
        info += "- \"Use ESM-1v for specific mutation scoring\"\n\n"
        
        info += "📁 File Requirements:\n"
        info += "- FASTA files: Protein sequences for sequence models\n"
        info += "- PDB files: Protein structures for structure models\n"
        info += "- CSV files: Mutation lists with columns: mutant (e.g., 'A123V:A124V')\n"
        info += "📊 Output: Model scoring CSV results only\n"
        
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
            structure_models = ["mifst", "prosst", "protssn", "esmif1", "saprot"]
            
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
        """AI: DEBUG function - lists all files in upload directory for troubleshooting"""
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
        """AI: DEBUG function - shows files in user's private storage directory"""
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
        """AI: STATUS function - check PBS job status by job ID"""
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
