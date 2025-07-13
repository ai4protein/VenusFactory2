"""
id: protein-mutation-tool
name: protein mutation tool
description: AI tool for protein mutation design. User MUST specify a model name. Returns CSV scoring results only. AI model score, not energy, the higher the score, the better the mutation.

⚠️ CRITICAL DATA INTEGRITY WARNING ⚠️
This tool handles real scientific data. AI agents MUST:
- NEVER invent, guess, or extrapolate data not present in files
- ONLY report factual information from actual CSV files and logs
- Always verify file existence before reading or analyzing
- If data is missing or unclear, explicitly state this
- Use exact values from files, do not round or approximate
- If analysis cannot be performed, clearly explain why
- Always check for empty files, missing data, or invalid formats

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

4. FILE TRACKING (NEW):
   - Files are automatically tracked after upload
   - Use get_tracked_files() to see available files
   - Use run_protein_analysis_with_tracked_files() for subsequent jobs
   - No need to re-upload files for multiple analyses

5. OUTPUT READING (NEW):
   - Use get_job_output(job_id) to read CSV results or logs
   - Use list_job_outputs() to see user's job history
   - Automatically shows CSV preview or log content

SUPPORTED MODELS:
- Sequence models (FASTA files): esm2, esm1v
- Structure models (PDB files): mifst, prosst, protssn, esmif1, saprot

USER EXAMPLES:
- "Use ESM-2 for saturation mutagenesis"
- "Run ESM-IF1 on my protein"
- "Design mutations with SaProt"
- "Use ESM-1v for specific scoring"
- "Run another analysis with the same files"
- "Show me my job results"
- "What are the results of job 123?"

OUTPUT: Only CSV scoring results. Use get_job_output(job_id) to check results and logs.

CRITICAL FOR AI:
- Always extract model name from user request
- Always call help_with_protein_analysis() first
- Only proceed if user confirms
- Never return file paths to user
- Use tracked files for subsequent analyses
- Use get_job_output(job_id) to show results to users
- Use list_job_outputs() to show user's job history

CRITICAL DATA INTEGRITY RULES:
- NEVER invent, guess, or extrapolate data not present in files
- ONLY report factual information from actual CSV files and logs
- If data is missing or unclear, explicitly state this
- Always verify file existence before attempting to read or analyze
- If analysis cannot be performed due to data limitations, clearly explain why
- Use exact values from files, do not round or approximate unless explicitly requested
- If a column doesn't exist in CSV, do not mention it
- Always check for empty files, missing data, or invalid formats
version: 3.0.3
"""
import os
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field

# Import centralized configuration and Slurm manager
from agent.config import (
    DATA_DIR, PROJECT_ROOT, UPLOAD_DIR, STORAGE_DIR, RESULT_DIR,
    TEMP_SCRIPT_DIR, FILE_TRACKER_FILE, ensure_directories,
    get_user_directories, get_user_email
)
from agent.slurm.slurm import slurm_manager

class Tools:
    class Valves(BaseModel):
        """Global configuration valves for Slurm job settings and user management"""
        # Slurm Job Configuration
        SLURM_PARTITION: str = Field(
            default="venus", 
            description="Slurm partition name"
        )
        SLURM_NCPUS: int = Field(
            default=4, 
            description="Number of CPU cores needed for the job"
        )
        SLURM_NGPUS: int = Field(
            default=1, 
            description="Number of GPUs needed for the job"
        )
        SLURM_MEM: str = Field(
            default="10G", 
            description="Memory requirement (e.g., '8G', '16G', '32G')"
        )
        SLURM_TIME: str = Field(
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
        
        # Create all required directories using centralized config
        ensure_directories()
        
        print(f"Protein mutation tool loaded. Project root: {PROJECT_ROOT}")
        print(f"Slurm configuration: partition={self.valves.SLURM_PARTITION}, ncpus={self.valves.SLURM_NCPUS}, ngpus={self.valves.SLURM_NGPUS}, mem={self.valves.SLURM_MEM}, time={self.valves.SLURM_TIME}")

    def _load_file_tracker(self) -> Dict:
        """Load the file tracker from JSON file"""
        if FILE_TRACKER_FILE.exists():
            try:
                with open(FILE_TRACKER_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_file_tracker(self, tracker: Dict):
        """Save the file tracker to JSON file"""
        try:
            with open(FILE_TRACKER_FILE, 'w') as f:
                json.dump(tracker, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save file tracker: {e}")

    def _record_file_path(self, original_filename: str, new_path: str, user_id: str, conversation_id: str):
        """Record a file's new path after moving"""
        tracker = self._load_file_tracker()
        
        # Create user entry if it doesn't exist
        if user_id not in tracker:
            tracker[user_id] = {}
        if conversation_id not in tracker[user_id]:
            tracker[user_id][conversation_id] = {}
        
        # Record the file path with timestamp
        tracker[user_id][conversation_id][original_filename] = {
            "path": new_path,
            "timestamp": datetime.now().isoformat(),
            "exists": Path(new_path).exists()
        }
        
        self._save_file_tracker(tracker)
        print(f"DEBUG: Recorded file path for {original_filename}: {new_path}")

    def _get_file_path(self, original_filename: str, user_id: str, conversation_id: str) -> Optional[str]:
        """Get the recorded path for a file"""
        tracker = self._load_file_tracker()
        
        try:
            file_info = tracker[user_id][conversation_id][original_filename]
            if file_info["exists"] and Path(file_info["path"]).exists():
                return file_info["path"]
            else:
                # Remove stale entry
                del tracker[user_id][conversation_id][original_filename]
                self._save_file_tracker(tracker)
                return None
        except KeyError:
            return None

    def _list_tracked_files(self, user_id: str, conversation_id: str) -> List[str]:
        """List all tracked files for a user/conversation"""
        tracker = self._load_file_tracker()
        
        try:
            files = []
            for filename, info in tracker[user_id][conversation_id].items():
                if info["exists"] and Path(info["path"]).exists():
                    files.append(filename)
            return files
        except KeyError:
            return []

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
            
            # Record the file path for future use
            self._record_file_path(original_filename, str(secure_file_path.absolute()), user_id, conversation_id)
            
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
        partition: Optional[str] = None,
        ncpus: Optional[int] = None,
        ngpus: Optional[int] = None,
        mem: Optional[str] = None,
        time: Optional[str] = None,
    ) -> str:
        """
        AI: ADVANCED function - use analyze_protein_mutations() instead for simplicity.
        
        This function:
        1. Claims uploaded files by exact filename
        2. Submits Slurm job for protein mutation design
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
        partition = partition or self.valves.SLURM_PARTITION
        ncpus = ncpus or self.valves.SLURM_NCPUS
        ngpus = ngpus or self.valves.SLURM_NGPUS
        mem = mem or self.valves.SLURM_MEM
        time = time or self.valves.SLURM_TIME
        
        print(f"DEBUG: Starting protein mutation design - task: {task_type}, model: {model_name}")
        print(f"DEBUG: Slurm settings - partition: {partition}, ncpus: {ncpus}, ngpus: {ngpus}, mem: {mem}, time: {time}")
        
        # Get user directories
        user_id, conversation_id = self.get_user_directories(__user__)

        # Auto-fill email if not provided
        if not user_email and self.valves.AUTO_FILL_EMAIL:
            user_email = self.get_user_email(__user__)
            if user_email:
                print(f"DEBUG: Auto-filled email: {user_email}")
        
        # Email is no longer required for job submission
        print(f"DEBUG: Email is ignored. Results will be available via get_job_output().")

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
            # Use placeholder for job ID in filename
            output_file = output_dir / f"{source_file.stem}_saturation_JOBID.csv"
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
            # Use placeholder for job ID in filename
            output_file = output_dir / f"{structure_file.stem}_specific_scores_JOBID.csv"
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
        
        # Step 3: Submit to Slurm using centralized manager
        try:
            # Prepare Slurm configuration from tool's Valves
            slurm_config = {
                'SLURM_PARTITION': partition,
                'SLURM_TIME': time,
                'SLURM_NCPUS': ncpus,
                'SLURM_NGPUS': ngpus,
                'SLURM_MEM': mem,
                'AUTO_FILL_EMAIL': self.valves.AUTO_FILL_EMAIL
            }
            
            # Submit job using centralized Slurm manager
            success, result = slurm_manager.submit_job(final_job_name, command, slurm_config, __user__)
            
            if success:
                job_id = result
                # Update output file path with actual job ID
                actual_output_file = output_file_path_str.replace("JOBID", job_id)
                
                return f"✅ '{task_type}' design job submitted successfully!\n- Job ID: {job_id}\n- Output file: {actual_output_file}\n- Slurm settings: partition={partition}, ncpus={ncpus}, ngpus={ngpus}, mem={mem}, time={time}\n\nUse get_job_output('{job_id}') to check progress and view results."
            else:
                return f"❌ Slurm job submission failed: {result}"

        except Exception as e:
            return f"❌ Error: Unknown error when submitting Slurm job: {e}"

    def analyze_protein_mutations(
        self,
        model: str,
        task: str = "saturation",
        __user__: Optional[Dict] = None,
    ) -> str:
        """
        AI: CALL THIS SECOND after user confirms in help_with_protein_analysis().
        
        This function:
        1. Automatically finds uploaded files or tracked files
        2. Runs protein mutation design
        3. Returns job submission result
        
        :param model: Model name (esm2, esm1v, mifst, prosst, protssn, esmif1, saprot)
        :param task: Task type (saturation or specific)
        :param __user__: User info (auto-provided)
        :return: Job submission result or error message
        """
        print(f"DEBUG: Simplified design called - model: {model}, task: {task}")
        
        # First, check for tracked files (preferred)
        tracked_files_info = self.get_tracked_files(__user__)
        print(f"DEBUG: Tracked files info: {tracked_files_info}")
        
        if "No tracked files found" not in tracked_files_info:
            # Use tracked files
            user_id, conversation_id = self.get_user_directories(__user__)
            tracked_files = self._list_tracked_files(user_id, conversation_id)
            
            if tracked_files:
                print(f"DEBUG: Using tracked files: {tracked_files}")
                
                # Determine task type and validate file count
                if task == "saturation":
                    if len(tracked_files) >= 1:
                        input_files = [tracked_files[0]]  # Use the first tracked file
                    else:
                        return "❌ Saturation design requires at least 1 protein file."
                elif task == "specific":
                    if len(tracked_files) >= 2:
                        # Find structure file and CSV file
                        structure_file = None
                        csv_file = None
                        for filename in tracked_files:
                            if filename.endswith(('.fasta', '.pdb')):
                                structure_file = filename
                            elif filename.endswith('.csv'):
                                csv_file = filename
                        
                        if structure_file and csv_file:
                            input_files = [structure_file, csv_file]
                        else:
                            return f"❌ Specific scoring requires 1 structure file (.fasta/.pdb) and 1 CSV file. Found: {tracked_files}"
                    else:
                        return "❌ Specific scoring requires at least 2 files (structure + CSV)."
                else:
                    return f"❌ Unknown task type: {task}. Use 'saturation' or 'specific'."
                
                print(f"DEBUG: Using tracked input files: {input_files}")
                
                # Run the design with tracked files
                return self.run_protein_analysis_with_tracked_files(
                    task_type=task,
                    model_name=model,
                    input_filenames=input_files,
                    __user__=__user__
                )
        
        # Fallback to uploaded files
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
        
        # Run the design with uploaded files
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
        
        # Check for tracked files first (preferred)
        tracked_files_info = self.get_tracked_files(__user__)
        
        # Check for uploaded files as fallback
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
        
        # Check if files are available (prefer tracked files)
        if "No tracked files found" not in tracked_files_info:
            # We have tracked files
            return f"""✅ AI: Found tracked files for {model.upper()} {task} design.

{tracked_files_info}

Ready to proceed with analysis. Would you like me to run the {task} design using {model.upper()}?"""
        elif "No files uploaded" in files_info:
            return f"""❌ AI: No protein files found for {model.upper()} {task} design.

Ask user to upload protein file:
- For sequence models (ESM-2, ESM-1v): FASTA file (.fasta) with protein sequence
- For structure models (ESM-IF1, SaProt, MIF-ST, ProSST, ProtSSN): PDB file (.pdb) with protein structure

After upload, ask again to run {model.upper()} design.

💡 Note: Once uploaded, files will be tracked for future analyses - no need to re-upload for multiple jobs!

{self.get_supported_models_and_files()}"""
        
        # Files are available, provide guidance
        guidance = f"""✅ AI: Found uploaded files for {model.upper()} {task} design.

{files_info}

AI will:
1. Use most recent protein file for design
2. Submit Slurm job for {model.upper()} {task} design
3. You can check results and logs using get_job_output(job_id)

Ask user: \"Would you like me to start the {model.upper()} design now? Just say 'yes' or 'proceed' and I'll run it immediately.\"

Or if you want to use specific files, tell me which ones to use.

💡 Note: Files will be tracked for future analyses - no need to re-upload for multiple jobs!"""
        
        return guidance

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
        
        info += "�� Usage Examples:\n"
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
        """AI: STATUS function - check Slurm job status by job ID using direct Slurm queries"""
        return slurm_manager.get_job_status(job_id)

    def analyze_csv_results(self, csv_file_path: str, job_id: str = None) -> str:
        """
        AI: CSV analysis function - provides intelligent analysis of mutation results
        
        CRITICAL AI INSTRUCTIONS:
        - ONLY report factual data from the CSV file
        - NEVER invent, guess, or extrapolate data not present in the file
        - If data is missing or unclear, explicitly state this
        - Use exact values from the CSV, do not round or approximate unless explicitly requested
        - If a column doesn't exist, do not mention it
        - If analysis cannot be performed due to data limitations, clearly state why
        - Always verify data exists before reporting it
        
        This function:
        1. Reads CSV file and provides statistical analysis
        2. Identifies key findings and patterns
        3. Suggests next steps for the user
        4. Provides visualization recommendations
        
        :param csv_file_path: Path to the CSV file
        :param job_id: Job ID for context
        :return: Comprehensive analysis report
        """
        try:
            import pandas as pd
            import numpy as np
            
            # Verify file exists and is readable
            if not Path(csv_file_path).exists():
                return f"❌ **File Not Found**: CSV file {csv_file_path} does not exist."
            
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            
            # Verify we have data
            if df.empty:
                return f"❌ **Empty File**: CSV file {csv_file_path} contains no data."
            
            analysis = f"📊 **CSV Analysis Report**\n\n"
            analysis += f"📁 **File**: {csv_file_path}\n"
            if job_id:
                analysis += f"🆔 **Job ID**: {job_id}\n"
            analysis += f"📈 **Total Rows**: {len(df):,}\n"
            analysis += f"📋 **Columns**: {len(df.columns)}\n\n"
            
            # Basic statistics - ONLY report what exists
            analysis += "📊 **Data Overview**:\n"
            analysis += f"- Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            analysis += f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB\n"
            missing_total = df.isnull().sum().sum()
            analysis += f"- Missing values: {missing_total} total\n\n"
            
            # Column analysis - ONLY report existing columns
            analysis += "🔍 **Column Analysis**:\n"
            for col in df.columns:
                col_type = df[col].dtype
                unique_count = df[col].nunique()
                missing_count = df[col].isnull().sum()
                
                analysis += f"- **{col}**: {col_type}, {unique_count} unique values"
                if missing_count > 0:
                    analysis += f", {missing_count} missing"
                analysis += "\n"
            analysis += "\n"
            
            # Score analysis - ONLY if score columns actually exist
            score_columns = [col for col in df.columns if 'score' in col.lower() or 'prediction' in col.lower()]
            if score_columns:
                analysis += "🎯 **Score Analysis**:\n"
                for score_col in score_columns:
                    # Verify column contains numeric data
                    scores = pd.to_numeric(df[score_col], errors='coerce')
                    valid_scores = scores.dropna()
                    
                    if len(valid_scores) > 0:
                        analysis += f"- **{score_col}**:\n"
                        analysis += f"  - Mean: {valid_scores.mean():.4f}\n"
                        analysis += f"  - Std: {valid_scores.std():.4f}\n"
                        analysis += f"  - Min: {valid_scores.min():.4f}\n"
                        analysis += f"  - Max: {valid_scores.max():.4f}\n"
                        analysis += f"  - Range: {valid_scores.max() - valid_scores.min():.4f}\n"
                        
                        # Only show top/bottom if we have enough data
                        if len(valid_scores) >= 5:
                            top_5 = valid_scores.nlargest(5)
                            bottom_5 = valid_scores.nsmallest(5)
                            analysis += f"  - Top 5 scores: {', '.join([f'{x:.4f}' for x in top_5.values])}\n"
                            analysis += f"  - Bottom 5 scores: {', '.join([f'{x:.4f}' for x in bottom_5.values])}\n"
                        else:
                            analysis += f"  - All scores: {', '.join([f'{x:.4f}' for x in valid_scores.values])}\n"
                        analysis += "\n"
                    else:
                        analysis += f"- **{score_col}**: No valid numeric data found\n\n"
            else:
                analysis += "🎯 **Score Analysis**: No score columns found in the data\n\n"
            
            # Mutation analysis - ONLY if mutation columns actually exist
            mutation_columns = [col for col in df.columns if 'mutation' in col.lower() or 'mutant' in col.lower()]
            if mutation_columns:
                analysis += "🧬 **Mutation Analysis**:\n"
                for mut_col in mutation_columns:
                    unique_mutations = df[mut_col].nunique()
                    analysis += f"- **{mut_col}**: {unique_mutations} unique mutations\n"
                    
                    # Only analyze if we have string data
                    if df[mut_col].dtype == 'object':
                        # Show actual examples from the data
                        sample_mutations = df[mut_col].dropna().head(5).tolist()
                        if sample_mutations:
                            analysis += f"  - Sample mutations: {', '.join(sample_mutations)}\n"
                analysis += "\n"
            else:
                analysis += "🧬 **Mutation Analysis**: No mutation columns found in the data\n\n"
            
            # Key findings - ONLY report what we can verify
            analysis += "🔍 **Key Findings**:\n"
            
            if score_columns and len(score_columns) > 0:
                best_score_col = score_columns[0]
                scores = pd.to_numeric(df[best_score_col], errors='coerce')
                valid_scores = scores.dropna()
                
                if len(valid_scores) > 0:
                    best_idx = valid_scores.idxmax()
                    worst_idx = valid_scores.idxmin()
                    
                    analysis += f"- **Best performing mutation**: "
                    if mutation_columns and len(mutation_columns) > 0:
                        mut_col = mutation_columns[0]
                        if mut_col in df.columns:
                            best_mutation = df.loc[best_idx, mut_col] if not pd.isna(df.loc[best_idx, mut_col]) else f"Row {best_idx}"
                            analysis += f"{best_mutation} (score: {valid_scores.max():.4f})\n"
                        else:
                            analysis += f"Row {best_idx} (score: {valid_scores.max():.4f})\n"
                    else:
                        analysis += f"Row {best_idx} (score: {valid_scores.max():.4f})\n"
                    
                    analysis += f"- **Worst performing mutation**: "
                    if mutation_columns and len(mutation_columns) > 0:
                        mut_col = mutation_columns[0]
                        if mut_col in df.columns:
                            worst_mutation = df.loc[worst_idx, mut_col] if not pd.isna(df.loc[worst_idx, mut_col]) else f"Row {worst_idx}"
                            analysis += f"{worst_mutation} (score: {valid_scores.min():.4f})\n"
                        else:
                            analysis += f"Row {worst_idx} (score: {valid_scores.min():.4f})\n"
                    else:
                        analysis += f"Row {worst_idx} (score: {valid_scores.min():.4f})\n"
                    
                    # Score distribution
                    score_range = valid_scores.max() - valid_scores.min()
                    analysis += f"- **Score range**: {score_range:.4f}\n"
                else:
                    analysis += "- **Score Analysis**: No valid numeric scores found\n"
            else:
                analysis += "- **Score Analysis**: No score columns available for analysis\n"
            
            # Data quality assessment
            if missing_total > 0:
                missing_pct = (missing_total / (len(df) * len(df.columns))) * 100
                analysis += f"- **Data quality**: {missing_pct:.1f}% missing values\n"
            else:
                analysis += f"- **Data quality**: Complete dataset (no missing values)\n"
            
            analysis += "\n"
            
            # Recommendations - ONLY suggest based on actual data
            analysis += "💡 **Recommendations**:\n"
            
            if score_columns and len(score_columns) > 0:
                analysis += "1. **Focus on high-scoring mutations** for experimental validation\n"
                analysis += "2. **Investigate low-scoring mutations** to understand failure modes\n"
                analysis += "3. **Consider score thresholds** for practical applications\n"
            
            if len(df) > 100:
                analysis += "4. **Large dataset**: Consider statistical significance testing\n"
            
            if mutation_columns and len(mutation_columns) > 0:
                analysis += "5. **Mutation patterns**: Analyze amino acid substitution preferences\n"
            
            analysis += "6. **Visualization**: Create score distribution plots and mutation maps\n"
            analysis += "7. **Experimental design**: Plan validation experiments for top candidates\n"
            
            analysis += "\n"
            
            # Export suggestions
            analysis += "📤 **Export Options**:\n"
            analysis += f"- Download full CSV: [Download Complete Results]({csv_file_path})\n"
            if score_columns and len(score_columns) > 0:
                analysis += "- Filter top 10%: Consider exporting high-scoring mutations separately\n"
            analysis += "- Statistical summary: Export descriptive statistics\n"
            
            return analysis
            
        except ImportError:
            return f"❌ **Analysis Error**: pandas library not available. Basic file info:\n- File: {csv_file_path}\n- Size: {Path(csv_file_path).stat().st_size} bytes\n- Lines: {len(open(csv_file_path).readlines())}"
        except Exception as e:
            return f"❌ **Analysis Error**: {str(e)}\n\n📁 **File Info**:\n- Path: {csv_file_path}\n- Exists: {Path(csv_file_path).exists()}\n- Size: {Path(csv_file_path).stat().st_size if Path(csv_file_path).exists() else 'N/A'} bytes"

    def get_job_output(self, job_id: str, __user__: Optional[Dict] = None) -> str:
        """
        Get job output - CSV results if available, otherwise log file content.
        User should use this to check progress and results. No email is sent.
        
        CRITICAL AI INSTRUCTIONS:
        - ONLY report factual data from actual files
        - NEVER invent or guess file contents
        - If files don't exist, clearly state this
        - If job is not found, say so explicitly
        - Only show actual file content, do not summarize or interpret unless explicitly requested
        - Always verify file existence before attempting to read
        
        :param job_id: Slurm job ID
        :param __user__: User info (auto-provided)
        :return: CSV content or log content
        """
        return slurm_manager.get_job_output(job_id)

    def list_job_outputs(self, __user__: Optional[Dict] = None) -> str:
        """
        List all available job outputs for the current user.
        Use this to see your job history and check results/logs. No email is sent.
        :param __user__: User info (auto-provided)
        :return: List of job outputs
        """
        return slurm_manager.list_job_outputs(__user__)

    def run_protein_analysis_with_tracked_files(
        self,
        task_type: Literal["saturation", "specific"],
        model_name: Literal["esm2", "esm1v", "mifst", "prosst", "protssn", "esmif1", "saprot"],
        input_filenames: List[str],
        user_email: Optional[str] = None,
        __user__: Optional[Dict] = None,
        job_name: Optional[str] = None,
        partition: Optional[str] = None,
        ncpus: Optional[int] = None,
        ngpus: Optional[int] = None,
        mem: Optional[str] = None,
        time: Optional[str] = None,
    ) -> str:
        """
        AI: Use this function when you have already moved files and want to run analysis.
        
        This function:
        1. Uses tracked file paths (no need to re-upload files)
        2. Submits Slurm job for protein mutation design
        3. Returns job submission result
        
        :param task_type: "saturation" (1 file) or "specific" (2 files)
        :param model_name: Model to use (esm2, esm1v, mifst, prosst, protssn, esmif1, saprot)
        :param input_filenames: Original filenames that were already moved
        :param user_email: Email for results (auto-filled if not provided)
        :param __user__: User info (auto-provided)
        :return: Job submission result
        """
        # Use configurable defaults
        partition = partition or self.valves.SLURM_PARTITION
        ncpus = ncpus or self.valves.SLURM_NCPUS
        ngpus = ngpus or self.valves.SLURM_NGPUS
        mem = mem or self.valves.SLURM_MEM
        time = time or self.valves.SLURM_TIME
        
        print(f"DEBUG: Starting protein mutation design with tracked files - task: {task_type}, model: {model_name}")
        print(f"DEBUG: Slurm settings - partition: {partition}, ncpus: {ncpus}, ngpus: {ngpus}, mem: {mem}, time: {time}")
        
        # Get user directories
        user_id, conversation_id = self.get_user_directories(__user__)

        # Auto-fill email if not provided
        if not user_email and self.valves.AUTO_FILL_EMAIL:
            user_email = self.get_user_email(__user__)
            if user_email:
                print(f"DEBUG: Auto-filled email: {user_email}")
        
        # Email is no longer required for job submission
        print(f"DEBUG: Email is ignored. Results will be available via get_job_output().")

        # Step 1: Get tracked file paths
        tracked_files = []
        failed_files = []
        
        for filename in input_filenames:
            print(f"DEBUG: Looking for tracked file: {filename}")
            tracked_path = self._get_file_path(filename, user_id, conversation_id)
            
            if tracked_path:
                print(f"DEBUG: Found tracked file: {tracked_path}")
                tracked_files.append(tracked_path)
            else:
                failed_files.append((filename, f"File '{filename}' not found in tracked files"))
                print(f"DEBUG: Failed to find tracked file: {filename}")

        print(f"DEBUG: All tracked files: {tracked_files}")
        print(f"DEBUG: Failed files: {failed_files}")
        
        # If any files failed to find, provide detailed error information
        if failed_files:
            error_msg = "❌ Error: Could not find the following tracked files:\n"
            for filename, error in failed_files:
                error_msg += f"- {filename}: {error}\n"
            
            # Add helpful debugging information for AI
            error_msg += "\n💡 AI Troubleshooting:\n"
            error_msg += "1. Check if files were previously uploaded and moved\n"
            error_msg += "2. Use get_tracked_files() to see available files\n"
            error_msg += "3. Ask user to upload files again if needed\n"
            error_msg += "4. Use run_protein_analysis() for fresh file uploads\n"
            
            return error_msg

        if not tracked_files:
            return "❌ Error: No tracked files were found. Please check your file uploads."

        # Step 2: Validate and build command (same as run_protein_analysis)
        command = ""
        output_file_path_str = ""
        sequence_models = ["esm2", "esm1v"]
        structure_models = ["mifst", "prosst", "protssn", "esmif1", "saprot"]
        
        if task_type == "saturation":
            if len(tracked_files) != 1:
                return "❌ Error: Saturation analysis needs exactly 1 input file (FASTA/PDB)."
            
            source_file = Path(tracked_files[0])
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
            # Use placeholder for job ID in filename
            output_file = output_dir / f"{source_file.stem}_saturation_JOBID.csv"
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
            if len(tracked_files) != 2:
                return "❌ Error: Specific scoring needs exactly 2 input files (1 structure file, 1 CSV file)."
            
            structure_file_str = next((f for f in tracked_files if Path(f).suffix.lower() in ['.pdb', '.fasta']), None)
            mutation_csv_str = next((f for f in tracked_files if Path(f).suffix.lower() == '.csv'), None)

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
            # Use placeholder for job ID in filename
            output_file = output_dir / f"{structure_file.stem}_specific_scores_JOBID.csv"
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
        
        # Step 3: Submit to Slurm using centralized manager
        try:
            # Prepare Slurm configuration from tool's Valves
            slurm_config = {
                'SLURM_PARTITION': partition,
                'SLURM_TIME': time,
                'SLURM_NCPUS': ncpus,
                'SLURM_NGPUS': ngpus,
                'SLURM_MEM': mem,
                'AUTO_FILL_EMAIL': self.valves.AUTO_FILL_EMAIL
            }
            
            # Submit job using centralized Slurm manager
            success, result = slurm_manager.submit_job(final_job_name, command, slurm_config, __user__)
            
            if success:
                job_id = result
                # Update output file path with actual job ID
                actual_output_file = output_file_path_str.replace("JOBID", job_id)
                
                return f"✅ '{task_type}' design job submitted successfully using tracked files!\n- Job ID: {job_id}\n- Output file: {actual_output_file}\n- Slurm settings: partition={partition}, ncpus={ncpus}, ngpus={ngpus}, mem={mem}, time={time}\n\nUse get_job_output('{job_id}') to check progress and view results."
            else:
                return f"❌ Slurm job submission failed: {result}"

        except Exception as e:
            return f"❌ Error: Unknown error when submitting Slurm job: {e}"

    def get_tracked_files(self, __user__: Optional[Dict] = None) -> str:
        """
        Get list of tracked files for the current user/conversation.
        
        :param __user__: User info (auto-provided)
        :return: List of tracked files
        """
        user_id, conversation_id = self.get_user_directories(__user__)
        tracked_files = self._list_tracked_files(user_id, conversation_id)
        
        if not tracked_files:
            return "No tracked files found for this session."
        
        result = "📁 Tracked files (ready for analysis):\n"
        for filename in tracked_files:
            result += f"• {filename}\n"
        
        result += f"\n💡 Use run_protein_analysis_with_tracked_files() with these filenames.\n"
        result += f"   Example: input_filenames=[\"{tracked_files[0]}\"]\n"
        
        return result

    def get_conversation_guide(self, __user__: Optional[Dict] = None) -> str:
        """
        AI: Conversation guide function - helps Agent understand user intent and provide workflow guidance
        
        This function provides Agent with:
        1. User current status analysis
        2. Next step action suggestions
        3. Common Q&A
        4. Workflow hints
        
        :param __user__: User info (auto-provided)
        :return: Conversation guidance information
        """
        user_id, conversation_id = self.get_user_directories(__user__)
        
        # Check user current status
        tracked_files = self._list_tracked_files(user_id, conversation_id)
        recent_files = self.get_recent_uploaded_files(__user__, minutes=60)
        
        guide = "🤖 **AI Agent Conversation Guide**\n\n"
        
        # Status analysis
        if tracked_files:
            guide += f"📁 **Current Status**: User has {len(tracked_files)} tracked files\n"
            guide += f"   - File list: {', '.join(tracked_files[:3])}{'...' if len(tracked_files) > 3 else ''}\n\n"
        elif "No files uploaded" not in recent_files:
            guide += "📁 **Current Status**: User just uploaded files\n\n"
        else:
            guide += "📁 **Current Status**: User hasn't uploaded files yet\n\n"
        
        # Next step suggestions
        guide += "💡 **Agent Action Suggestions**:\n"
        if tracked_files:
            guide += "1. Ask user what type of analysis they want (saturation/specific mutations)\n"
            guide += "2. Recommend suitable models (ESM-2, SaProt, ProSST, etc.)\n"
            guide += "3. Run analysis after confirmation\n"
            guide += "4. Actively monitor task progress\n"
        else:
            guide += "1. Guide user to upload protein files\n"
            guide += "2. Explain file requirements (FASTA/PDB format)\n"
            guide += "3. Provide model selection suggestions\n"
        
        guide += "\n🔍 **Smart Monitoring**:\n"
        guide += "- Actively ask if user needs to check task progress\n"
        guide += "- Automatically prompt user to view results when task completes\n"
        guide += "- Provide result explanation and download links\n"
        
        guide += "\n💬 **Conversation Tips**:\n"
        guide += "- Use friendly language, avoid technical jargon\n"
        guide += "- Actively provide options and suggestions\n"
        guide += "- Give timely task status feedback\n"
        guide += "- Explain result meanings\n"
        
        return guide

    def check_user_tasks(self, __user__: Optional[Dict] = None) -> str:
        """
        AI: User task status check - let Agent actively monitor user tasks
        
        This function helps Agent:
        1. Check if user has running tasks
        2. Identify completed tasks
        3. Provide proactive suggestions
        
        :param __user__: User info (auto-provided)
        :return: Task status summary
        """
        log_dir = DATA_DIR / "slurm_logs"
        if not log_dir.exists():
            return "📊 **Task Status**: No task records found"
        
        try:
            # Find all log files
            job_ids = set()
            for log_file in log_dir.glob("*"):
                if log_file.is_file() and log_file.suffix in ['.out', '.err']:
                    job_id = log_file.stem
                    if job_id.isdigit():
                        job_ids.add(job_id)
            
            if not job_ids:
                return "📊 **Task Status**: No task records found"
            
            # Check task status
            running_jobs = []
            completed_jobs = []
            failed_jobs = []
            
            for job_id in sorted(job_ids, key=lambda x: int(x), reverse=True)[:5]:  # Recent 5 tasks
                try:
                    result = subprocess.run(f"squeue -j {job_id}", shell=True, capture_output=True, text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            running_jobs.append(job_id)
                        else:
                            # Check if completed
                            out_log = log_dir / f"{job_id}.out"
                            if out_log.exists():
                                try:
                                    with open(out_log, 'r') as f:
                                        content = f.read()
                                        if "Job finished" in content and "exit code 0" in content:
                                            completed_jobs.append(job_id)
                                        else:
                                            failed_jobs.append(job_id)
                                except:
                                    failed_jobs.append(job_id)
                            else:
                                failed_jobs.append(job_id)
                except:
                    failed_jobs.append(job_id)
            
            status = "📊 **User Task Status**:\n\n"
            
            if running_jobs:
                status += f"⏳ **Running**: {len(running_jobs)} tasks\n"
                status += f"   - Job IDs: {', '.join(running_jobs)}\n"
                status += "   💡 Suggestion: Actively ask if user needs to check progress\n\n"
            
            if completed_jobs:
                status += f"✅ **Completed**: {len(completed_jobs)} tasks\n"
                status += f"   - Job IDs: {', '.join(completed_jobs)}\n"
                status += "   💡 Suggestion: Actively provide result viewing and download links\n\n"
            
            if failed_jobs:
                status += f"❌ **Failed**: {len(failed_jobs)} tasks\n"
                status += f"   - Job IDs: {', '.join(failed_jobs)}\n"
                status += "   💡 Suggestion: Check error logs, provide solutions\n\n"
            
            status += "🤖 **Agent Action Suggestions**:\n"
            if running_jobs:
                status += f"- Proactive ask: \"Your task {running_jobs[0]} is running, would you like to check progress?\"\n"
            if completed_jobs:
                status += f"- Proactive offer: \"Task {completed_jobs[0]} is completed, let me show you the results\"\n"
            if failed_jobs:
                status += f"- Proactive help: \"Task {failed_jobs[0]} encountered an issue, let me check the error information\"\n"
            
            return status
            
        except Exception as e:
            return f"❌ Error checking task status: {e}"

    def auto_analyze_mutation_results(self, job_id: str = None, __user__: Optional[Dict] = None) -> str:
        """
        AI: Auto-analyze mutation results - automatically finds and analyzes user's mutation results
        
        CRITICAL AI INSTRUCTIONS:
        - ONLY analyze files that actually exist
        - NEVER invent or guess file contents or analysis results
        - If no files are found, clearly state this
        - If job ID is invalid, say so explicitly
        - Only report factual data from actual CSV files
        - Always verify file existence before analysis
        - If analysis cannot be performed, clearly explain why
        
        This function:
        1. Uses job ID from context or finds the most recent mutation results
        2. Analyzes CSV files without requiring user to specify paths
        3. Provides comprehensive analysis report
        4. Suggests next steps based on results
        
        :param job_id: Job ID from context (optional)
        :param __user__: User info (auto-provided)
        :return: Comprehensive analysis report
        """
        user_id, conversation_id = self.get_user_directories(__user__)
        
        # If job_id is provided, analyze that specific job
        if job_id:
            # Check if job is completed
            log_dir = DATA_DIR / "slurm_logs"
            out_log = log_dir / f"{job_id}.out"
            
            if not out_log.exists():
                return f"❌ **Job Not Found**: Job {job_id} not found in logs."
            
            # Check if job completed successfully
            try:
                with open(out_log, 'r') as f:
                    content = f.read()
                    if "Job finished" not in content or "exit code 0" not in content:
                        return f"❌ **Job Not Completed**: Job {job_id} may still be running or failed. Check status first."
            except:
                return f"❌ **Error Reading Job Log**: Cannot read log for job {job_id}."
            
            # Look for CSV file for this specific job
            result_dir = RESULT_DIR
            if not result_dir.exists():
                return f"❌ **No Results Found**: Results directory does not exist."
            
            csv_files = []
            for csv_file in result_dir.rglob(f"*{job_id}*.csv"):
                csv_files.append(csv_file)
            
            if not csv_files:
                return f"❌ **No Results Found**: No CSV files found for job {job_id}. The job may not have produced output files."
            
            # Analyze the found CSV file
            csv_file = csv_files[0]  # Use the first matching file
            
            # Verify file exists and is valid
            if not csv_file.exists():
                return f"❌ **File Not Found**: CSV file {csv_file} does not exist."
            
            file_size = csv_file.stat().st_size
            if file_size == 0:
                return f"❌ **Empty File**: CSV file {csv_file} is empty (0 bytes)."
            
            analysis_report = f"🔍 **Analysis of Job {job_id} Results**\n\n"
            analysis_report += f"📁 **File**: {csv_file.name}\n"
            analysis_report += f"📅 **Last Modified**: {datetime.fromtimestamp(csv_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}\n"
            analysis_report += f"📏 **File Size**: {file_size} bytes\n\n"
            
            # Get detailed analysis
            detailed_analysis = self.analyze_csv_results(str(csv_file), job_id)
            analysis_report += detailed_analysis
            
            return analysis_report
        
        # If no job_id provided, find the most recent results
        result_dir = RESULT_DIR
        if not result_dir.exists():
            return "❌ **No Results Found**: Results directory does not exist. Please run a mutation analysis first."
        
        # Find all CSV files for this user
        csv_files = []
        user_results_dir = result_dir / user_id / conversation_id
        
        # Check user-specific directory first
        if user_results_dir.exists():
            for csv_file in user_results_dir.rglob("*.csv"):
                csv_files.append(csv_file)
        
        # If no user-specific files, check all results
        if not csv_files:
            for csv_file in result_dir.rglob("*.csv"):
                csv_files.append(csv_file)
        
        if not csv_files:
            return "❌ **No Results Found**: No CSV result files found. Please run a mutation analysis first."
        
        # Sort by modification time (most recent first)
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        analysis_report = "🔍 **Auto-Analysis of Latest Mutation Results**\n\n"
        
        # Analyze the most recent file
        most_recent_file = csv_files[0]
        analysis_report += f"📁 **Analyzing**: {most_recent_file.name}\n"
        analysis_report += f"📅 **Last Modified**: {datetime.fromtimestamp(most_recent_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Get detailed analysis
        detailed_analysis = self.analyze_csv_results(str(most_recent_file))
        analysis_report += detailed_analysis
        
        # If there are multiple files, provide summary
        if len(csv_files) > 1:
            analysis_report += f"\n📋 **Other Available Results**:\n"
            for i, csv_file in enumerate(csv_files[1:4], 2):  # Show next 3 files
                mod_time = datetime.fromtimestamp(csv_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                analysis_report += f"{i}. {csv_file.name} (Modified: {mod_time})\n"
            
            if len(csv_files) > 4:
                analysis_report += f"... and {len(csv_files) - 4} more files\n"
            
            analysis_report += f"\n💡 **Tip**: You can ask me to analyze specific job results by mentioning the job ID.\n"
        
        return analysis_report

    def extract_job_id_from_context(self, user_message: str, __user__: Optional[Dict] = None) -> str:
        """
        AI: Extract job ID from user message or context
        
        CRITICAL AI INSTRUCTIONS:
        - ONLY extract job IDs that actually exist in the system
        - NEVER invent or guess job IDs
        - If no job ID is found, clearly state this
        - Always verify job existence before returning it
        - Only return factual information from actual job logs
        
        This function:
        1. Looks for job ID patterns in user message
        2. Checks recent job history for context
        3. Returns the most relevant job ID
        
        :param user_message: User's message
        :param __user__: User info (auto-provided)
        :return: Extracted job ID or guidance message
        """
        import re
        
        # Look for job ID patterns in the message
        job_id_patterns = [
            r'job\s+(\d+)',  # "job 12345"
            r'(\d{4,})',     # Any 4+ digit number
            r'#(\d+)',       # "#12345"
            r'id\s+(\d+)',   # "id 12345"
        ]
        
        for pattern in job_id_patterns:
            matches = re.findall(pattern, user_message.lower())
            if matches:
                return matches[0]  # Return the first found job ID
        
        # If no job ID found in message, check recent jobs
        log_dir = DATA_DIR / "slurm_logs"
        if log_dir.exists():
            job_ids = set()
            for log_file in log_dir.glob("*"):
                if log_file.is_file() and log_file.suffix in ['.out', '.err']:
                    job_id = log_file.stem
                    if job_id.isdigit():
                        # Verify job actually exists by checking if log file has content
                        try:
                            if log_file.stat().st_size > 0:
                                job_ids.add(job_id)
                        except:
                            pass  # Skip if we can't check file size
            
            if job_ids:
                # Return the most recent job ID
                most_recent_job = sorted(job_ids, key=lambda x: int(x), reverse=True)[0]
                return f"💡 **Context**: No specific job ID mentioned. Using most recent job {most_recent_job} for analysis."
        
        return "❌ **No Job ID Found**: Please specify a job ID or run a mutation analysis first."

    def suggest_next_actions(self, user_message: str, __user__: Optional[Dict] = None) -> str:
        """
        AI: Smart suggestion function - provides next action suggestions based on user message
        
        This function helps Agent:
        1. Understand user intent
        2. Provide appropriate response suggestions
        3. Proactively guide user through workflow
        
        :param user_message: User's message
        :param __user__: User info (auto-provided)
        :return: Action suggestions
        """
        message_lower = user_message.lower()
        
        suggestions = "🤖 **AI Agent Action Suggestions**:\n\n"
        
        # Analyze user intent
        if any(word in message_lower for word in ["upload", "file", "上传", "文件"]):
            suggestions += "📁 **Detected file upload intent**\n"
            suggestions += "💡 Suggested actions:\n"
            suggestions += "1. Confirm files are uploaded\n"
            suggestions += "2. Ask user what analysis they want to perform\n"
            suggestions += "3. Recommend suitable models\n\n"
        
        elif any(word in message_lower for word in ["mutation", "design", "analysis", "突变", "设计", "分析"]):
            suggestions += "🔬 **Detected mutation analysis intent**\n"
            suggestions += "💡 Suggested actions:\n"
            suggestions += "1. Analyze user's protein analysis requirements\n"
            suggestions += "2. Confirm model and task type\n"
            suggestions += "3. Run protein mutation analysis\n\n"
        
        elif any(word in message_lower for word in ["progress", "status", "finish", "进度", "状态", "完成"]):
            suggestions += "📊 **Detected progress query intent**\n"
            suggestions += "💡 Suggested actions:\n"
            suggestions += "1. Check user's task status\n"
            suggestions += "2. Show detailed results and logs\n"
            suggestions += "3. Provide download links for completed results\n\n"
        
        elif any(word in message_lower for word in ["result", "download", "view", "结果", "下载", "查看"]):
            suggestions += "📋 **Detected result viewing intent**\n"
            suggestions += "💡 Suggested actions:\n"
            suggestions += "1. List user's available results\n"
            suggestions += "2. Display specific job results\n"
            suggestions += "3. Provide result explanation and download links\n\n"
        
        elif any(word in message_lower for word in ["analyze", "analysis", "分析", "解析", "解读"]):
            suggestions += "🔍 **Detected analysis intent**\n"
            suggestions += "💡 Suggested actions:\n"
            suggestions += "1. **Extract job ID from user message or use most recent job**\n"
            suggestions += "2. **Automatically analyze the corresponding results**\n"
            suggestions += "3. **Provide comprehensive analysis report with key findings**\n"
            suggestions += "4. **Suggest next steps based on analysis results**\n\n"
        
        else:
            suggestions += "❓ **Unclear intent detected**\n"
            suggestions += "💡 Suggested actions:\n"
            suggestions += "1. Ask user for specific requirements\n"
            suggestions += "2. Provide function options (upload files/analyze mutations/check progress/view results)\n"
            suggestions += "3. Get conversation guidance for better understanding\n\n"
        
        # Add general suggestions
        suggestions += "🎯 **General Tips**:\n"
        suggestions += "- Maintain friendly and professional tone\n"
        suggestions += "- Proactively provide help and suggestions\n"
        suggestions += "- Give timely task status feedback\n"
        suggestions += "- Explain technical terms in simple language\n"
        suggestions += "- Provide multiple access methods for results\n"
        
        return suggestions
