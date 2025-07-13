"""
Core Slurm operations module for VenusAgent system.
Handles job submission, status checking, and file management.
"""

import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

# Import global configuration
from agent.config import (
    DATA_DIR, PROJECT_ROOT, UPLOAD_DIR, STORAGE_DIR, RESULT_DIR,
    TEMP_SCRIPT_DIR, SLURM_LOG_DIR, SLURM_FAILURE_LOG_DIR,
    CLEANUP_SETTINGS, ensure_directories, get_user_directories, get_user_email
)

# Import file management module
from agent.file_manager import file_manager


class SlurmManager:
    """
    Core Slurm operations manager.
    Handles job submission, status checking, and file management.
    """
    
    def __init__(self):
        """Initialize the Slurm manager"""
        self.file_manager = file_manager
        
        # Create all required directories using centralized config
        ensure_directories()
        
        print(f"Slurm manager loaded. Project root: {PROJECT_ROOT}")
        print(f"Using centralized configuration from agent.config")
    
    def submit_job(self, job_name: str, command: str, slurm_config: Dict[str, Any], user_info: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Submit a job to Slurm.
        
        Args:
            job_name: Name of the job
            command: Command to execute
            slurm_config: Slurm configuration from tool's Valves
            user_info: User information
            
        Returns:
            Tuple of (success, job_id_or_error_message)
        """
        try:
            # Get user directories
            user_id, conversation_id, _, _ = get_user_directories(user_info)
            
            # Create timestamp for unique identification
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_base = SLURM_LOG_DIR / f"{job_name}_{timestamp}"
            log_file_base.parent.mkdir(parents=True, exist_ok=True)
            
            # Create Slurm script using provided configuration
            slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={slurm_config.get('SLURM_PARTITION', 'venus')}
#SBATCH --time={slurm_config.get('SLURM_TIME', '24:00:00')}
#SBATCH --cpus-per-task={slurm_config.get('SLURM_NCPUS', 4)}
#SBATCH --gres=gpu:{slurm_config.get('SLURM_NGPUS', 1)}
#SBATCH --mem={slurm_config.get('SLURM_MEM', '10G')}
#SBATCH --output={log_file_base.parent}/%j.out
#SBATCH --error={log_file_base.parent}/%j.err

echo "Job started on $(hostname) at $(date)"
cd {PROJECT_ROOT} || exit 1

source /opt/conda/etc/profile.d/conda.sh
conda activate venus-env

{command}

echo "Job finished at $(date) with exit code $?"
"""
            
            # Write script to temporary file
            script_path = TEMP_SCRIPT_DIR / f"{job_name}_{timestamp}.slurm"
            script_path.write_text(slurm_script_content)
            
            # Submit job
            result = subprocess.run(f"sbatch {script_path}", shell=True, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split()[-1]  # Extract job ID from "Submitted batch job 12345"
            
            return True, job_id
            
        except subprocess.CalledProcessError as e:
            return False, f"Slurm job submission failed: {e.stderr}"
        except Exception as e:
            return False, f"Error creating or submitting Slurm script: {e}"
    
    def get_job_status(self, job_id: str) -> str:
        """
        Get detailed status of a Slurm job.
        
        Args:
            job_id: Job ID to check
            
        Returns:
            Status message
        """
        try:
            # Check if job exists in Slurm
            result = subprocess.run(f"squeue -j {job_id}", shell=True, capture_output=True, text=True)
            
            if result.returncode == 0 and job_id in result.stdout:
                # Job is still running or queued
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header line
                    job_info = lines[1].split()
                    if len(job_info) >= 5:
                        status = job_info[4]
                        return f"🔄 **Job {job_id} Status**: {status}\n\nUse `get_job_output('{job_id}')` to check progress."
            
            # Job not found in queue, check if completed
            out_log = SLURM_LOG_DIR / f"{job_id}.out"
            err_log = SLURM_LOG_DIR / f"{job_id}.err"
            
            if out_log.exists() or err_log.exists():
                return f"✅ **Job {job_id} completed**\n\nUse `get_job_output('{job_id}')` to view results."
            else:
                return f"❌ **Job {job_id} not found**\n\nJob may have failed or been removed from queue."
                
        except Exception as e:
            return f"❌ Error checking job status: {e}"
    
    def get_job_output(self, job_id: str) -> str:
        """
        Get job output and logs.
        
        Args:
            job_id: Job ID to get output for
            
        Returns:
            Job output message
        """
        try:
            # Check for log files
            out_log = SLURM_LOG_DIR / f"{job_id}.out"
            err_log = SLURM_LOG_DIR / f"{job_id}.err"
            
            output = f"📊 **Job {job_id} Output**\n\n"
            
            # Read output log
            if out_log.exists():
                try:
                    with open(out_log, 'r') as f:
                        content = f.read()
                        if content.strip():
                            output += f"📄 **Output Log** ({out_log.stat().st_size} bytes):\n"
                            output += f"```\n{content}\n```\n\n"
                        else:
                            output += "📄 **Output Log**: Empty\n\n"
                except Exception as e:
                    output += f"❌ **Error reading output log**: {e}\n\n"
            else:
                output += "📄 **Output Log**: Not found\n\n"
            
            # Read error log
            if err_log.exists():
                try:
                    with open(err_log, 'r') as f:
                        content = f.read()
                        if content.strip():
                            output += f"⚠️ **Error Log** ({err_log.stat().st_size} bytes):\n"
                            output += f"```\n{content}\n```\n\n"
                        else:
                            output += "⚠️ **Error Log**: Empty\n\n"
                except Exception as e:
                    output += f"❌ **Error reading error log**: {e}\n\n"
            else:
                output += "⚠️ **Error Log**: Not found\n\n"
            
            # Check for result files
            result_files = []
            if RESULT_DIR.exists():
                for result_file in RESULT_DIR.rglob(f"*{job_id}*"):
                    if result_file.is_file():
                        result_files.append(result_file)
            
            if result_files:
                output += "📁 **Result Files**:\n"
                for result_file in result_files:
                    output += f"• {result_file.name} ({result_file.stat().st_size} bytes)\n"
                output += "\n"
            else:
                output += "📁 **Result Files**: None found\n\n"
            
            return output
            
        except Exception as e:
            return f"❌ Error getting job output: {e}"
    
    def list_job_outputs(self, user_info: Optional[Dict] = None) -> str:
        """
        List all job outputs for the user.
        
        Args:
            user_info: User information
            
        Returns:
            List of job outputs
        """
        try:
            if not SLURM_LOG_DIR.exists():
                return "📁 No job logs found."
            
            # Get user directories
            user_id, conversation_id, _, _ = get_user_directories(user_info)
            
            # Find all log files
            log_files = []
            for log_file in SLURM_LOG_DIR.glob("*.out"):
                try:
                    job_id = log_file.stem
                    file_size = log_file.stat().st_size
                    mod_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    age = datetime.now() - mod_time
                    age_str = f"{age.total_seconds() / 3600:.1f} hours ago"
                    
                    log_files.append({
                        'job_id': job_id,
                        'size': file_size,
                        'age': age_str,
                        'mod_time': mod_time
                    })
                except Exception as e:
                    print(f"Error processing log file {log_file}: {e}")
            
            if not log_files:
                return "📁 No job outputs found."
            
            # Sort by modification time (newest first)
            log_files.sort(key=lambda x: x['mod_time'], reverse=True)
            
            output = "📊 **Your Jobs**:\n\n"
            for log_file in log_files[:10]:  # Show only the 10 most recent
                output += f"• Job {log_file['job_id']} ({log_file['size']} bytes, {log_file['age']})\n"
            
            if len(log_files) > 10:
                output += f"\n... and {len(log_files) - 10} more jobs\n"
            
            output += f"\n💡 Use `get_job_output(job_id)` to view specific job results."
            
            return output
            
        except Exception as e:
            return f"❌ Error listing job outputs: {e}"
    
    def get_system_status(self) -> str:
        """
        Get comprehensive system status.
        
        Returns:
            System status report
        """
        try:
            status = "🔧 **System Status Report**\n\n"
            
            # Check Slurm status
            try:
                result = subprocess.run("sinfo", shell=True, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    status += "✅ **Slurm**: Running\n"
                else:
                    status += "❌ **Slurm**: Not responding\n"
            except Exception as e:
                status += f"❌ **Slurm**: Error - {e}\n"
            
            # Check directories
            directories = [
                ("Upload Directory", UPLOAD_DIR),
                ("Storage Directory", STORAGE_DIR),
                ("Result Directory", RESULT_DIR),
                ("Log Directory", SLURM_LOG_DIR),
                ("Temp Script Directory", TEMP_SCRIPT_DIR)
            ]
            
            status += "\n📁 **Directory Status**:\n"
            for name, directory in directories:
                if directory.exists():
                    status += f"✅ {name}: {directory}\n"
                else:
                    status += f"❌ {name}: Missing\n"
            
            # Check recent jobs
            try:
                if SLURM_LOG_DIR.exists():
                    recent_logs = list(SLURM_LOG_DIR.glob("*.out"))
                    status += f"\n📊 **Recent Jobs**: {len(recent_logs)} log files found\n"
                else:
                    status += "\n📊 **Recent Jobs**: No logs found\n"
            except Exception as e:
                status += f"\n📊 **Recent Jobs**: Error - {e}\n"
            
            return status
            
        except Exception as e:
            return f"❌ Error getting system status: {e}"
    
    def cleanup_old_files(self) -> str:
        """
        Clean up old temporary files and logs.
        
        Returns:
            Cleanup result message
        """
        try:
            cleaned_files = []
            failed_cleanups = []
            
            # Clean up old temporary scripts (older than retention period)
            cutoff_time = datetime.now() - timedelta(hours=CLEANUP_SETTINGS["TEMP_SCRIPT_RETENTION_HOURS"])
            
            for script_file in TEMP_SCRIPT_DIR.glob("*.slurm"):
                try:
                    if script_file.stat().st_mtime < cutoff_time.timestamp():
                        script_file.unlink()
                        cleaned_files.append(f"Temporary script: {script_file.name}")
                except Exception as e:
                    failed_cleanups.append(f"Failed to clean {script_file.name}: {e}")
            
            # Clean up old log files (older than retention period)
            cutoff_time = datetime.now() - timedelta(days=CLEANUP_SETTINGS["LOG_RETENTION_DAYS"])
            
            for log_file in SLURM_LOG_DIR.glob("*.out"):
                try:
                    if log_file.stat().st_mtime < cutoff_time.timestamp():
                        log_file.unlink()
                        cleaned_files.append(f"Old log file: {log_file.name}")
                except Exception as e:
                    failed_cleanups.append(f"Failed to clean {log_file.name}: {e}")
            
            for log_file in SLURM_LOG_DIR.glob("*.err"):
                try:
                    if log_file.stat().st_mtime < cutoff_time.timestamp():
                        log_file.unlink()
                        cleaned_files.append(f"Old error log: {log_file.name}")
                except Exception as e:
                    failed_cleanups.append(f"Failed to clean {log_file.name}: {e}")
            
            # Generate report
            if cleaned_files and not failed_cleanups:
                return f"✅ **Cleanup Successful**\n\nCleaned {len(cleaned_files)} files:\n" + "\n".join([f"- {f}" for f in cleaned_files])
            elif cleaned_files and failed_cleanups:
                return f"⚠️ **Partial Cleanup**\n\nCleaned {len(cleaned_files)} files:\n" + "\n".join([f"- {f}" for f in cleaned_files]) + f"\n\nFailed to clean {len(failed_cleanups)} files:\n" + "\n".join([f"- {f}" for f in failed_cleanups])
            elif not cleaned_files and not failed_cleanups:
                return "ℹ️ **No Cleanup Needed**\n\nNo old files found to clean."
            else:
                return f"❌ **Cleanup Failed**\n\nFailed to clean {len(failed_cleanups)} files:\n" + "\n".join([f"- {f}" for f in failed_cleanups])
            
        except Exception as e:
            return f"❌ **Cleanup Error**: {e}"

    # File management methods (delegated to file_manager)
    def get_user_directories(self, __user__: Optional[Dict] = None) -> tuple[str, str]:
        """Get user-specific directories based on user info"""
        user_id, conversation_id, _, _ = get_user_directories(__user__)
        return user_id, conversation_id

    def get_user_email(self, __user__: Optional[Dict] = None) -> Optional[str]:
        """Get user email from user info or return None"""
        return get_user_email(__user__)

    def claim_and_move_file(self, original_filename: str, __user__: Optional[Dict] = None) -> str:
        """Move uploaded file to user's private storage"""
        return self.file_manager.claim_and_move_file(original_filename, __user__)

    def get_tracked_files(self, __user__: Optional[Dict] = None) -> str:
        """Get list of tracked files for the user"""
        return self.file_manager.get_tracked_files(__user__)

    def get_recent_uploaded_files(self, __user__: Optional[Dict] = None, minutes: int = 30) -> str:
        """Get list of recently uploaded files"""
        return self.file_manager.get_recent_uploaded_files(__user__, minutes)

    def get_file_path(self, original_filename: str, __user__: Optional[Dict] = None) -> Optional[str]:
        """Get the secure path of a tracked file"""
        return self.file_manager.get_file_path(original_filename, __user__)

    def list_tracked_files(self, __user__: Optional[Dict] = None) -> List[str]:
        """List all tracked files for a user"""
        return self.file_manager.list_tracked_files(__user__)

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate a file for upload"""
        return self.file_manager.validate_file(file_path)

    def get_user_files(self, __user__: Optional[Dict] = None) -> str:
        """Get list of files in user's private storage directory"""
        return self.file_manager.get_user_files(__user__)

    def list_upload_files(self, __user__: Optional[Dict] = None) -> str:
        """List all files in upload directory for troubleshooting"""
        return self.file_manager.list_upload_files(__user__)


# Create global instance
slurm_manager = SlurmManager() 