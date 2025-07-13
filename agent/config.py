"""
Global configuration file for VenusAgent system - centralized path and directory management.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# ============================================================================
# CORE PATH CONFIGURATION
# ============================================================================

# DATA_DIR: default DATA_DIR for OpenWebUI
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/VenusFactory/openwebui_data"))

# PROJECT_ROOT: the absolute path of the project root
PROJECT_ROOT = "/app/VenusFactory"

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

# UPLOAD_DIR: the public, temporary upload "inbox"
UPLOAD_DIR = DATA_DIR.expanduser() / "uploads"

# STORAGE_DIR: the root directory for all user private files
STORAGE_DIR = DATA_DIR.expanduser() / "storage"

# RESULT_DIR: the directory for analysis results
RESULT_DIR = DATA_DIR.expanduser() / "results"

# TEMP_SCRIPT_DIR: the directory for temporary generated Slurm scripts
TEMP_SCRIPT_DIR = DATA_DIR.expanduser() / "temp_slurm_scripts"

# FILE_TRACKER_FILE: the JSON file for tracking moved file paths
FILE_TRACKER_FILE = DATA_DIR.expanduser() / "file_tracker.json"

# SLURM_LOG_DIR: the directory for Slurm job logs
SLURM_LOG_DIR = DATA_DIR.expanduser() / "slurm_logs"

# SLURM_FAILURE_LOG_DIR: the directory for job failure logs
SLURM_FAILURE_LOG_DIR = DATA_DIR.expanduser() / "slurm_failure_logs"

# ============================================================================
# FILE UPLOAD SETTINGS
# ============================================================================

UPLOAD_SETTINGS = {
    "MAX_FILE_SIZE": 100 * 1024 * 1024,  # 100MB
    "ALLOWED_EXTENSIONS": [".pdb", ".csv", ".fasta", ".txt"],
    "UPLOAD_TIMEOUT": 300,  # 5 minutes
}

# ============================================================================
# JOB SETTINGS
# ============================================================================

JOB_SETTINGS = {
    "MAX_RECOMMENDATIONS": 1000,
    "MIN_RECOMMENDATIONS": 1,
    "DEFAULT_RECOMMENDATIONS": 30,
    "DEFAULT_STRATEGY": "ensemble_round",
    "VALID_STRATEGIES": [
        "ensemble_round", 
        "ensemble_top", 
        "individual_best", 
        "frequency_based", 
        "diversity_based"
    ]
}

# ============================================================================
# ANALYSIS SETTINGS
# ============================================================================

ANALYSIS_SETTINGS = {
    "CSV_ANALYSIS_ENABLED": True,
    "AUTO_ANALYSIS_ENABLED": True,
    "MAX_CSV_SIZE": 50 * 1024 * 1024,  # 50MB
    "ANALYSIS_TIMEOUT": 60,  # 60 seconds
}

# ============================================================================
# VALIDATION SETTINGS
# ============================================================================

VALIDATION_SETTINGS = {
    "PDB_MAX_SIZE": 100 * 1024 * 1024,  # 100MB
    "CSV_MAX_SIZE": 50 * 1024 * 1024,   # 50MB
    "ENABLE_COMPREHENSIVE_VALIDATION": True,
    "ENABLE_SLURM_TEST": True,
}

# ============================================================================
# CLEANUP SETTINGS
# ============================================================================

CLEANUP_SETTINGS = {
    "TEMP_SCRIPT_RETENTION_HOURS": 24,
    "LOG_RETENTION_DAYS": 7,
    "FAILURE_LOG_RETENTION_DAYS": 30,
    "AUTO_CLEANUP_ENABLED": True,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directories() -> bool:
    """
    Ensure all required directories exist.
    
    Returns:
        bool: True if all directories were created successfully
    """
    directories = [
        UPLOAD_DIR,
        STORAGE_DIR,
        RESULT_DIR,
        TEMP_SCRIPT_DIR,
        SLURM_LOG_DIR,
        SLURM_FAILURE_LOG_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return True

def get_user_directories(user_info: Optional[Dict] = None) -> Tuple[str, str, Path, Path]:
    """
    Get user-specific directories based on user info.
    
    Args:
        user_info: User information dictionary
        
    Returns:
        Tuple of (user_id, conversation_id, user_storage_dir, user_result_dir)
    """
    if user_info and isinstance(user_info, dict):
        user_id = user_info.get("id", "default")
        conversation_id = user_info.get("conversation_id", "default")
    else:
        user_id = "default"
        conversation_id = "default"
    
    user_storage_dir = STORAGE_DIR / user_id / conversation_id
    user_result_dir = RESULT_DIR / user_id / conversation_id
    
    return user_id, conversation_id, user_storage_dir, user_result_dir

def get_file_paths() -> Dict[str, Path]:
    """
    Get all important file paths.
    
    Returns:
        Dictionary of path names to Path objects
    """
    return {
        "data_dir": DATA_DIR,
        "project_root": PROJECT_ROOT,
        "upload_dir": UPLOAD_DIR,
        "storage_dir": STORAGE_DIR,
        "result_dir": RESULT_DIR,
        "temp_script_dir": TEMP_SCRIPT_DIR,
        "file_tracker_file": FILE_TRACKER_FILE,
        "slurm_log_dir": SLURM_LOG_DIR,
        "slurm_failure_log_dir": SLURM_FAILURE_LOG_DIR,
    }

def validate_configuration() -> List[str]:
    """
    Validate the configuration and return any issues.
    
    Returns:
        List of validation issues (empty if all valid)
    """
    issues = []
    
    # Check if DATA_DIR is accessible
    try:
        DATA_DIR.expanduser().mkdir(parents=True, exist_ok=True)
    except Exception as e:
        issues.append(f"Cannot create DATA_DIR {DATA_DIR}: {e}")
    
    # Check if PROJECT_ROOT exists
    if not Path(PROJECT_ROOT).exists():
        issues.append(f"PROJECT_ROOT {PROJECT_ROOT} does not exist")
    
    # Check Slurm commands
    import subprocess
    try:
        subprocess.run(["which", "sbatch"], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        issues.append("sbatch command not found - Slurm may not be installed")
    
    try:
        subprocess.run(["which", "squeue"], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        issues.append("squeue command not found - Slurm may not be installed")
    
    return issues

def get_configuration_summary() -> Dict:
    """
    Get a summary of the current configuration.
    
    Returns:
        Dictionary containing configuration summary
    """
    summary = {
        "paths": get_file_paths(),
        "upload_settings": UPLOAD_SETTINGS,
        "job_settings": JOB_SETTINGS,
        "analysis_settings": ANALYSIS_SETTINGS,
        "validation_settings": VALIDATION_SETTINGS,
        "cleanup_settings": CLEANUP_SETTINGS,
    }
    
    # Add validation issues
    validation_issues = validate_configuration()
    if validation_issues:
        summary["validation_issues"] = validation_issues
    
    return summary

def get_user_email(user_info: Optional[Dict] = None) -> Optional[str]:
    """
    Get user email from user info.
    
    Args:
        user_info: User information dictionary
        
    Returns:
        Email address if available, None otherwise
    """
    if not user_info:
        return None
    
    # Try different possible email fields
    email = (
        user_info.get("email") or 
        user_info.get("user_email") or 
        user_info.get("mail") or 
        None
    )
    
    return email

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def is_valid_file_extension(filename: str) -> bool:
    """
    Check if file extension is allowed.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if extension is allowed, False otherwise
    """
    file_ext = Path(filename).suffix.lower()
    return file_ext in UPLOAD_SETTINGS["ALLOWED_EXTENSIONS"]

def is_valid_file_size(file_size: int) -> bool:
    """
    Check if file size is within limits.
    
    Args:
        file_size: Size of the file in bytes
        
    Returns:
        True if file size is valid, False otherwise
    """
    return file_size <= UPLOAD_SETTINGS["MAX_FILE_SIZE"]

def is_valid_recommendations_count(count: int) -> bool:
    """
    Check if recommendations count is valid.
    
    Args:
        count: Number of recommendations
        
    Returns:
        True if count is valid, False otherwise
    """
    return JOB_SETTINGS["MIN_RECOMMENDATIONS"] <= count <= JOB_SETTINGS["MAX_RECOMMENDATIONS"]

def is_valid_strategy(strategy: str) -> bool:
    """
    Check if strategy is valid.
    
    Args:
        strategy: Strategy name to validate
        
    Returns:
        True if strategy is valid, False otherwise
    """
    return strategy in JOB_SETTINGS["VALID_STRATEGIES"] 