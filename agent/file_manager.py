"""
File management module for VenusAgent system.
Handles file uploads, tracking, and user directory management.
"""

import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Import global configuration
from agent.config import (
    UPLOAD_DIR, STORAGE_DIR, RESULT_DIR,
    UPLOAD_SETTINGS, ensure_directories, get_user_directories,
    is_valid_file_extension, is_valid_file_size
)


class FileManager:
    """
    File management operations.
    """
    
    def __init__(self):
        """Initialize the file manager"""
        # Create all required directories using centralized config
        ensure_directories()
        
        print(f"File manager loaded.")
    
    def claim_and_move_file(self, original_filename: str, user_info: Optional[Dict] = None) -> str:
        """
        Move uploaded file to user's private storage and return secure path.
        
        Args:
            original_filename: Original filename in upload directory
            user_info: User information
            
        Returns:
            Success message with secure path or error message
        """
        try:
            # Get user directories
            user_id, conversation_id, user_storage_dir, _ = get_user_directories(user_info)
            
            # Ensure user storage directory exists
            user_storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if file exists in upload directory
            upload_path = UPLOAD_DIR / original_filename
            if not upload_path.exists():
                return f"❌ File '{original_filename}' not found in upload directory."
            
            # Check if file is already in user's storage
            secure_path = user_storage_dir / original_filename
            if secure_path.exists():
                return f"✅ File already tracked at `{secure_path}`"
            
            # Move file to user's private storage
            shutil.move(str(upload_path), str(secure_path))
            
            print(f"DEBUG: Moved {original_filename} to {secure_path}")
            return f"✅ File moved to secure location: `{secure_path}`"
            
        except Exception as e:
            return f"❌ Error moving file '{original_filename}': {e}"
    
    def get_tracked_files(self, user_info: Optional[Dict] = None) -> str:
        """
        Get list of tracked files for the user.
        
        Args:
            user_info: User information
            
        Returns:
            List of tracked files
        """
        try:
            # Get user directories
            user_id, conversation_id, user_storage_dir, _ = get_user_directories(user_info)
            
            if not user_storage_dir.exists():
                return "📁 No tracked files found for this user."
            
            # List files in user's storage directory
            files = []
            for file_path in user_storage_dir.iterdir():
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
                    age_str = f"{file_age.total_seconds() / 60:.1f} minutes ago"
                    files.append(f"• {file_path.name} ({file_size} bytes, {age_str})")
            
            if not files:
                return "📁 No tracked files found for this user."
            
            return f"📁 **Your Tracked Files**:\n" + "\n".join(files)
            
        except Exception as e:
            return f"❌ Error getting tracked files: {e}"
    
    def get_recent_uploaded_files(self, user_info: Optional[Dict] = None, minutes: int = 30) -> str:
        """
        Get list of recently uploaded files.
        
        Args:
            user_info: User information
            minutes: How many minutes back to look
            
        Returns:
            List of recent uploaded files
        """
        try:
            if not UPLOAD_DIR.exists():
                return "📁 Upload directory does not exist."
            
            files = []
            now = datetime.now()
            cutoff_time = now - timedelta(minutes=minutes)
            
            for filepath in UPLOAD_DIR.iterdir():
                if filepath.is_file():
                    file_age = now - datetime.fromtimestamp(filepath.stat().st_mtime)
                    if file_age <= timedelta(minutes=minutes):
                        age_str = f"{file_age.total_seconds() / 60:.1f} minutes ago"
                        files.append(f"• {filepath.name} ({filepath.stat().st_size} bytes, {age_str})")
            
            if not files:
                return f"📁 No files uploaded in the last {minutes} minutes."
            
            return f"📁 **Recently Uploaded Files** (last {minutes} minutes):\n" + "\n".join(files)
            
        except Exception as e:
            return f"❌ Error getting uploaded files: {e}"
    
    def get_file_path(self, original_filename: str, user_info: Optional[Dict] = None) -> Optional[str]:
        """
        Get the secure path of a tracked file.
        
        Args:
            original_filename: Original filename
            user_info: User information
            
        Returns:
            Secure file path if found, None otherwise
        """
        try:
            # Get user directories
            user_id, conversation_id, user_storage_dir, _ = get_user_directories(user_info)
            
            # Check if file exists in user's storage directory
            secure_path = user_storage_dir / original_filename
            if secure_path.exists():
                return str(secure_path)
            
            return None
        except Exception as e:
            print(f"Error getting file path: {e}")
            return None
    
    def list_tracked_files(self, user_info: Optional[Dict] = None) -> List[str]:
        """
        List all tracked files for a user.
        
        Args:
            user_info: User information
            
        Returns:
            List of tracked file names
        """
        try:
            # Get user directories
            user_id, conversation_id, user_storage_dir, _ = get_user_directories(user_info)
            
            if not user_storage_dir.exists():
                return []
            
            # List all files in user's storage directory
            files = []
            for file_path in user_storage_dir.iterdir():
                if file_path.is_file():
                    files.append(file_path.name)
            
            return files
        except Exception as e:
            print(f"Error listing tracked files: {e}")
            return []
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate a file for upload.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            file_path_obj = Path(file_path)
            
            # Check if file exists
            if not file_path_obj.exists():
                return False, f"File not found: {file_path}"
            
            # Check file size
            file_size = file_path_obj.stat().st_size
            if not is_valid_file_size(file_size):
                max_size_mb = UPLOAD_SETTINGS["MAX_FILE_SIZE"] / (1024 * 1024)
                return False, f"File too large ({file_size / (1024 * 1024):.1f}MB): {file_path}. Maximum size is {max_size_mb}MB."
            
            # Check file extension
            if not is_valid_file_extension(file_path_obj.name):
                allowed_extensions = ", ".join(UPLOAD_SETTINGS["ALLOWED_EXTENSIONS"])
                return False, f"File type not allowed: {file_path}. Allowed extensions: {allowed_extensions}"
            
            return True, "File is valid."
            
        except Exception as e:
            return False, f"Error validating file: {e}"
    
    def get_user_files(self, user_info: Optional[Dict] = None) -> str:
        """
        Get list of files in user's private storage directory.
        
        Args:
            user_info: User information
            
        Returns:
            List of user files
        """
        try:
            user_id, conversation_id, user_storage_dir, _ = get_user_directories(user_info)
            
            if not user_storage_dir.exists():
                return "📁 No files found in your private directory."
            
            files = list(user_storage_dir.iterdir())
            if not files:
                return "📁 No files found in your private directory."
            
            file_list = "\n".join([f"- {f.name} ({f.stat().st_size} bytes)" for f in files])
            return f"📁 Files in your private directory:\n{file_list}"
            
        except Exception as e:
            return f"❌ Error getting user files: {e}"
    
    def list_upload_files(self, user_info: Optional[Dict] = None) -> str:
        """
        List all files in upload directory for troubleshooting.
        
        Args:
            user_info: User information
            
        Returns:
            List of upload files
        """
        try:
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
            
        except Exception as e:
            return f"❌ Error listing upload files: {e}"


# Create global instance
file_manager = FileManager() 