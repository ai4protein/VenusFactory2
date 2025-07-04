"""
id: pbs-job-manager
name: PBS Job Manager
description: AI tool for querying and managing PBS job execution status. Provides job status, details, and execution feedback.

AI INTERACTION GUIDE:
When user asks about PBS job status or execution, follow this workflow:

1. DETECT JOB: Extract job ID from user's request
   - Direct job ID: "job 12345", "PBS job 67890"
   - Recent jobs: "my recent jobs", "latest job"
   - All jobs: "all my jobs", "job history"

2. CALL SEQUENCE:
   a) For specific job: get_job_status(job_id)
   b) For recent jobs: get_recent_jobs(user_info)
   c) For all jobs: get_all_user_jobs(user_info)
   d) For job details: get_job_details(job_id)

SUPPORTED QUERIES:
- "Check job 12345 status"
- "Show my recent jobs"
- "What's the status of my latest job?"
- "Show all my PBS jobs"

OUTPUT: Job status, execution details, and user-friendly feedback

CRITICAL FOR AI:
- Always extract job ID when user provides one
- Use user context for job queries
- Provide clear status explanations
- Never expose sensitive system information
version: 1.0.0
"""
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# DATA_DIR: default DATA_DIR for OpenWebUI
DATA_DIR = Path(os.getenv("DATA_DIR", "~/openwebui_data"))
# JOB_TRACKER_FILE: the JSON file for tracking PBS job status
JOB_TRACKER_FILE = DATA_DIR.expanduser() / "job_tracker.json"

class JobStatus(BaseModel):
    """PBS job status information"""
    job_id: str
    job_name: str
    status: str
    queue: str
    user: str
    submit_time: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    exit_code: Optional[int] = None
    cpu_time: Optional[str] = None
    wall_time: Optional[str] = None
    memory_used: Optional[str] = None
    output_file: Optional[str] = None
    error_file: Optional[str] = None

class Tools:
    class Valves(BaseModel):
        """Global configuration valves for PBS job management"""
        # Job tracking
        MAX_JOBS_TO_SHOW: int = Field(
            default=10,
            description="Maximum number of jobs to show in recent jobs list"
        )
        JOB_HISTORY_DAYS: int = Field(
            default=30,
            description="Number of days to look back for job history"
        )
        # Status refresh
        REFRESH_STATUS: bool = Field(
            default=True,
            description="Whether to refresh job status from PBS system"
        )

    def __init__(self):
        """Initialize the PBS job manager tool"""
        self.valves = self.Valves()
        
        # Ensure job tracker file exists
        if not JOB_TRACKER_FILE.exists():
            self._create_job_tracker()
        
        print(f"PBS Job Manager loaded. Job tracker: {JOB_TRACKER_FILE}")

    def _create_job_tracker(self):
        """Create initial job tracker file"""
        initial_tracker = {
            "active_jobs": {},
            "completed_jobs": {},
            "failed_jobs": {}
        }
        JOB_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(JOB_TRACKER_FILE, 'w') as f:
            json.dump(initial_tracker, f, indent=4)

    def get_job_status(self, job_id: str, __user__: Optional[Dict] = None) -> str:
        """
        AI: Get detailed status of a specific PBS job.
        
        This function:
        1. Queries PBS system for job status
        2. Checks job tracker for additional info
        3. Returns comprehensive job status
        
        :param job_id: PBS job ID (e.g., "12345.hostname")
        :param __user__: User info (auto-provided)
        :return: Detailed job status information
        """
        print(f"DEBUG: Getting status for job {job_id}")
        
        # Clean job ID (remove hostname if present)
        clean_job_id = job_id.split('.')[0]
        
        try:
            # Query PBS system for job status
            if self.valves.REFRESH_STATUS:
                pbs_status = self._query_pbs_job_status(clean_job_id)
            else:
                pbs_status = None
            
            # Get job info from tracker
            tracker_info = self._get_job_from_tracker(clean_job_id)
            
            if not pbs_status and not tracker_info:
                return f"❌ Job {job_id} not found in PBS system or job tracker."
            
            # Combine PBS and tracker information
            status_info = self._combine_job_info(pbs_status, tracker_info, clean_job_id)
            
            return self._format_job_status(status_info)
            
        except Exception as e:
            print(f"DEBUG: Error getting job status: {e}")
            return f"❌ Error querying job {job_id}: {e}"

    def get_recent_jobs(self, __user__: Optional[Dict] = None, max_jobs: Optional[int] = None) -> str:
        """
        AI: Get list of user's recent PBS jobs.
        
        This function:
        1. Queries PBS for user's recent jobs
        2. Combines with tracker information
        3. Returns formatted job list
        
        :param __user__: User info (auto-provided)
        :param max_jobs: Maximum jobs to show (default: 10)
        :return: List of recent jobs with status
        """
        max_jobs = max_jobs or self.valves.MAX_JOBS_TO_SHOW
        print(f"DEBUG: Getting recent jobs (max: {max_jobs})")
        
        try:
            # Get user info
            user_id = self._get_user_id(__user__)
            
            # Query PBS for recent jobs
            if self.valves.REFRESH_STATUS:
                pbs_jobs = self._query_pbs_user_jobs(user_id)
            else:
                pbs_jobs = []
            
            # Get jobs from tracker
            tracker_jobs = self._get_user_jobs_from_tracker(user_id)
            
            # Combine and sort jobs
            all_jobs = self._combine_job_lists(pbs_jobs, tracker_jobs)
            recent_jobs = sorted(all_jobs, key=lambda x: x.get('submit_time', ''), reverse=True)[:max_jobs]
            
            if not recent_jobs:
                return f"📊 No recent jobs found for user {user_id}."
            
            return self._format_job_list(recent_jobs, f"Recent {len(recent_jobs)} jobs")
            
        except Exception as e:
            print(f"DEBUG: Error getting recent jobs: {e}")
            return f"❌ Error querying recent jobs: {e}"

    def get_all_user_jobs(self, __user__: Optional[Dict] = None, days: Optional[int] = None) -> str:
        """
        AI: Get all user's PBS jobs from the last N days.
        
        This function:
        1. Queries PBS for user's job history
        2. Filters by time period
        3. Returns comprehensive job history
        
        :param __user__: User info (auto-provided)
        :param days: Number of days to look back (default: 30)
        :return: Complete job history for the period
        """
        days = days or self.valves.JOB_HISTORY_DAYS
        print(f"DEBUG: Getting all jobs from last {days} days")
        
        try:
            # Get user info
            user_id = self._get_user_id(__user__)
            
            # Query PBS for job history
            if self.valves.REFRESH_STATUS:
                pbs_jobs = self._query_pbs_user_jobs(user_id, days)
            else:
                pbs_jobs = []
            
            # Get jobs from tracker
            tracker_jobs = self._get_user_jobs_from_tracker(user_id, days)
            
            # Combine and sort jobs
            all_jobs = self._combine_job_lists(pbs_jobs, tracker_jobs)
            filtered_jobs = self._filter_jobs_by_time(all_jobs, days)
            
            if not filtered_jobs:
                return f"📊 No jobs found for user {user_id} in the last {days} days."
            
            return self._format_job_list(filtered_jobs, f"All jobs from last {days} days")
            
        except Exception as e:
            print(f"DEBUG: Error getting all jobs: {e}")
            return f"❌ Error querying job history: {e}"

    def get_job_details(self, job_id: str, __user__: Optional[Dict] = None) -> str:
        """
        AI: Get comprehensive details of a PBS job including logs and output.
        
        This function:
        1. Gets basic job status
        2. Attempts to read job output/error files
        3. Returns detailed job information
        
        :param job_id: PBS job ID
        :param __user__: User info (auto-provided)
        :return: Comprehensive job details with logs
        """
        print(f"DEBUG: Getting detailed info for job {job_id}")
        
        # Get basic status first
        basic_status = self.get_job_status(job_id, __user__)
        
        if basic_status.startswith("❌"):
            return basic_status
        
        try:
            # Clean job ID
            clean_job_id = job_id.split('.')[0]
            
            # Get job info from tracker
            tracker_info = self._get_job_from_tracker(clean_job_id)
            
            details = [basic_status]
            
            # Try to read output file
            if tracker_info and 'output_file_path' in tracker_info:
                output_file = Path(tracker_info['output_file_path'])
                if output_file.exists():
                    try:
                        with open(output_file, 'r') as f:
                            output_content = f.read().strip()
                        if output_content:
                            details.append(f"\n📄 Job Output (last 20 lines):\n{self._get_last_lines(output_content, 20)}")
                    except Exception as e:
                        details.append(f"\n⚠️ Could not read output file: {e}")
            
            # Try to read PBS log files
            pbs_logs = self._get_pbs_log_files(clean_job_id)
            if pbs_logs:
                details.append(f"\n📋 PBS Log Files:\n{pbs_logs}")
            
            return "\n".join(details)
            
        except Exception as e:
            print(f"DEBUG: Error getting job details: {e}")
            return f"{basic_status}\n\n❌ Error getting additional details: {e}"

    def get_job_summary(self, __user__: Optional[Dict] = None) -> str:
        """
        AI: Get summary statistics of user's PBS jobs.
        
        This function:
        1. Analyzes user's job history
        2. Provides statistics and insights
        3. Returns job summary report
        
        :param __user__: User info (auto-provided)
        :return: Job summary statistics
        """
        print(f"DEBUG: Getting job summary")
        
        try:
            # Get user info
            user_id = self._get_user_id(__user__)
            
            # Get all jobs from tracker
            tracker_jobs = self._get_user_jobs_from_tracker(user_id, 30)  # Last 30 days
            
            if not tracker_jobs:
                return f"📊 No job history found for user {user_id} in the last 30 days."
            
            # Calculate statistics
            total_jobs = len(tracker_jobs)
            completed_jobs = len([j for j in tracker_jobs if j.get('status') == 'Completed'])
            failed_jobs = len([j for j in tracker_jobs if j.get('status') == 'Failed'])
            running_jobs = len([j for j in tracker_jobs if j.get('status') in ['Running', 'Queued']])
            
            # Calculate success rate
            success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
            
            summary = f"""📊 PBS Job Summary for {user_id} (Last 30 days):

🔢 Total Jobs: {total_jobs}
✅ Completed: {completed_jobs}
❌ Failed: {failed_jobs}
🔄 Running/Queued: {running_jobs}
📈 Success Rate: {success_rate:.1f}%

💡 Insights:"""
            
            if success_rate >= 90:
                summary += "\n- Excellent job success rate!"
            elif success_rate >= 75:
                summary += "\n- Good job success rate"
            elif success_rate >= 50:
                summary += "\n- Moderate job success rate - consider reviewing failed jobs"
            else:
                summary += "\n- Low success rate - recommend checking job configurations"
            
            if running_jobs > 0:
                summary += f"\n- {running_jobs} jobs currently active"
            
            return summary
            
        except Exception as e:
            print(f"DEBUG: Error getting job summary: {e}")
            return f"❌ Error generating job summary: {e}"

    def _query_pbs_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Query PBS system for job status"""
        try:
            result = subprocess.run(
                f"qstat -f {job_id}",
                shell=True, capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                return None
            
            return self._parse_qstat_output(result.stdout)
            
        except subprocess.TimeoutExpired:
            print(f"DEBUG: PBS query timeout for job {job_id}")
            return None
        except Exception as e:
            print(f"DEBUG: Error querying PBS for job {job_id}: {e}")
            return None

    def _query_pbs_user_jobs(self, user_id: str, days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query PBS system for user's jobs"""
        try:
            # Build qstat command
            cmd = f"qstat -u {user_id}"
            if days:
                cmd += f" -s all"  # Show all jobs, we'll filter by time later
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
            
            if result.returncode != 0:
                return []
            
            return self._parse_qstat_list_output(result.stdout)
            
        except subprocess.TimeoutExpired:
            print(f"DEBUG: PBS query timeout for user {user_id}")
            return []
        except Exception as e:
            print(f"DEBUG: Error querying PBS for user {user_id}: {e}")
            return []

    def _parse_qstat_output(self, output: str) -> Optional[Dict[str, Any]]:
        """Parse qstat -f output"""
        try:
            lines = output.strip().split('\n')
            job_info = {}
            
            for line in lines:
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'Job_Id':
                        job_info['job_id'] = value
                    elif key == 'Job_Name':
                        job_info['job_name'] = value
                    elif key == 'job_state':
                        job_info['status'] = self._map_pbs_status(value)
                    elif key == 'queue':
                        job_info['queue'] = value
                    elif key == 'Job_Owner':
                        job_info['user'] = value
                    elif key == 'ctime':
                        job_info['submit_time'] = value
                    elif key == 'stime':
                        job_info['start_time'] = value
                    elif key == 'mtime':
                        job_info['end_time'] = value
                    elif key == 'exit_status':
                        job_info['exit_code'] = int(value) if value.isdigit() else None
                    elif key == 'resources_used.cput':
                        job_info['cpu_time'] = value
                    elif key == 'resources_used.walltime':
                        job_info['wall_time'] = value
                    elif key == 'resources_used.mem':
                        job_info['memory_used'] = value
                    elif key == 'Output_Path':
                        job_info['output_file'] = value
                    elif key == 'Error_Path':
                        job_info['error_file'] = value
            
            return job_info if job_info else None
            
        except Exception as e:
            print(f"DEBUG: Error parsing qstat output: {e}")
            return None

    def _parse_qstat_list_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse qstat list output"""
        try:
            lines = output.strip().split('\n')
            jobs = []
            
            # Skip header lines
            for line in lines[2:]:  # Skip first two header lines
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 6:
                        job_info = {
                            'job_id': parts[0],
                            'job_name': parts[1],
                            'user': parts[2],
                            'status': self._map_pbs_status(parts[4]),
                            'queue': parts[3],
                            'submit_time': ' '.join(parts[5:]) if len(parts) > 5 else ''
                        }
                        jobs.append(job_info)
            
            return jobs
            
        except Exception as e:
            print(f"DEBUG: Error parsing qstat list output: {e}")
            return []

    def _map_pbs_status(self, pbs_status: str) -> str:
        """Map PBS status to user-friendly status"""
        status_map = {
            'Q': 'Queued',
            'R': 'Running',
            'E': 'Exiting',
            'C': 'Completed',
            'H': 'Held',
            'S': 'Suspended',
            'W': 'Waiting',
            'T': 'Transit'
        }
        return status_map.get(pbs_status, pbs_status)

    def _get_job_from_tracker(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job information from tracker file"""
        try:
            if not JOB_TRACKER_FILE.exists():
                return None
            
            with open(JOB_TRACKER_FILE, 'r') as f:
                tracker = json.load(f)
            
            # Check active jobs
            if job_id in tracker.get('active_jobs', {}):
                return tracker['active_jobs'][job_id]
            
            # Check completed jobs
            if job_id in tracker.get('completed_jobs', {}):
                return tracker['completed_jobs'][job_id]
            
            # Check failed jobs
            if job_id in tracker.get('failed_jobs', {}):
                return tracker['failed_jobs'][job_id]
            
            return None
            
        except Exception as e:
            print(f"DEBUG: Error reading job tracker: {e}")
            return None

    def _get_user_jobs_from_tracker(self, user_id: str, days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get user's jobs from tracker file"""
        try:
            if not JOB_TRACKER_FILE.exists():
                return []
            
            with open(JOB_TRACKER_FILE, 'r') as f:
                tracker = json.load(f)
            
            user_jobs = []
            cutoff_time = None
            if days:
                cutoff_time = datetime.now() - timedelta(days=days)
            
            # Check all job categories
            for category in ['active_jobs', 'completed_jobs', 'failed_jobs']:
                for job_id, job_info in tracker.get(category, {}).items():
                    if job_info.get('user_id') == user_id:
                        # Filter by time if specified
                        if cutoff_time:
                            submit_time = datetime.fromisoformat(job_info.get('submission_time', '1970-01-01'))
                            if submit_time < cutoff_time:
                                continue
                        
                        job_info['job_id'] = job_id
                        user_jobs.append(job_info)
            
            return user_jobs
            
        except Exception as e:
            print(f"DEBUG: Error reading user jobs from tracker: {e}")
            return []

    def _combine_job_info(self, pbs_info: Optional[Dict], tracker_info: Optional[Dict], job_id: str) -> Dict[str, Any]:
        """Combine PBS and tracker information"""
        combined = {
            'job_id': job_id,
            'job_name': 'Unknown',
            'status': 'Unknown',
            'user': 'Unknown',
            'submit_time': 'Unknown',
            'start_time': None,
            'end_time': None,
            'exit_code': None,
            'cpu_time': None,
            'wall_time': None,
            'memory_used': None,
            'output_file': None,
            'error_file': None
        }
        
        # Update with tracker info first
        if tracker_info:
            combined.update({
                'job_name': tracker_info.get('job_name', combined['job_name']),
                'status': tracker_info.get('status', combined['status']),
                'user': tracker_info.get('user_id', combined['user']),
                'submit_time': tracker_info.get('submission_time', combined['submit_time']),
                'output_file_path': tracker_info.get('output_file_path')
            })
        
        # Update with PBS info (overrides tracker)
        if pbs_info:
            combined.update(pbs_info)
        
        return combined

    def _combine_job_lists(self, pbs_jobs: List[Dict], tracker_jobs: List[Dict]) -> List[Dict]:
        """Combine PBS and tracker job lists"""
        combined = {}
        
        # Add tracker jobs
        for job in tracker_jobs:
            job_id = job.get('job_id', '')
            if job_id:
                combined[job_id] = job
        
        # Add/update with PBS jobs
        for job in pbs_jobs:
            job_id = job.get('job_id', '')
            if job_id:
                if job_id in combined:
                    combined[job_id].update(job)
                else:
                    combined[job_id] = job
        
        return list(combined.values())

    def _filter_jobs_by_time(self, jobs: List[Dict], days: int) -> List[Dict]:
        """Filter jobs by time period"""
        cutoff_time = datetime.now() - timedelta(days=days)
        filtered = []
        
        for job in jobs:
            submit_time_str = job.get('submit_time', '')
            if submit_time_str:
                try:
                    # Try to parse various time formats
                    if 'T' in submit_time_str:
                        submit_time = datetime.fromisoformat(submit_time_str.replace('Z', '+00:00'))
                    else:
                        submit_time = datetime.strptime(submit_time_str, '%a %b %d %H:%M:%S %Y')
                    
                    if submit_time >= cutoff_time:
                        filtered.append(job)
                except:
                    # If we can't parse the time, include the job
                    filtered.append(job)
            else:
                # If no submit time, include the job
                filtered.append(job)
        
        return filtered

    def _format_job_status(self, job_info: Dict[str, Any]) -> str:
        """Format job status for display"""
        status_emoji = {
            'Queued': '⏳',
            'Running': '🔄',
            'Completed': '✅',
            'Failed': '❌',
            'Held': '⏸️',
            'Suspended': '⏸️',
            'Exiting': '🔄',
            'Waiting': '⏳',
            'Transit': '🔄'
        }
        
        emoji = status_emoji.get(job_info['status'], '❓')
        
        result = f"""{emoji} Job {job_info['job_id']} Status: {job_info['status']}

📋 Job Details:
• Name: {job_info['job_name']}
• User: {job_info['user']}
• Queue: {job_info.get('queue', 'Unknown')}
• Submitted: {job_info['submit_time']}"""

        if job_info.get('start_time'):
            result += f"\n• Started: {job_info['start_time']}"
        
        if job_info.get('end_time'):
            result += f"\n• Ended: {job_info['end_time']}"
        
        if job_info.get('exit_code') is not None:
            result += f"\n• Exit Code: {job_info['exit_code']}"
        
        if job_info.get('cpu_time'):
            result += f"\n• CPU Time: {job_info['cpu_time']}"
        
        if job_info.get('wall_time'):
            result += f"\n• Wall Time: {job_info['wall_time']}"
        
        if job_info.get('memory_used'):
            result += f"\n• Memory Used: {job_info['memory_used']}"
        
        if job_info.get('output_file_path'):
            result += f"\n• Output File: {job_info['output_file_path']}"
        
        return result

    def _format_job_list(self, jobs: List[Dict[str, Any]], title: str) -> str:
        """Format job list for display"""
        if not jobs:
            return f"📊 {title}: No jobs found."
        
        status_emoji = {
            'Queued': '⏳',
            'Running': '🔄',
            'Completed': '✅',
            'Failed': '❌',
            'Held': '⏸️',
            'Suspended': '⏸️'
        }
        
        result = f"📊 {title}:\n\n"
        
        for job in jobs:
            emoji = status_emoji.get(job.get('status', ''), '❓')
            job_id = job.get('job_id', 'Unknown')
            job_name = job.get('job_name', 'Unknown')
            status = job.get('status', 'Unknown')
            submit_time = job.get('submit_time', 'Unknown')
            
            result += f"{emoji} {job_id} - {job_name}\n"
            result += f"   Status: {status} | Submitted: {submit_time}\n\n"
        
        return result

    def _get_user_id(self, __user__: Optional[Dict] = None) -> str:
        """Extract user ID from user info"""
        if not __user__:
            return "unknown_user"
        
        return (
            __user__.get("id") or 
            __user__.get("user_id") or 
            __user__.get("username") or 
            "unknown_user"
        )

    def _get_pbs_log_files(self, job_id: str) -> str:
        """Get PBS log file information"""
        try:
            # Try to find PBS log files
            log_patterns = [
                f"/var/spool/pbs/server_priv/jobs/{job_id}.*",
                f"/var/spool/pbs/server_priv/jobs/{job_id}",
                f"~/pbs_logs/{job_id}.*"
            ]
            
            found_logs = []
            for pattern in log_patterns:
                import glob
                matches = glob.glob(pattern)
                found_logs.extend(matches)
            
            if found_logs:
                return "\n".join([f"• {log}" for log in found_logs[:3]])  # Show first 3
            else:
                return "No PBS log files found"
                
        except Exception as e:
            return f"Could not search for log files: {e}"

    def _get_last_lines(self, content: str, num_lines: int) -> str:
        """Get last N lines of content"""
        lines = content.split('\n')
        if len(lines) <= num_lines:
            return content
        else:
            return '\n'.join(lines[-num_lines:])

    def help_with_job_query(self, user_request: str, __user__: Optional[Dict] = None) -> str:
        """
        AI: CALL THIS FIRST when user asks about PBS job status or execution.
        
        This function:
        1. Analyzes user's job query request
        2. Determines what type of job information they want
        3. Provides guidance on how to proceed
        
        :param user_request: User's full request (e.g., "Check my job status")
        :param __user__: User info (auto-provided)
        :return: Guidance on how to proceed with job query
        """
        print(f"DEBUG: Job query help called with request: {user_request}")
        
        request_lower = user_request.lower()
        
        # Determine query type
        if any(word in request_lower for word in ['job', 'pbs', 'status']):
            if any(word in request_lower for word in ['recent', 'latest', 'my']):
                return """✅ AI: User wants recent job information.

Call get_recent_jobs() to show their recent PBS jobs."""
            
            elif any(word in request_lower for word in ['all', 'history', 'summary']):
                return """✅ AI: User wants job history or summary.

Call get_all_user_jobs() for complete history or get_job_summary() for statistics."""
            
            else:
                return """✅ AI: User wants job status information.

Ask user to specify:
- Job ID (e.g., "12345") for specific job status
- "recent" for recent jobs
- "all" for job history
- "summary" for job statistics"""
        
        else:
            return """❌ AI: No job-related query detected.

Ask user what they want to know about their PBS jobs:
- Specific job status (provide job ID)
- Recent jobs
- Job history
- Job summary/statistics"""
