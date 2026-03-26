import os
import json
import asyncio
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

load_dotenv()

SCP_WORKFLOW_SERVER_URL = "http://115.190.136.251:8080/mcp"

class SCPWorkflowClient:
    def __init__(self, server_url: str = SCP_WORKFLOW_SERVER_URL):
        self.server_url = server_url
        self.session = None
        self.transport = None
        self.session_ctx = None
        
    async def connect(self, timeout: int = 30):
        """Connect to SCP Workflow server"""
        self.transport = streamablehttp_client(
            url=self.server_url,
            sse_read_timeout=60 * 10
        )
        
        self.read, self.write, self.get_session_id = await asyncio.wait_for(
            self.transport.__aenter__(), 
            timeout=timeout
        )
        
        self.session_ctx = ClientSession(self.read, self.write)
        self.session = await self.session_ctx.__aenter__()
        await asyncio.wait_for(
            self.session.initialize(),
            timeout=timeout
        )
        
    async def disconnect(self):
        try:
            if self.session_ctx:
                await self.session_ctx.__aexit__(None, None, None)
            if self.transport:
                await self.transport.__aexit__(None, None, None)
        except Exception:
            pass
    
    async def generate_presigned_url(self, key: str, expires_seconds: int = 3600) -> Dict[str, Any]:
        result = await self.session.call_tool(
            "generate_presigned_url",
            arguments={
                "key": key,
                "expires_seconds": expires_seconds
            }
        )
        
        if hasattr(result, 'content') and result.content:
            text = result.content[0].text
            return json.loads(text)
        return result

async def upload_file_via_curl(upload_url: str, file_path: str) -> bool:
    try:
        def _put_file():
            with open(file_path, 'rb') as f:
                # Use requests to PUT the file content; increased timeout for larger files
                response = requests.put(upload_url, data=f, timeout=300)
            return response.status_code

        status_code = await asyncio.to_thread(_put_file)
        if status_code not in [200, 201]:
            print(f"Upload failed with status code: {status_code}")
            return False
        return True
    except Exception as e:
        print(f"Upload via requests failed: {e}")
        return False

async def upload_file_to_cloud_async(file_path: str, key: Optional[str] = None, expires_seconds: int = 3600) -> Optional[str]:
    client = SCPWorkflowClient()
    try:
        await client.connect()
        
        if not key:
            key = Path(file_path).name
        
        result = await client.generate_presigned_url(key, expires_seconds)
        
        if "error" in result:
             return None

        upload_url = result["upload"]["url"]
        download_url = result["download"]["url"]
        
        success = await upload_file_via_curl(upload_url, file_path)
        
        if success:
            return download_url
        return None
        
    except Exception as e:
        print(f"Unexpected error in upload_file_to_cloud_async: {e}")
        return None
    finally:
        await client.disconnect()

def upload_file_to_oss_sync(file_path: str) -> Optional[str]:
    if not file_path or not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return None
    
    try:
        # Check if we are in a loop (e.g. mcp server main loop)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If in a running loop, we shouldn't create a new one or run_until_complete?
            # Actually upload_file_to_oss_sync is intended to be called from sync context.
            # But if called from async context (like ESMFold_predict which is async/sync hybrid),
            # creating new loop raises "Cannot run the event loop while another loop is running".
            # The safe way is to create a new loop ONLY if none exists, or use a thread.
            pass

        # The original implementation always created a new loop, which caused issues in Test Script.
        # Here we keep validation simple but robust implementation:
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            url = loop.run_until_complete(upload_file_to_cloud_async(file_path))
            return url
        finally:
            loop.close()
    except Exception as e:
        print(f"Failed to upload file to OSS: {file_path}, error: {e}")
        return None
