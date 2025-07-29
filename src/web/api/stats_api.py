from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stats_manager import global_stats_manager

app = FastAPI(title="VenusFactory Stats API", version="1.0.0")

class UsageEvent(BaseModel):
    module: str
    timestamp: str = None

@app.post("/api/stats/track")
async def track_usage(event: UsageEvent):
    """追踪功能使用次数"""
    if not event.timestamp:
        event.timestamp = datetime.now().isoformat()
    
    success = global_stats_manager.track_usage(event.module)
    if not success:
        raise HTTPException(status_code=400, detail=f"Invalid module: {event.module}")
    
    return {
        "status": "success", 
        "message": f"Tracked {event.module} usage",
        "timestamp": event.timestamp
    }

@app.get("/api/stats")
async def get_stats():
    """获取统计数据"""
    stats = global_stats_manager.get_stats()
    return stats

@app.get("/api/stats/reset")
async def reset_stats():
    """重置统计数据"""
    success = global_stats_manager.reset_stats()
    if success:
        return {"status": "success", "message": "Stats reset successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reset stats")

@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "VenusFactory Stats API",
        "version": "1.0.0",
        "endpoints": {
            "track_usage": "POST /api/stats/track",
            "get_stats": "GET /api/stats",
            "reset_stats": "GET /api/stats/reset"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 