import json
import os
from datetime import datetime

class StatsManager:
    def __init__(self):
        self.stats_file = "data/usage_stats.json"
        self.backup_file = "data/usage_stats_backup.json"
        self.ensure_data_dir()
        self.load_stats()
    
    def ensure_data_dir(self):
        """确保数据目录存在"""
        os.makedirs("data", exist_ok=True)
    
    def load_stats(self):
        """加载统计数据"""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
            elif os.path.exists(self.backup_file):
                with open(self.backup_file, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
                self.save_stats()
            else:
                self.stats = self.get_default_stats()
                self.save_stats()
        except Exception as e:
            print(f"Error loading stats: {e}")
            self.stats = self.get_default_stats()
    
    def save_stats(self):
        """保存统计数据"""
        try:
            temp_file = self.stats_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
            
            if os.path.exists(self.stats_file):
                import shutil
                shutil.copy2(self.stats_file, self.backup_file)
            
            os.replace(temp_file, self.stats_file)
        except Exception as e:
            print(f"Error saving stats: {e}")
    
    def track_usage(self, module):
        """追踪功能使用次数"""
        if module in self.stats:
            self.stats[module] += 1
            self.stats["last_updated"] = datetime.now().isoformat()
            self.save_stats()
            return True
        return False
    
    def get_stats(self):
        """获取统计数据"""
        return self.stats.copy()
    
    def get_default_stats(self):
        """获取默认统计数据"""
        return {
            "mutation_prediction": 0,
            "function_analysis": 0,
            "total_visits": 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def reset_stats(self):
        """重置统计数据"""
        self.stats = self.get_default_stats()
        self.save_stats()
        return True

# 全局实例
global_stats_manager = StatsManager() 