#!/usr/bin/env python3
"""
VenusFactory 启动脚本
同时启动主应用和统计API服务
"""

import subprocess
import sys
import time
import threading
import os
from pathlib import Path

def start_stats_api():
    """启动统计API服务"""
    try:
        print("🚀 启动统计API服务...")
        api_process = subprocess.Popen([
            sys.executable, "api/stats_api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待API服务启动
        time.sleep(3)
        
        if api_process.poll() is None:
            print("✅ 统计API服务启动成功 (端口: 8000)")
            return api_process
        else:
            print("❌ 统计API服务启动失败")
            return None
    except Exception as e:
        print(f"❌ 启动统计API服务时出错: {e}")
        return None

def start_main_app():
    """启动主应用"""
    try:
        print("🚀 启动VenusFactory主应用...")
        main_process = subprocess.Popen([
            sys.executable, "src/webui.py"
        ])
        
        print("✅ VenusFactory主应用启动成功 (端口: 7860)")
        return main_process
    except Exception as e:
        print(f"❌ 启动主应用时出错: {e}")
        return None

def main():
    """主函数"""
    print("=" * 50)
    print("🎯 VenusFactory 启动器")
    print("=" * 50)
    
    # 检查必要文件
    required_files = [
        "src/webui.py",
        "api/stats_api.py",
        "utils/stats_manager.py"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 缺少必要文件: {file_path}")
            return
    
    print("✅ 所有必要文件检查通过")
    
    # 启动统计API服务
    api_process = start_stats_api()
    if not api_process:
        print("⚠️  统计API服务启动失败，将使用模拟数据")
    
    # 启动主应用
    main_process = start_main_app()
    if not main_process:
        print("❌ 主应用启动失败")
        if api_process:
            api_process.terminate()
        return
    
    print("\n" + "=" * 50)
    print("🎉 VenusFactory 启动完成!")
    print("📊 主应用: http://localhost:7860")
    if api_process:
        print("📈 统计API: http://localhost:8000")
    print("=" * 50)
    print("按 Ctrl+C 停止所有服务")
    
    try:
        # 等待进程结束
        while True:
            if main_process.poll() is not None:
                print("主应用已停止")
                break
            if api_process and api_process.poll() is not None:
                print("统计API服务已停止")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 正在停止服务...")
        
        if main_process:
            main_process.terminate()
            print("✅ 主应用已停止")
        
        if api_process:
            api_process.terminate()
            print("✅ 统计API服务已停止")
        
        print("👋 所有服务已停止")

if __name__ == "__main__":
    main() 