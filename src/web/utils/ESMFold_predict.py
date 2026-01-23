import os
import json
import base64
import asyncio
from typing import Dict, Any, Optional, Union, Tuple

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

from dotenv import load_dotenv

load_dotenv()

DRUGSDA_MODEL_SERVER_URL = os.getenv("DRUGSDA_MODEL_SERVER_URL")
DRUGSDA_TOOL_SERVER_URL = os.getenv("DRUGSDA_TOOL_SERVER_URL")

class DrugSDAClient:    
    """DrugSDA MCP客户端类"""
    
    def __init__(self, server_url: str):
        """
        初始化DrugSDA客户端
        
        参数:
            server_url: 服务器URL
        """
        self.server_url = server_url
        self.session = None
        self.session_ctx = None
        self.transport = None
        
    async def connect(self, verbose: bool = True) -> bool:
        """
        连接到DrugSDA服务器
        
        参数:
            verbose: 是否打印详细信息
            
        返回:
            连接是否成功
        """
        if verbose:
            print(f"连接到服务器: {self.server_url}")
        try:
            self.transport = streamablehttp_client(
                url=self.server_url,
                headers={"SCP-HUB-API-KEY": os.getenv("SCP_HUB_API_KEY")}
            )
            self.read, self.write, self.get_session_id = await self.transport.__aenter__()
            
            self.session_ctx = ClientSession(self.read, self.write)
            self.session = await self.session_ctx.__aenter__()

            await self.session.initialize()
            session_id = self.get_session_id()
            
            if verbose:
                print(f"✓ 连接成功 (会话ID: {session_id})")
            return True
            
        except Exception as e:
            if verbose:
                print(f"✗ 连接失败: {e}")
                import traceback
                traceback.print_exc()
            return False
    
    async def disconnect(self, verbose: bool = True) -> None:
        """
        断开与DrugSDA服务器的连接
        
        参数:
            verbose: 是否打印详细信息
        """
        try:
            if self.session_ctx:
                await self.session_ctx.__aexit__(None, None, None)
            if hasattr(self, 'transport'):
                await self.transport.__aexit__(None, None, None)
            if verbose:
                print("✓ 已断开连接")
        except Exception as e:
            if verbose:
                print(f"✗ 断开连接时出错: {e}")
    
    def parse_result(self, result: Any) -> Any:
        """
        解析MCP工具调用结果
        
        参数:
            result: MCP工具调用结果
            
        返回:
            解析后的结果
        """
        try:
            if hasattr(result, 'content') and result.content:
                content = result.content[0]
                if hasattr(content, 'text'):
                    return json.loads(content.text)
            return str(result)
        except Exception as e:
            return {"error": f"解析结果失败: {e}", "raw": str(result)}


def base64_to_pdb_file(base64_string: str, file_name: str = "protein_structure.pdb", 
                       save_dir: str = "./protein_structures") -> Optional[str]:
    """
    将Base64字符串转换为PDB文件并保存到本地
    
    参数:
        base64_string: Base64编码的字符串
        file_name: 保存的文件名
        save_dir: 保存的目录
        
    返回:
        保存的文件路径，如果失败则返回None
    """
    try:
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 解码Base64字符串
        pdb_content = base64.b64decode(base64_string)
        
        # 构建完整的文件路径
        file_path = os.path.join(save_dir, file_name)
        
        # 将内容写入文件
        with open(file_path, "wb") as f:
            f.write(pdb_content)
            
        print(f"✓ PDB文件已保存到: {file_path}")
        return file_path
    
    except Exception as e:
        print(f"✗ 保存PDB文件失败: {str(e)}")
        return None


async def predict_protein_structure(sequence: str, output_dir: str = "./protein_structures", 
                                   verbose: bool = True) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    使用ESMFold预测蛋白质结构并保存为PDB文件
    
    参数:
        sequence: 蛋白质序列
        output_dir: 输出目录
        verbose: 是否打印详细信息
        
    返回:
        (保存的PDB文件路径, 预测结果信息)，如果失败则返回(None, None)
    """
    if verbose:
        print("开始蛋白质结构预测...")
    
    # 连接到DrugSDA-Model服务器
    model_client = DrugSDAClient(DRUGSDA_MODEL_SERVER_URL)
    if not await model_client.connect(verbose=verbose):
        if verbose:
            print("连接DrugSDA-Model服务器失败")
        return None, None
    
    # 连接到DrugSDA-Tool服务器
    tool_client = DrugSDAClient(DRUGSDA_TOOL_SERVER_URL)
    if not await tool_client.connect(verbose=verbose):
        if verbose:
            print("连接DrugSDA-Tool服务器失败")
        await model_client.disconnect(verbose=verbose)
        return None, None
    
    try:
        # 步骤1: 使用ESMFold预测蛋白质结构
        if verbose:
            print("\n步骤1: 使用ESMFold预测蛋白质结构...")
        result = await model_client.session.call_tool(
            "pred_protein_structure_esmfold",
            arguments={
                "sequence": sequence
            }
        )
        
        result_data = model_client.parse_result(result)
        if "error" in result_data:
            if verbose:
                print(f"✗ 预测失败: {result_data['error']}")
            return None, None
            
        if verbose:
            print(result_data)
        pdb_path = result_data["pdb_path"]
        if verbose:
            print("蛋白质结构文件路径: ", pdb_path)
        
        # 步骤2: 将PDB文件转换为Base64字符串
        if verbose:
            print("\n步骤2: 将PDB文件转换为Base64字符串...")
        result = await tool_client.session.call_tool(
            "server_file_to_base64",
            arguments={
                "file_path": pdb_path
            }
        )
        
        result_data = tool_client.parse_result(result)
        if "error" in result_data:
            if verbose:
                print(f"✗ 转换为Base64失败: {result_data['error']}")
            return None, None
            
        file_name = result_data["file_name"]
        base64_string = result_data["base64_string"]
        if verbose:
            print(f"文件转换为Base64: {file_name}, Base64前20个字符: {base64_string[:20]}...")
        
        # 步骤3: 将Base64字符串转换为本地PDB文件
        if verbose:
            print("\n步骤3: 将Base64字符串转换为本地PDB文件...")
        local_file_path = base64_to_pdb_file(base64_string, file_name, output_dir)
        
        if not local_file_path:
            if verbose:
                print("✗ 保存PDB文件失败")
            return None, None
        
        # 返回结果
        result_info = {
            "file_name": file_name,
            "server_path": pdb_path,
            "local_path": local_file_path,
            "sequence_length": len(sequence)
        }
        
        if verbose:
            print(f"\n✓ 预测成功! PDB文件已保存到: {local_file_path}")
        
        return local_file_path, result_info
        
    except Exception as e:
        if verbose:
            print(f"✗ 预测过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
        return None, None
        
    finally:
        # 断开所有连接
        await tool_client.disconnect(verbose=verbose)
        await model_client.disconnect(verbose=verbose)


def predict_structure_sync(sequence: str, output_dir: str = "./protein_structures", 
                          verbose: bool = True) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    同步版本的蛋白质结构预测函数（用于非异步环境）
    
    参数:
        sequence: 蛋白质序列
        output_dir: 输出目录
        verbose: 是否打印详细信息
        
    返回:
        (保存的PDB文件路径, 预测结果信息)，如果失败则返回(None, None)
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(predict_protein_structure(sequence, output_dir, verbose))



if __name__ == "__main__":
    example_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVVRIELKGIDFKEDGNILGHKLEYNYNSHNVYITADKQKNGIKANFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
    
    pdb_path, result_info = predict_structure_sync(example_sequence)
    
    if pdb_path:
        print(f"预测成功! PDB文件路径: {pdb_path}")
        print(f"结果信息: {result_info}")
    else:
        print("预测失败!")
