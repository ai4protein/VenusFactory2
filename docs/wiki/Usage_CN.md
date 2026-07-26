# Agent · CLI · API · 训练方法

回到 [Wiki 首页](./Home.md) · [English](./Usage.md)

## Agent-0.1

自然语言编排工具链（LangGraph + LangChain）。示例：

```
您: "为 PDB:1ABC 设计耐热突变"
→ 下载结构 → ESM-2 扫描 → 稳定性评分 → 排序报告
```

| 类别 | 能力 |
|:-----|:-----|
| 分析 | 突变 · 功能/稳定性 · 结构 |
| 数据 | 多库检索 · 格式转换 · 批量 |
| 规划 | 多步自动化 · 工具编排 |
| 研究 | 文献 · 家族分析 · 报告 |

需要 LLM API Key（界面 Settings 或 `.env`，见 `.env.example`）。  
详细手册：[AgentManual_CN.md](../manual/AgentManual_CN.md)

## CLI（示例）

脚本多在 `script/tools/` 下：

```bash
# 训练（路径以仓库实际文件为准）
bash script/tools/train/train_plm_lora.sh \
  --model facebook/esm2_t33_650M_UR50D \
  --dataset DeepSol --batch_size 32

# 评估
bash script/tools/evaluate/eval.sh \
  --model_path ckpt/DeepSol/best_model \
  --test_dataset DeepSol
```

数据库下载等脚本见 `script/tools/database/`。

## REST API

```bash
python src/api_server.py   # → http://localhost:5000/docs
```

```bash
curl -X POST http://localhost:5000/api/mutation/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTAYIA...", "mutations": ["A23V", "K45R"]}'
```

具体路由以 `/docs` 为准（版本可能演进）。

## Python 调用（示意）

```python
from src.tools.mutation import predict_mutation_effects
from src.tools.predict import predict_protein_function

results = predict_mutation_effects(
    sequence="MKTAYIAKQR...",
    mutations=["A5V", "K9R"],
    model="esm2",
)
```

导入路径以当前代码为准。

## PEFT 训练方法对照

| 方法 | 内存 | 速度 | 表现 | 适合 |
|:-----|:----:|:----:|:----:|:-----|
| LoRA | 低 | 快 | 良好 | 通用 |
| QLoRA | 极低 | 慢 | 良好 | 显存紧张 |
| DoRA | 低 | 中 | 更好 | LoRA 改进 |
| AdaLoRA | 低 | 中 | 更好 | 自适应秩 |
| SES-Adapter | 中 | 中 | 更好 | 选择性调优 |
| IA3 | 极低 | 快 | 良好 | 轻量 |
| Freeze | 低 | 快 | 良好 | 简单调优 |

详见 [TrainingManual_ZH.md](../manual/TrainingManual_ZH.md)。
