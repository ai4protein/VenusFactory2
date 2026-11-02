<!-- Compatibility view; edit zh/data.md -->

# 检索数据与证据

## 在 Chat 提出需求

下载作为工具能力提供，不再单独占用侧栏页面。在 Agent 中说明蛋白、物种和所需数据，并要求保留编号和文件，例如：

> 获取人溶菌酶经过审阅的 UniProt 序列及可用 PDB 结构，保留数据库编号，将序列保存为 FASTA。

已注册工具可覆盖 UniProt、NCBI、RCSB、AlphaFold DB、InterPro 等开启的服务。网络与文献检索使用后台配置的 Tavily、SerpAPI 等提供方，无需把搜索密钥粘贴到 Chat。

## 按问题选择数据库

| 想查询什么 | 可用能力 |
| --- | --- |
| 靶点、疾病和药物的联系 | **Open Targets**：按基因、疾病或药物名检索，查看关联、可成药性和作用机制 |
| 基因编号与跨物种信息 | **Ensembl**：按基因名和物种解析编号，查询注释、交叉引用、直系同源基因和序列 |
| 基因表达与富集 | **gget**：Ensembl/UniProt 信息、序列、PDB、ARCHS4 表达、Bgee 同源关系、Open Targets 和 Enrichr |
| 论文与摘要 | **Europe PMC**：摘要、DOI/PMID、发表年份、开放获取与撤稿信息 |
| 小分子信息 | **PubChem**：名称、CID/SMILES、分子式、分子量与理化描述符 |

可以直接提问：

> 查询人类 BRCA2，整理 Open Targets 中的疾病关联和药物候选，附上来源。

> 查找人类 BRCA2 在小鼠中的直系同源基因，并获取蛋白序列。

> 在 Europe PMC 查找近期 PET 水解酶论文，比较各研究的测定条件。

> 从 PubChem 查询咖啡因的 SMILES 和分子性质。

这些数据库查询本身不需要额外的大模型或搜索服务密钥。本地和在线均可使用，
由 Administration 控制开关。Agent 和 Scientist 可以按任务搜索调用。
Researcher 通过同一个 Europe PMC 工具检索文献；UniProt 提供明确的 Ensembl 基因映射时，
还会采集基因注释，并为人类蛋白补充相关的 Open Targets 疾病关联和药物候选。

Open Targets 面向人类治疗靶点，比较记录前先确认编号和物种。数据库关联不等于因果证明，
PubChem 理化描述符也不等于 ADMET 或结合能力预测。结果保留来源链接、检索时间，以及
服务提供的版本信息；需要更多结果时继续翻页。gget 提供选定的公开查询模块，
不包含大型数据集下载或付费 AI 模块。

## 序列与结构检索

NCBI BLAST 用于序列相似性检索，远程 Foldseek 用于结构相似性检索，gget 适配器提供支持的基因、序列和注释查询。提供正确编号、物种、序列或结构，只有蛋白名时可能需要澄清。

远程服务可能先返回作业编号，继续查询该编号即可，不要重复提交同一检索。公共服务器的限流和故障与平台自身是否可用不同。

## 保留来源

保存数据库编号、来源链接、查询背景和相关版本。检索摘要不等于论文全文，同源命中也不等于功能已被实验验证。需要全面整合时使用 [Researcher](./zh/researcher.md)。

分析沙箱不联网，Agent 先通过注册的检索或下载工具获得文件，再进行本地分析。本地 VenusMine/Foldseek 挖掘与远程 Foldseek 服务不同，前者在线版不可用。
