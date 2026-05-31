# Download — 生物数据库批量检索

**Download** 标签让你能从主要数据库批量拉序列、结构和元数据，不用挨个去网页点。所有六个源工作流一致：选 **Single ID** 或 **From File**、点运行、拿归档。

## 数据源一览

| 数据源 | 路由 | ID 示例 | 你能拿到什么 |
| :--- | :--- | :--- | :--- |
| UniProt | `/download/uniprot` | `P00734` | 蛋白序列（FASTA） |
| NCBI | `/download/ncbi` | `NP_000517.1` | 蛋白序列（FASTA） |
| RCSB Structure | `/download/rcsb-structure` | `1a0j` | 三维结构（`.pdb` / `.cif`） |
| AlphaFold | `/download/alphafold` | `P00734` | 预测结构（`.pdb`） |
| RCSB Metadata | `/download/rcsb-metadata` | `1a0j` | 每条 entry 的 JSON 元数据 |
| InterPro | `/download/interpro` | `IPR000001` | 每条 entry 的 JSON 元数据 |

---

## 通用布局

每个下载页都用同一套表单：

| 字段 | 说明 |
| :--- | :--- |
| **Download Method** | `Single ID` 或 `From File`。 |
| **ID 输入（Single）** | 输入一个 ID — label / placeholder 因源而异。 |
| **文件上传（From File）** | 拖入 `.txt`（每行一个 ID）或从 **Workspace** 选。点 **Use Example** 加载示例列表。会显示前 20 条预览。 |
| **Save Error File** | （默认开）失败 ID 写到输出目录的 `failed.txt`。 |
| **源相关选项** | 见下方各源说明。 |

运行完会显示状态消息和结果归档链接。结构类源还会内嵌一个 3D viewer。

---

## 1. UniProt Sequences

| 字段 | 说明 |
| :--- | :--- |
| **ID 格式** | UniProt accession（如 `P00734`）。 |
| **额外选项** | **Merge FASTA** — 把所有命中合并成一个 multi-record FASTA。 |

**磁盘布局**（Single + 不合并）：
```
download/uniprot_sequences/
├── P00734.fasta      # 每个 ID 一个 FASTA
└── merged.fasta      # 仅在 merge 启用时
```

---

## 2. NCBI Sequences

| 字段 | 说明 |
| :--- | :--- |
| **ID 格式** | RefSeq / GenBank 蛋白 accession（如 `NP_000517.1`、`XP_011541001.1`）。 |
| **额外选项** | **Merge FASTA** — 合并为单文件。 |

磁盘布局同 UniProt。

---

## 3. RCSB Structures

| 字段 | 说明 |
| :--- | :--- |
| **ID 格式** | 4 字符 PDB 编号（如 `1a0j`）。 |
| **File Type** | `pdb` 或 `cif`（mmCIF，大型结构推荐）。 |

**磁盘布局：**
```
download/rcsb_structures/
└── 1a0j.pdb          # 或 1a0j.cif
```

下载的结构会通过 Molstar 内嵌渲染，方便先看一眼再下载归档。

---

## 4. AlphaFold Structures

| 字段 | 说明 |
| :--- | :--- |
| **ID 格式** | UniProt accession（如 `P00734`）。 |
| **额外展示** | Molstar viewer 旁边显示每个结构的 pLDDT 和 B-factor 统计。 |

**磁盘布局：**
```
download/alphafold_structures/
└── P00734.pdb        # AlphaFold 预测结构
```

---

## 5. RCSB Metadata

| 字段 | 说明 |
| :--- | :--- |
| **ID 格式** | PDB 编号。 |
| **返回** | 每条 entry 的 JSON：分辨率、实验方法、文献信息、链组成等。 |

```
download/rcsb_metadata/
└── 1a0j.json
```

---

## 6. InterPro Metadata

| 字段 | 说明 |
| :--- | :--- |
| **ID 格式** | InterPro accession（如 `IPR000001`）。 |
| **返回** | domain 详情 + 该 domain 关联的 UniProt ID 列表。 |

```
download/interpro_domain/
└── IPR000001/
    ├── detail.json    # 蛋白详细信息
    ├── meta.json      # accession + 蛋白计数
    └── uids.txt       # 该 domain 下的 UniProt ID 列表
```

---

## 输入文件格式

**ID 列表（UniProt / NCBI / RCSB / AlphaFold / InterPro）：** 每行一个 ID。

```
P00734
P61823
Q8WZ42
```

**InterPro 批量从 JSON**（旧格式）：含 `metadata.accession` 的对象数组。

```json
[
    {"metadata": {"accession": "IPR000001"}},
    {"metadata": {"accession": "IPR000002"}}
]
```

---

## 错误文件

开启 **Save Error File** 时，失败 ID 写入输出目录的 `failed.txt`：

```
P00734 - Download failed: 404 Not Found
1a0j   - Connection timeout
```

---

## 小贴士

- **分批 50–200 个 ID。** 公共 API 会限速；后端虽已做节流，但超大列表仍然慢。
- **大结构用 `cif`。** 旧的 PDB 格式装不下现代大型组装体，`cif` 可以。
- **AlphaFold ≠ 实验结构。** 始终看 pLDDT 面板——低置信区域（橙 / 红）不要过度解读。
- **Merge FASTA** 只在你想要单文件喂下游（如 MMseqs / BLAST）时有用。要每 ID 一个文件就关掉。
