# VenusFactory 下载模块使用指南

## InterPro 元数据
**描述**: 从InterPro数据库下载蛋白质结构域信息

**数据源**: [InterPro数据库](https://www.ebi.ac.uk/interpro/)

**下载选项**:
- 单个ID: 下载特定InterPro结构域数据（例如IPR000001）
- 通过JSON: 使用包含多个InterPro条目的JSON文件进行批量下载

**输出格式**:
```
download/interpro_domain/
└── IPR000001/
    ├── detail.json    # 详细蛋白质信息
    ├── meta.json      # 元数据（包含编号和蛋白质计数）
    └── uids.txt       # 关联的UniProt ID列表
```

## RCSB 元数据
**描述**: 从RCSB蛋白质数据库下载结构元数据

**数据源**: [RCSB PDB](https://www.rcsb.org/)

**下载选项**:
- 单个ID: 下载特定PDB条目的元数据（例如1a0j）
- 通过文件: 使用包含PDB ID的文本文件进行批量下载

**输出格式**:
```
download/rcsb_metadata/
└── 1a0j.json         # 包含结构元数据：
                     # - 分辨率
                     # - 实验方法
                     # - 出版物信息
                     # - 链信息
```

## UniProt 序列
**描述**: 从UniProt数据库下载蛋白质序列

**数据源**: [UniProt](https://www.uniprot.org/)

**下载选项**:
- 单个ID: 下载特定UniProt条目的序列（例如P00734）
- 通过文件: 使用包含UniProt ID的文本文件批量下载
- 合并选项: 将所有序列合并为单个FASTA文件

**输出格式**:
```
download/uniprot_sequences/
├── P00734.fasta      # 单独FASTA文件（未合并时）
└── merged.fasta      # 合并后的序列文件（启用合并选项时）
```

## RCSB 结构
**描述**: 从RCSB PDB下载3D结构文件

**数据源**: [RCSB PDB](https://www.rcsb.org/)

**下载选项**:
- 单个ID: 下载特定PDB条目的结构
- 通过文件: 使用包含PDB ID的文本文件批量下载
- 文件类型:
    * cif: mmCIF格式（推荐）
    * pdb: 传统PDB格式
    * xml: PDBML/XML格式
    * sf: 结构因子
    * mr: NMR约束数据
- 解压选项: 自动解压下载文件

**输出格式**:
```
download/rcsb_structures/
├── 1a0j.pdb          # 解压后的结构文件（启用解压时）
└── 1a0j.pdb.gz       # 压缩的结构文件（未解压时）
```

## AlphaFold2 结构
**描述**: 从AlphaFold蛋白质结构数据库下载预测结构

**数据源**: [AlphaFold DB](https://alphafold.ebi.ac.uk/)

**下载选项**:
- 单个ID: 下载特定UniProt条目的结构
- 通过文件: 使用包含UniProt ID的文本文件批量下载
- 索引层级: 根据ID前缀组织子目录

**输出格式**:
```
download/alphafold2_structures/
└── P/               # 索引层级=1
    └── P0/          # 索引层级=2
        └── P00734.pdb  # AlphaFold预测结构
```

## 通用功能
- **错误处理**: 所有组件支持生成错误日志文件
- **输出目录**: 可自定义输出路径
- **批处理**: 支持通过文件输入多个ID
- **进度跟踪**: 实时显示下载进度和状态更新

## 输入文件格式
1. **PDB ID列表**（用于RCSB下载）:
```
1a0j
4hhb
1hho
```

2. **UniProt ID列表**（用于UniProt和AlphaFold）:
```
P00734
P61823
Q8WZ42
```

3. **InterPro JSON**（用于批量InterPro下载）:
```json
[
    {
        "metadata": {
            "accession": "IPR000001"
        }
    },
    {
        "metadata": {
            "accession": "IPR000002"
        }
    }
]
```

## 错误日志文件
启用错误日志后，失败记录将保存到输出目录的`failed.txt`:
```
P00734 - Download failed: 404 Not Found
1a0j - Connection timeout
```