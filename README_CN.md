<div align="right">
  <a href="README.md">English</a> | <a href="README_CN.md">简体中文</a>
</div>

<p align="center">
  <img src="img/banner_2503.png" width="70%" alt="VenusFactory2 Banner">
</p>

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/AI4Protein/VenusFactory2?style=flat-square)](https://github.com/AI4Protein/VenusFactory2/stargazers)
[![GitHub license](https://img.shields.io/github/license/AI4Protein/VenusFactory2?style=flat-square)](https://github.com/AI4Protein/VenusFactory2/blob/main/LICENSE)
[![Wiki](https://img.shields.io/badge/Wiki-docs-brightgreen?style=flat-square)](docs/wiki/Home.md)
[![Demo](https://img.shields.io/badge/Demo-Website-blue?style=flat-square)](https://venusfactory.bio/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.27303-b31b1b?style=flat-square)](https://arxiv.org/abs/2603.27303)

**Agent 驱动的蛋白质工程平台**  
Web UI · REST API · CLI · 40+ 模型 · 11+ 数据库

</div>

<details>
<summary>📨 微信群 / 反馈</summary>
<p align="center"><img src="img/wechat.png" width="50%" alt="WeChat"></p>
</details>

---

## 这是什么？

VenusFactory2 把蛋白质语言模型、数据库与 Agent 编排放在同一平台：上传序列/结构即可预测，也可用自然语言驱动多步分析，或训练自己的微调模型。

在线试用：[venusfactory.bio](https://venusfactory.bio/)

<p align="center">
  <img src="img/web_v2/train.png" width="90%" alt="VenusFactory2 训练界面">
</p>

<p align="center">
  <img src="img/web_v2/agent.png" width="90%" alt="VenusFactory2 Agent 界面">
</p>

---

## 快速开始

| 前置 | 说明 |
|:-----|:-----|
| Python | **≥3.12**（一键安装默认创建 3.12 的 `.venv`） |
| Node.js | **25.x** + npm（构建 WebUI v2） |
| 磁盘 | 建议预留数 GB～十余 GB（含 PyTorch） |
| GPU | 可选；有 NVIDIA 则装 CUDA 轮子，否则 CPU |

国内 HF 较慢时可先：`export HF_ENDPOINT=https://hf-mirror.com`

```bash
git clone --recurse-submodules https://github.com/AI4Protein/VenusFactory2.git && cd VenusFactory2
# 若已普通 clone：git submodule update --init --recursive
python scripts/setup_quickstart.py          # 交互一键安装（推荐）
# python scripts/setup_quickstart.py -y     # CI / 无交互

source .venv/bin/activate
python src/webui_v2.py --host 0.0.0.0 --port 7861
# → http://localhost:7861
```

安装器会扫描旧环境（`.venv` / 前端 / 权重），可选复用或清理重装。一路回车即走推荐方案。

**装完之后**

1. 打开网页 → Quick Tools 无需 LLM Key 即可试用  
2. Agent：在界面 Settings 配置 API Key，或 `cp .env.example .env` 后填写（见 `.env.example`）  
3. 验证环境：`python scripts/check_env.py`

---

## 文档

分三层，从上往下看即可：

| 层级 | 入口 |
|:-----|:-----|
| 文档总览 | **[docs/README.md](docs/README.md)** |
| 安装 · 目录 · WebUI 地图 · CLI/API | [docs/wiki/Home.md](docs/wiki/Home.md) |
| 各功能点操作手册 | [docs/manual/](docs/manual/) |
| Agent Skills（作者指南） | [src/agent/skills/AGENTS.md](src/agent/skills/AGENTS.md) · [docs/agent/SKILLS_INDEX.md](docs/agent/SKILLS_INDEX.md) · [docs/agent/CONTRIBUTING_SKILLS.md](docs/agent/CONTRIBUTING_SKILLS.md) |

English: [README.md](README.md) · [docs/README.md](docs/README.md)

---

## 引用

```bibtex
@article{tan2026venusfactory2,
  title={Self-evolving AI agents for protein discovery and directed evolution},
  author={Tan, Yang and Zhang, Lingrong and Li, Mingchen and Yu, Yuanxi and Zhong, Bozitao and Zhou, Bingxin and Dong, Nanqing and Hong, Liang},
  journal={arXiv preprint arXiv:2603.27303},
  year={2026}
}

@inproceedings{tan2025venusfactory,
  title={VenusFactory: An Integrated System for Protein Engineering with Data Retrieval and Language Model Fine-Tuning},
  author={Tan, Yang and Liu, Chen and Gao, Jingyuan and Banghao, Wu and Li, Mingchen and Wang, Ruilin and Zhang, Lingrong and Yu, Huiqun and Fan, Guisheng and Hong, Liang and others},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  pages={230--241},
  year={2025}
}
```

---

## 协议与致谢

采用 **VenusFactory 非商业许可**：学术/非商业免费；商用需邮件授权 → **<hongl3lilang@sjtu.edu.cn>**。详见 [`LICENSE`](./LICENSE) / [`LICENSE_CN.md`](./LICENSE_CN.md)。

由上海交通大学 [Liang's Lab](https://ins.sjtu.edu.cn/people/lhong/index.html) 维护。  
[Website](https://venusfactory.bio/) · [YouTube](https://www.youtube.com/watch?v=MT6lPH5kgCc) · [Issues](https://github.com/AI4Protein/VenusFactory2/issues)
