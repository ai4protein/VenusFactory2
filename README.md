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

**Agent-driven protein engineering platform**  
Web UI · REST API · CLI · 40+ models · 11+ databases

</div>

<details>
<summary>📨 WeChat / feedback</summary>
<p align="center"><img src="img/wechat.png" width="50%" alt="WeChat"></p>
</details>

---

## What is it?

VenusFactory2 puts protein language models, biological databases, and an Agent orchestrator in one place: upload a sequence/structure to predict, drive multi-step analysis in natural language, or fine-tune your own models.

Try online: [venusfactory.bio](https://venusfactory.bio/)

<p align="center">
  <img src="img/web_v2/train.png" width="90%" alt="VenusFactory2 Training Interface">
</p>

<p align="center">
  <img src="img/web_v2/agent.png" width="90%" alt="VenusFactory2 Agent Interface">
</p>

---

## Quick Start

| Prerequisite | Notes |
|:-------------|:------|
| Python | **≥3.12** (installer creates a 3.12 `.venv` by default) |
| Node.js | **25.x** + npm (WebUI v2 build) |
| Disk | Reserve several–15+ GB (PyTorch wheels dominate) |
| GPU | Optional; NVIDIA → CUDA wheels, else CPU |

Slow Hugging Face access: `export HF_ENDPOINT=https://hf-mirror.com`

```bash
git clone https://github.com/AI4Protein/VenusFactory2.git && cd VenusFactory2
python scripts/setup_quickstart.py          # interactive one-click install
# python scripts/setup_quickstart.py -y     # CI / non-interactive

source .venv/bin/activate
python src/webui_v2.py --host 0.0.0.0 --port 7861
# → http://localhost:7861
```

The installer detects older setups (`.venv` / frontend / weights) and can reuse or wipe them. Press Enter for the recommended path.

**After launch**

1. Open the UI → Quick Tools work without an LLM key  
2. Agent: set an API key in Settings, or `cp .env.example .env` (see `.env.example`)  
3. Optional check: `python scripts/check_env.py`

---

## Documentation

Three layers — start at the top; dig only when needed:

| Layer | Where |
|:------|:------|
| Map of all docs | **[docs/README.md](docs/README.md)** |
| Install · catalogs · WebUI map · CLI/API | [docs/wiki/Home.md](docs/wiki/Home.md) |
| Per-feature how-tos | [docs/manual/](docs/manual/) |
| Agent skills (authors) | [src/agent/skills/AGENTS.md](src/agent/skills/AGENTS.md) · [docs/agent/SKILLS_INDEX.md](docs/agent/SKILLS_INDEX.md) · [docs/agent/CONTRIBUTING_SKILLS.md](docs/agent/CONTRIBUTING_SKILLS.md) |

中文：[README_CN.md](README_CN.md) · [docs/README.md](docs/README.md)

---

## Citation

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

## License & acknowledgement

Released under the **VenusFactory Non-Commercial License**: free for academic / non-commercial use; commercial use needs written approval → **<hongl3lilang@sjtu.edu.cn>**. See [`LICENSE`](./LICENSE) / [`LICENSE_CN.md`](./LICENSE_CN.md).

Maintained by [Liang's Lab](https://ins.sjtu.edu.cn/people/lhong/index.html), Shanghai Jiao Tong University.  
[Website](https://venusfactory.bio/) · [YouTube](https://www.youtube.com/watch?v=MT6lPH5kgCc) · [Issues](https://github.com/AI4Protein/VenusFactory2/issues)
