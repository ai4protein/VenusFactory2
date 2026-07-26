#!/usr/bin/env python3
"""VenusFactory2 one-click installer (interactive by default).

  python scripts/setup_quickstart.py
  python scripts/setup_quickstart.py -y
  python scripts/setup_quickstart.py --lang zh   # optional Chinese UI
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
_FRONTEND = _REPO_ROOT / "frontend"
_DIST = _FRONTEND / "dist"
_INSTALL_DIR = _REPO_ROOT / ".install"
_VENV_DIR = _REPO_ROOT / ".venv"

# Minimal imports that mean "deps actually finished", not just torch wheels.
_SMOKE_IMPORTS = ("torch", "torch_geometric", "transformers")

_PRESET_BYTES = {
    "demo": 410_000,
    "predict-core": 163 * 1024 * 1024,
    "proteinmpnn": 64 * 1024 * 1024,
    "predict-all": 405 * 1024 * 1024,
    "all": 469 * 1024 * 1024,
}

# Rough download/install budgets (bytes) for planning / disk checks.
_DEPS_BYTES_GPU = 4 * 1024**3
_DEPS_BYTES_CPU = 2 * 1024**3
_FRONTEND_BYTES = 400 * 1024 * 1024

# UI language: English by default. Chinese only with explicit --lang zh.
_LANG = "en"


def _use_zh() -> bool:
    return _LANG == "zh"


def _t(en: str, zh: str) -> str:
    return zh if _use_zh() else en


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _section(title: str) -> None:
    _print()
    _print(f"==> {title}")


def _run(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> int:
    _print(f"$ {' '.join(cmd)}")
    if dry_run:
        return 0
    return int(subprocess.run(cmd, cwd=str(cwd or _REPO_ROOT)).returncode)


def _which(name: str) -> str | None:
    return shutil.which(name)


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _resolve_lang(cli: str) -> str:
    """English is the default installer UI. Opt into Chinese with --lang zh only."""
    if cli == "zh":
        return "zh"
    return "en"


def _venv_python() -> Path | None:
    candidate = (
        _VENV_DIR / "Scripts" / "python.exe"
        if os.name == "nt"
        else _VENV_DIR / "bin" / "python"
    )
    return candidate if candidate.is_file() else None


def _planned_venv_python() -> str:
    return str(
        _VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )


def _fmt_bytes(num: int | float) -> str:
    n = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{int(n)} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def _dir_size_bytes(path: Path, *, limit_files: int = 5000) -> int:
    total = 0
    count = 0
    try:
        for p in path.rglob("*"):
            if not p.is_file():
                continue
            try:
                total += p.stat().st_size
            except OSError:
                continue
            count += 1
            if count >= limit_files:
                break
    except OSError:
        return total
    return total


def _disk_free_bytes(path: Path) -> int | None:
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return None


def _python_imports_ok(python: str, modules: tuple[str, ...]) -> tuple[bool, str]:
    if not Path(python).exists():
        return False, "interpreter missing"
    code = "; ".join(f"import {m}" for m in modules)
    proc = subprocess.run(
        [python, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True, "ok"
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, (err[-1] if err else f"exit {proc.returncode}")


def _env_ready(python: str) -> tuple[bool, str]:
    return _python_imports_ok(python, _SMOKE_IMPORTS)


def _python_has_torch(python: str) -> bool:
    ok, _ = _python_imports_ok(python, ("torch",))
    return ok


def _detect_torch_type(requested: str) -> tuple[str, str]:
    if str(_INSTALL_DIR) not in sys.path:
        sys.path.insert(0, str(_INSTALL_DIR))
    from torch_detect import resolve_install_type

    return resolve_install_type(requested)


def _node_major() -> int | None:
    node = _which("node")
    if not node:
        return None
    out = subprocess.run(["node", "-v"], capture_output=True, text=True).stdout.strip()
    # v25.1.0
    if out.startswith("v"):
        out = out[1:]
    try:
        return int(out.split(".")[0])
    except (ValueError, IndexError):
        return None


def _conda_active() -> str | None:
    return os.getenv("CONDA_PREFIX") or None


class InstallPlan:
    def __init__(
        self,
        *,
        install_deps: bool = True,
        force_install_deps: bool = False,
        clean_venv: bool = False,
        clean_frontend: bool = False,
        torch_type: str = "auto",
        env: str = "venv",
        preset: str = "predict-core",
        skip_ckpts: bool = False,
        skip_frontend: bool = False,
        force_frontend: bool = False,
        with_frpc: bool = False,
        check_env: bool = False,
        hf_mirror: bool = False,
        port: int = 7861,
        dry_run: bool = False,
        offer_launch: bool = True,
    ) -> None:
        self.install_deps = install_deps
        self.force_install_deps = force_install_deps
        self.clean_venv = clean_venv
        self.clean_frontend = clean_frontend
        self.torch_type = torch_type
        self.env = env
        self.preset = preset
        self.skip_ckpts = skip_ckpts
        self.skip_frontend = skip_frontend
        self.force_frontend = force_frontend
        self.with_frpc = with_frpc
        self.check_env = check_env
        self.hf_mirror = hf_mirror
        self.port = port
        self.dry_run = dry_run
        self.offer_launch = offer_launch


class EnvSurvey:
    def __init__(self) -> None:
        self.venv_exists = _VENV_DIR.is_dir()
        self.venv_python = _venv_python()
        if self.venv_python:
            self.venv_ready, self.venv_detail = _env_ready(str(self.venv_python))
            self.venv_torch = _python_has_torch(str(self.venv_python))
        else:
            self.venv_ready, self.venv_detail = False, "missing"
            self.venv_torch = False
        self.system_ready, self.system_detail = _env_ready(sys.executable)
        self.system_torch = _python_has_torch(sys.executable)
        self.frontend_dist = (_DIST / "index.html").is_file()
        self.frontend_node_modules = (_FRONTEND / "node_modules").is_dir()
        self.ckpt_dir = _REPO_ROOT / "ckpt"
        self.ckpt_bytes = _dir_size_bytes(self.ckpt_dir) if self.ckpt_dir.is_dir() else 0
        self.has_old_env = bool(
            self.venv_exists
            or self.frontend_dist
            or self.frontend_node_modules
            or self.ckpt_bytes > 0
        )
        self.partial_venv = bool(self.venv_exists and self.venv_torch and not self.venv_ready)

    def lines(self) -> list[str]:
        lines: list[str] = []
        if self.venv_exists:
            if self.venv_ready:
                status = _t("healthy (torch+PyG+transformers)", "健康（torch+PyG+transformers）")
            elif self.partial_venv:
                status = _t(
                    f"PARTIAL (torch ok, missing deps: {self.venv_detail})",
                    f"半残（有 torch，缺依赖: {self.venv_detail}）",
                )
            else:
                status = _t(
                    f"broken ({self.venv_detail})",
                    f"损坏 ({self.venv_detail})",
                )
            lines.append(f".venv/              {_t('present', '存在')} ({status})")
        else:
            lines.append(f".venv/              {_t('not found', '不存在')}")
        lines.append(
            f"system python       "
            f"{'OK' if self.system_ready else _t('incomplete/missing', '不完整/缺失')} "
            f"({sys.executable})"
        )
        fe = _t("built", "已构建") if self.frontend_dist else _t("missing", "缺失")
        if self.frontend_node_modules:
            fe += "; node_modules"
        lines.append(f"frontend/dist        {fe}")
        if self.ckpt_bytes > 0:
            lines.append(f"ckpt/                ~{_fmt_bytes(self.ckpt_bytes)}")
        else:
            lines.append(f"ckpt/                {_t('empty', '空')}")
        return lines


def remove_path(path: Path, *, dry_run: bool, label: str) -> bool:
    _section(_t(f"Remove {label}", f"删除 {label}"))
    if not path.exists():
        _print(_t(f"Nothing to remove: {path}", f"无需删除: {path}"))
        return True
    _print(f"{_t('Removing', '正在删除')}: {path}")
    if dry_run:
        _print(_t("(dry-run) skip delete", "（dry-run）跳过删除"))
        return True
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:
        _print(_t(f"ERROR: failed to remove {path}: {exc}", f"错误：无法删除 {path}: {exc}"))
        _print(
            _t(
                "Close programs using this path, then re-run with --clean-venv / --clean-frontend.",
                "请关闭占用该路径的程序后，用 --clean-venv / --clean-frontend 重试。",
            )
        )
        return False
    _print(_t("Removed.", "已删除。"))
    return True


def _prompt_choice(
    title: str,
    options: list[tuple[str, str]],
    *,
    default_key: str,
) -> str:
    keys = [k for k, _ in options]
    if default_key not in keys:
        default_key = keys[0]
    _print()
    _print(title)
    default_idx = keys.index(default_key) + 1
    for i, (_key, label) in enumerate(options, start=1):
        marker = _t(" ← default", " ← 默认") if _key == default_key else ""
        _print(f"  {i}) {label}{marker}")
    while True:
        raw = input(
            _t(
                f"Select [1-{len(options)}] (Enter={default_idx}): ",
                f"请选择 [1-{len(options)}]（回车={default_idx}）: ",
            )
        ).strip()
        if not raw:
            return default_key
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        if raw in keys:
            return raw
        _print(_t(f"Invalid choice: {raw!r}", f"无效选项: {raw!r}"))


def _prompt_yes_no(title: str, *, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    _print()
    while True:
        raw = input(f"{title} [{hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes", "是"):
            return True
        if raw in ("n", "no", "否"):
            return False
        _print(_t("Please answer y or n.", "请输入 y 或 n。"))


def _estimate_plan_bytes(plan: InstallPlan, torch_profile: str) -> int:
    total = 0
    if plan.install_deps or plan.force_install_deps or plan.clean_venv:
        total += _DEPS_BYTES_GPU if torch_profile == "cu128" else _DEPS_BYTES_CPU
    if not plan.skip_frontend:
        total += _FRONTEND_BYTES
    if not plan.skip_ckpts:
        total += _PRESET_BYTES.get(plan.preset, _PRESET_BYTES["predict-core"])
    return total


def _print_plan_summary(plan: InstallPlan, detected: str) -> None:
    est = _estimate_plan_bytes(plan, detected)
    _print()
    _print(_t("Plan summary:", "安装计划："))
    if plan.clean_venv:
        _print(_t("  • remove ./.venv then reinstall deps", "  • 删除 ./.venv 后重装依赖"))
    elif plan.install_deps:
        _print(
            _t(
                f"  • Python deps → ./{plan.env} (torch={detected}"
                f"{', force' if plan.force_install_deps else ''})",
                f"  • Python 依赖 → ./{plan.env}（torch={detected}"
                f"{'，强制重装' if plan.force_install_deps else ''}）",
            )
        )
    else:
        _print(_t("  • reuse existing Python env", "  • 复用现有 Python 环境"))
    if plan.skip_frontend:
        _print(_t("  • frontend: skip", "  • 前端：跳过"))
    else:
        _print(
            _t(
                f"  • frontend: {'wipe + ' if plan.clean_frontend else ''}build WebUI v2",
                f"  • 前端：{'清理后' if plan.clean_frontend else ''}构建 WebUI v2",
            )
        )
    if plan.skip_ckpts:
        _print(_t("  • weights: on-demand later", "  • 权重：稍后按需下载"))
    else:
        _print(
            _t(
                f"  • weights: {plan.preset} (~{_fmt_bytes(_PRESET_BYTES.get(plan.preset, 0))})",
                f"  • 权重：{plan.preset}（约 {_fmt_bytes(_PRESET_BYTES.get(plan.preset, 0))}）",
            )
        )
    if plan.hf_mirror:
        _print("  • HF_ENDPOINT=https://hf-mirror.com")
    _print(
        _t(
            f"  • rough download/install budget: ~{_fmt_bytes(est)} · often 15–45 min",
            f"  • 粗估下载/安装量：约 {_fmt_bytes(est)} · 常见 15–45 分钟",
        )
    )
    _print(
        _t(
            "  • Tip: Ctrl+C is safe; re-run the same command to continue/reuse.",
            "  • 提示：可随时 Ctrl+C；再次运行同一命令可继续/复用。",
        )
    )


def _apply_env_action(plan: InstallPlan, action: str, survey: EnvSurvey) -> None:
    if action == "reuse":
        plan.clean_venv = False
        plan.force_install_deps = False
        # Only the target venv health matters when env=venv.
        plan.install_deps = not survey.venv_ready
        if survey.partial_venv:
            plan.force_install_deps = True
            plan.install_deps = True
        return
    if action == "reinstall_venv":
        plan.clean_venv = True
        plan.install_deps = True
        plan.force_install_deps = True
        plan.env = "venv"
        return
    if action == "force_deps":
        plan.clean_venv = False
        plan.install_deps = True
        plan.force_install_deps = True
        return
    if action == "wipe_all":
        plan.clean_venv = True
        plan.clean_frontend = True
        plan.install_deps = True
        plan.force_install_deps = True
        plan.force_frontend = True
        plan.env = "venv"


def build_interactive_plan(base: InstallPlan) -> InstallPlan:
    detected, reason = _detect_torch_type("auto")
    survey = EnvSurvey()
    _print()
    _print(_t("VenusFactory2 installer", "VenusFactory2 安装向导"))
    _print(f"repo     : {_REPO_ROOT}")
    _print(f"torch    : {detected} ({reason})")
    conda = _conda_active()
    if conda:
        _print(
            _t(
                f"WARN: conda env active ({conda}). Default install still uses ./.venv — "
                "activate it after install, or choose Custom → system/conda.",
                f"注意：当前已激活 conda（{conda}）。默认仍会装到 ./.venv——"
                "装完请 source .venv，或选「自定义」装到当前 conda。",
            )
        )
    _section(_t("Existing environment", "已有环境"))
    for line in survey.lines():
        _print(f"  • {line}")
    _print()
    _print(_t("Tip: press Enter to accept the default.", "提示：直接回车即接受默认/推荐项。"))

    # Single recommended default; broken/partial → default to fresh repair.
    if survey.venv_exists and (not survey.venv_ready):
        default_mode = "fresh"
        mode_options = [
            (
                "fresh",
                _t(
                    "Repair: remove .venv and reinstall [recommended]",
                    "修复：删除 .venv 并重装【推荐】",
                ),
            ),
            (
                "recommended",
                _t("Continue with all-in-one (may reuse)", "继续一键安装（可能复用）"),
            ),
            (
                "wipe",
                _t("Full wipe (.venv + frontend) then reinstall", "全量清理（.venv+前端）后重装"),
            ),
            ("custom", _t("Custom", "自定义")),
        ]
    elif survey.has_old_env:
        default_mode = "recommended"
        mode_options = [
            (
                "recommended",
                _t("Recommended all-in-one [recommended]", "推荐一键安装【推荐】"),
            ),
            (
                "fresh",
                _t("Remove .venv and reinstall cleanly", "删除 .venv 后干净重装"),
            ),
            (
                "wipe",
                _t("Full wipe (.venv + frontend) then reinstall", "全量清理（.venv+前端）后重装"),
            ),
            ("custom", _t("Custom", "自定义")),
        ]
    else:
        default_mode = "recommended"
        mode_options = [
            (
                "recommended",
                _t("Recommended all-in-one [recommended]", "推荐一键安装【推荐】"),
            ),
            ("custom", _t("Custom", "自定义")),
        ]

    mode = _prompt_choice(
        _t("Install mode:", "安装模式："),
        mode_options,
        default_key=default_mode,
    )

    plan = InstallPlan(
        install_deps=True,
        torch_type="auto",
        env="venv",
        preset="predict-core",
        skip_ckpts=False,
        skip_frontend=False,
        with_frpc=False,
        check_env=True,
        port=base.port,
        dry_run=base.dry_run,
        offer_launch=True,
    )

    if mode == "recommended":
        if survey.venv_ready:
            # Silent reuse — show in summary only.
            _apply_env_action(plan, "reuse", survey)
        elif survey.venv_exists:
            _apply_env_action(plan, "reinstall_venv", survey)
        else:
            plan.install_deps = True
    elif mode == "fresh":
        _apply_env_action(plan, "reinstall_venv", survey)
        if survey.frontend_dist or survey.frontend_node_modules:
            plan.clean_frontend = _prompt_yes_no(
                _t(
                    "Also remove frontend/dist (+ node_modules)?",
                    "是否同时删除 frontend/dist（及 node_modules）？",
                ),
                default=False,
            )
            plan.force_frontend = plan.clean_frontend
    elif mode == "wipe":
        _apply_env_action(plan, "wipe_all", survey)
    else:
        # Custom (simplified)
        plan = InstallPlan(port=base.port, dry_run=base.dry_run, offer_launch=True)
        if survey.venv_exists:
            env_action = _prompt_choice(
                _t("Python environment:", "Python 环境："),
                [
                    (
                        "reuse",
                        _t("Reuse .venv if healthy", "复用健康的 .venv")
                        if survey.venv_ready
                        else _t("Try reuse .venv", "尝试复用 .venv"),
                    ),
                    (
                        "reinstall_venv",
                        _t("Remove .venv and reinstall", "删除 .venv 并重装"),
                    ),
                    (
                        "force_deps",
                        _t("Keep .venv, force reinstall packages", "保留 .venv，强制重装包"),
                    ),
                    ("skip_deps", _t("Do not touch Python deps", "不改动 Python 依赖")),
                ],
                default_key="reinstall_venv" if not survey.venv_ready else "reuse",
            )
            if env_action == "skip_deps":
                plan.install_deps = False
            else:
                _apply_env_action(plan, env_action, survey)
        else:
            plan.install_deps = _prompt_yes_no(
                _t("Install Python deps? [recommended: yes]", "安装 Python 依赖？【推荐：是】"),
                default=True,
            )
        if plan.install_deps:
            plan.torch_type = _prompt_choice(
                _t("Torch profile:", "Torch 方案："),
                [
                    ("auto", _t(f"auto → {detected} [recommended]", f"自动 → {detected}【推荐】")),
                    ("cu128", "cu128 (NVIDIA CUDA 12.8)"),
                    ("cpu", _t("cpu (no NVIDIA / macOS / old driver)", "cpu（无 NVIDIA / macOS / 老驱动）")),
                ],
                default_key="auto",
            )
            if not plan.clean_venv:
                default_env = "system" if _conda_active() else "venv"
                plan.env = _prompt_choice(
                    _t("Install target:", "安装目标："),
                    [
                        ("venv", _t("./.venv via uv [recommended]", "./.venv（uv）【推荐】")),
                        ("system", _t("current interpreter (conda/system)", "当前解释器（conda/系统）")),
                    ],
                    default_key=default_env,
                )
        plan.skip_frontend = not _prompt_yes_no(
            _t("Build WebUI frontend? [recommended: yes]", "构建 WebUI 前端？【推荐：是】"),
            default=True,
        )
        preset = _prompt_choice(
            _t("Weights to download now:", "现在下载的权重："),
            [
                (
                    "predict-core",
                    _t(
                        "predict-core ~163MB — typical prediction [recommended]",
                        "predict-core ~163MB — 常用预测【推荐】",
                    ),
                ),
                ("demo", _t("demo ~0.4MB — smoke test", "demo ~0.4MB — 冒烟")),
                ("proteinmpnn", _t("proteinmpnn ~64MB — sequence design", "proteinmpnn ~64MB — 序列设计")),
                ("predict-all", _t("predict-all ~405MB", "predict-all ~405MB")),
                ("all", _t("all ~469MB", "all ~469MB")),
                ("skip", _t("skip — download on first use", "跳过 — 首次使用时再下")),
            ],
            default_key="predict-core",
        )
        if preset == "skip":
            plan.skip_ckpts = True
        else:
            plan.preset = preset
        plan.check_env = True

    # HF mirror (China / slow HF)
    if not plan.skip_ckpts and not os.getenv("HF_ENDPOINT"):
        plan.hf_mirror = _prompt_yes_no(
            _t(
                "Use Hugging Face mirror hf-mirror.com? (recommended in mainland China)",
                "是否使用 Hugging Face 镜像 hf-mirror.com？（中国大陆推荐）",
            ),
            default=False,
        )

    detected2, _ = _detect_torch_type(plan.torch_type)
    _print_plan_summary(plan, detected2)
    if not _prompt_yes_no(_t("Continue?", "继续安装？"), default=True):
        _print(_t("Aborted.", "已取消。"))
        raise SystemExit(130)
    return plan


def plan_from_args(args: argparse.Namespace) -> InstallPlan:
    install_deps = bool(
        args.install_deps or args.force_install_deps or args.yes or args.clean_venv
    )
    survey = EnvSurvey()
    force = bool(args.force_install_deps or args.clean_venv)
    clean_venv = bool(args.clean_venv)
    # -y auto-heal: broken venv (no torch) → wipe; partial (torch only) → force reinstall.
    if args.yes and args.env == "venv" and survey.venv_exists and not survey.venv_ready:
        install_deps = True
        force = True
        if not survey.venv_torch:
            clean_venv = True
    return InstallPlan(
        install_deps=install_deps,
        force_install_deps=force,
        clean_venv=clean_venv,
        clean_frontend=bool(args.clean_frontend),
        torch_type=args.torch_type,
        env=args.env,
        preset=args.preset,
        skip_ckpts=bool(args.skip_ckpts),
        skip_frontend=bool(args.skip_frontend),
        force_frontend=bool(args.force_frontend or args.clean_frontend),
        with_frpc=bool(args.with_frpc),
        check_env=bool(args.check_env or args.yes),
        hf_mirror=bool(args.hf_mirror),
        port=args.port,
        dry_run=bool(args.dry_run),
        offer_launch=(not args.yes) and _is_interactive(),
    )


def _early_prereqs(plan: InstallPlan, *, dry_run: bool) -> list[str]:
    """Checks that must pass before long downloads."""
    problems: list[str] = []
    py = sys.version_info
    _print(f"Python     : {sys.executable} ({py.major}.{py.minor}.{py.micro})")
    if plan.env == "system" and (py.major, py.minor) < (3, 12):
        problems.append(
            _t(
                f"system install needs Python >= 3.12 (found {py.major}.{py.minor}). "
                "Use default .venv install or upgrade Python.",
                f"装到当前解释器需要 Python >= 3.12（当前 {py.major}.{py.minor}）。"
                "请用默认 .venv，或升级 Python。",
            )
        )

    if not plan.skip_frontend:
        major = _node_major()
        npm = _which("npm")
        if major is None or not npm:
            problems.append(
                _t(
                    "Node.js 25.x + npm required for WebUI. Install: "
                    "https://nodejs.org/ or `nvm install 25`. "
                    "Or re-run with --skip-frontend.",
                    "构建 WebUI 需要 Node.js 25.x + npm。安装：https://nodejs.org/ 或 "
                    "`nvm install 25`。也可加 --skip-frontend 跳过前端。",
                )
            )
        else:
            npm_v = subprocess.run(["npm", "-v"], capture_output=True, text=True).stdout.strip()
            _print(f"Node/npm   : v{major}.x / {npm_v}")
            if major != 25:
                problems.append(
                    _t(
                        f"Node major {major} found; WebUI engines require Node 25.x. "
                        "Install Node 25 or use --skip-frontend.",
                        f"检测到 Node 主版本 {major}；WebUI 需要 Node 25.x。"
                        "请安装 Node 25，或使用 --skip-frontend。",
                    )
                )

    detected, _ = _detect_torch_type(plan.torch_type)
    need = _estimate_plan_bytes(plan, detected)
    free = _disk_free_bytes(_REPO_ROOT)
    if free is not None:
        _print(f"Disk free  : {_fmt_bytes(free)} (plan ~{_fmt_bytes(need)})")
        if free < need * 1.05 and not dry_run:
            problems.append(
                _t(
                    f"Not enough free disk (need ~{_fmt_bytes(need)}, have {_fmt_bytes(free)}). "
                    "Free space, or use --preset demo / --skip-ckpts.",
                    f"磁盘空间不足（约需 {_fmt_bytes(need)}，可用 {_fmt_bytes(free)}）。"
                    "请腾出空间，或改用 --preset demo / --skip-ckpts。",
                )
            )
    return problems


def install_deps(*, torch_type: str, env: str, dry_run: bool) -> int:
    _section(_t("Install Python deps", "安装 Python 依赖"))
    resolved, reason = _detect_torch_type(torch_type)
    _print(f"{_t('Detected', '检测结果')}: {resolved} ({reason})")
    return _run(
        [
            sys.executable,
            str(_REPO_ROOT / "install.py"),
            "--env",
            env,
            "--type",
            resolved,
            "--skip-frpc",
        ],
        dry_run=dry_run,
    )


def build_frontend(*, force: bool, dry_run: bool) -> int:
    _section(_t("Build frontend (WebUI v2)", "构建前端（WebUI v2）"))
    if (_DIST / "index.html").is_file() and not force:
        _print(_t(f"Already built: {_DIST}", f"已构建: {_DIST}"))
        return 0
    if not _FRONTEND.is_dir():
        _print(f"ERROR: missing {_FRONTEND}")
        return 1
    code = _run(["npm", "install"], cwd=_FRONTEND, dry_run=dry_run)
    if code != 0:
        return code
    return _run(["npm", "run", "build"], cwd=_FRONTEND, dry_run=dry_run)


def download_ckpts(python: str, preset: str, *, dry_run: bool) -> int:
    _section(_t(f"Download checkpoints ({preset})", f"下载权重（{preset}）"))
    return _run(
        [python, str(_REPO_ROOT / "scripts" / "download_ckpts.py"), "--preset", preset],
        dry_run=dry_run,
    )


def download_frpc(python: str, *, dry_run: bool) -> int:
    _section(_t("Download Gradio frpc", "下载 Gradio frpc"))
    return _run(
        [python, str(_REPO_ROOT / "scripts" / "download_frpc.py"), "--to-gradio-cache"],
        dry_run=dry_run,
    )


def run_check_env(python: str, *, dry_run: bool) -> int:
    _section(_t("Environment check", "环境检查"))
    return _run([python, str(_REPO_ROOT / "scripts" / "check_env.py")], dry_run=dry_run)


def print_next_steps(*, python: str, port: int, used_venv: bool) -> None:
    _section(_t("Ready — launch WebUI v2", "就绪 — 启动 WebUI v2"))
    launch_python = python
    if used_venv:
        vpy = _venv_python()
        launch_python = str(vpy) if vpy else _planned_venv_python()
        if os.name == "nt":
            _print("  .\\.venv\\Scripts\\activate")
        else:
            _print("  source .venv/bin/activate")
    _print(f"  {launch_python} src/webui_v2.py --host 0.0.0.0 --port {port}")
    _print()
    _print(f"{_t('Open', '浏览器打开')}: http://localhost:{port}")
    _print(
        _t(
            "First steps: Quick Tools (no LLM key) · Agent needs Settings API key.",
            "建议：先试「快速工具」（无需 LLM Key）；Agent 需在 Settings 配置密钥。",
        )
    )
    if _conda_active():
        _print(
            _t(
                f"WARN: conda still active — prefer the .venv python above, not bare `python`.",
                f"注意：conda 仍处于激活状态 — 请用上面的 .venv python，不要直接用裸 `python`。",
            )
        )
    _print("Docs: docs/wiki/Home.md")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-y", "--yes", action="store_true", help="Non-interactive recommended install.")
    p.add_argument("--non-interactive", action="store_true", help="Disable prompts.")
    p.add_argument(
        "--lang",
        choices=["en", "zh"],
        default="en",
        help="Installer UI language (default: en). Use zh only if you want Chinese prompts.",
    )
    p.add_argument("--install-deps", action="store_true")
    p.add_argument("--force-install-deps", action="store_true")
    p.add_argument("--clean-venv", action="store_true")
    p.add_argument("--clean-frontend", action="store_true")
    p.add_argument("--torch-type", choices=["auto", "cpu", "cu128"], default="auto")
    p.add_argument("--env", choices=["venv", "system"], default="venv")
    p.add_argument("--preset", default="predict-core")
    p.add_argument("--skip-ckpts", action="store_true")
    p.add_argument("--skip-frontend", action="store_true")
    p.add_argument("--force-frontend", action="store_true")
    p.add_argument("--with-frpc", action="store_true")
    p.add_argument("--check-env", action="store_true")
    p.add_argument("--strict-check", action="store_true", help="Fail if check_env fails.")
    p.add_argument("--hf-mirror", action="store_true", help="Set HF_ENDPOINT=hf-mirror.com")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-launch-prompt", action="store_true")
    return p


def _should_prompt(args: argparse.Namespace) -> bool:
    if args.yes or args.non_interactive:
        return False
    automation = any(
        [
            args.install_deps,
            args.force_install_deps,
            args.clean_venv,
            args.clean_frontend,
            args.skip_ckpts,
            args.skip_frontend,
            args.force_frontend,
            args.with_frpc,
            args.check_env,
            args.hf_mirror,
            args.torch_type != "auto",
            args.env != "venv",
            args.preset != "predict-core",
        ]
    )
    return (not automation) and _is_interactive()


def execute_plan(plan: InstallPlan, *, strict_check: bool = False) -> int:
    global _LANG  # noqa: PLW0603 — module UI language
    os.chdir(_REPO_ROOT)
    _print(_t("VenusFactory2 quickstart", "VenusFactory2 一键安装"))
    _print(f"repo: {_REPO_ROOT}")
    if plan.dry_run:
        _print(_t("(dry-run: no side effects)", "（dry-run：无实际副作用）"))

    if plan.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        _print("HF_ENDPOINT=https://hf-mirror.com")

    resolved, reason = _detect_torch_type(plan.torch_type)
    _section(_t("Scan platform", "扫描平台"))
    _print(f"OS         : {sys.platform}")
    _print(f"Torch plan : {resolved} ({reason})")
    survey = EnvSurvey()
    for line in survey.lines():
        _print(f"env        : {line}")
    if _conda_active():
        _print(
            _t(
                f"conda      : active ({_conda_active()})",
                f"conda      : 已激活（{_conda_active()}）",
            )
        )

    # Heal partial venv automatically when installing into venv.
    if plan.env == "venv" and survey.partial_venv and not plan.clean_venv:
        _print(
            _t(
                "Detected partial .venv (torch present, other deps missing) → force reinstall packages.",
                "检测到半残 .venv（有 torch，缺其他依赖）→ 将强制重装包。",
            )
        )
        plan.force_install_deps = True
        plan.install_deps = True

    _section(_t("Preflight checks", "安装前检查"))
    early = _early_prereqs(plan, dry_run=plan.dry_run)
    if early:
        for p in early:
            if plan.dry_run:
                _print(f"WARN: {p}")
            else:
                _print(f"ERROR: {p}")
        if not plan.dry_run:
            _print()
            _print(_t("Fix above, then re-run:", "请先修复以上问题，然后重跑："))
            _print("  python scripts/setup_quickstart.py")
            return 1

    if plan.clean_venv:
        if not remove_path(_VENV_DIR, dry_run=plan.dry_run, label="./.venv"):
            return 1
        plan.install_deps = True
        plan.force_install_deps = True
        plan.env = "venv"

    if plan.clean_frontend:
        if not remove_path(_DIST, dry_run=plan.dry_run, label="frontend/dist"):
            return 1
        if not remove_path(_FRONTEND / "node_modules", dry_run=plan.dry_run, label="frontend/node_modules"):
            return 1
        plan.force_frontend = True

    python = sys.executable
    used_venv = False
    existing = _venv_python()
    if plan.env == "venv" and existing is not None:
        python = str(existing)
        used_venv = True

    want_install = plan.install_deps or plan.force_install_deps
    ready, detail = _env_ready(python) if Path(python).exists() else (False, "missing")

    if want_install and (plan.force_install_deps or not ready):
        code = install_deps(torch_type=plan.torch_type, env=plan.env, dry_run=plan.dry_run)
        if code != 0:
            _print(
                _t(
                    "Dependency install failed. Re-run the same command after fixing network/disk. "
                    "For a clean slate: python scripts/setup_quickstart.py -y --clean-venv",
                    "依赖安装失败。修好网络/磁盘后重跑同一命令。"
                    "彻底重来：python scripts/setup_quickstart.py -y --clean-venv",
                )
            )
            return code
        if plan.env == "venv":
            vpy = _venv_python()
            if vpy is not None:
                python = str(vpy)
                used_venv = True
            elif plan.dry_run:
                used_venv = True
                python = _planned_venv_python()
                _print(f"(dry-run) {_t('would use', '将使用')}: {python}")
            else:
                _print(_t("ERROR: .venv python missing after install.", "错误：安装后找不到 .venv python。"))
                return 1
        if not plan.dry_run:
            ok, smoke_detail = _env_ready(python)
            if not ok:
                _print(
                    _t(
                        f"Post-install smoke failed ({smoke_detail}). Try --clean-venv.",
                        f"安装后冒烟失败（{smoke_detail}）。请尝试 --clean-venv。",
                    )
                )
                return 1
            _print(_t("Smoke OK: torch + torch_geometric + transformers", "冒烟通过：torch + PyG + transformers"))
    elif want_install and ready:
        _section(_t("Install Python deps", "安装 Python 依赖"))
        _print(_t(f"Env already ready via {python}; skip.", f"环境已就绪（{python}），跳过安装。"))

    if not plan.skip_frontend:
        code = build_frontend(force=plan.force_frontend, dry_run=plan.dry_run)
        if code != 0:
            _print(
                _t(
                    "Frontend build failed. Fix Node 25.x, or continue with --skip-frontend. "
                    "Python deps/weights already done are kept.",
                    "前端构建失败。请修好 Node 25.x，或加 --skip-frontend。"
                    "已完成的 Python 依赖/权重会保留。",
                )
            )
            return code
    else:
        _section(_t("Build frontend", "构建前端"))
        _print(_t("Skipped.", "已跳过。"))

    ckpt_ok = True
    if not plan.skip_ckpts:
        # Prefer venv python for download scripts after install.
        code = download_ckpts(python if Path(python).exists() else sys.executable, plan.preset, dry_run=plan.dry_run)
        if code != 0:
            ckpt_ok = False
            _print(
                _t(
                    "Checkpoint download failed (deps/frontend may already be OK). "
                    "Retry with: export HF_ENDPOINT=https://hf-mirror.com "
                    "&& python scripts/download_ckpts.py --preset "
                    f"{plan.preset}",
                    "权重下载失败（依赖/前端可能已装好）。可重试："
                    "export HF_ENDPOINT=https://hf-mirror.com && "
                    f"python scripts/download_ckpts.py --preset {plan.preset}",
                )
            )
    else:
        _section(_t("Download checkpoints", "下载权重"))
        _print(_t("Skipped (on-demand download still enabled by default).", "已跳过（默认仍可按需自动下载）。"))

    if plan.with_frpc:
        download_frpc(python if Path(python).exists() else sys.executable, dry_run=plan.dry_run)

    if plan.check_env and Path(python).exists():
        code = run_check_env(python, dry_run=plan.dry_run)
        if code != 0 and not plan.dry_run:
            _print(
                _t(
                    "check_env reported issues (WARN). You can still try launching the WebUI.",
                    "check_env 报告了问题（警告）。仍可尝试启动 WebUI。",
                )
            )
            if strict_check:
                return code

    print_next_steps(python=python, port=plan.port, used_venv=used_venv and plan.env == "venv")

    if (
        plan.offer_launch
        and not plan.dry_run
        and ckpt_ok
        and _is_interactive()
        and Path(python).exists()
    ):
        if _prompt_yes_no(
            _t("Launch WebUI now?", "现在启动 WebUI？"),
            default=True,
        ):
            return int(
                subprocess.run(
                    [
                        python,
                        str(_REPO_ROOT / "src" / "webui_v2.py"),
                        "--host",
                        "0.0.0.0",
                        "--port",
                        str(plan.port),
                    ]
                ).returncode
            )

    return 0 if ckpt_ok else 1


def main(argv: list[str] | None = None) -> int:
    global _LANG
    args = build_parser().parse_args(argv)
    _LANG = _resolve_lang(args.lang)
    os.chdir(_REPO_ROOT)

    if _should_prompt(args):
        try:
            plan = build_interactive_plan(
                InstallPlan(port=args.port, dry_run=args.dry_run, offer_launch=not args.no_launch_prompt)
            )
        except EOFError:
            _print(_t("No TTY input. Use -y or flags.", "无交互输入。请用 -y 或显式参数。"))
            return 2
    else:
        plan = plan_from_args(args)
        if args.yes and not args.install_deps and not args.force_install_deps:
            plan.install_deps = True
            plan.check_env = True
        if args.no_launch_prompt:
            plan.offer_launch = False
        if args.hf_mirror:
            plan.hf_mirror = True
        if _conda_active() and plan.env == "venv" and args.yes:
            _print(
                _t(
                    f"WARN: conda active ({_conda_active()}) while installing into ./.venv. "
                    "After install, use .venv/bin/python (see next steps).",
                    f"注意：conda 已激活（{_conda_active()}），但将安装到 ./.venv。"
                    "装完请使用 .venv/bin/python（见后续提示）。",
                )
            )

    return execute_plan(plan, strict_check=bool(args.strict_check))


if __name__ == "__main__":
    raise SystemExit(main())
