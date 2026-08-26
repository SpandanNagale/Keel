"""Repo detection: inspect the working directory before asking anything."""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from keel.models import DetectionResult

_DEP_NAME_RE = re.compile(r"[<>=\[; ]")

_WEB_FRAMEWORKS = {
    "fastapi", "flask", "django", "express", "koa", "nestjs", "@nestjs/core",
    "hapi", "restify", "aiohttp", "starlette", "sanic", "gin", "echo",
}
_CLI_FRAMEWORKS = {"typer", "click", "argparse", "commander", "yargs", "cobra", "clap"}
_DATA_LIBS = {
    "pandas", "airflow", "apache-airflow", "dagster", "prefect", "polars",
    "pyspark", "dbt", "numpy", "beautifulsoup4", "scrapy",
}


def _dep_name(raw: str) -> str:
    return _DEP_NAME_RE.split(raw.strip(), 1)[0].strip().strip('"').strip("'")


def detect_project(root: Path) -> DetectionResult:
    result = DetectionResult()

    pkg_json = root / "package.json"
    pyproject = root / "pyproject.toml"
    requirements = root / "requirements.txt"
    cargo = root / "Cargo.toml"
    gomod = root / "go.mod"
    git_config = root / ".git" / "config"

    if pkg_json.exists():
        result.project_type = "node"
        result.language = "javascript"
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8"))
            deps = list(data.get("dependencies", {}).keys())
            deps += list(data.get("devDependencies", {}).keys())
            result.dependencies = deps
            result.scripts = {k: v for k, v in data.get("scripts", {}).items()}
            if "typescript" in deps or (root / "tsconfig.json").exists():
                result.language = "typescript"
        except (json.JSONDecodeError, OSError):
            pass
    elif pyproject.exists():
        result.project_type = "python"
        result.language = "python"
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            deps = data.get("project", {}).get("dependencies", [])
            result.dependencies = [_dep_name(d) for d in deps if _dep_name(d)]
        except (tomllib.TOMLDecodeError, OSError):
            pass
    elif requirements.exists():
        result.project_type = "python"
        result.language = "python"
        try:
            lines = requirements.read_text(encoding="utf-8").splitlines()
            result.dependencies = [
                _dep_name(line) for line in lines
                if line.strip() and not line.strip().startswith("#") and _dep_name(line)
            ]
        except OSError:
            pass
    elif cargo.exists():
        result.project_type = "rust"
        result.language = "rust"
    elif gomod.exists():
        result.project_type = "go"
        result.language = "go"

    if git_config.exists():
        try:
            content = git_config.read_text(encoding="utf-8")
            match = re.search(r"url\s*=\s*.*?([^/\\\s]+?)(\.git)?\s*$", content, re.MULTILINE)
            if match:
                result.repo_name = match.group(1)
        except OSError:
            pass

    result.has_tests = (root / "tests").is_dir() or (root / "test").is_dir()
    result.has_dockerfile = (root / "Dockerfile").exists()
    result.has_env_example = (root / ".env.example").exists()

    result.summary = _build_summary(result)
    return result


def _build_summary(result: DetectionResult) -> str:
    if not result.language:
        return ""

    parts = [f"a {result.language} project"]
    if result.dependencies:
        parts.append(f"using {', '.join(result.dependencies[:5])}")
    if result.has_tests:
        parts.append("with a tests/ directory")
    if result.has_dockerfile:
        parts.append("with a Dockerfile")
    return " ".join(parts)


def suggest_template(detection: DetectionResult, prompt: str) -> str:
    deps_lower = {d.lower() for d in detection.dependencies}
    prompt_lower = prompt.lower()

    if deps_lower & _WEB_FRAMEWORKS or any(
        w in prompt_lower for w in ("api", "endpoint", "webhook", "rest service", " server")
    ):
        return "web-api"
    if deps_lower & _DATA_LIBS or any(
        w in prompt_lower for w in ("pipeline", "etl", "scrape", "cluster", "ingest", "batch", "crawl")
    ):
        return "data-pipeline"
    if deps_lower & _CLI_FRAMEWORKS or any(
        w in prompt_lower for w in ("cli", "command-line", "command line", "script that", "tool that renames")
    ):
        return "cli"
    return "default"
