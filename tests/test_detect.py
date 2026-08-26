from pathlib import Path

from keel.detect import detect_project, suggest_template
from keel.models import DetectionResult


def test_detects_python_project_from_pyproject(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["httpx>=0.27", "pydantic"]\n',
        encoding="utf-8",
    )
    result = detect_project(tmp_path)
    assert result.project_type == "python"
    assert result.language == "python"
    assert "httpx" in result.dependencies
    assert "pydantic" in result.dependencies
    assert "python" in result.summary


def test_detects_node_project_from_package_json(tmp_path: Path):
    (tmp_path / "package.json").write_text(
        '{"dependencies": {"express": "^4.0.0"}, "devDependencies": {"typescript": "^5.0.0"}, "scripts": {"start": "node index.js"}}',
        encoding="utf-8",
    )
    result = detect_project(tmp_path)
    assert result.project_type == "node"
    assert result.language == "typescript"
    assert "express" in result.dependencies
    assert result.scripts["start"] == "node index.js"


def test_empty_directory_detects_nothing(tmp_path: Path):
    result = detect_project(tmp_path)
    assert result.project_type is None
    assert result.summary == ""


def test_has_tests_and_dockerfile_flags(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "Dockerfile").touch()
    result = detect_project(tmp_path)
    assert result.has_tests is True
    assert result.has_dockerfile is True


def test_suggest_template_web_api():
    assert suggest_template(DetectionResult(), "build a REST API for a todo list") == "web-api"


def test_suggest_template_data_pipeline():
    assert suggest_template(DetectionResult(), "scrape prices and cluster them daily") == "data-pipeline"


def test_suggest_template_cli():
    assert suggest_template(DetectionResult(), "a command-line tool that renames photos") == "cli"


def test_suggest_template_default_fallback():
    assert suggest_template(DetectionResult(), "help me organize my notes") == "default"


def test_suggest_template_from_detected_dependency():
    detection = DetectionResult(dependencies=["fastapi", "uvicorn"])
    assert suggest_template(detection, "build something") == "web-api"
