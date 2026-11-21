#!/usr/bin/env python3
"""Launch the Droplet Detector Streamlit Web UI.

Usage:
    python scripts/run_ui.py
"""
import os
import sys
import subprocess
from pathlib import Path


def dashboard_python(repo_root: Path) -> str:
    """Prefer the project's installed dependencies over the system Python."""
    candidates = (
        repo_root / ".venv" / "Scripts" / "python.exe",
        repo_root / ".venv" / "bin" / "python",
    )
    return next((str(path) for path in candidates if path.is_file()), sys.executable)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    app_path = repo_root / "app.py"
    if not app_path.exists():
        print(f"Error: app.py not found at {app_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        dashboard_python(repo_root),
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
    ]
    environment = os.environ.copy()
    source = str(repo_root / "src")
    prior_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source + os.pathsep + prior_path if prior_path else source
    print(f"Starting Droplet Detector Dashboard: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True, cwd=repo_root, env=environment)
    except KeyboardInterrupt:
        print("\nStopping Dashboard.")


if __name__ == "__main__":
    main()
