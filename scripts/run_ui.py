#!/usr/bin/env python3
"""Launch the Droplet Detector Streamlit Web UI.

Usage:
    python scripts/run_ui.py
"""
import sys
import subprocess
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    app_path = repo_root / "app.py"
    if not app_path.exists():
        print(f"Error: app.py not found at {app_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
    ]
    print(f"Starting Droplet Detector Dashboard: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopping Dashboard.")


if __name__ == "__main__":
    main()
