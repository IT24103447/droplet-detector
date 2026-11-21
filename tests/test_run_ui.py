"""The system Python must launch the dashboard with project dependencies."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_ui


class LauncherTests(unittest.TestCase):
    def test_system_python_uses_project_environment_and_working_directory(self):
        with tempfile.TemporaryDirectory(prefix="droplet project ") as directory:
            root = Path(directory)
            interpreter = root / ".venv/Scripts/python.exe"
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()
            (root / "app.py").touch()
            with patch.object(run_ui, "__file__", str(root / "scripts/run_ui.py")), \
                    patch.object(run_ui.subprocess, "run") as launch:
                run_ui.main()
            launch.assert_called_once()
            args, kwargs = launch.call_args
            self.assertEqual(Path(args[0][0]), interpreter)
            self.assertEqual(args[0][1:4], ["-m", "streamlit", "run"])
            self.assertEqual(kwargs["cwd"], root)
            self.assertEqual(Path(kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0]), root / "src")

    def test_falls_back_to_current_interpreter_without_project_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(run_ui.dashboard_python(Path(directory)), run_ui.sys.executable)


if __name__ == "__main__":
    unittest.main()
