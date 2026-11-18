import compileall
import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    compiled = compileall.compile_dir(
        project_root / "src",
        quiet=1,
    )
    compiled = (
        compileall.compile_file(
            project_root / "main.py",
            quiet=1,
        )
        and compiled
    )
    if not compiled:
        return 1

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(project_root / "tests"),
            "-q",
        ],
        cwd=project_root,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
