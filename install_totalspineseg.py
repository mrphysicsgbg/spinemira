#!/usr/bin/env python3
"""Install TotalSpineSeg using uv, or venv and pip when uv is unavailable."""

import argparse
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "venv_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "tools/totalspineseg/.venv",
        help="Virtual environment directory (default: tools/totalspineseg/.venv beside this script)",
    )
    args = parser.parse_args()

    uv = shutil.which("uv")
    venv_dir = args.venv_dir.expanduser().resolve()
    if not (venv_dir / "pyvenv.cfg").is_file():
        if venv_dir.exists():
            parser.error(
                f"Destination already exists and is not a virtual environment: {venv_dir}"
            )
        if uv:
            subprocess.run([uv, "venv", "--python", "3.11", str(venv_dir)], check=True)
        else:
            python = (
                sys.executable
                if sys.version_info[:2] == (3, 11)
                else shutil.which("python3.11")
            )
            if python is None:
                parser.error(
                    "Install Python 3.11 or uv to create the TotalSpineSeg environment"
                )
            subprocess.run([python, "-m", "venv", str(venv_dir)], check=True)

    venv_python = str(venv_dir / "bin/python")
    if uv:
        install_command = [uv, "pip", "install", "--python", venv_python]
    else:
        print("uv is unavailable; using venv and pip.")
        subprocess.run([venv_python, "-m", "ensurepip", "--upgrade"], check=True)
        install_command = [venv_python, "-m", "pip", "install"]

    subprocess.run(
        install_command + ["totalspineseg[nnunetv2]>=20260623", "kornia<0.8"],
        check=True,
    )

    print(f"\nTotalSpineSeg installed in {venv_dir}")
    print(f"Segmentation command: {venv_dir / 'bin/totalspineseg'}")
    print("To add the command to PATH in this shell, run:")
    print(f'export PATH={shlex.quote(str(venv_dir / "bin"))}:"$PATH"')
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
