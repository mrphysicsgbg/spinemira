#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys


JUPYTER_PORT = 18888


def ensure_uv():
    """Verify uv is installed and available on PATH."""
    if shutil.which("uv") is None:
        print(
            "Error: uv is not installed or not available on your PATH.\n",
            file=sys.stderr,
        )
        sys.exit(1)


def run(*cmd):
    ensure_uv()
    subprocess.run(cmd, check=True)


def install():
    """Install the uv environment."""
    print("Creating virtual environment using uv")
    run("uv", "sync")


def install_dev():
    """Install the uv environment with dev dependencies and pre-commit hooks."""
    print("Creating virtual environment using uv")
    run("uv", "sync", "--group", "dev")

    print("Installing pre-commit hooks")
    run("uv", "run", "pre-commit", "install")


def check():
    """Run code quality tools."""
    print("Checking uv lock file consistency with 'pyproject.toml'")
    run("uv", "lock", "--check")

    print("Linting code: Running pre-commit")
    run("uv", "run", "pre-commit", "run", "-a")


def test():
    """Test the code with pytest."""
    print("Testing code: Running pytest")
    run("uv", "run", "pytest")


def clean_build():
    """Clean build artifacts."""
    shutil.rmtree("dist", ignore_errors=True)


def build():
    """Build wheel file using uv."""
    clean_build()

    print("Creating wheel file")
    run("uv", "build")


def jupyter_kernel():
    """Install kernel for Jupyter."""
    print("Install Jupyter kernel")
    run(
        "uv",
        "run",
        "python",
        "-m",
        "ipykernel",
        "install",
        "--user",
        "--name",
        "spinemira",
        "--display-name",
        "Python (spinemira)",
    )


def jupyter():
    """Run Jupyter Lab."""
    print("Starting Jupyter Lab")
    run(
        "uv",
        "run",
        "jupyter",
        "lab",
        "--no-browser",
        f"--port={JUPYTER_PORT}",
    )


COMMANDS = {
    "install": install,
    "install-dev": install_dev,
    "check": check,
    "test": test,
    "build": build,
    "clean-build": clean_build,
    "jupyter-kernel": jupyter_kernel,
    "jupyter": jupyter,
}


def main():
    parser = argparse.ArgumentParser(description="Project task runner")
    parser.add_argument(
        "command",
        nargs="?",
        choices=COMMANDS.keys(),
        help="Command to run",
    )

    args = parser.parse_args()

    if args.command is None:
        print("Available commands:\n")
        for name, func in COMMANDS.items():
            print(f"{name:<20} {func.__doc__}")
        return

    COMMANDS[args.command]()


if __name__ == "__main__":
    main()
