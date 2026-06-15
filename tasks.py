#!/usr/bin/env python3

import argparse
import shutil
import subprocess
import sys


JUPYTER_PORT = 18888


def ensure_poetry():
    """Verify Poetry is installed and available on PATH."""
    if shutil.which("poetry") is None:
        print(
            "Error: Poetry is not installed or not available on your PATH.\n"
            "Install Poetry from https://python-poetry.org/docs/#installation",
            file=sys.stderr,
        )
        sys.exit(1)


def run(*cmd):
    subprocess.run(cmd, check=True)


def install():
    """Install the poetry environment."""
    print("Creating virtual environment using pyenv and poetry")
    run("poetry", "install")


def install_dev():
    """Install the poetry environment with dev dependencies and pre-commit hooks."""
    print("Creating virtual environment using pyenv and poetry")
    run("poetry", "install", "--with", "dev")

    print("Installing pre-commit hooks")
    run("poetry", "run", "pre-commit", "install")


def check():
    """Run code quality tools."""
    print(
        "Checking Poetry lock file consistency with 'pyproject.toml': Running poetry check --lock"
    )
    run("poetry", "check", "--lock")

    print("Linting code: Running pre-commit")
    run("poetry", "run", "pre-commit", "run", "-a")


def test():
    """Test the code with pytest."""
    print("Testing code: Running pytest")
    run("poetry", "run", "pytest")


def clean_build():
    """Clean build artifacts."""
    shutil.rmtree("dist", ignore_errors=True)


def build():
    """Build wheel file using poetry."""
    clean_build()

    print("Creating wheel file")
    run("poetry", "build")


def jupyter_kernel():
    """Install kernel for Jupyter."""
    print("Install Jupyter kernel")
    run(
        "poetry",
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
        "poetry",
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
