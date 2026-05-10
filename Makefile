PORT := 18888

.PHONY: install
install: ## Install the poetry environment
	@echo "Creating virtual environment using pyenv and poetry"
	@poetry install

.PHONE: install-dev
install-dev: ## Install the poetry environment with dev dependencies and pre-commit hooks
	@echo "Creating virtual environment using pyenv and poetry"
	@poetry install --with dev
	@echo "Installing pre-commit hooks"
	@poetry run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "Checking Poetry lock file consistency with 'pyproject.toml': Running poetry check --lock"
	@poetry check --lock
	@echo "Linting code: Running pre-commit"
	@poetry run pre-commit run -a

.PHONY: test
test: ## Test the code with pytest
	@echo "Testing code: Running pytest"
	@poetry run pytest

.PHONY: build
build: clean-build ## Build wheel file using poetry
	@echo "Creating wheel file"
	@poetry build

.PHONY: clean-build
clean-build: ## clean build artifacts
	@rm -rf dist

.PHONY: jupyter-kernel
jupyter-kernel: ## Install kernel for Jupyter
	@echo "Install Jupyter kernel"
	@poetry run python -m ipykernel install --user --name spinemira --display-name "Python (spinemira)"

.PHONY: jupyter
jupyter: ## Run Jupyter Lab
	@echo "Staring Jupyter Lab"
	@poetry run jupyter lab --no-browser --port=$(PORT)

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
