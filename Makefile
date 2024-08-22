# === Constants ===

# the source directories which are being checked
SRC_DIRS = ./src ./tests

# === Package and Dependencies ===

# Upgrading pip, setuptools and wheel
.PHONY: upgrade-pip
upgrade-pip:
	@echo Upgrading pip, setuptools, and wheel ...
	python -m pip install --upgrade pip setuptools wheel

# Installing the required dependencies and building the package
.PHONY: install
install: upgrade-pip
	@echo Installing the required dependencies and building the package ...
	python -m pip install --upgrade .

.PHONY: install-dev
install-dev: upgrade-pip
	@echo Installing the required dependencies and building the package for development ...
	python -m pip install --upgrade .["dev"]

.PHONY: install-ci
install-ci: upgrade-pip
	@echo Installing the required dependencies for CI ...
	python -m pip install --upgrade .["git_ci"]

# === Source File Checks ===

# black format checking
.PHONY: black-check
black-check:
	@echo Checking code formatting with 'black' ...
	black --check --diff --color $(SRC_DIRS)

# isort import checking
.PHONY: isort-check
isort-check:
	@echo Checking import sorting with 'isort' ...
	isort --check --diff --color $(SRC_DIRS)

# pyright static type checking
.PHONY: pyright-check
pyright-check:
	@echo Checking types statically with 'pyright' ...
	pyright $(SRC_DIRS)

# mypy static type checking
.PHONY: mypy-check
mypy-check:
	@echo Checking types statically with 'mypy' ...
	mypy $(SRC_DIRS)

# pycodestyle style checking
.PHONY: pycodestyle-check
pycodestyle-check:
	@echo Checking code style with 'pycodestyle' ...
	pycodestyle $(SRC_DIRS) --max-line-length=88 --ignore=E203,W503,E704

# ruff lint checking
.PHONY: ruff-check
ruff-check:
	@echo Checking code style with 'ruff' ...
	ruff check $(SRC_DIRS)

# All checks combined
.PHONY: check
check: black-check isort-check pyright-check mypy-check pycodestyle-check ruff-check

# === Test Commands ===

# Running a selected test (serial)
.PHONY: test
test:
	@echo Running specific test with pytest ...
	pytest -k "$(TEST)" -x

# Running a selected test (parallel)
.PHONY: test-parallel
test-parallel:
	@echo Running specific test with pytest in parallel ...
	pytest -k "$(TEST)" -n="auto" -x

# Running the tests
.PHONY: test-htmlcov
test-htmlcov:
	@echo Running the tests with HTML coverage report ...
	pytest --cov=pyftir ./tests -n="auto" --cov-report=html -x --no-jit

.PHONY: test-xmlcov
test-xmlcov:
	@echo Running the tests with XML coverage report ...
	pytest --cov=pyftir ./tests -n="auto" --cov-report=xml -x --no-jit
