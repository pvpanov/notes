.PHONY: py-fmt py-lint dev-setup help

# Default target
help:
	@echo "Available targets:"
	@echo "  py-fmt      - Format Python code using ruff"
	@echo "  py-lint      - Lint Python code using ruff"
	@echo "  dev-setup    - Install pip 25.1.1 and requirements"
	@echo "  help         - Show this help message"

# Format Python code
py-fmt:
	ruff format notes/

# Lint Python code
py-lint:
	ruff check notes/

# Development setup
dev-setup:
	pip install --upgrade pip==25.1.1
	pip install -r requirements.txt 