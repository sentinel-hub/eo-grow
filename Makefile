# Makefile for creating a new release of the package and uploading it to PyPI

help:
	@echo "Use 'make upload' to upload the package to PyPI"

upload:
	rm -r dist | true
	python -m build --sdist --wheel
	twine upload --skip-existing dist/*

# For testing:
test-upload:
	rm -r dist | true
	python -m build --sdist --wheel
	twine upload --repository testpypi --skip-existing dist/*
