.PHONY: build clean run-example

PACKAGE_NAME = tinykern
VERSION = 0.1.0
WHEEL = dist/$(PACKAGE_NAME)-$(VERSION)-cp311-cp311-linux_x86_64.whl
EXAMPLE = examples/basic.py
TINYKERN_MAIN = tinykern/main.py

build:
	uv build --wheel
	uv pip install $(WHEEL)

clean:
	uv pip uninstall $(PACKAGE_NAME)

run-example:
	uv run $(EXAMPLE)

jit-run:
	uv run $(TINYKERN_MAIN)