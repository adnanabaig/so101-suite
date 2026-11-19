.PHONY: test shellcheck

PYTHON ?= python3

test: shellcheck
	$(PYTHON) -m unittest discover -s tests -v

shellcheck:
	bash -n collect-host-facts.sh scripts/build-container.sh scripts/container-image.sh \
		scripts/probe-container.sh scripts/run-container.sh
