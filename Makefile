.PHONY: setup-sim check-sim setup-h12 check-h12 test run-planner figures

UV_ENV ?= .venv312
H12_UV_ENV ?= .venv312-h12
PYTHON ?= python3

setup-sim:
	tools/setup_uv_env.sh --profile sim-hand --python 3.12 --env $(UV_ENV)

check-sim:
	UV_PROJECT_ENVIRONMENT=$(UV_ENV) uv run $(PYTHON) tools/check_profile_imports.py --profile sim-hand

setup-h12:
	tools/setup_uv_env.sh --profile sim-h12 --python 3.12 --env $(H12_UV_ENV)

check-h12:
	UV_PROJECT_ENVIRONMENT=$(H12_UV_ENV) uv run $(PYTHON) tools/check_profile_imports.py --profile sim-h12

test:
	UV_PROJECT_ENVIRONMENT=$(UV_ENV) uv run --extra test pytest -q

run-planner:
	UV_PROJECT_ENVIRONMENT=$(UV_ENV) uv run $(PYTHON) -m rh56_controller.grasp_viz

figures:
	UV_PROJECT_ENVIRONMENT=$(UV_ENV) uv run $(PYTHON) tools/generate_paper_figures.py
