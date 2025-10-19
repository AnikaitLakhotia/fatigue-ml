# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11]
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Poetry
        run: |
          python -m pip install --upgrade pip
          pip install poetry

      - name: Install dependencies
        # use -n to avoid creating virtualenv inside CI runner; keeps environment isolated in job
        run: |
          poetry config virtualenvs.create false || true
          poetry install --no-interaction --no-ansi

      - name: Run tests
        run: |
          poetry run pytest -q

      - name: Upload test results (if any)
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: pytest-report
          path: tests
