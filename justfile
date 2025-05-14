# use PowerShell instead of sh:
# set shell := ["powershell.exe", "-c"]

description:
  @echo 'This file is used to run all automated commands in annotile!'

format: ruff-check-fix ruff-format mypy

ruff-format:
    poetry run ruff format annotile tests

ruff-check:
    poetry run ruff check annotile tests

ruff-check-fix:
    poetry run ruff check annotile tests --fix

mypy:
    poetry run mypy annotile tests

test:
    poetry run pytest