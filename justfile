# use PowerShell instead of sh:
set shell := ["powershell.exe", "-c"]

description:
  @echo 'This file is used to run all automated commands in annotile!'

format: black ruff pydoclint mypy

black:
    poetry run black annotile tests

ruff:
    poetry run ruff check annotile tests

pydoclint:
    poetry run pydoclint

mypy:
    poetry run mypy annotile tests

test:
    poetry run pytest