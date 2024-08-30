# Annotile
<hr>

Tile and restitch images and labels for computer vision models.

# Getting Started
<hr>

## Installation
<hr>

This is a WIP, not all steps are covered.

1. Install [Just](https://github.com/casey/just) (currently set for windows).
2. Install `poetry` using `pipx`.
3. Install dependencies using poetry.

```
cd Annotile
poetry install 
```

## Dev tools
<hr>

Just is used for running formatting and testing jobs.

Run `just --list` for a list of all relevant jobs. 

Formatting command:
```
just format
```

Testing command:
```
just test
```