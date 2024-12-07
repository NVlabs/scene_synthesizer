# Development

## Run Tests

From the root directory of the repo, run the following:
```
python -m pytest
```

## Generate Documentation

```
make docs
xdg-open docs/_out/html/index.html
```

## Code Formatting

* Formatter: [Black](https://github.com/psf/black), Version `23.7.0`
* Linter: [Flake8](https://github.com/PyCQA/flake8)<br>
  Use following `settings.json` in vscode (from [here](https://dev.to/ldsands/the-best-linter-for-black-in-vs-code-54a0)):
  ```
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": [
        "--ignore=E203",
        "--ignore=E266",
        "--ignore=E501",
        "--ignore=W503",
        "--max-line-length=88",
        "--select = B,C,E,F,W,T4,B9",
        "--max-complexity = 18"
    ],
  ```
* Docstrings for keyword arguments and variable length argument lists, see suggestion: https://stackoverflow.com/a/32273299

## IPython Console Debugging

```
%config Application.log_level='DEBUG'
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('scene_synthesizer').setLevel(logging.DEBUG)
```
