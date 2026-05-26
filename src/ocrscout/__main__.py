"""Allow ``python -m ocrscout`` to invoke the CLI.

Mostly a convenience for environments where the ``ocrscout`` console
script isn't on ``PATH`` (e.g. a checkout that hasn't been ``pip install
-e``'d). The shipped console script is preferred — see ``[project.scripts]``
in pyproject.toml.
"""

from ocrscout.cli import main

if __name__ == "__main__":
    main()
