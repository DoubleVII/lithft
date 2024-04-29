
from typing import Any

def CLI(*args: Any, **kwargs: Any) -> Any:
    from jsonargparse import CLI

    kwargs.setdefault("as_positional", False)

    return CLI(*args, **kwargs)