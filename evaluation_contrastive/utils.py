import os
import json
import functools
import inspect
from dotenv import load_dotenv

# always cache from the top level in this package
ROOT = os.path.dirname(os.path.realpath(__file__)) + "/../"
load_dotenv(ROOT + ".env")


def cache(func=None, *, cache_dir=None):
    """
    Decorator that caches the result of a function call to a file.

    Arguments in the wrapper:
        cache (str | None): The cache filename to use. Defaults to "default".
                           If None or False, caching is disabled.
    """
    if func is None:
        return functools.partial(cache, cache_dir=cache_dir)

    @functools.wraps(func)
    def wrapper(*args, cache="default", **kwargs):
        if len(args) > 0:
            params = list(inspect.signature(func).parameters.keys())
            if not (len(args) == 1 and params and params[0] in ("self", "cls")):
                raise ValueError(
                    "Only keyword arguments are supported for cached functions"
                )

        # If cache is explicitly None, disable caching
        if not cache:
            return func(*args, **kwargs)

        # Ensure cache directory exists
        os.makedirs(ROOT + "cache", exist_ok=True)
        if cache_dir is None:
            cache_file = ROOT + f"cache/{cache}.json"
        else:
            os.makedirs(ROOT + f"cache/{cache_dir}", exist_ok=True)
            cache_file = ROOT + f"cache/{cache_dir}/{cache}.json"

        key = ", ".join([f"{k}={v}" for k, v in kwargs.items()])

        # Try to load from cache
        data = {}
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    pass
            if key in data:
                return data[key]

        # Call the actual function
        result = func(*args, **kwargs)

        # Save to cache
        data[key] = result
        with open(cache_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return result

    return wrapper


def load_data(
    human_scores_only=True,
    require_human_scores=True,
) -> dict[str, list[dict]]:
    # piggy-back on top of evaluation_bandit's wrapper around subset2evaluate
    import evaluation_bandit.utils

    data_all = evaluation_bandit.utils.load_data(
        human_scores_only=human_scores_only,
        require_human_scores=require_human_scores,
        wmt_years={"wmt25"},
    )
    return {k.removeprefix("wmt25_"): v for k, v in data_all.items()}
