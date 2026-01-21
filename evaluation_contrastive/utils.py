import os
import json
import functools
import inspect
from dotenv import load_dotenv

# always cache from the top level in this package
ROOT = os.path.dirname(os.path.realpath(__file__)) + "/../"
load_dotenv(ROOT + ".env")


def cache_read(cache_dir=None, cache="default", **kwargs):
    if cache is None:
        return None

    if cache_dir is None:
        cache_file = ROOT + f"cache/{cache}.json"
    else:
        cache_file = ROOT + f"cache/{cache_dir}/{cache}.json"

    key = ", ".join([f"{k}={v}" for k, v in kwargs.items()])

    # Try to load from cache
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            data = json.load(f)
        if key in data:
            return data[key]
    else:
        return None


def cache_write(result, cache_dir=None, cache="default", **kwargs):
    if cache is None:
        return None

    key = ", ".join([f"{k}={v}" for k, v in kwargs.items()])

    os.makedirs(ROOT + "cache", exist_ok=True)
    if cache_dir is None:
        cache_file = ROOT + f"cache/{cache}.json"
    else:
        os.makedirs(ROOT + f"cache/{cache_dir}", exist_ok=True)
        cache_file = ROOT + f"cache/{cache_dir}/{cache}.json"

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[key] = result
    with open(cache_file, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return result


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
