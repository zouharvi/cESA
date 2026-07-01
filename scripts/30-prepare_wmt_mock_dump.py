# %%

import json
import collections
import os
import statistics

os.chdir(os.path.dirname(__file__)+"/../")

data = []
model_scores = collections.defaultdict(list)
with open("humeval/collected/main_enja.json", "r") as f:
    data_raw = json.load(f)
    for campaign_name, campaign_data in data_raw.items():
        for doc in campaign_data:
            if doc["annotation"] == "__RESET__":
                continue
            k = len(doc["item"][0]["tgt"])
            for item in doc["item"]:
                item["item_id"] = item["item_id"].removesuffix("_#_dup0").removesuffix("_#_dup1")
            if k != 3:
                continue

            # store all data
            data.append(doc)

            # compute total model ranking
            for item, item_ann in zip(doc["item"], doc["annotation"]):
                for model, annotation in item_ann.items():
                    model = model.removesuffix("'")
                    model_scores[model].append(annotation["score"])

model_scores = {model: statistics.mean(scores) for model, scores in model_scores.items()}
# sort
model_scores = dict(sorted(model_scores.items(), key=lambda x: x[1], reverse=True))
model_rank = {model: rank for rank, (model, _) in enumerate(model_scores.items(), start=1)}

# %%

import copy
import random
import scipy.stats

def compute_model_distribution(data):
    model_scores = collections.defaultdict(list)
    for doc in data:
        for item, item_ann in zip(doc["item"], doc["annotation"]):
            for model, annotation in item_ann.items():
                model = model.removesuffix("'")
                model_scores[model].append(annotation["score"])
    model_scores = {model: len(scores) for model, scores in model_scores.items()}
    # sort by number of scored items
    model_scores = dict(sorted(model_scores.items(), key=lambda x: x[1], reverse=True))
    model_rank = {model: rank for rank, (model, _) in enumerate(model_scores.items(), start=1)}
    return model_rank

data_local = copy.deepcopy(data)

tau_prev = -1
for i in range(5_000):
    # randomly pop one item
    item = data_local.pop(random.randint(0, len(data_local) - 1))

    model_rank_new = compute_model_distribution(data_local)
    # kendalltau
    tau, _p_value = scipy.stats.kendalltau(
        list(model_rank.values()),
        [model_rank_new[model] for model in model_rank],
    )
    print(tau, len(data_local))

    if tau > tau_prev or random.random() < 0.01: # type: ignore
        # commit
        tau_prev = tau
    else:
        # revert transaction
        data_local.append(item)

# %%

with open("humeval/collected/mock_enja_v1.json", "w") as f:
    json.dump(data_local, f, indent=2, ensure_ascii=False)