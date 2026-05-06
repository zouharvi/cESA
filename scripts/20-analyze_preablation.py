# %%

import json

with open("../humeval/collected/preablation.json", "r") as f:
    data_raw = json.load(f)
    data = []
    data_cESA = []
    data_ESA = []
    for campaign_name, campaign_data in data_raw.items():
        protocol = "cESA" if "cESA" in campaign_name else "ESA"
        for item in campaign_data:
            item["protocol"] = protocol

        data += campaign_data
        if protocol == "cESA":
            data_cESA += campaign_data
        else:
            data_ESA += campaign_data

# %%
import statistics
import collections


def compute_times(actions):
    times = sorted([x["time"] for x in actions])
    deltas = [(times[i] - times[i - 1]) for i in range(1, len(times))]
    return sum([min(60, delta) for delta in deltas])


def analyze_protocol(data):
    data = [
        doc
        for doc in data
        if "_#_tutorial_#_" not in doc["item"][0]["item_id"]
        and "_#_attention-check_#_" not in doc["item"][0]["item_id"]
    ]
    print("Count:", len(data))
    times = [compute_times(doc["actions"]) for doc in data]
    print("Time (avg/doc):", f"{statistics.mean(times) / 60:.1f}", "minutes")
    words = [sum([len(seg["src_text"].split()) for seg in doc["item"]]) for doc in data]
    print("Time (avg/word):", f"{sum(times) / sum(words):.3f}", "s")

    model_scores = collections.defaultdict(list)
    model_errors = collections.defaultdict(list)
    for doc in data:
        for item in doc["annotation"]:
            for model, annotation in item.items():
                model_scores[model].append(annotation["score"])
                model_errors[model].append(len(annotation["error_spans"]))
    model_scores = {k: statistics.mean(v) for k, v in model_scores.items()}
    model_errors = {k: statistics.mean(v) for k, v in model_errors.items()}
    print("Average score by model:", model_scores)
    print("Overall average score:", f"{statistics.mean(model_scores.values()):.2f}")
    print("Overall number of errors:", f"{statistics.mean(model_errors.values()):.2f}")

    # TODO: inter- and intra-annotator agreement


print("ESA")
analyze_protocol(data)
