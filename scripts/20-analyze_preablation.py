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
import matplotlib.pyplot as plt
import os
import numpy as np
import evaluation_contrastive.utils_fig
import scipy.stats
import random

os.makedirs("computed/img/", exist_ok=True)


def compute_times(actions):
    times = sorted([x["time"] for x in actions])
    deltas = [(times[i] - times[i - 1]) for i in range(1, len(times))]
    return sum([delta if delta < 3 * 60 else 30 for delta in deltas if delta])


results = {}


def get_model_ranking(model_scores):
    return {k: statistics.mean(v.values()) for k, v in model_scores.items()}


def compute_kendalltau(model_scores1, model_scores2):
    return scipy.stats.kendalltau(
        list(model_scores1.values()),
        [model_scores2[model] for model in model_scores1],
    ).correlation


def analyze_protocol(data, key):
    data = [
        doc
        for doc in data
        if "_#_tutorial_#_" not in doc["item"][0]["item_id"]
        and "_#_attention-check_#_" not in doc["item"][0]["item_id"]
    ]
    segments = sum([len(doc["item"]) for doc in data])
    print("Count:", len(data))
    times = [compute_times(doc["actions"]) for doc in data]
    print("Time (avg/doc):", f"{statistics.mean(times) / 60:.1f}", "minutes")
    print("Time (avg/seg):", f"{statistics.mean(times) / segments / 60:.1f}", "minutes")
    words = [sum([len(seg["src_text"].split()) for seg in doc["item"]]) for doc in data]
    print("Time (avg/word):", f"{sum(times) / sum(words):.3f}", "s")

    model_scores = collections.defaultdict(dict)
    model_errors = collections.defaultdict(dict)
    duplicate_annotations = []
    for doc in data:
        for item, item_ann in zip(doc["item"], doc["annotation"]):
            item_id = item["item_id"]
            for model, annotation in item_ann.items():
                if item_id in model_scores[model]:
                    duplicate_annotations.append(
                        [
                            model_scores[model][item_id],
                            annotation["score"],
                        ]
                    )
                    model_scores[model][item_id] = (
                        annotation["score"] + model_scores[model][item_id]
                    ) / 2
                else:
                    model_scores[model][item_id] = annotation["score"]
                model_errors[model][item_id] = len(annotation["error_spans"])
    model_scores_avg = {k: statistics.mean(v.values()) for k, v in model_scores.items()}
    model_errors_avg = {k: statistics.mean(v.values()) for k, v in model_errors.items()}
    print("Average score by model:", model_scores_avg)
    print("Overall average score:", f"{statistics.mean(model_scores_avg.values()):.2f}")
    print("Overall errors:", f"{statistics.mean(model_errors_avg.values()):.2f}")

    # TODO: inter- and intra-annotator agreement

    # compute intra-annotator agreement
    intra_aa_mse = statistics.mean([abs(a - b) for a, b in duplicate_annotations])

    # compute stability
    model_scores_avg = {k: statistics.mean(v.values()) for k, v in model_scores.items()}
    item_ids = list(
        {
            item_id
            for model_scores in model_scores.values()
            for item_id in model_scores.keys()
        }
    )
    stability_taus = []
    for subset_p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for _ in range(100):
            item_ids_local = random.sample(item_ids, int(len(item_ids) * subset_p))
            model_scores_local = {
                model: {
                    item_id: v[item_id] for item_id in item_ids_local if item_id in v
                }
                for model, v in model_scores.items()
            }
            model_scores_avg_local = {
                model: statistics.mean(v.values()) if v.values() else 0
                for model, v in model_scores_local.items()
            }
            stability_taus.append(
                compute_kendalltau(model_scores_avg, model_scores_avg_local)
            )

    # plot histogram of model scores
    scores_all = [x for ll in model_scores.values() for x in ll.values()]
    plt.figure(figsize=(2.5, 1))
    plt.hist(
        scores_all,
        bins=np.linspace(0, 100, 21),
        color="black",
    )
    plt.axis("off")
    plt.text(-5, 0, "0", fontsize=18)
    plt.text(100, 0, "100", fontsize=18)
    plt.ylim(0, len(scores_all) / 5)
    plt.tight_layout()
    plt.savefig(f"computed/img/{key}_scores.svg", transparent=True)

    results[key] = {
        "time": f"{sum(times) / segments:.1f}s",
        "inter-aa": "TODO",
        "intra-aa": f"{intra_aa_mse:.0f}",
        "stability": f"{statistics.mean(stability_taus):.3f}",
    }


print("ESA")
analyze_protocol(data_ESA, "small-esa")

with open("computed/analysis.json", "w") as f:
    json.dump(results, f, indent=2)
