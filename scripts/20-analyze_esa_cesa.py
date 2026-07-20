# %%

import json
import collections
import os

os.chdir(os.path.dirname(__file__)+"/../")

small_enja = collections.defaultdict(list)
with open("humeval/annotations/preablation_enja.json", "r") as f:
    data_raw = json.load(f)
    for campaign_name, campaign_data in data_raw.items():
        protocol = "cESA" if "cESA" in campaign_name else "ESA"
        for item in campaign_data:
            item["protocol"] = f"small-enja-{protocol}"

        small_enja[protocol] += campaign_data

big_enja_cESA_k = collections.defaultdict(list)
with open("humeval/annotations/main_enja.json", "r") as f:
    data_raw = json.load(f)
    for campaign_name, campaign_data in data_raw.items():
        for item in campaign_data:
            if item["annotation"] == "__RESET__":
                continue
            k = len(item["item"][0]["tgt"])
            protocol = f"big-enja-cESA{k}"
            item["protocol"] = protocol
            big_enja_cESA_k[k].append(item)


big_enit_cESA_k = collections.defaultdict(list)
with open("humeval/annotations/main_enit.json", "r") as f:
    data_raw = json.load(f)
    for campaign_name, campaign_data in data_raw.items():
        for item in campaign_data:
            if item["annotation"] == "__RESET__":
                continue
            k = len(item["item"][0]["tgt"])
            protocol = f"big-enit-cESA{k}"
            item["protocol"] = protocol
            big_enit_cESA_k[k].append(item)

# %%
import statistics
import matplotlib.pyplot as plt
import evaluation_contrastive.utils_fig
import os
import numpy as np
import scipy.stats
import random
import itertools
import krippendorff # type: ignore

os.makedirs("computed/img/", exist_ok=True)
results = {}

def compute_times(actions):
    times = sorted([x["time"] for x in actions])
    deltas = [(times[i] - times[i - 1]) for i in range(1, len(times))]
    return sum([delta if delta < 3 * 60 else 3 * 60 for delta in deltas if delta])

def get_model_ranking(model_scores):
    return {k: statistics.mean(v.values()) for k, v in model_scores.items()}

def compute_kendalltau(model_scores1, model_scores2):
    return scipy.stats.kendalltau(
        list(model_scores1.values()),
        [model_scores2[model] for model in model_scores1],
    ).correlation # type: ignore

def compute_pval(model_scores1, model_scores2):
    return scipy.stats.ttest_rel(
        model_scores1, model_scores2,
    ).pvalue # type: ignore

def compute_ci(data, confidence=0.99):
    ci = scipy.stats.bootstrap((data,), np.mean, confidence_level=confidence, method='BCa')
    return (ci.confidence_interval.high - ci.confidence_interval.low) / 2

def analyze_protocol(data, key, k=1, domain=None):
    if domain is not None:
        key = f"{key}-{domain}"
    print("\n"+key)
    data = [
        doc
        for doc in data
        if "_#_tutorial_#_" not in doc["item"][0]["item_id"]
        and "_#_attention-check_#_" not in doc["item"][0]["item_id"]
        and (domain is None or f"_#_{domain}_#_" in doc["item"][0]["item_id"])  
    ]
    segments = sum([len(doc["item"]) for doc in data])
    print("Count:", len(data))
    if not data:
        return
    times = [compute_times(doc["actions"]) for doc in data]
    words = [sum([len(seg["src_text"].split()) for seg in doc["item"]]) for doc in data]
    print("Time (avg/word):", f"{sum(times) / sum(words) / k:.3f}", "s")

    model_scores = collections.defaultdict(lambda: collections.defaultdict(list))
    model_errors = collections.defaultdict(lambda: collections.defaultdict(list))
    model_scores_user = collections.defaultdict(list)
    for doc in data:
        for item, item_ann in zip(doc["item"], doc["annotation"]):
            item_id = item["item_id"]
            item_id = item_id.removesuffix("_#_dup0").removesuffix("_#_dup1")
            for model, ann in item_ann.items():
                model = model.removesuffix("'")
                model_scores_user[("all", item_id, model)].append(ann["score"])
                model_scores_user[(doc["user_id"], item_id, model)].append(ann["score"])
                model_scores[model][item_id].append(ann["score"])
                model_errors[model][item_id].append(len(ann["error_spans"]))
    model_scores_avg = {
        model: statistics.mean([statistics.mean(ll) for ll in model_v.values()])
        for model, model_v in model_scores.items()
    }
    print("Average score by model:", model_scores_avg)

    # compute inter-annotator agreement
    inter_aa_mse = [
            abs(a - b)
            for (
                user,
                _item,
                _model,
            ), duplicate_annotations_user in model_scores_user.items()
            if user == "all" and len(duplicate_annotations_user) >= 2
            for a, b in itertools.combinations(duplicate_annotations_user, 2)
        ]

    inter_aa_mse_ci = compute_ci(inter_aa_mse)
    inter_aa_mse = statistics.mean(inter_aa_mse)

    # compute intra-annotator agreement
    intra_aa_mse = [
            abs(a - b)
            for (
                user,
                _item,
                _model,
            ), duplicate_annotations_user in model_scores_user.items()
            if user != "all" and len(duplicate_annotations_user) >= 2
            for a, b in itertools.combinations(duplicate_annotations_user, 2)
        ]
    intra_aa_mse_ci = compute_ci(intra_aa_mse)
    intra_aa_mse = statistics.mean(intra_aa_mse)

    # compute krippendorff's alpha
    users_all = set([user for (user, _item, _model) in model_scores_user.keys() if user != "all"])
    items_all = set([(item, model) for (_user, item, model) in model_scores_user.keys()])
    user_to_index = {user: i for i, user in enumerate(sorted(users_all))}
    item_to_index = {item: i for i, item in enumerate(sorted(items_all))}
    kr_data = np.full((len(users_all), len(items_all)), np.nan)

    for (user, item, model), duplicate_annotations_user in model_scores_user.items():
        if user == "all":
            continue
        user_index = user_to_index[user]
        item_index = item_to_index[(item, model)]
        kr_data[user_index, item_index] = statistics.mean(duplicate_annotations_user)
    kr_alpha = krippendorff.alpha(reliability_data=kr_data, level_of_measurement='interval')

    # compute stability
    # model_scores_avg = {k: statistics.mean(v.values()) for k, v in model_scores.items()}
    item_ids = list(
        {
            item_id
            for model_scores in model_scores.values()
            for item_id in model_scores.keys()
        }
    )
    stability_taus = []
    for _ in range(100):
        stability_taus_local = []
        for subset_p in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for _ in range(100):
                item_ids_local = random.sample(item_ids, int(len(item_ids) * subset_p))
                model_scores_local = {
                    model: {
                        item_id: v[item_id] for item_id in item_ids_local if item_id in v
                    }
                    for model, v in model_scores.items()
                }
                model_scores_avg_local = {
                    model: statistics.mean([statistics.mean(ll) for ll in model_v.values()])
                    if model_v.values()
                    else 0
                    for model, model_v in model_scores_local.items()
                }
                stability_taus_local.append(
                    compute_kendalltau(model_scores_avg, model_scores_avg_local)
                )
        stability_taus.append(statistics.mean(stability_taus_local))
    stability_taus_ci = compute_ci(stability_taus)

    # compute clusters
    pvals = []
    models_sorted = sorted(model_scores_avg.keys(), key=lambda x: model_scores_avg[x], reverse=True)
    for (model1, model2) in itertools.product(models_sorted, repeat=2):
        if model1 == model2:
            continue
        model_scores_flat = [
            (statistics.mean(model_scores[model1][item_id]), statistics.mean(model_scores[model2][item_id]))
            for item_id in model_scores[model1].keys() & model_scores[model2].keys()
        ]
        model_scores_1 = [x for x, _ in model_scores_flat]
        model_scores_2 = [x for _, x in model_scores_flat]
        pvals.append(compute_pval(model_scores_1, model_scores_2))

    # plot histogram of model scores
    scores_all = [x for ll in model_scores.values() for y in ll.values() for x in y]
    plt.figure(figsize=(3.2, 1))
    plt.hist(
        scores_all,
        bins=list(np.linspace(0, 100, 21)),
        color="black",
    )
    plt.axis("off")
    plt.text(-5, 0, "0", fontsize=18)
    plt.text(100, 0, "100", fontsize=18)
    plt.ylim(0, len(scores_all) / 5)
    plt.tight_layout()
    plt.savefig(f"computed/img/{key}_scores.svg", transparent=True)


    # plot histogram of model errors
    errors_all = [x for ll in model_errors.values() for y in ll.values() for x in y]

    times_perseg_ci = compute_ci([t / segments / k for t in times])
    times_pererr_ci = compute_ci([t / sum(errors_all) for t in times])


    results[key] = {
        "time_perseg": f"{sum(times) / segments / k:.1f}s",
        "time_perseg_ci": f"{times_perseg_ci:.2f}",
        "time_perword": f"{sum(times) / sum(words) / k:.3f}s",
        "time_pererr": f"{sum(times) / sum(errors_all):.1f}s",
        "time_pererr_ci": f"{times_pererr_ci:.2f}",
        "inter-aa": f"{inter_aa_mse:.1f}",
        "inter-aa_ci": f"{inter_aa_mse_ci:.3f}",
        "intra-aa": f"{intra_aa_mse:.1f}",
        "intra-aa_ci": f"{intra_aa_mse_ci:.3f}",
        "kr_alpha": f"{kr_alpha:.3f}",
        "stability": f"{statistics.mean(stability_taus):.3f}",
        "stability_ci": f"{stability_taus_ci:.3f}",
        "model_ranking": {model: mean for model, mean in sorted(model_scores_avg.items(), key=lambda x: x[1], reverse=True)},
        "pvals": f"{statistics.mean(pvals):.3f}",
        "pvals_sig": f"{statistics.mean([float(pval < 0.05) for pval in pvals]):.0%}",
        "errors_avg": f"{statistics.mean(errors_all):.1f}",
    }

analyze_protocol(small_enja["ESA"], "small-enja-esa")
analyze_protocol(small_enja["cESA"], "small-enja-cesa1")
for k in range(1, 4+1):
    for domain in [None, "speech", "social", "news"]:
        analyze_protocol(big_enit_cESA_k[k], f"big-enit-cesa{k}", k=k, domain=domain)
        analyze_protocol(big_enja_cESA_k[k], f"big-enja-cesa{k}", k=k, domain=domain)

# add model ordering across all big-enit-*
results["big-enit-ordering"] = sorted(
    results["big-enit-cesa1"]["model_ranking"].keys(),
    key=lambda model: statistics.mean([
        float(results[f"big-enit-cesa{k}"]["model_ranking"].get(model, 0))
        for k in range(1, 4+1)
    ]),
    reverse=True,
)
results["big-enja-ordering"] = sorted(
    results["big-enja-cesa1"]["model_ranking"].keys(),
    key=lambda model: statistics.mean([
        float(results[f"big-enja-cesa{k}"]["model_ranking"].get(model, 0))
        for k in range(1, 4+1)
    ]),
    reverse=True,
)

with open("computed/analysis.json", "w") as f:
    json.dump(results, f, indent=2)
