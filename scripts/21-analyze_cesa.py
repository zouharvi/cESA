# %%
import collections
import json
import os
from random import random
import statistics

os.chdir(os.path.dirname(__file__)+"/../")

big_enja = collections.defaultdict(list)
with open("humeval/annotations/main_enja.json", "r") as f:
    data_raw = json.load(f)
    for campaign_name, campaign_data in data_raw.items():
        for item in campaign_data:
            if item["annotation"] == "__RESET__":
                continue
            k = len(item["item"][0]["tgt"])
            item["protocol"] = k
            big_enja[k].append(item)


big_enit = collections.defaultdict(list)
with open("humeval/annotations/main_enit.json", "r") as f:
    data_raw = json.load(f)
    for campaign_name, campaign_data in data_raw.items():
        for item in campaign_data:
            if item["annotation"] == "__RESET__":
                continue
            k = len(item["item"][0]["tgt"])
            item["protocol"] = k
            big_enit[k].append(item)


# %%
import numpy as np
import matplotlib.pyplot as plt
import evaluation_contrastive.utils_fig
import os

os.makedirs("computed/img", exist_ok=True)

# position bias

def plot(data, key, pos):
    scores = []
    for doc in data:
        for item, item_ann in zip(doc["item"], doc["annotation"]):
            for model_i, (model, annotation) in enumerate(item_ann.items()):
                if model_i != pos - 1:
                    continue
                if annotation["score"] is not None:
                    scores.append(annotation["score"])
    
    plt.figure(figsize=(1.7, 1))
    plt.hist(
        scores,
        bins=list(np.linspace(0, 100, 21)),
        width=10,
        color="black",
    )
    plt.axis("off")
    plt.ylim(0, len(scores) / 5)
    mean = float(np.mean(scores))
    plt.text(
        mean,
        len(scores) / 5,
        f"{mean:.1f}",
        fontsize=18,
        ha="right",
        va="center",
        color="#727D46",
    )
    plt.vlines(
        mean,
        ymin=0,
        ymax=len(scores) / 6,
        color="#727D46",
        linestyle="--",
    )
    plt.tight_layout()
    plt.savefig(f"computed/img/position_bias_{key}.svg", transparent=True)


plot(big_enit[1], "enit,1,1", pos=1)
plot(big_enit[2], "enit,2,1", pos=1)
plot(big_enit[2], "enit,2,2", pos=2)
plot(big_enit[3], "enit,3,1", pos=1)
plot(big_enit[3], "enit,3,2", pos=2)
plot(big_enit[3], "enit,3,3", pos=3)
plot(big_enit[4], "enit,4,1", pos=1)
plot(big_enit[4], "enit,4,2", pos=2)
plot(big_enit[4], "enit,4,3", pos=3)
plot(big_enit[4], "enit,4,4", pos=4)

plot(big_enja[1], "enja,1,1", pos=1)
plot(big_enja[2], "enja,2,1", pos=1)
plot(big_enja[2], "enja,2,2", pos=2)
plot(big_enja[3], "enja,3,1", pos=1)
plot(big_enja[3], "enja,3,2", pos=2)
plot(big_enja[3], "enja,3,3", pos=3)
plot(big_enja[4], "enja,4,1", pos=1)
plot(big_enja[4], "enja,4,2", pos=2)
plot(big_enja[4], "enja,4,3", pos=3)
plot(big_enja[4], "enja,4,4", pos=4)


# %%
import json
import collections
import itertools
import sacrebleu
import tqdm

# similarity and dominance bias

output = {
    key: {
        "similarity_mean": {},
        "similarity_max": {},
        "dominance_mean": {},
        "dominance_max": {},
        "intramae": {},
        "intermae": {},
        "withinscreen-pairwise": {},
        "acrossscreen-pairwise": {},
    } for key in ["enja", "enit"]
}

CHRF = sacrebleu.metrics.chrf.CHRF()
def chrf(s1, s2):
    """Pairwise chrf for tgt similarity."""
    return (CHRF.sentence_score(s1, [s2]).score + CHRF.sentence_score(s2, [s1]).score) / 2

for data_key, data in [("enja", big_enja), ("enit", big_enit)]:
    for k in tqdm.tqdm([1, 2, 3, 4]):
        # collect "screens"
        docs_to_screens = collections.defaultdict(list)
        model_average = collections.defaultdict(list)
        for item in data[k]:
            for item_seg, item_ann in zip(item["item"], item["annotation"]):
                screen = {}
                for model, annotation in item_ann.items():
                    if annotation["score"] is not None and "_#_tutorial_#_" not in item_seg["item_id"]:
                        model_average[model.removesuffix("'")].append(annotation["score"])
                        screen[model.removesuffix("'")] = {"score": annotation["score"], "tgt": item_seg["tgt"][model]}
                if screen:
                    item_id = item_seg["item_id"].removesuffix("_#_dup0").removesuffix("_#_dup1")
                    docs_to_screens[item_id].append(screen)
        model_average = {k: np.mean(v) for k, v in model_average.items()}

        # inter- and intra-MAE
        results_intramae = []
        results_intermae = []
        for _doc, screens in docs_to_screens.items():
            if len(screens) < 2:
                continue
            for screen_a, screen_b in itertools.product(screens, screens):
                if screen_a == screen_b:
                    continue
                if k != 1:
                    screen_a_intramae = statistics.mean([abs(x1-x2) for x1, x2 in itertools.combinations([v["score"] for v in screen_a.values()], 2)])
                    screen_b_intramae = statistics.mean([abs(x1-x2) for x1, x2 in itertools.combinations([v["score"] for v in screen_b.values()], 2)])
                    screens_intramae = (screen_a_intramae + screen_b_intramae) / 2
                    results_intramae.append(screens_intramae)
                
                screens_intermae = statistics.mean([abs(x1-x2) for x1, x2 in itertools.product([v["score"] for v in screen_a.values()], [v["score"] for v in screen_b.values()])])
                results_intermae.append(screens_intermae)
        if k != 1:
            output[data_key]["intramae"][k] = np.mean(results_intramae)
        output[data_key]["intermae"][k] = np.mean(results_intermae)

        results_withinscreen = []
        results_acrossscreen = []
        for _doc, screens in docs_to_screens.items():
            # compute within-screeen --- global model ranking
            for screen in screens:
                for model1, model2 in itertools.combinations(screen.keys(), 2):
                    results_withinscreen.append(int((model_average[model1] > model_average[model2]) == (screen[model1]["score"] > screen[model2]["score"])))
                
            # compute across-screeen --- global model ranking
            for screen_a, screen_b in itertools.product(screens, screens):
                if screen_a == screen_b:
                    continue

                for model1, model2 in itertools.product(screen_a.keys(), screen_b.keys()):
                    if model1 == model2:
                        continue
                    results_acrossscreen.append(int((model_average[model1] > model_average[model2]) == (screen_a[model1]["score"] > screen_b[model2]["score"])))

        if k != 1:
            output[data_key]["withinscreen-pairwise"][k] = np.mean(results_withinscreen)
        output[data_key]["acrossscreen-pairwise"][k] = np.mean(results_acrossscreen)

        if k == 1:
            continue

        # compute similarity
        results = []
        for vvv in ["mean", "max"]:
            results = []
            for _doc, screens in docs_to_screens.items():
                if len(screens) < 2:
                    continue
                for screen_a, screen_b in itertools.product(screens, screens):
                    if screen_a == screen_b:
                        continue

                    # model needs to be in both
                    for model in screen_a.keys() & screen_b.keys():
                        screen_a_sim = [
                            chrf(model1_tgt, model2_tgt)
                            for model1_tgt, model2_tgt in itertools.combinations(
                                [v["tgt"] for v in screen_a.values()],
                                2
                            )
                        ]
                        screen_b_sim = [
                            chrf(model1_tgt, model2_tgt)
                            for model1_tgt, model2_tgt in itertools.combinations(
                                [v["tgt"] for v in screen_b.values()],
                                2
                            )
                        ]
                        if vvv == "mean":
                            screen_a_vvv = np.mean(screen_a_sim)
                            screen_b_vvv = np.mean(screen_b_sim)
                        elif vvv == "max":
                            screen_a_vvv = np.max(screen_a_sim)
                            screen_b_vvv = np.max(screen_b_sim)
                        else:
                            raise ValueError(vvv)

                        # check if higher similarity is causing the model score to be lower
                        results.append(
                            int(screen_a_vvv > screen_b_vvv and screen_a[model]["score"] < screen_b[model]["score"])
                            -
                            # subtract the opposite case
                            int(screen_a_vvv > screen_b_vvv and screen_a[model]["score"] > screen_b[model]["score"])
                        )

            output[data_key]["similarity_" + vvv][k] = np.mean(results)

            # compute dominance
            results = []
            for _doc, screens in docs_to_screens.items():
                if len(screens) < 2:
                    continue
                for screen_a, screen_b in itertools.product(screens, screens):
                    if screen_a == screen_b:
                        continue
                    # model needs to be in both
                    for model in screen_a.keys() & screen_b.keys():
                        if vvv == "mean":
                            screen_a_vvv = np.mean([v["score"] for v in screen_a.values()])
                            screen_b_vvv = np.mean([v["score"] for v in screen_b.values()])
                        elif vvv == "max":
                            screen_a_vvv = np.max([v["score"] for v in screen_a.values()])
                            screen_b_vvv = np.max([v["score"] for v in screen_b.values()])
                        else:
                            raise ValueError(vvv)

                        # check if dominance is causing the model score to be lower
                        results.append(
                            int(screen_a_vvv > screen_b_vvv and screen_a[model]["score"] < screen_b[model]["score"])
                            -
                            int(screen_a_vvv > screen_b_vvv and screen_a[model]["score"] > screen_b[model]["score"])
                        )

            output[data_key]["dominance_" + vvv][k] = np.mean(results)


with open("computed/similarity_dominance_bias.json", "w") as f:
    json.dump(output, f, indent=2)


# %%


# statistical modelling

import choix
import numpy as np
import tqdm
import scipy.stats
import copy
import random

def relative_ranking(n_models, data, **kwargs):
    ranks = collections.defaultdict(list)
    for screen in data:
        for i, model in enumerate(screen):
            ranks[model].append(i)
    ranks = {k: np.mean(v) for k, v in ranks.items()}
    return [ranks[i] for i in range(n_models)]

output = collections.defaultdict(dict)
for data_key, data in [("enja", big_enja), ("enit", big_enit)]:
    for k in tqdm.tqdm([1, 2, 3, 4]):

        model_average = collections.defaultdict(list)
        data_across_screens = collections.defaultdict(dict)
        data_within_screens = []

        for item in data[k]:
            for item_seg, item_ann in zip(item["item"], item["annotation"]):
                screen = {}
                for model, annotation in item_ann.items():
                    model = model.removesuffix("'")
                    if annotation["score"] is not None and "_#_tutorial_#_" not in item_seg["item_id"]:
                        model_average[model].append(annotation["score"])
                        data_across_screens[item_seg["item_id"]][model] = annotation["score"]
                        screen[model] = annotation["score"]
                if screen:
                    data_within_screens.append(screen)

        model_average = {k: np.mean(v) for k, v in model_average.items()}
        model_to_i = {k: i for i, k in enumerate(sorted(model_average.keys()))}
        i_to_model = {i: k for k, i in model_to_i.items()}

        # make a "large" screen out of each item
        data_across_screens = list(data_across_screens.values())
        # encode models as integers for Plackett-Luce
        data_across_screens = [
            tuple([model_to_i[model] for model, score in sorted(screen.items(), key=lambda x: x[1], reverse=True)])
            for screen in data_across_screens
        ]
        data_within_screens = [
            tuple([model_to_i[model] for model, score in sorted(screen.items(), key=lambda x: x[1], reverse=True)])
            for screen in data_within_screens
        ]

        # Plackett-Luce

        # across-screen ranking
        pl_params = choix.ilsr_rankings(len(model_average), data_across_screens, max_iter=10_000)
        model_average_pred = {i_to_model[i]: score for i, score in enumerate(pl_params)}
        tau = scipy.stats.kendalltau(
            [model_average[model] for model in sorted(model_average.keys())],
            [model_average_pred[model] for model in sorted(model_average.keys())],
        )
        print(f"k={k} across-screens:      {tau[0]:.3f}")
        output[data_key][f"across-screens,k={k}"] = tau[0]

        if k == 1:
            continue

        # # across-screen-mock ranking
        # # create mock tuples
        # data_across_screens_mock = []
        # for _screen in data_across_screens:
        #     for _ in range(100):
        #         screen = copy.deepcopy(_screen)
        #         while screen:
        #             sample_i = tuple(random.sample(range(len(screen)), k=min(len(screen), k)))
        #             data_across_screens_mock.append(tuple([screen[i] for i in sample_i]))
        #             screen = tuple([screen[i] for i in range(len(screen)) if i not in sample_i])
        # pl_params = choix.ilsr_rankings(len(model_average), data_across_screens_mock, max_iter=10_000)
        # model_average_pred = {i_to_model[i]: score for i, score in enumerate(pl_params)}
        # tau = scipy.stats.kendalltau(
        #     [model_average[model] for model in sorted(model_average.keys())],
        #     [model_average_pred[model] for model in sorted(model_average.keys())],
        # )
        # print(f"k={k} across-screens-mock: {tau[0]:.3f}")
        # output[data_key][f"across-screens-mock,k={k}"] = tau[0]

        # within-screen ranking
        pl_params = choix.ilsr_rankings(len(model_average), data_within_screens, max_iter=10_000)
        model_average_pred = {i_to_model[i]: score for i, score in enumerate(pl_params)}
        tau = scipy.stats.kendalltau(
            [model_average[model] for model in sorted(model_average.keys())],
            [model_average_pred[model] for model in sorted(model_average.keys())],
        )
        print(f"k={k} within-screens:      {tau[0]:.3f}")
        output[data_key][f"within-screens,k={k}"] = tau[0]

with open("computed/modelling.json", "w") as f:
    json.dump(output, f, indent=2)

# %%

# annotation sequence pattern

# TODO:

# score, error
output = []
counter = 0
for col in range(4):
    output_col = []
    for row in range(3):
        output_col.append(
            (
                counter + 1 + np.random.normal(0, counter / 10),
                counter + np.random.normal(0, counter / 2),
            )
        )
        counter += 2
    output.append(output_col)

output = np.array(output).transpose((1, 0, 2))
output = [[(f"{x[0]:.1f}", f"{x[1]:.1f}") for x in y] for y in output]

with open("computed/sequence_pattern.json", "w") as f:
    json.dump(output, f, indent=2)
