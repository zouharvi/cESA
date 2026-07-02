# %%
import collections
import json
import os

os.chdir(os.path.dirname(__file__)+"/../")

big_enja = collections.defaultdict(list)
with open("humeval/collected/main_enja.json", "r") as f:
    data_raw = json.load(f)
    for campaign_name, campaign_data in data_raw.items():
        for item in campaign_data:
            if item["annotation"] == "__RESET__":
                continue
            k = len(item["item"][0]["tgt"])
            item["protocol"] = k
            big_enja[k].append(item)


big_enit = collections.defaultdict(list)
with open("humeval/collected/main_enit.json", "r") as f:
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
    } for key in ["enja", "enit"]
}

CHRF = sacrebleu.metrics.chrf.CHRF()
def chrf(s1, s2):
    """Pairwise chrf for tgt similarity."""
    return (CHRF.sentence_score(s1, [s2]).score + CHRF.sentence_score(s2, [s1]).score) / 2

for data_key, data in [("enja", big_enja), ("enit", big_enit)]:
    for k in tqdm.tqdm([2, 3, 4]):
        # collect "screens"
        docs_to_screens = collections.defaultdict(list)
        for item in data[k]:
            for item_seg, item_ann in zip(item["item"], item["annotation"]):
                screen = {}
                for model, annotation in item_ann.items():
                    if annotation["score"] is not None:
                        screen[model.removesuffix("'")] = {"score": annotation["score"], "tgt": item_seg["tgt"][model]}
                if len(screen) < 2:
                    continue
                if screen:
                    item_id = item_seg["item_id"].removesuffix("_#_dup0").removesuffix("_#_dup1")
                    docs_to_screens[item_id].append(screen)
        
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
