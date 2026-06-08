# %%
import collections
import json

data_cESA_k = collections.defaultdict(list)
with open("../humeval/collected/main_enit.json", "r") as f:
    data_raw = json.load(f)
    for campaign_name, campaign_data in data_raw.items():
        for item in campaign_data:
            k = len(item["item"][0]["tgt"])
            protocol = "cESA_k" + str(k)
            item["protocol"] = protocol
            data_cESA_k[k].append(item)

# %%
import numpy as np
import matplotlib.pyplot as plt
import evaluation_contrastive.utils_fig
import os

os.makedirs("computed/img", exist_ok=True)

# position bias


def plot(key):
    scores = np.clip(np.random.normal(65, 30, 300), 0, 150)
    scores = np.digitize(scores, np.arange(10, 100, 10)) * 10
    plt.figure(figsize=(2, 1))
    plt.hist(
        scores,
        bins=list(np.linspace(0, 100, 21)),
        width=10,
        color="black",
    )
    plt.axis("off")
    plt.ylim(0, len(scores) / 5)
    mean = scores.mean()
    plt.text(
        mean,
        len(scores) / 5,
        f"{mean:.1f}",
        fontsize=18,
        ha="right",
        va="center",
        color=evaluation_contrastive.utils_fig.COLORS[0],
    )
    plt.vlines(
        mean,
        ymin=0,
        ymax=len(scores) / 6,
        color=evaluation_contrastive.utils_fig.COLORS[0],
        linestyle="--",
    )
    plt.tight_layout()
    plt.savefig(f"computed/img/position_bias_{key}.svg", transparent=True)


plot("k=1,1")
plot("k=2,1")
plot("k=2,2")
plot("k=3,1")
plot("k=3,2")
plot("k=3,3")
plot("k=4,1")
plot("k=4,2")
plot("k=4,3")
plot("k=4,4")


# %%
import json
import collections

# similarity and dominance bias

output = {}

for bias in ["similarity", "dominance"]:
    output_col = collections.defaultdict(dict)
    for k in [2, 3, 4]:
        for sim in ["mean", "max"]:
            output_col[sim][k] = f"{np.random.normal(0.5, 0.01):.1%}"
    output[bias] = output_col

with open("computed/similarity_dominance_bias.json", "w") as f:
    json.dump(output, f, indent=2)


# %%

# annotation sequence pattern

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

# %%
