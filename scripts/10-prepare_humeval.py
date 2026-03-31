# %%

import subset2evaluate.utils

data_enit = subset2evaluate.utils.load_data_wmt("wmt25", "en-it_IT", normalize=False)
data_enja = subset2evaluate.utils.load_data_wmt(
    "wmt25", "en-ja_JP", normalize=False, include_ref=True
)

# %%

# 200 hours = 2 * time_per_char_enzh * sum([len(item["src"].split()) for item in data_enzh]) * len(data_enzh[0]["scores"])
time_per_word_enja = (
    200
    * 60
    * 60
    / (
        2
        * sum([len(item["src"].split()) for item in data_enja])
        * len(data_enja[0]["scores"])
    )
)
time_per_word_enja = 1 / (4447 / 60 / 60)

print(time_per_word_enja)
print(data_enja[0]["scores"].keys())

# %%
print(len(data_enja[0]["scores"]), "models")

import collections

data_by_domain = collections.defaultdict(list)

for line in data_enja:
    domain = line["domain"]
    data_by_domain[domain].append(line)

print({domain: len(data) for domain, data in data_by_domain.items()})
print(len(data_enja))

# %%

for cap in [10, 20]:
    data_enja_capped = [
        segment
        for domain, data in data_by_domain.items()
        for segment in data[:cap]
        if domain != "literary"
    ]

    print(
        sum([len(item["src"].split()) for item in data_enja_capped])
        * (len(data_enja[0]["scores"]) + 1)
        * 2
        * time_per_word_enja
        / (60 * 60)
    )

# %%
import statistics

# average number of characters in source
print(statistics.mean([len(item["src"].split()) for item in data_enja_capped]))

# %%
import copy
import json

data_to_doc = collections.defaultdict(list)
for item in data_enja_capped:
    print(item["doc"])
    data_to_doc[item["doc"]].append(item)

data_to_doc = {doc: data for doc, data in data_to_doc.items() if len(data) >= 2}

data_local = [
    [
        {
            "src": item["src"],
            "tgt": {model: item["tgt"][model] for model in list(data_enja[0]["scores"].keys())[:models]},
        }
        for item in doc
    ]
    for models, doc in zip([1, 2, 3, 4, 5, 6], data_to_doc.values())
]

with open("../humeval/cESA_guidelines.html", "r") as f:
    html_guidelines = f.read()

data_pearmut = {
    "info": {
        "assignment": "task-based",
        "protocol": "ESA",
        "textfield": "hidden",
        "instructions": html_guidelines,
    },
    "campaign_id": "example_cESA",
    "data": [
        copy.deepcopy(data_local),
        copy.deepcopy(data_local),
        copy.deepcopy(data_local),
    ]
}

with open("../humeval/example_cESA.json", "w") as f:
    json.dump(data_pearmut, f, indent=4, ensure_ascii=False)