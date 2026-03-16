# %%

import subset2evaluate.utils

data_enit = subset2evaluate.utils.load_data_wmt("wmt25", "en-it_IT", normalize=False)
data_enzh = subset2evaluate.utils.load_data_wmt("wmt25", "en-zh_CN", normalize=False)

# %%

# 200 hours = 2 * time_per_char_enit * sum([len(item["src"].split()) for item in data_enit]) * len(data_enit[0]["scores"])
time_per_char_enit = (
    200
    * 60
    * 60
    / (
        2
        * sum([len(item["src"].split()) for item in data_enit])
        * len(data_enit[0]["scores"])
    )
)
# 200 hours = 2 * time_per_char_enzh * sum([len(item["src"].split()) for item in data_enzh]) * len(data_enzh[0]["scores"])
time_per_char_enzh = (
    200
    * 60
    * 60
    / (
        2
        * sum([len(item["src"].split()) for item in data_enzh])
        * len(data_enzh[0]["scores"])
    )
)

print(time_per_char_enit, time_per_char_enzh)

# %%
print(len(data_enzh[0]["scores"]), "models")

import collections

data_by_domain = collections.defaultdict(list)

for line in data_enzh:
    domain = line["domain"]
    data_by_domain[domain].append(line)

print({domain: len(data) for domain, data in data_by_domain.items()})
print(len(data_enzh))

# %%

data_enzh_capped = [
    segment
    for domain, data in data_by_domain.items()
    for segment in data[:15]
    if domain != "literary"
]

print(
    sum([len(item["src"].split()) for item in data_enzh_capped])
    * (len(data_enzh_capped[0]["scores"]) + 1)
    * time_per_char_enzh
    * 2
    / (60 * 60)
)
