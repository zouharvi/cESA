# %%

# ESA vs cESA ablation

import subset2evaluate.utils
import collections
import random
import json


data = subset2evaluate.utils.load_data_wmt(
    "wmt25", "en-ja_JP", normalize=False, include_ref=True
)
MODELS = [
    "refA",
    "Gemini-2.5-Pro",
    "ONLINE-B",
    "Laniqo",
]

time_per_word = 1 / (4447 / 60 / 60)

doc_to_data = collections.defaultdict(list)
for item in data:
    if item["domain"] == "literary":
        continue
    doc_to_data[item["doc"]].append(item)
doc_to_data = {doc: data for doc, data in doc_to_data.items()}
for doc, data_local in sorted(doc_to_data.items(), key=lambda x: x[0]):
    print(doc, len(data_local))

PHASES = []
PHASES.append(
    [
        ("en-ja_JP_#_news_#_guardian.228996", 6),
        ("en-ja_JP_#_news_#_guardian.230737", 4),
        ("en-ja_JP_#_news_#_guardian.231311", 4),
        ("en-ja_JP_#_social_#_112502991207286008", 5),
        ("en-ja_JP_#_speech_#_vid_BrC0v6K3H4I", 1),
        ("en-ja_JP_#_speech_#_vid_Ft41IG2BAG8", 1),
        ("en-ja_JP_#_speech_#_vid_GYfhGLrrDts", 1),
        ("en-ja_JP_#_speech_#_vid_HjRhgaz1xTI", 1),
        ("en-ja_JP_#_speech_#_vid_JIq55zBGNX4", 1),
        ("en-ja_JP_#_speech_#_vid_JoTLTGv8kqA", 1),
    ]
)
PHASES.append(
    [
        ("en-ja_JP_#_news_#_brisbanetimes.com.au.306576", 6),
        ("en-ja_JP_#_news_#_guardian.228996", 6),
        ("en-ja_JP_#_social_#_114151944720213193", 7),
        ("en-ja_JP_#_social_#_114157282362077575", 5),
        ("en-ja_JP_#_speech_#_vid_27keISTaqYw", 1),
        ("en-ja_JP_#_speech_#_vid_2cLeDVfEqG4", 1),
        ("en-ja_JP_#_speech_#_vid_3vpEaAjDgtI", 1),
        ("en-ja_JP_#_speech_#_vid_6dP6bHX73_k", 1),
        ("en-ja_JP_#_speech_#_vid_7Aw4Q46omiM", 1),
        ("en-ja_JP_#_speech_#_vid_8I8msBYGNR4", 1),
        ("en-ja_JP_#_speech_#_vid_8qZFupajBuo", 1),
    ]
)
PHASES.append(
    [
        ("en-ja_JP_#_news_#_guardian.231314", 6),
        ("en-ja_JP_#_news_#_newrepublic.com.12619", 4),
        ("en-ja_JP_#_social_#_114300646822630777", 4),
        ("en-ja_JP_#_social_#_114417630342798842", 4),
        ("en-ja_JP_#_speech_#_vid_Qu8m9h-AC0I", 1),
        ("en-ja_JP_#_speech_#_vid_RIjzeZ1xk1c", 1),
        ("en-ja_JP_#_speech_#_vid_Si9CnnVI8sk", 1),
        ("en-ja_JP_#_speech_#_vid_YTxmx8hyJtA", 1),
        ("en-ja_JP_#_speech_#_vid__S0GOwEJXUA", 1),
    ]
)
PHASES.append(
    [
        ("en-ja_JP_#_news_#_newrepublic.com.12623", 2),
        ("en-ja_JP_#_news_#_nytimes.153341", 5),
        ("en-ja_JP_#_social_#_114174389242714730", 6),
        ("en-ja_JP_#_speech_#_vid_Lb7Dhf7oo4Y", 1),
        ("en-ja_JP_#_speech_#_vid_M7v3ZfP4fm0", 1),
        ("en-ja_JP_#_speech_#_vid_OmK3XTCDbbs", 1),
        ("en-ja_JP_#_speech_#_vid_PJLN1kAzbyw", 1),
        ("en-ja_JP_#_speech_#_vid_QgTkUWYd82M", 1),
    ]
)

data_by_domain = collections.defaultdict(list)
for line in data:
    domain = line["domain"]
    data_by_domain[domain].append(line)

print({domain: len(data) for domain, data in data_by_domain.items()})

data_phases = []
for phase in PHASES:
    data_phase = []
    for doc, num_segments in phase:
        data_phase.append(doc_to_data[doc][:num_segments])
    data_phases.append(data_phase)


Item = dict
Doc = list[Item]
# DocAll --- docs across the same configuration
DocAll = list[Doc]

data_phases_out_flat = []
for phase, data_phase in enumerate(data_phases):
    data_phase_out: list[list[DocAll]] = []
    for data_doc in data_phase:
        random.seed(0)
        data_item: list[DocAll] = []
        # add one random duplicate model
        models = random.sample(MODELS, len(MODELS)) + random.sample(MODELS, 1)
        for dup_i in [0, 1]:
            data_item_config: DocAll = []
            for model in models:
                data_doc_local: Doc = []
                for item_i, item in enumerate(data_doc):
                    if "_#_social_#_" in item["doc"]:
                        src_img = item["doc"].split("_#_")[-1]
                        src = f'<img style="width: 100%;" src="https://vilda.net/t/wmt25/assets/en/social/{src_img}-anon/{src_img}-anon_{item_i + 1}.png">'
                    elif "_#_speech_#_" in item["doc"]:
                        src_vid = item["doc"].split("_#_")[-1]
                        src = f'<video style="width: 100%;" src="https://vilda.net/t/wmt25/assets/en/speech/{src_vid}.mp4" controls>'
                    else:
                        src = item["src"].replace("\\n", "\n")
                    data_doc_local.append(
                        {
                            "src": src,
                            "src_text": item["src"],
                            "tgt": {
                                model: item["tgt"][model]
                            },
                            "doc": item["doc"] + f"-dup{dup_i}",
                        }
                    )
                data_item_config.append(data_doc_local)
            data_item.append(data_item_config)
        data_phase_out.append(data_item)
    time = sum(
        len(item["src_text"].split()) * len(item["tgt"]) * time_per_word
        for data_item in data_phase_out
        for data_item_config in data_item
        for doc in data_item_config
        for item in doc
    )
    print(f"Phase time: {time / (60 * 60) * 2:.1f} hours per all configurations")
    data_phases_out_flat.append(data_phase_out)

# %%
import statistics


with open("/home/vilda/pearmut/examples/tutorials/cesa_jaen.json", "r") as f:
    data_tutorial_cesa = json.load(f)["data"][0]


with open("/home/vilda/pearmut/examples/tutorials/esa_jaen.json", "r") as f:
    data_tutorial_esa = json.load(f)["data"][0]


for phase, data_phase_out in enumerate(data_phases_out_flat):
    num_users = 5
    data_phase_out: list[list[DocAll]]
    tasks = [[] for _ in range(num_users)]
    for data_item in data_phase_out:
        data_item: list[DocAll]
        # sort by fewest segments to most segments so that we assign the longest ones last (to make sure we don't end up with a long one at the end that doesn't fit in any user's queue)
        tasks.sort(key=lambda task: sum([len(doc) for doc in task]))
        tasks_to_expand = tasks[:len(data_item)]
        # make sure that we assign different conditions of the same doc to different users
        for task, data_item_config in zip(tasks_to_expand, data_item):
            data_item_config: DocAll
            task.extend(data_item_config)

    print([len(task) for task in tasks])
    print(f"{statistics.mean([
        sum([len(item["src_text"].split()) * len(item["tgt"]) * time_per_word for doc in task for item in doc])
        for task in tasks
    ]) / (60 * 60):.1f} hours per user")
    print(f"{sum([
        sum([len(item["src_text"].split()) * len(item["tgt"]) * time_per_word for doc in task for item in doc])
        for task in tasks
    ]) / (60 * 60) * 2:.1f} hours total")
    print()

    # shuffle queue
    for task in tasks:
        random.shuffle(task)

    data_pearmut_cesa = {
        "info": {
            "assignment": "task-based",
            "protocol": "cESA",
            # "assets": {
            #     "source": "wmt25_genmt_assets/assets/en",
            #     "destination": "assets/wmt25_genmt_assets",
            # },
        },
        "campaign_id": f"preablation_cESA_phase{phase}",
        "data": [
            list(data_tutorial_cesa) + task
            for task in tasks
        ],
    }
    data_pearmut_esa = {
        "info": {
            "assignment": "task-based",
            "protocol": "ESA",
            # "assets": {
            #     "source": "wmt25_genmt_assets/assets/en",
            #     "destination": "assets/wmt25_genmt_assets",
            # },
        },
        "campaign_id": f"preablation_ESA_phase{phase}",
        "data": [
            list(data_tutorial_esa) + task
            for task in tasks
        ],
    }

    with open(f"../humeval/preablation_cESA_phase{phase}.json", "w") as f:
        json.dump(data_pearmut_cesa, f, indent=4, ensure_ascii=False)


    with open(f"../humeval/preablation_ESA_phase{phase}.json", "w") as f:
        json.dump(data_pearmut_esa, f, indent=4, ensure_ascii=False)

# %%
import os
import requests
import zipfile
import io

os.makedirs("../humeval/wmt25_genmt_assets", exist_ok=True)
zipfile.ZipFile(
    io.BytesIO(
        requests.get(
            "https://data.statmt.org/wmt25/general-mt/wmt25_genmt_assets.zip"
        ).content
    )
).extractall("../humeval/wmt25_genmt_assets")
