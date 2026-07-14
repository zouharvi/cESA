# %%

import subset2evaluate.utils # type: ignore
import collections
import random
import json
import os

os.chdir(os.path.dirname(__file__) + "/../")


data = subset2evaluate.utils.load_data_wmt(
    "wmt25", "en-it_IT", normalize=False, include_ref=True, require_human=False,
)
# dict_keys(['Claude-4', 'ONLINE-G', 'CommandR7B', 'ONLINE-W', 'AyaExpanse-32B', 'EuroLLM-22B', 'DeepSeek-V3', 'GemTrans', 'AyaExpanse-8B', 'Mistral-7B', 'Gemma-3-12B', 'UvA-MT', 'Laniqo', 'IR-MultiagentMT', 'CommandA', 'Llama-3.1-8B', 'TowerPlus-9B', 'EuroLLM-9B', 'Shy', 'ONLINE-B', 'NLLB', 'Qwen2.5-7B', 'TowerPlus-72B', 'Mistral-Medium', 'Gemma-3-27B', 'TranssionTranslate', 'CommandA-MT', 'GPT-4.1', 'Qwen3-235B', 'SalamandraTA', 'IRB-MT', 'refA', 'Llama-4-Maverick', 'Gemini-2.5-Pro'])
MODELS = [
    "GPT-4.1", # refA
    "Gemini-2.5-Pro",
    "GemTrans",  # "Algharb",
    "Mistral-Medium",
    "CommandA-MT",
    "DeepSeek-V3",
    "Claude-4",
    "ONLINE-B",
    "Gemma-3-27B", # "Yolu",
    "Laniqo",
]

time_per_word = 1 / (4447 / 60 / 60)
print(time_per_word)

# 1 en-it_IT_#_news_#_brisbanetimes.com.au.306576 6
# 1 en-it_IT_#_news_#_guardian.228996 9
# 1 en-it_IT_#_news_#_guardian.230737 8
# 1 en-it_IT_#_news_#_guardian.231311 9
# 2 en-it_IT_#_news_#_guardian.231314 10
# 2 en-it_IT_#_news_#_newrepublic.com.12619 4
# 2 en-it_IT_#_news_#_newrepublic.com.12623 2
# 2 en-it_IT_#_news_#_nytimes.153341 5
# 1 en-it_IT_#_social_#_112502991207286008 5
# 1 en-it_IT_#_social_#_114151944720213193 7
# 1 en-it_IT_#_social_#_114157282362077575 5
# 2 en-it_IT_#_social_#_114174389242714730 8
# 2 en-it_IT_#_social_#_114300646822630777 12
# 2 en-it_IT_#_social_#_114417630342798842 21
# 1 en-it_IT_#_speech_#_vid_27keISTaqYw 1
# 1 en-it_IT_#_speech_#_vid_2cLeDVfEqG4 1
# 1 en-it_IT_#_speech_#_vid_3vpEaAjDgtI 1
# 1 en-it_IT_#_speech_#_vid_6dP6bHX73_k 1
# 1 en-it_IT_#_speech_#_vid_7Aw4Q46omiM 1
# 1 en-it_IT_#_speech_#_vid_8I8msBYGNR4 1
# 1 en-it_IT_#_speech_#_vid_8qZFupajBuo 1
# 1 en-it_IT_#_speech_#_vid_BrC0v6K3H4I 1
# 1 en-it_IT_#_speech_#_vid_Ft41IG2BAG8 1
# 1 en-it_IT_#_speech_#_vid_GYfhGLrrDts 1
# 1 en-it_IT_#_speech_#_vid_HjRhgaz1xTI 1
# 1 en-it_IT_#_speech_#_vid_JIq55zBGNX4 1
# 1 en-it_IT_#_speech_#_vid_JoTLTGv8kqA 1
# 2 en-it_IT_#_speech_#_vid_Lb7Dhf7oo4Y 1
# 2 en-it_IT_#_speech_#_vid_M7v3ZfP4fm0 1
# 2 en-it_IT_#_speech_#_vid_OmK3XTCDbbs 1
# 2 en-it_IT_#_speech_#_vid_PJLN1kAzbyw 1
# 2 en-it_IT_#_speech_#_vid_QgTkUWYd82M 1
# 2 en-it_IT_#_speech_#_vid_Qu8m9h-AC0I 1
# 2 en-it_IT_#_speech_#_vid_RIjzeZ1xk1c 1
# 2 en-it_IT_#_speech_#_vid_Si9CnnVI8sk 1
# 2 en-it_IT_#_speech_#_vid_YTxmx8hyJtA 1
# 2 en-it_IT_#_speech_#_vid__S0GOwEJXUA 1
# en-it_IT_#_speech_#_vid__r0K17ipoac 1
# en-it_IT_#_speech_#_vid_fWRlPSl5itQ 1
# en-it_IT_#_speech_#_vid_m6CcZN4S8mk 1
# en-it_IT_#_speech_#_vid_mDjddYiHd7A 1
# en-it_IT_#_speech_#_vid_n-mmxFdva4w 1
# en-it_IT_#_speech_#_vid_nNjtLybPRjM 1
# en-it_IT_#_speech_#_vid_nU09s4CuaRg 1
# en-it_IT_#_speech_#_vid_qhwgP27R4xw 1
# en-it_IT_#_speech_#_vid_r-aI0JD0bCE 1
# en-it_IT_#_speech_#_vid_rU8nY03Lo5s 1
# en-it_IT_#_speech_#_vid_u2JNW0Ftyzg 1
# en-it_IT_#_speech_#_vid_v2NNTNAXRWY 1
# en-it_IT_#_speech_#_vid_x48axn9yc9k 1

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
        ("en-it_IT_#_news_#_guardian.228996", 6),
        ("en-it_IT_#_news_#_guardian.230737", 4),
        ("en-it_IT_#_news_#_guardian.231311", 4),
        ("en-it_IT_#_social_#_112502991207286008", 5),
        ("en-it_IT_#_speech_#_vid_BrC0v6K3H4I", 1),
        ("en-it_IT_#_speech_#_vid_Ft41IG2BAG8", 1),
        ("en-it_IT_#_speech_#_vid_GYfhGLrrDts", 1),
        ("en-it_IT_#_speech_#_vid_HjRhgaz1xTI", 1),
        ("en-it_IT_#_speech_#_vid_JIq55zBGNX4", 1),
        ("en-it_IT_#_speech_#_vid_JoTLTGv8kqA", 1),
    ]
)
PHASES.append(
    [
        ("en-it_IT_#_news_#_brisbanetimes.com.au.306576", 6),
        ("en-it_IT_#_news_#_guardian.228996", 6),
        ("en-it_IT_#_social_#_114151944720213193", 7),
        ("en-it_IT_#_social_#_114157282362077575", 5),
        ("en-it_IT_#_speech_#_vid_27keISTaqYw", 1),
        ("en-it_IT_#_speech_#_vid_2cLeDVfEqG4", 1),
        ("en-it_IT_#_speech_#_vid_3vpEaAjDgtI", 1),
        ("en-it_IT_#_speech_#_vid_6dP6bHX73_k", 1),
        ("en-it_IT_#_speech_#_vid_7Aw4Q46omiM", 1),
        ("en-it_IT_#_speech_#_vid_8I8msBYGNR4", 1),
        ("en-it_IT_#_speech_#_vid_8qZFupajBuo", 1),
    ]
)
PHASES.append(
    [
        ("en-it_IT_#_news_#_guardian.231314", 6),
        ("en-it_IT_#_news_#_newrepublic.com.12619", 4),
        ("en-it_IT_#_social_#_114300646822630777", 4),
        ("en-it_IT_#_social_#_114417630342798842", 4),
        ("en-it_IT_#_speech_#_vid_Qu8m9h-AC0I", 1),
        ("en-it_IT_#_speech_#_vid_RIjzeZ1xk1c", 1),
        ("en-it_IT_#_speech_#_vid_Si9CnnVI8sk", 1),
        ("en-it_IT_#_speech_#_vid_YTxmx8hyJtA", 1),
        ("en-it_IT_#_speech_#_vid__S0GOwEJXUA", 1),
    ]
)
PHASES.append(
    [
        ("en-it_IT_#_news_#_newrepublic.com.12623", 2),
        ("en-it_IT_#_news_#_nytimes.153341", 5),
        ("en-it_IT_#_social_#_114174389242714730", 6),
        ("en-it_IT_#_speech_#_vid_Lb7Dhf7oo4Y", 1),
        ("en-it_IT_#_speech_#_vid_M7v3ZfP4fm0", 1),
        ("en-it_IT_#_speech_#_vid_OmK3XTCDbbs", 1),
        ("en-it_IT_#_speech_#_vid_PJLN1kAzbyw", 1),
        ("en-it_IT_#_speech_#_vid_QgTkUWYd82M", 1),
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
    random.seed(0)
    for data_doc in data_phase:
        data_item: list[DocAll] = []
        for contrastive_k in [1, 2, 3, 4]:
            # we might have to dip into duplicate models, so we sample from the doubled list
            while True:
                models = []
                models += random.sample(MODELS, len(MODELS))
                models += random.sample([model + "'" for model in MODELS], 2)
                random.shuffle(models)
                models_groups = []
                # even for standard ESA add +2 duplicate for intra AA because they divide 10
                while sum(len(models) for models in models_groups) < 12:
                    models_groups.append(models[:contrastive_k])
                    models = models[contrastive_k:]
                assert len(models) == 0
                # ensure we don't show the same model twice in the same screen
                if all(
                    len({model.removesuffix("'") for model in model_group})
                    == contrastive_k
                    for model_group in models_groups
                ):
                    break

            for dup_i in [0, 1]:
                data_item_config: DocAll = []
                for model_group in models_groups:
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
                                    model: item["tgt"][model.removesuffix("'")].replace("\\n", "\n")
                                    for model in model_group
                                },
                                "item_id": item["doc"] + f"_#_s{item_i}_#_dup{dup_i}",
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
    print(f"Phase time: {time / (60 * 60):.1f} hours per all configurations")
    data_phases_out_flat.append(data_phase_out)

# %%
import pearmut.constants
import statistics

instructions = (
    pearmut.constants.PROTOCOL_INSTRUCTIONS["cESA"]
    + """
<style>
.output_candidate, .output_src {
    width: 345px !important;
    flex: unset;
}
.output_tgt {
    font-size: 10pt;
}
</style>
"""
)

with open("/home/vilda/pearmut/examples/tutorials/cesa_iten.json", "r") as f:
    data_tutorial = json.load(f)["data"][0]


for phase, data_phase_out in enumerate(data_phases_out_flat):
    num_users = 12
    data_phase_out: list[list[DocAll]]
    tasks = [[] for _ in range(num_users)]
    for data_item in data_phase_out:
        data_item: list[DocAll]
        # sort by fewest segments to most segments so that we assign the longest ones last (to make sure we don't end up with a long one at the end that doesn't fit in any user's queue)
        tasks.sort(key=lambda task: sum([len(doc) for doc in task]))
        tasks_to_expand = tasks[: len(data_item)]
        # make sure that we assign different conditions of the same doc to different users
        for task, data_item_config in zip(tasks_to_expand, data_item):
            data_item_config: DocAll
            task.extend(data_item_config)

    print([len(task) for task in tasks])
    time = statistics.mean(
        [
            sum(
                [
                    len(item["src_text"].split()) * len(item["tgt"]) * time_per_word
                    for doc in task
                    for item in doc
                ]
            )
            for task in tasks
        ]
    ) / (60 * 60)
    print(f"{time:.1f} hours per user")
    time = sum(
        [
            sum(
                [
                    len(item["src_text"].split()) * len(item["tgt"]) * time_per_word
                    for doc in task
                    for item in doc
                ]
            )
            for task in tasks
        ]
    ) / (60 * 60)
    print(f"{time:.1f} hours total")
    print()

    # shuffle queue
    for task in tasks:
        random.shuffle(task)

    # prepend tutorial
    tasks = [list(data_tutorial) + task for task in tasks]

    data_pearmut = {
        "info": {
            "assignment": "task-based",
            "protocol": "cESA",
            # "assets": {
            #     "source": "wmt25_genmt_assets/assets/en",
            #     "destination": "assets/wmt25_genmt_assets",
            # },
            "instructions": instructions,
        },
        "campaign_id": f"main_cESA_phase{phase + 1}_v4_enit",
        "data": tasks,
    }
    
    with open(f"humeval/main_enit_cESA_phase{phase + 1}.json", "w") as f:
        json.dump(data_pearmut, f, indent=4, ensure_ascii=False)

# %%
import os
import requests
import zipfile
import io

os.makedirs("humeval/wmt25_genmt_assets", exist_ok=True)
zipfile.ZipFile(
    io.BytesIO(
        requests.get(
            "https://data.statmt.org/wmt25/general-mt/wmt25_genmt_assets.zip"
        ).content
    )
).extractall("humeval/wmt25_genmt_assets")

# %%
import json

# sanity check

with open("humeval/main_cESA_phase1.json", "r") as f:
    data = json.load(f)["data"]

for user, data_user in enumerate(data):
    print(len(data_user))
    for doc in data_user:
        # if "item_name"
        print(
            doc[0]["item_id"].replace("_#_s0", "")
            + "_#_"
            + str(list(doc[0]["tgt"].keys()))
        )
    print()
