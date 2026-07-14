# %%

import glob
import re
import os

os.chdir(os.path.dirname(__file__) + "/../")

re_url = re.compile(r'"src": (.*?)src=\\"(.*?)\\"')
observed_assets = set()
for f in glob.glob("humeval/campaigns/*.json"):
    with open(f, "r") as f:
        data_txt = f.read()
    urls = re_url.findall(data_txt)
    for url in urls:
        url = url[1].removeprefix("https://vilda.net/t/wmt25/assets/en/")
        observed_assets.add(url)

os.makedirs("humeval/campaigns/assets_pruned", exist_ok=True)

for asset in observed_assets:
    asset_src = f"humeval/wmt25_genmt_assets/assets/en/{asset}"
    os.makedirs(os.path.dirname(f"humeval/campaigns/assets_pruned/{asset}"), exist_ok=True)
    os.system(f"cp {asset_src} humeval/campaigns/assets_pruned/{asset}")


# %%

with open("../humeval/annotations/main_enja_old.json", "r") as f:
    data_txt = f.read()

data_txt = (
    data_txt
    .replace("main_cESA_phase1_v2", "main_cESA_phase1_v2_enja")
    .replace("main_cESA_phase2_v2", "main_cESA_phase2_v2_enja")
    .replace("main_cESA_phase3_v2", "main_cESA_phase3_v2_enja")
    .replace("main_cESA_phase4_v2", "main_cESA_phase4_v2_enja")
)
with open("../humeval/annotations/main_enja.json", "w") as f:
    f.write(data_txt)