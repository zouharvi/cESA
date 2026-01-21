# %%

from evaluation_contrastive import utils
from evaluation_contrastive import llm_judge
import importlib

importlib.reload(llm_judge)
importlib.reload(utils)

# dict_keys(['cs-de_DE', 'cs-uk_UA', 'en-ar_EG', 'en-bho_IN', 'en-cs_CZ', 'en-et_EE', 'en-is_IS', 'en-it_IT', 'en-ja_JP', 'en-ko_KR', 'en-mas_KE', 'en-ru_RU', 'en-sr_Cyrl_RS', 'en-uk_UA', 'en-zh_CN', 'ja-zh_CN'])
data = utils.load_data()["en-cs_CZ"]


# %%

metric = llm_judge.LLMJudge()
for item in data[:20]:
    metric.score(src=item["src"], tgt=[item["tgt"]["Claude-4"]], cache="en-cs_CZ")
