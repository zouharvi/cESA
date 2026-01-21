# %%

from evaluation_contrastive import utils
from evaluation_contrastive import llm_judge
import random
import tqdm

# dict_keys(['cs-de_DE', 'cs-uk_UA', 'en-ar_EG', 'en-bho_IN', 'en-cs_CZ', 'en-et_EE', 'en-is_IS', 'en-it_IT', 'en-ja_JP', 'en-ko_KR', 'en-mas_KE', 'en-ru_RU', 'en-sr_Cyrl_RS', 'en-uk_UA', 'en-zh_CN', 'ja-zh_CN'])
data = utils.load_data()["en-cs_CZ"]


def evaluate_kway(data, k_way=4, cache=None):
    metric = llm_judge.LLMJudge()
    data_out = []
    for item in tqdm.tqdm(data[:20]):
        models = list(item["tgt"].keys())
        random.Random(item["src"]).shuffle(models)

        model_scores = {}
        scores_log = []
        # chunk into 4-tuples
        for i in tqdm.tqdm(range(0, len(models), k_way)):
            print(i)
            models_chunk = models[i : i + k_way]
            scores = metric.score(
                src=item["src"],
                tgt=[item["tgt"][m] for m in models_chunk],
                cache="en-cs_CZ",
                shuffle=False,
            )
        scores_log.append({model: score for model, score in zip(models_chunk, scores)})
        for model, score in zip(models_chunk, scores):
            model_scores[model] = score

    data_out.append(
        {
            "src": item["src"],
            "tgt": item["tgt"],
            "scores": model_scores,
            "scores_log": scores_log,
        }
    )
    return data_out


data_4way = evaluate_kway(data, k_way=4)
data_1way = evaluate_kway(data, k_way=1)


# %%
from evaluation_contrastive import llm_judge

metric = llm_judge.LLMJudge()

response = metric.score(
    src="Moje zbraně jsou moje slova.",
    tgt=["My words are my weapons.", "My weapons are my words."],
    cache=None,
)


metric.score(
    src="Moje zbraně jsou moje slova.",
    tgt=["My words are my weapons.", "My weapons are my words."],
    cache=None,
)

metric.score(
    src="Moje zbraně jsou moje slova.",
    tgt=[
        "weapons weapons",
        "My words are my weapons.",
        "My weapons are my words.",
        "CCCC",
    ],
    cache="en-cs_CZ",
    shuffle=True,
)

# %%
