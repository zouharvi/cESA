import os
from evaluation_contrastive import utils
import cohere
import json
import random

PROMPT = """
You are a judge for machine translation quality.
Given a source text and a list of various translation, assign a score from 0 to 100 to each.
Try to be as objective as possible.

{SOURCE}

{TRANSLATIONS}

Output only the list of scores (same as number of translations) such that it can be parsed by json.loads().
"""


class LLMJudgeCohere:
    def __init__(self):
        # read api key from COHERE_API_KEY environment variable
        api_key = os.getenv("COHERE_API_KEY")
        if api_key is None:
            raise ValueError("COHERE_API_KEY environment variable is not set")

        self.client = cohere.ClientV2(
            api_key=api_key,
            log_warning_experimental_features=False,
        )

    def score(self, src: str, tgt: list[str], shuffle=False, cache=None) -> list[float]:
        assert isinstance(src, str)
        assert isinstance(tgt, list)

        tgt_frozen = sorted(tgt) if shuffle else tgt

        if result := utils.cache_read(
            cache=cache, cache_dir="cohere", src=src, tgt=tgt_frozen
        ):
            return result

        if shuffle:
            shuffle_i = list(range(len(tgt)))
            random.shuffle(shuffle_i)
            shuffle_i_inv = {j: i for i, j in enumerate(shuffle_i)}
            tgt = [tgt[i] for i in shuffle_i]

        prompt = PROMPT.format(
            SOURCE="Source:\n" + src,
            TRANSLATIONS="\n\n".join(
                [f"Translation {i + 1}:\n{t}" for i, t in enumerate(tgt)]
            ),
        )
        response = self.client.chat(
            messages=[
                {"role": "user", "content": prompt},
            ],
            model="command-a-reasoning-08-2025",
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "scores": {
                            "type": "array",
                            "items": {"type": "number", "min": 0, "max": 100},
                        }
                    },
                    "required": ["scores"],
                },
            },
        )
        for content in response.message.content:
            if content.type == "text":
                scores = json.loads(content.text)["scores"]
                if shuffle:
                    scores = [scores[shuffle_i_inv[i]] for i in range(len(scores))]

        utils.cache_write(
            scores, cache=cache, cache_dir="cohere", src=src, tgt=tgt_frozen
        )
        return scores


LLMJudge = LLMJudgeCohere
