import os
from evaluation_contrastive import utils
import cohere


class LLMJudgeCohere:
    def __init__(self):
        # read api key from COHERE_API_KEY environment variable
        api_key = os.getenv("COHERE_API_KEY")
        if api_key is None:
            raise ValueError("COHERE_API_KEY environment variable is not set")

        self.client = cohere.Client(api_key=api_key)

    @utils.cache(cache_dir="cohere")
    def score(self, src: str, tgt: list[str]) -> list[float]:
        assert isinstance(src, str)
        assert isinstance(tgt, list)
        # TODO: implement call
        import time

        time.sleep(1)
        return [len(tgt) for tgt in tgt]


LLMJudge = LLMJudgeCohere
