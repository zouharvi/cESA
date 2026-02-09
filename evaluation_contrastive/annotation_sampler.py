import abc
import random
import statistics
import numpy as np
import itertools
from evaluation_bandit import utils

Result = dict[str, float]


class Sampler(abc.ABC):
    def __init__(self, models: list[str], K: int = 2):
        self.models = models
        self.K = K

    @abc.abstractmethod
    def record_match(self, result: Result):
        pass

    @abc.abstractmethod
    def next_match(self) -> list[str]:
        pass

    @abc.abstractmethod
    def model_skill(self, model: str) -> float:
        pass


class SamplerRandom(Sampler):
    """
    Fully randomly select a match.
    """

    def __init__(self, models: list[str], K: int = 2):
        super().__init__(models, K)
        self.scores = {model: [] for model in models}
        self.random = random.Random(hash(tuple(models)))

    def model_skill(self, model: str) -> float:
        """
        Simple proportion of wins (or partial wins).
        """

        if not self.scores[model]:
            return np.mean(0.5)
        else:
            return np.mean(self.scores[model])

    def match_desireability(self, model1: str, model2: str):
        return self.random.random()

    def record_match(self, model1: str, model2: str, result: Result):
        assert result >= 0 and result <= 1
        self.scores[model1].append(result)
        self.scores[model2].append(1 - result)


class SamplerRandomUniform(SamplerRandom):
    """
    Randomly select a match that's balanced between all models.
    """

    def __init__(self, models: list[str], K: int = 2):
        super().__init__(models, K)
        self.queue = itertools.cycle(
            (tuple(sorted(pair)) for pair in itertools.combinations(models, 2))
        )
        self.next = next(self.queue)

    def match_desireability(self, model1: str, model2: str):
        if (model1, model2) == self.next:
            return 1.0
        else:
            return 0.0

    def record_match(self, model1: str, model2: str, result: Result):
        super().record_match(model1, model2, result)
        assert self.next == (model1, model2)
        self.next = next(self.queue)


class SamplerCloseUniform(Sampler):
    """
    Randomly select a match that's balanced between all models, but
    prioritizes matches between models with similar skill.
    Results from K-way evaluation are interpreted as unbiased scores.
    """

    def __init__(self, models: list[str], K: int = 2):
        super().__init__(models, K)
        self.scores = {model: [] for model in models}

    def next_match(self) -> list[str]:
        """
        Scan through all K-tuples in sorted models and select the K-tuple with lowest
        number of matches so far.
        """
        models_sorted = sorted(self.models, key=self.model_skill)
        best_tuple = None
        min_matches = float("inf")
        for i in range(len(models_sorted) - self.K + 1):
            candidate_tuple = tuple(models_sorted[i : i + self.K])
            total_matches = sum(len(self.scores[model]) for model in candidate_tuple)
            if total_matches < min_matches:
                min_matches = total_matches
                best_tuple = candidate_tuple

        return list(best_tuple)

    def record_match(self, result: Result):
        for model, score in result.items():
            self.scores[model].append(score)

    def model_skill(self, model: str) -> float:
        return statistics.mean(self.scores[model]) if self.scores[model] else 0


class SamplerCloseCluster(Sampler):
    """
    Randomly select a match that's balanced between all models, but
    prioritizes matches between models with similar skill.
    Results from K-way evaluation are interpreted as unbiased scores.
    """

    def __init__(self, models: list[str], K: int = 2):
        super().__init__(models, K)
        self.scores = {model: [] for model in models}

    def next_match(self) -> list[str]:
        """
        Scan through all K-tuples in sorted models and select the K-tuple with
        max pvalue (highest uncertainty) based on current skill evidence.
        """
        models_sorted = sorted(self.models, key=self.model_skill)
        candidate_tuples = []
        for i in range(len(models_sorted) - self.K + 1):
            candidate_tuple = tuple(models_sorted[i : i + self.K])
            pval = sum(
                utils.pval(self.scores[a], self.scores[b])
                for a, b in zip(candidate_tuple, candidate_tuple[1:])
            )
            if pval > 0.05 * (self.K - 1):
                candidate_tuples.append(candidate_tuple)

        if candidate_tuples:
            return random.choice(candidate_tuples)
        else:
            print("Warning: No candidate tuples with p-value > 0.05 found.")
            # randomly select a tuple
            return list(itertools.combinations(models_sorted, self.K))

    def record_match(self, result: Result):
        for model, score in result.items():
            self.scores[model].append(score)

    def model_skill(self, model: str) -> float:
        return statistics.mean(self.scores[model]) if self.scores[model] else 0
