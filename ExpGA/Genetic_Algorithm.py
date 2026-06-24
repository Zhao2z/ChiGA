"""
Standard Genetic Algorithm for ExpGA — LIME-Guided Fairness Testing (Baseline).

This is a conventional GA without feature-importance-guided mutation.
It uses standard selection, crossover, and uniform random mutation,
relying on LIME to pre-filter seed samples rather than guiding mutation
during evolution.
"""

import numpy as np


class GA:
    """Standard Genetic Algorithm.

    Parameters
    ----------
    nums : array-like
        Initial population, shape (pop_size, n_features).
    bound : array-like
        Feature bounds, shape (n_features, 2) — (min, max) per feature.
    func : callable
        Fitness function f(sample) -> float.
    DNA_SIZE : int, optional
        Number of features (auto-detected if None).
    cross_rate : float
        Crossover probability (default 0.8).
    mutation : float
        Per-gene mutation probability (default 0.003).
    """

    def __init__(self, nums, bound, func, DNA_SIZE=None,
                 cross_rate=0.8, mutation=0.003):
        nums = np.array(nums)
        bound = np.array(bound)
        self.bound = bound

        min_nums, max_nums = np.array(list(zip(*bound)))
        self.var_len = var_len = max_nums - min_nums
        bits = np.ceil(np.log2(var_len + 1))

        if DNA_SIZE is None:
            DNA_SIZE = int(np.max(bits))
        self.DNA_SIZE = DNA_SIZE

        self.POP_SIZE = len(nums)
        self.POP = nums
        self.copy_POP = nums.copy()
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.func = func

    def get_fitness(self, non_negative=False):
        """Evaluate fitness for all individuals."""
        result = [self.func(self.POP[i]) for i in range(len(self.POP))]
        if non_negative:
            min_fit = np.min([r[0] for r in result])
            result = [(r[0] - min_fit, r[1]) for r in result]
        return result

    def select(self):
        """Fitness-proportional (roulette wheel) selection."""
        fitness = self.get_fitness()
        fit = np.array([item[0] for item in fitness])
        total = np.sum(fit)
        if total > 0:
            probs = fit / total
        else:
            probs = np.ones(len(fit)) / len(fit)

        self.POP = self.POP[
            np.random.choice(np.arange(self.POP.shape[0]),
                             size=self.POP.shape[0], replace=True, p=probs)
        ]

    def crossover(self):
        """Single-point crossover: with probability `cross_rate`,
        two parents exchange a random feature segment."""
        for people in self.POP:
            if np.random.rand() < self.cross_rate:
                partner = np.random.randint(0, self.POP.shape[0])
                start = np.random.randint(0, len(self.bound))
                length = np.random.randint(0, len(self.bound) - start)
                people[start:start + length] = \
                    self.POP[partner, start:start + length]

    def mutate(self):
        """Uniform random mutation: each feature is independently
        re-sampled from its valid range with probability `mutation`."""
        for people in self.POP:
            for point in range(self.DNA_SIZE):
                if np.random.rand() < self.mutation:
                    people[point] = np.random.randint(
                        self.bound[point][0], self.bound[point][1],
                    )

    def evolution(self):
        """Perform one full generation: selection → crossover → mutation."""
        self.select()
        self.crossover()
        self.mutate()
