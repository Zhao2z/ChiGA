"""
Genetic Algorithm for ChiGA — Chi-squared Guided Fairness Testing.

This GA extends the standard operators (selection, crossover, mutation) with
feature-importance-guided mutation.  The χ² scores from the training data
determine which features are more (or less) likely to be mutated, steering the
search toward regions of the input space where discriminatory behaviour is
more probable.
"""

import numpy as np


class GA:
    """Genetic Algorithm with chi-squared-guided mutation.

    Parameters
    ----------
    nums : array-like
        Initial population, shape (pop_size, n_features).
    bound : array-like
        Feature bounds, shape (n_features, 2) — (min, max) per feature.
    func : callable
        Fitness function f(sample) -> (float, int) where the int is 1 if
        discriminatory, 0 otherwise.
    datasetName : str
        Dataset identifier (used for logging).
    sensitive_param : int
        1-based index of the sensitive attribute.
    DNA_SIZE : int, optional
        Number of features (set to len(bound) by default).
    cross_rate : float
        Crossover probability (default 0.8).
    mutation : float
        Base mutation probability (default 0.003).
    mutation_positive : array-like, optional
        Per-feature probabilities for mutating *discriminatory* samples
        (derived from χ² scores).
    mutation_negative : array-like, optional
        Per-feature probabilities for mutating *non-discriminatory* samples
        (inverse of χ² scores).
    """

    def __init__(self, nums, bound, func, datasetName, sensitive_param,
                 DNA_SIZE=None, cross_rate=0.8, mutation=0.003,
                 mutation_positive=None, mutation_negative=None):
        nums = np.array(nums)
        bound = np.array(bound)

        self.bound = bound
        self.sensitive_param = sensitive_param
        self.EPOCHS = 0

        min_nums, max_nums = np.array(list(zip(*bound)))
        self.var_len = var_len = max_nums - min_nums
        bits = np.ceil(np.log2(var_len + 1))

        if DNA_SIZE is None:
            DNA_SIZE = int(np.max(bits))
        self.DNA_SIZE = DNA_SIZE

        self.POP_SIZE = len(nums)

        # Population tracking
        self.total_samples: set = set()
        self.total_samples_list: list = []
        self.disc_samples: set = set()
        self.disc_samples_list: list = []

        self.POP = nums
        self.SIZE = self.POP.shape[0]
        self.copy_POP = nums.copy()
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.func = func
        self.mutation_positive = mutation_positive
        self.mutation_negative = mutation_negative
        self.IsDisc = np.array([])

    def get_fitness(self, non_negative=False):
        """Evaluate fitness for all individuals in the population.

        Returns
        -------
        fitness_list : list of float
            Fitness value per individual.
        """
        fitness_list = []
        disc_list = []
        for i in range(len(self.POP)):
            f, p = self.func(self.POP[i])
            fitness_list.append(f)
            disc_list.append(p)

            sample = np.reshape([int(x) for x in self.POP[i]], (1, -1))
            self.total_samples.add(tuple(map(tuple, sample)))
            if p == 1:
                self.disc_samples.add(tuple(map(tuple, sample)))

        self.IsDisc = np.array(disc_list)

        if non_negative:
            min_fit = np.min(fitness_list, axis=0)
            fitness_list = [v - min_fit for v in fitness_list]
        return fitness_list

    def select(self):
        """Fitness-proportional (roulette wheel) selection."""
        fitness = self.get_fitness()
        fit = np.array(fitness)
        total = np.sum(fit)
        if total > 0:
            probs = fit / total
        else:
            probs = np.ones(len(fit)) / len(fit)

        self.POP = self.POP[
            np.random.choice(np.arange(self.POP.shape[0]),
                             size=self.SIZE, replace=True, p=probs)
        ]
        self.get_fitness()

    def mutate(self):
        """χ²-guided mutation.

        For non-discriminatory samples: mutate features with low χ² scores
        (mutation_negative distribution) — explore new regions.

        For discriminatory samples: mutate features with high χ² scores
        (mutation_positive distribution) — refine around promising regions.
        """
        for index in range(len(self.POP)):
            if self.IsDisc[index] == 0:
                # Non-discriminatory: explore by mutating low-importance features
                n_mutations = np.random.randint(1, 5)
                mutation_idx = np.random.choice(
                    np.arange(self.DNA_SIZE), size=n_mutations,
                    replace=False, p=self.mutation_negative,
                )
                for a in mutation_idx:
                    self.POP[index][a] = np.random.randint(
                        self.bound[a][0], self.bound[a][1],
                    )
            else:
                # Discriminatory: refine by mutating high-importance features
                n_mutations = np.random.randint(1, 5)
                mutation_idx = np.random.choice(
                    np.arange(self.DNA_SIZE), size=n_mutations,
                    replace=False, p=self.mutation_positive,
                )
                for a in mutation_idx:
                    self.POP[index][a] = np.random.randint(
                        self.bound[a][0], self.bound[a][1],
                    )

    def crossover(self):
        """Single-point crossover between pairs of individuals.

        With probability `cross_rate`, two parents exchange a random
        contiguous segment of features.
        """
        new_list = self.POP.tolist()
        for people in self.POP:
            if np.random.rand() < self.cross_rate:
                partner_idx = np.random.randint(0, self.POP.shape[0])
                start = np.random.randint(0, len(self.bound))
                length = np.random.randint(0, len(self.bound) - start)

                child_a = np.array(people).tolist()
                child_b = np.array(self.POP[partner_idx]).reshape(-1).tolist()

                child_a[start:start + length], child_b[start:start + length] = \
                    child_b[start:start + length], child_a[start:start + length]

                new_list.append(child_a)
                new_list.append(child_b)

        self.POP = np.array(new_list)

    def evolution(self):
        """Perform one full generation: selection → mutation → crossover."""
        self.select()
        self.mutate()
        self.crossover()
