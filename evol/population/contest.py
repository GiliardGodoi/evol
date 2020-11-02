from copy import copy
from itertools import cycle, islice
from math import ceil
from random import randint
from typing import (Any, Callable, Generator, Iterable, Iterator, List,
                    Optional, Sequence)

from evol import Individual
from evol.conditions import Condition
from evol.exceptions import StopEvolution

from .base import BasePopulation


class Contest:
    """A single contest among a group of competitors.

    This is encapsulated in an object so that scores for many sets of
    competitors can be evaluated concurrently without resorting to a
    dict or some similar madness to correlate score vectors with an
    ever-widening matrix of contests and competitors.

    :param competitors: Iterable of Individuals in this Contest.
    """

    def __init__(self, competitors: Iterable[Individual]):
        self.competitors = list(competitors)

    def assign_scores(self, scores: Sequence[float]) -> None:
        for competitor, score in zip(self.competitors, scores):
            competitor.fitness += score

    @property
    def competitor_chromosomes(self):
        return [competitor.chromosome for competitor in self.competitors]

    @classmethod
    def generate(cls, individuals: Sequence[Individual],
                 individuals_per_contest: int, contests_per_round: int) -> List['Contest']:
        """Generate contests for a round of evaluations.

        :param individuals: A sequence of competing Individuals.
        :param individuals_per_contest: Number of Individuals participating in each Contest.
        :param contests_per_round: Minimum number of contests each individual
            takes part in for each evaluation round. The actual number of contests
            per round is a multiple of individuals_per_contest.
        :return: List of Contests
        """
        contests = []
        n_rounds = ceil(contests_per_round / individuals_per_contest)
        for _ in range(n_rounds):
            offsets = [0] + [randint(0, len(individuals) - 1) for _ in range(individuals_per_contest - 1)]
            generators = [islice(cycle(individuals), offset, None) for offset in offsets]
            for competitors in islice(zip(*generators), len(individuals)):
                contests.append(Contest(competitors))
        return contests


class ContestPopulation(BasePopulation):
    """Population which is evaluated through contests.

    This variant of the Population is used when individuals cannot be
    evaluated on a one-by-one basis, but instead can only be compared to
    each other. This is typically the case for AI that performs some task
    (i.e. plays a game), but can be useful in many other cases.

    For each round of evaluation, each individual participates in a given
    number of contests, in which a given number of individuals take part.
    The resulting scores of these contests are summed to form the fitness.

    Since the fitness of an individual is dependent on the other individuals
    in the population, the fitness of all individuals is recalculated when
    new individuals are present, and the fitness of all individuals is reset
    when the population is modified (e.g. by calling survive, mutate etc).

    :param chromosomes: Iterable of initial chromosomes of the Population.
    :param eval_function: Function that reduces a chromosome to a fitness.
    :param maximize: If True, fitness will be maximized, otherwise minimized.
        Defaults to True.
    :param individuals_per_contest: Number of individuals that take part in
        each contest. The size of the population must be divisible by this
        number. Defaults to 2.
    :param contests_per_round: Minimum number of contests each individual
        takes part in for each evaluation round. The actual number of contests
        per round is a multiple of individuals_per_contest. Defaults to 10.
    :param generation: Generation of the Population. This is incremented after
        echo survive call. Defaults to 0.
    :param intended_size: Intended size of the Population. The population will
        be replenished to this size by .breed(). Defaults to the number of
        chromosomes provided.
    :param checkpoint_target: Target for the serializer of the Population. If
        a serializer is provided, this target is ignored. Defaults to None.
    :param serializer: Serializer for the Population. If None, a new
        SimpleSerializer is created. Defaults to None.
    :param concurrent_workers: If > 1, evaluate individuals in {concurrent_workers}
        separate processes. If None, concurrent_workers is set to n_cpus. Defaults to 1.
    """
    eval_function: Callable[..., Sequence[float]]  # This population expects a different eval signature

    def __init__(self,
                 chromosomes: Iterable,
                 eval_function: Callable[..., Sequence[float]],
                 maximize: bool = True,
                 individuals_per_contest=2,
                 contests_per_round=10,
                 generation: int = 0,
                 intended_size: Optional[int] = None,
                 checkpoint_target: Optional[int] = None,
                 serializer=None,
                 concurrent_workers: Optional[int] = 1):
        super().__init__(chromosomes=chromosomes,
                         eval_function=eval_function,
                         maximize=maximize,
                         generation=generation,
                         intended_size=intended_size,
                         checkpoint_target=checkpoint_target,
                         serializer=serializer,
                         concurrent_workers=concurrent_workers)
        self.contests_per_round = contests_per_round
        self.individuals_per_contest = individuals_per_contest

    def __copy__(self):
        result = self.__class__(chromosomes=[],
                                eval_function=self.eval_function,
                                maximize=self.maximize,
                                contests_per_round=self.contests_per_round,
                                individuals_per_contest=self.individuals_per_contest,
                                serializer=self.serializer,
                                intended_size=self.intended_size,
                                generation=self.generation,
                                concurrent_workers=1)
        result.individuals = [copy(individual) for individual in self.individuals]
        result.pool = self.pool
        result.concurrent_workers = self.concurrent_workers
        result.documented_best = None
        result.id = self.id
        return result

    def evaluate(self,
                 lazy: bool = False,
                 contests_per_round: Optional[int] = None,
                 individuals_per_contest: Optional[int] = None) -> 'ContestPopulation':
        """Evaluate the individuals in the population.

        This evaluates the fitness of all individuals. For each round of
        evaluation, each individual participates in a given number of
        contests, in which a given number of individuals take part.
        The resulting scores of these contests are summed to form the fitness.
        This means that the score of the individual is influenced by other
        chromosomes in the population.

        Note that in the `ContestPopulation` two settings are passed at
        initialisation which affect how we are evaluating individuals:
        contests_per_round and individuals_per_contest. You may overwrite them
        here if you wish.

        If lazy is True, the fitness is only evaluated when a fitness value
        is not yet known for all individuals.
        In most situations adding an explicit evaluation step is not needed, as
        lazy evaluation is implicitly included in the operations that need it
        (most notably in the survive operation).

        :param lazy: If True, do no re-evaluate the fitness if the fitness is known.
        :param contests_per_round: If set, overwrites the population setting for the
        number of contests there will be every round.
        :param individuals_per_contest: If set, overwrites the population setting for
        number of individuals to have in a contest during the evaluation.
        :return: self
        """
        if contests_per_round is None:
            contests_per_round = self.contests_per_round
        if individuals_per_contest is None:
            individuals_per_contest = self.individuals_per_contest
        if lazy and all(individual.fitness is not None for individual in self):
            return self
        for individual in self.individuals:
            individual.fitness = 0
        contests = Contest.generate(individuals=self.individuals, individuals_per_contest=individuals_per_contest,
                                    contests_per_round=contests_per_round)
        if self.pool is None:
            for contest in contests:
                contest.assign_scores(self.eval_function(*contest.competitor_chromosomes))
        else:
            f = self.eval_function  # We cannot refer to self in the map
            results = self.pool.map(lambda c: f(*c.competitor_chromosomes), contests)
            for result, contest in zip(results, contests):
                contest.assign_scores(result)
        return self

    def map(self, func: Callable[..., Individual], **kwargs) -> 'ContestPopulation':
        """Apply the provided function to each individual in the population.

        Resets the fitness of all individuals.

        :param func: A function to apply to each individual in the population,
            which when called returns a modified individual.
        :param kwargs: Arguments to pass to the function.
        :return: self
        """
        BasePopulation.map(self, func=func, **kwargs)
        self.reset_fitness()
        return self

    def filter(self, func: Callable[..., bool], **kwargs) -> 'ContestPopulation':
        """Add a filter step to the Evolution.

        Filters the individuals in the population using the provided function.
        Resets the fitness of all individuals.

        :param func: Function to filter the individuals in the population by,
            which returns a boolean when called on an individual.
        :param kwargs: Arguments to pass to the function.
        :return: self
        """
        BasePopulation.filter(self, func=func, **kwargs)
        self.reset_fitness()
        return self

    def survive(self,
                fraction: Optional[float] = None,
                n: Optional[int] = None,
                luck: bool = False) -> 'ContestPopulation':
        """Let part of the population survive.

        Remove part of the population. If both fraction and n are specified,
        the minimum resulting population size is taken. Resets the fitness
        of all individuals.

        :param fraction: Fraction of the original population that survives.
            Defaults to None.
        :param n: Number of individuals of the population that survive.
            Defaults to None.
        :param luck: If True, individuals randomly survive (with replacement!)
            with chances proportional to their fitness. Defaults to False.
        :return: self
        """
        BasePopulation.survive(self, fraction=fraction, n=n, luck=luck)
        self.reset_fitness()
        return self

    def reset_fitness(self):
        """Reset the fitness of all individuals."""
        for individual in self:
            individual.fitness = None
