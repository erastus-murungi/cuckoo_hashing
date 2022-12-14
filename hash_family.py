import random
from abc import ABC, abstractmethod
from typing import Callable, Final, Hashable

import numpy as np

p: Final[int] = 0x1FFFFFFFFFFFFFFF  # Mersenne prime (2^61 - 1)

WORD_SIZE: Final[int] = 64  # word size


def gen_multiplier_increment_tuples(
    count: int, multiplier_gen: Callable[[], int]
) -> set[tuple[int, int]]:
    tuples: set[tuple[int, int]] = set()
    while len(tuples) != count:
        increment = random.randint(0, p)
        tuples.add((multiplier_gen(), increment))
    return tuples


class HashFamily(ABC):
    def __init__(self, size: int):
        self._size = size

    @abstractmethod
    def gen(self):
        pass

    @abstractmethod
    def __call__(self, table_index: int, item: Hashable, buckets_per_index: int) -> int:
        pass

    def size(self):
        return self._size


class HashFamilyModular(HashFamily):
    def __init__(self, size: int):
        super().__init__(size)
        self.a = None
        self.b = None

    def __call__(self, table_index, item, buckets_per_index):
        a, b = self.a[table_index], self.b[table_index]
        h = (a * hash(item) + b) & p
        return h % buckets_per_index

    def gen(self):
        self.a, self.b = gen_multiplier_increment_tuples(
            self.size(), lambda: random.randint(1, p)
        )

    def __repr__(self):
        return f"(ax + b) % p where a = {self.a}, b = {self.b}, p = {p}"


class HashFamilyShift(HashFamily):
    def __init__(self, size: int):
        super().__init__(size)
        self.a = None
        self.b = None

    @staticmethod
    def get_rand_odd(w: int) -> int:
        x = random.getrandbits(w)
        while not (x & 1):
            x = random.getrandbits(w)
        return x

    @staticmethod
    def is_power_of_two(x: int) -> int:
        return not (x & (x - 1))

    def __call__(self, table_index, item, buckets_per_index):
        assert self.is_power_of_two(
            buckets_per_index
        ), f"table size must be a power of 2, {buckets_per_index} is not"

        m = buckets_per_index.bit_length() - 1
        mask = (1 << (WORD_SIZE + m)) - 1
        a, b = self.a[table_index], self.b[table_index]
        h = (a * hash(item) + b) & mask
        return h >> WORD_SIZE

    def gen(self):
        self.a, self.b = gen_multiplier_increment_tuples(
            self.size(), lambda: self.get_rand_odd(WORD_SIZE)
        )


class HashFamilyTabulation(HashFamily):
    def __init__(self, size: int):
        super().__init__(size)
        self.tables: np.ndarray = np.empty(0)

    def __call__(self, table_index: int, item: Hashable, buckets_per_index: int) -> int:
        x = hash(item)
        h0 = x & 0xFF
        h1 = (x >> 8) & 0xFF
        h2 = (x >> 16) & 0xFF
        h3 = (x >> 24) & 0xFF
        h4 = (x >> 32) & 0xFF
        h5 = (x >> 40) & 0xFF
        h6 = (x >> 48) & 0xFF
        h7 = (x >> 56) & 0xFF
        t = self.tables[table_index]
        return (
            int(
                t[0][h0]
                ^ t[1][h1]
                ^ t[2][h2]
                ^ t[3][h3]
                ^ t[4][h4]
                ^ t[5][h5]
                ^ t[6][h6]
                ^ t[7][h7]
            )
            % buckets_per_index
        )

    def gen(self):
        self.tables = np.random.randint(
            0, 0xFFFFFFFFFFFFFFFF, size=(self.size(), 8, 256), dtype=np.uint64
        )

    def __repr__(self):
        return f"Tabulation hashing where T = {self.tables}"


class HashFamilyPolynomial(HashFamily):
    """
    https://cseweb.ucsd.edu/classes/fa13/cse290-b/notes-lecture2.pdf
    """

    def __init__(self, size: int, k: int = 3):
        super().__init__(size)
        self.coefficients = np.empty(0)
        self.k = k

    def gen(self):
        self.coefficients = np.random.randint(
            1, np.iinfo(np.uint64).max, dtype=np.uint64, size=self.k
        )

    def __call__(self, table_index: int, item: Hashable, buckets_per_index: int) -> int:
        hash_item = hash(item)
        xi = hash_item
        res = 0
        for coefficient in self.coefficients:
            res = res + (coefficient * xi) % p
            xi += hash_item
        return res % buckets_per_index
