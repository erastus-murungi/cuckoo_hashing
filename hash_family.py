import random
from abc import ABC, abstractmethod
from typing import Callable, Final, Hashable

import numpy as np

p: Final[int] = 0x1FFFFFFFFFFFFFFF  # Mersenne prime (2^61 - 1)

WORD_SIZE: Final[int] = 64  # word size


def gen_multiplier_increment_pair(
    count: int, multiplier_gen: Callable[[], int]
) -> set[tuple[int, int]]:
    pairs = set()
    while len(pairs) != count:
        increment = random.randint(0, p)
        pairs.add((multiplier_gen(), increment))
    return pairs


class HashFamily(ABC):
    def __init__(self, size: int):
        self._size = size

    @abstractmethod
    def gen(self):
        pass

    @abstractmethod
    def __call__(self, column_index: int, item: Hashable, table_size: int) -> int:
        pass

    def size(self):
        return self._size


class HashFamilyModular(HashFamily):
    def __init__(self, size: int):
        super().__init__(size)
        self.a = None
        self.b = None

    def __call__(self, column_index, item, table_size):
        a, b = self.a[column_index], self.b[column_index]
        h = (a * hash(item) + b) & p
        return h % table_size

    def gen(self):
        self.a, self.b = gen_multiplier_increment_pair(
            self.size(), lambda: random.randint(1, p)
        )


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

    def __call__(self, column_index, item, table_size):
        assert self.is_power_of_two(
            table_size
        ), f"table size must be a power of 2, {table_size} is not"

        m = table_size.bit_length() - 1
        mask = (1 << (WORD_SIZE + m)) - 1
        a, b = self.a[column_index], self.b[column_index]
        h = (a * hash(item) + b) & mask
        return h >> WORD_SIZE

    def gen(self):
        self.a, self.b = gen_multiplier_increment_pair(
            self.size(), lambda: self.get_rand_odd(WORD_SIZE)
        )


class HashFamilyTabulation(HashFamily):
    def __init__(self, size: int):
        super().__init__(size)
        self.tables = None

    def __call__(self, column_index: int, item: Hashable, table_size: int) -> int:
        x = hash(item)
        h0 = x & 0xFF
        h1 = (x >> 8) & 0xFF
        h2 = (x >> 16) & 0xFF
        h3 = (x >> 24) & 0xFF
        h4 = (x >> 32) & 0xFF
        h5 = (x >> 40) & 0xFF
        h6 = (x >> 48) & 0xFF
        h7 = (x >> 56) & 0xFF
        t = self.tables[column_index]
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
            % table_size
        )

    def gen(self):
        self.tables = np.random.randint(
            0, 0xFFFFFFFFFFFFFFFF, size=(self.size(), 8, 256), dtype=np.uint64
        )
