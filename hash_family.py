import random
from abc import ABC, abstractmethod
from typing import Callable, Final

p: Final[int] = 0x1FFFFFFFFFFFFFFF  # Mersenne prime (2^61 - 1)


def gen_multiplier_increment(
    count: int, multiplier_gen: Callable[[], int]
) -> set[tuple[int, int]]:
    pairs = set()
    while len(pairs) != count:
        increment = random.randint(0, p)
        pairs.add((multiplier_gen(), increment))
    return pairs


class HashFamily(ABC):
    def __init__(self, size):
        self._size = size

    @abstractmethod
    def gen(self):
        pass

    @abstractmethod
    def __call__(self, x, k: int, table_size: int) -> int:
        pass

    def size(self):
        return self._size


class HashFamilyModular(HashFamily):
    def __init__(self, size: int):
        super().__init__(size)
        self.a = None
        self.b = None

    def hash(self, item, k):
        a, b = self.a[k], self.b[k]
        return (a * hash(item) + b) % p

    def __call__(self, item, k, table_size):
        a, b = self.a[k], self.b[k]
        h = (a * hash(item) + b) % p
        return h % table_size

    def gen(self):
        self.a, self.b = gen_multiplier_increment(
            self.size(), lambda: random.randint(1, p)
        )


class HashFamilyShift(HashFamily):
    WORD_SIZE: Final[int] = 64  # word size

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

    def hash(self, item, k):
        a, b = self.a[k], self.b[k]
        return a * hash(item) + b

    def __call__(self, item, k, table_size):
        assert self.is_power_of_two(
            table_size
        ), f"table size must be a power of 2, {table_size} is not"
        m = table_size.bit_length() - 1
        mask = (1 << (self.WORD_SIZE + m)) - 1
        a, b = self.a[k], self.b[k]
        h = (a * hash(item) + b) & mask
        return h >> self.WORD_SIZE

    def gen(self):
        self.a, self.b = gen_multiplier_increment(
            self.size(), lambda: self.get_rand_odd(self.WORD_SIZE)
        )


if __name__ == "__main__":
    hm = HashFamilyShift(2)
    hm.gen()
    for i in range(100):
        print(hm(i, 0, 128))
