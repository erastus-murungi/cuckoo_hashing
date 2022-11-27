import random
import string

import numpy as np

from cuckoo_hash_table import CuckooHashTable, CuckooHashTableDAry


class TestCuckooHashTable:
    num_items = 50_000
    random_ints = np.random.randint(1, 1000000, num_items)
    random_strings = tuple(
        "".join(random.choice(string.ascii_letters) for _ in range(10))
        for _ in range(num_items)
    )

    def test_with_random_insertions(self):
        table = CuckooHashTable()
        for key, value in zip(self.random_ints, self.random_strings):
            table[key] = value
            assert key in table, key
            assert -key not in table

    def test_with_random_insertions_then_deletions(self):
        table = CuckooHashTable()
        for index, (key, value) in enumerate(
            zip(self.random_ints, self.random_strings)
        ):
            if index and index % 5 == 0:
                del table[self.random_ints[index - 1]]
                assert self.random_ints[index - 1] not in table
            else:
                table[key] = value
                assert key in table, key
                assert -key not in table

    def test_dary_cuckoo_with_random_insertions(self):
        table = CuckooHashTableDAry()
        for key, value in zip(self.random_ints, self.random_strings):
            table[key] = value
            assert key in table, key
            assert -key not in table

    def test_dary_cuckoo_with_random_insertions_then_deletions(self):
        table = CuckooHashTableDAry()
        for index, (key, value) in enumerate(
            zip(self.random_ints, self.random_strings)
        ):
            if index and index % 5 == 0:
                del table[self.random_ints[index - 1]]
                assert self.random_ints[index - 1] not in table
            else:
                table[key] = value
                assert key in table, key
                assert -key not in table
