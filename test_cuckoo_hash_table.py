import random
import string

import numpy as np

from cuckoo_hash_table import (
    CuckooHashTable,
    CuckooHashTableBucketed,
    CuckooHashTableDAry,
    CuckooHashTableDAryRandomWalk,
    CuckooHashTableStashed,
)


class TestCuckooHashTable:
    num_items = 100_000
    random_ints = np.random.randint(1, 1000000, num_items)
    random_strings = tuple(
        "".join(random.choice(string.ascii_letters) for _ in range(10))
        for _ in range(num_items)
    )

    def _test_with_random_insertions(self, table):
        for key, value in zip(self.random_ints, self.random_strings):
            table[key] = value
            assert key in table, key
            assert -key not in table

    def _test_with_random_insertions_then_deletions(self, table):
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

    def test_cuckoo_dary_cycle_eviction_policy_with_random_insertions(self):
        self._test_with_random_insertions(CuckooHashTableDAry())

    def test_cuckoo_dary_cycle_eviction_policy_with_random_insertions_then_deletions(
        self,
    ):
        self._test_with_random_insertions_then_deletions(CuckooHashTableDAry())

    def test_cuckoo_dary_random_walk_eviction_policy_with_random_insertions(self):
        self._test_with_random_insertions(CuckooHashTableDAryRandomWalk())

    def test_cuckoo_dary_random_walk_eviction_policy_with_random_insertions_then_deletions(
        self,
    ):
        self._test_with_random_insertions_then_deletions(
            CuckooHashTableDAryRandomWalk()
        )

    def test_cuckoo_with_random_insertions(self):
        self._test_with_random_insertions(CuckooHashTable())

    def test_cuckoo_with_random_insertions_then_deletions(self):
        self._test_with_random_insertions_then_deletions(CuckooHashTable())

    def test_cuckoo_bucketed_with_random_insertions(self):
        self._test_with_random_insertions(CuckooHashTableBucketed())

    def test_cuckoo_bucketed_with_random_insertions_then_deletions(self):
        self._test_with_random_insertions_then_deletions(CuckooHashTableBucketed())

    def test_cuckoo_stashed_with_random_insertions(self):
        self._test_with_random_insertions(CuckooHashTableStashed())

    def test_cuckoo_stashed_with_random_insertions_then_deletions(self):
        self._test_with_random_insertions_then_deletions(CuckooHashTableStashed())
