import random
import string

import numpy as np
import pytest as pytest

from cuckoo_hash_table import CuckooHashTable


@pytest.fixture
def num_items():
    return 100000


def test_with_random_insertions(num_items):
    random_ints = np.random.randint(1, 1000000, num_items)
    random_strings = (
        "".join(random.choice(string.ascii_letters) for _ in range(10))
        for _ in range(num_items)
    )
    table = CuckooHashTable()
    for key, value in zip(random_ints, random_strings):
        table[key] = value
        assert key in table, key
        assert -key not in table


def test_with_random_insertions_then_deletions(num_items):
    random_ints = np.random.randint(1, 1000000, num_items)
    random_strings = (
        "".join(random.choice(string.ascii_letters) for _ in range(10))
        for _ in range(num_items)
    )
    table = CuckooHashTable()
    for index, (key, value) in enumerate(zip(random_ints, random_strings)):
        if index and index % 5 == 0:
            del table[random_ints[index - 1]]
            assert random_ints[index - 1] not in table
        else:
            table[key] = value
            assert key in table, key
            assert -key not in table
