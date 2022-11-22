import random
import struct
from array import array
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Final, Optional

from hash_family import HashFamilyShift

MIN_SIZE: Final[int] = 8
NO_ENTRY: Final[int] = -1
WORD_SIZE: Final[int] = 64  # word size

is64: Final[bool] = (struct.calcsize("P") * 8) == 64


def usable_fraction(x: int) -> int:
    return x >> 1


@dataclass(slots=True)
class Entry:
    key: Any
    value: Any

    def match(self, key):
        if key is self.key:
            return True
        return key == self.key


class CuckooHashTable(MutableMapping):
    __slots__ = (
        "_index",
        "_entries",
        "_num_entries",
        "_hash_family",
        "_rehashing_depth_limit",
        "_rehashes",
        "_allowed_rehash_attempts",
    )

    def __init__(
        self, *, rehashing_depth_limit: int = 150, allowed_rehash_attempts: int = 20
    ):
        self._gen_index(MIN_SIZE)
        self._entries = []
        self._hash_family = HashFamilyShift(2)
        self._hash_family.gen()
        self._num_entries = 0
        self._rehashing_depth_limit = rehashing_depth_limit
        self._rehashes = 0
        self._allowed_rehash_attempts = allowed_rehash_attempts

    @property
    def n_buckets(self) -> int:
        return len(self._index[0])

    def _gen_index(self, size=None):
        if size is None:
            size = int(self.n_buckets * 2)

        if size <= 0x7F:  # signed 8-bit
            index = array("b", [NO_ENTRY]) * size
        elif size <= 0x7FFF:  # signed 16-bit
            index = array("h", [NO_ENTRY]) * size
        elif size <= 0x7FFFFFFF:  # signed 32-bit
            index = array("l", [NO_ENTRY]) * size
        elif is64 and size <= 0x7FFFFFFFFFFFFFFF:  # signed 64-bit
            index = array("q", [NO_ENTRY]) * size
        else:
            index = [NO_ENTRY] * size
        self._index = (
            index,
            index[:],
        )

    def _get_entry_specific_column(self, key, column_index: int) -> Optional[Entry]:
        offset = self._hash_family(key, column_index, self.n_buckets)
        entry_index = self._index[column_index][offset]
        if (
            entry_index != NO_ENTRY
            and self._entries[entry_index] is not None
            and self._entries[entry_index].match(key)
        ):
            return self._entries[entry_index]
        return None

    def _get_entry(self, key) -> Optional[Entry]:
        return self._get_entry_specific_column(
            key, 0
        ) or self._get_entry_specific_column(key, 1)

    def __contains__(self, key) -> bool:
        return self._get_entry(key) is not None

    def __getitem__(self, key):
        if (entry := self._get_entry(key)) is None:
            return None
        return entry.value

    def _gen_and_append_entry(self, key, value):
        self._num_entries += 1
        self._entries.append(Entry(key, value))

    def _should_grow_index(self) -> bool:
        return len(self._entries) >= usable_fraction(self.n_buckets)

    def _insert_entry_index(self, key, entry_index: int):
        if self._should_grow_index():
            self._rehash(expand=True)
            return True
        else:
            return self._insert(key, entry_index)

    def __setitem__(self, key, value):
        if (entry := self._get_entry(key)) is not None:
            # overwrite old value
            entry.value = value
        else:
            self._gen_and_append_entry(key, value)
            self._insert_entry_index(key, len(self._entries) - 1)

    def _maybe_get_key(
        self, key, entry_index: int, column_index: int
    ) -> Optional[tuple[Entry, int]]:
        index = self._index[column_index]
        offset = self._hash_family(key, column_index, self.n_buckets)

        if index[offset] == NO_ENTRY:
            index[offset] = entry_index
            return None

        next_entry_index = index[offset]
        index[offset] = entry_index
        return self._entries[next_entry_index].key, next_entry_index

    def _insert(self, key, entry_index: int) -> bool:
        k = 0
        for _ in range(self._rehashing_depth_limit):
            maybe_key_entry_index = self._maybe_get_key(key, entry_index, k)
            if maybe_key_entry_index is None:
                return False
            key, entry_index = maybe_key_entry_index
            k = not k

        if self._rehashes >= self._allowed_rehash_attempts:
            self._rehashes = 0
            self._rehash(expand=True)
        else:
            self._rehash()
        return True

    def _rehash(self, *, expand=False):
        # Rehash using a new hash function and recurse to try insert again.
        if expand:
            self._gen_index()
        else:
            self._gen_index(self.n_buckets)

        self._rehashes += 1
        self._hash_family.gen()

        for entry_index, entry in self._enumerate_entries():
            if self._insert_entry_index(entry.key, entry_index):
                return

    def __delitem__(self, key):
        for k, index in enumerate(self._index):
            offset = self._hash_family(key, k, self.n_buckets)
            entry_index = index[offset]
            if entry_index != NO_ENTRY and self._entries[entry_index].match(key):
                index[offset] = NO_ENTRY
                self._entries[entry_index] = None
                self._num_entries -= 1
                return True
        raise KeyError(f"{key} not found")

    def _enumerate_entries(self):
        for entry_index, entry in enumerate(self._entries):
            if entry is not None:
                yield entry_index, entry

    def _nonnull_entries(self):
        yield from filter(None, self._entries)

    def __iter__(self):
        yield from (entry.key for entry in self._nonnull_entries())

    def items(self):
        yield from ((entry.key, entry.value) for entry in self._nonnull_entries())

    def __len__(self):
        return self._num_entries

    def __repr__(self):
        if not self._num_entries:
            return "{}"
        else:
            return "\n".join(
                f"{index:<10} {key!r:<15} | {value!r}"
                for index, (key, value) in enumerate(self.items())
            )


if __name__ == "__main__":
    import string

    import numpy as np

    # np.random.seed(0)
    # random.seed(0)

    n = 15000
    rand_nums = np.random.randint(1, 1000000, n)
    rand_strings = (
        "".join(random.choice(string.ascii_letters) for _ in range(10))
        for _ in range(n)
    )
    ck = CuckooHashTable()
    for i, (_key, v) in enumerate(zip(rand_nums, rand_strings)):
        if i and i % 5 == 0:
            print(f"pop {i}: {(_key, v)} n_buckets = {ck.n_buckets}")
            ck.pop(rand_nums[i - 1])
            assert rand_nums[i - 1] not in ck
            continue
        print(f"insert {i}: {(_key, v)} n_buckets = {ck.n_buckets}")
        ck[_key] = v

        assert _key in ck, _key
        assert -_key not in ck
    print(ck)
