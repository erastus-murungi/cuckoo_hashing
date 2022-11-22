from array import array
from collections.abc import MutableMapping
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Final, Hashable, Optional

from hash_family import HashFamilyTabulation


class CuckooHashTable(MutableMapping):
    """
    A class to represent on ordered dictionary which resolves collisions using cuckoo hashing

    self._entries is a list of CuckooHashTable.Entry objects or None (to represent deleted entries).

    self._index is composed of 2 hash tables of equal length.
    self._index[i] holds the indices of entries in self._entries
    The size of on index table is self.index_size

    - Type of each value in self._index[i] is varies on self.index_size:
        * int8  for          self.index_size <= 128
        * int16 for 256   <= self.index_size <= 2**15
        * int32 for 2**16 <= self.index_size <= 2**31
        * int64 for 2**32 <= self.index_size
        * pyint for 2**64 <= ...


    `column_index` will be used as selector of hash_functions and index_tables

    """

    @dataclass(slots=True)
    class Entry:
        """
        An entry is a simple key-value pair
        """

        key: Hashable
        value: Any

        def match_key(self, candidate_key: Hashable) -> bool:
            """
            This method matches some candidate key with the key belonging to self

            This method is an attempt to speed up the matching process using referential equality checking,
            when the keys could potentially complex objects whose hashes are expensive to compute.

            Parameters
            ----------
            :param candidate_key: a key to try and match
            :return: true if `key` matches `self.key` and false otherwise
            """
            return candidate_key is self.key or candidate_key == self.key

    #  a constant representing the minimum size of an index
    MIN_SIZE: Final[int] = 8
    # a constant representing the lack of a value in the index table
    NO_ENTRY: Final[int] = -1
    # the assumed word-size in the platform architecture
    WORD_SIZE: Final[int] = 64

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
        self._gen_index(CuckooHashTable.MIN_SIZE)
        self._entries = []
        self._hash_family = HashFamilyTabulation(2)
        self._hash_family.gen()
        self._num_entries = 0
        self._rehashing_depth_limit = rehashing_depth_limit
        self._rehashes = 0
        self._allowed_rehash_attempts = allowed_rehash_attempts

    @property
    def index_size(self) -> int:
        return len(self._index[0])

    def _gen_index(self, size=None):
        if size is None:
            size = int(self.index_size * 2)

        if size <= 0x7F:  # signed 8-bit
            index = array("b", [CuckooHashTable.NO_ENTRY]) * size
        elif size <= 0x7FFF:  # signed 16-bit
            index = array("h", [CuckooHashTable.NO_ENTRY]) * size
        elif size <= 0x7FFFFFFF:  # signed 32-bit
            index = array("l", [CuckooHashTable.NO_ENTRY]) * size
        elif size <= 0x7FFFFFFFFFFFFFFF:  # signed 64-bit
            index = array("q", [CuckooHashTable.NO_ENTRY]) * size
        else:
            index = [CuckooHashTable.NO_ENTRY] * size
        self._index = (
            index,
            index[:],
        )

    def _get_entry_specific_column(
        self, column_index: int, key: Hashable
    ) -> Optional[Entry]:
        offset = self._hash_family(column_index, key, self.index_size)
        entry_index = self._index[column_index][offset]
        if (
            entry_index != CuckooHashTable.NO_ENTRY
            and self._entries[entry_index] is not None
            and self._entries[entry_index].match_key(key)
        ):
            return self._entries[entry_index]
        return None

    def _get_entry(self, key) -> Optional[Entry]:
        return self._get_entry_specific_column(
            0, key
        ) or self._get_entry_specific_column(1, key)

    def __contains__(self, key) -> bool:
        return self._get_entry(key) is not None

    def __getitem__(self, key):
        if (entry := self._get_entry(key)) is None:
            return None
        return entry.value

    def _gen_and_append_entry(self, key, value):
        self._num_entries += 1
        self._entries.append(CuckooHashTable.Entry(key, value))

    @staticmethod
    def usable_fraction(x: int) -> int:
        return x >> 1

    def _should_grow_index(self) -> bool:
        return len(self._entries) >= self.usable_fraction(self.index_size)

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
        offset = self._hash_family(column_index, key, self.index_size)

        if index[offset] == CuckooHashTable.NO_ENTRY:
            index[offset] = entry_index
            return None

        next_entry_index = index[offset]
        index[offset] = entry_index
        return self._entries[next_entry_index].key, next_entry_index

    def _insert(self, key, entry_index: int) -> bool:
        column_index = cycle((0, 1))
        for _ in range(self._rehashing_depth_limit):
            maybe_key_entry_index = self._maybe_get_key(
                key, entry_index, next(column_index)
            )
            if maybe_key_entry_index is None:
                return False
            key, entry_index = maybe_key_entry_index

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
            self._gen_index(self.index_size)

        self._rehashes += 1
        self._hash_family.gen()

        for entry_index, entry in self._enumerate_entries():
            if self._insert_entry_index(entry.key, entry_index):
                return

    def __delitem__(self, key):
        for k, index in enumerate(self._index):
            offset = self._hash_family(k, key, self.index_size)
            entry_index = index[offset]
            if entry_index != CuckooHashTable.NO_ENTRY and self._entries[
                entry_index
            ].match_key(key):
                index[offset] = CuckooHashTable.NO_ENTRY
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
