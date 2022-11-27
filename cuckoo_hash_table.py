import random
from array import array
from collections.abc import MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from itertools import cycle
from pprint import pprint
from typing import Any, Final, Hashable, Optional

from hash_family import HashFamily, HashFamilyTabulation


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


class CuckooHashTable(MutableMapping):
    """
    A class to represent on ordered dictionary which resolves collisions using cuckoo hashing

    self._entries is a list of CuckooHashTable.Entry objects or None (to represent deleted entries).

    self._index is composed of 2 hash tables of equal length.
    self._index[i] holds the indices of entries in self._entries
    The size of on index table is self.index_size

    - Type of each value in self._buckets[i] is varies on self.index_size:
        * int8  for          self.index_size <= 128
        * int16 for 256   <= self.index_size <= 2**15
        * int32 for 2**16 <= self.index_size <= 2**31
        * int64 for 2**32 <= self.index_size <= 2**63
        * pyint for 2**64 <= ...


    `column_index` will be used as selector of hash_functions and index_tables

    """

    #  a constant representing the minimum size of an index
    MIN_SIZE: Final[int] = 8
    # a constant representing the lack of a value in the index table
    NO_ENTRY: Final[int] = -1
    # the assumed word-size in the platform architecture
    WORD_SIZE: Final[int] = 64

    __slots__ = (
        "buckets",
        "_entries",
        "_num_entries",
        "_hash_family",
        "_rehashes",
        "_allowed_rehash_attempts",
        "_usable_fraction",
    )

    def __init__(
        self,
        *,
        allowed_rehash_attempts: int = 20,
        usable_fraction: float = 0.5,
        hash_family: HashFamily = None,
    ):
        self._hash_family = hash_family or HashFamilyTabulation(2)
        self._hash_family.gen()

        self._entries = []

        self._num_entries = 0
        self._rehashes = 0
        self._allowed_rehash_attempts = allowed_rehash_attempts
        self._usable_fraction = usable_fraction
        self._gen_index(CuckooHashTable.MIN_SIZE)

    @property
    def entries_per_bucket(self) -> int:
        return len(self.buckets[0])

    def _gen_index(self, entries_per_bucket=None, growth_factor: float = 2):
        if entries_per_bucket is None:
            entries_per_bucket = int(self.entries_per_bucket * growth_factor)
        capacity = entries_per_bucket * self._hash_family.size()

        if capacity <= 0x7F:  # signed 8-bit
            buckets = array("b", [CuckooHashTable.NO_ENTRY]) * entries_per_bucket
        elif capacity <= 0x7FFF:  # signed 16-bit
            buckets = array("h", [CuckooHashTable.NO_ENTRY]) * entries_per_bucket
        elif capacity <= 0x7FFFFFFF:  # signed 32-bit
            buckets = array("l", [CuckooHashTable.NO_ENTRY]) * entries_per_bucket
        elif capacity <= 0x7FFFFFFFFFFFFFFF:  # signed 64-bit
            buckets = array("q", [CuckooHashTable.NO_ENTRY]) * entries_per_bucket
        else:
            buckets = [CuckooHashTable.NO_ENTRY] * entries_per_bucket
        self.buckets = (buckets,) + tuple(
            buckets[:] for _ in range(self._hash_family.size() - 1)
        )

    def _get_entry_specific_column(
        self, column_index: int, key: Hashable
    ) -> Optional[Entry]:
        offset = self._hash_family(column_index, key, self.entries_per_bucket)
        entry_index = self.buckets[column_index][offset]
        if (
            entry_index != CuckooHashTable.NO_ENTRY
            and self._entries[entry_index] is not None
            and self._entries[entry_index].match_key(key)
        ):
            return self._entries[entry_index]
        return None

    def _get_entry(self, key) -> Optional[Entry]:
        for column_index in range(self._hash_family.size()):
            if (
                maybe_entry := self._get_entry_specific_column(column_index, key)
            ) is not None:
                return maybe_entry
        return None

    def __contains__(self, key) -> bool:
        return self._get_entry(key) is not None

    def __getitem__(self, key):
        if (entry := self._get_entry(key)) is None:
            return None
        return entry.value

    def _gen_and_append_entry(self, key, value):
        self._num_entries += 1
        self._entries.append(Entry(key, value))

    def load_factor(self) -> int:
        return len(self._entries) / (self._hash_family.size() * self.entries_per_bucket)

    def _should_grow_index(self) -> bool:
        return self.load_factor() >= self._usable_fraction

    def _insert_entry_index(self, key: Hashable, entry_index: int) -> bool:
        """
        Parameters
        ----------
        :param key: A key to insert into the buckets. It is needed to compute hashes
        :param entry_index: the entry index to insert
        :return: True if a rehash has been triggered
        """

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
        index = self.buckets[column_index]
        offset = self._hash_family(column_index, key, self.entries_per_bucket)

        if index[offset] == CuckooHashTable.NO_ENTRY:
            index[offset] = entry_index
            return None

        next_entry_index = index[offset]
        index[offset] = entry_index
        return self._entries[next_entry_index].key, next_entry_index

    def _get_eviction_policy(self):
        c = cycle(range(self._hash_family.size()))
        return lambda current_pos: next(c)

    def _insert(self, key: Hashable, entry_index: int) -> bool:
        column_index = 0
        find_next = self._get_eviction_policy()
        max_insert_failures = max(
            self._hash_family.size() * 2, 6 * self._num_entries.bit_length()
        )
        for _ in range(max_insert_failures):
            maybe_key_entry_index = self._maybe_get_key(key, entry_index, column_index)
            if maybe_key_entry_index is None:
                return False
            key, entry_index = maybe_key_entry_index
            column_index = find_next(column_index)

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
            self._gen_index(self.entries_per_bucket)

        self._rehashes += 1
        self._hash_family.gen()

        for entry_index, entry in self._enumerate_entries():
            if self._insert_entry_index(entry.key, entry_index):
                return

    def __delitem__(self, key: Hashable):
        for column_index, index in enumerate(self.buckets):
            offset = self._hash_family(column_index, key, self.entries_per_bucket)
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


class CuckooHashTableDAry(CuckooHashTable):
    DEFAULT_D = 4

    def __init__(self, hash_family=None, usable_fraction=0.9):
        super().__init__(
            hash_family=hash_family or HashFamilyTabulation(self.DEFAULT_D),
            usable_fraction=usable_fraction,
        )


class CuckooHashTableDAryRandomWalk(CuckooHashTableDAry):
    """
    A variation of cuckoo hashing where the table to evict and element from is chosen at random
    """

    def __init__(self, hash_family=None, usable_fraction=0.9):
        super().__init__(
            hash_family=hash_family or HashFamilyTabulation(self.DEFAULT_D),
            usable_fraction=usable_fraction,
        )

    def _get_eviction_policy(self):
        def random_walk_policy(current_index):
            index = random.randint(0, self._hash_family.size() - 1)
            while index == current_index:
                index = random.randint(0, self._hash_family.size() - 1)
            return index

        return random_walk_policy


class CuckooHashTableBucketed(CuckooHashTable):
    DEFAULT_SLOTS_PER_BUCKET = 2

    def __init__(self, slots_per_bucket: Optional[int] = None):
        self.slots_per_bucket = slots_per_bucket or self.DEFAULT_SLOTS_PER_BUCKET
        super().__init__()

    def _gen_index(self, entries_per_bucket=None, growth_factor: float = 2):
        if entries_per_bucket is None:
            entries_per_bucket = int(self.entries_per_bucket * growth_factor)
        capacity = entries_per_bucket * self._hash_family.size()
        type_codes = ["b", "h", "l", "q"]
        item_sizes = [0x7F, 0x7FFF, 0x7FFFFFFF, 0x7FFFFFFFFFFFFFFF]
        index = 0
        while capacity >= item_sizes[index]:
            index += 1
        buckets = [array(type_codes[index]) for _ in range(entries_per_bucket)]
        self.buckets = (buckets,) + tuple(
            deepcopy(buckets) for _ in range(self._hash_family.size() - 1)
        )

    def _get_entry_specific_column(
        self, column_index: int, key: Hashable
    ) -> Optional[Entry]:
        offset = self._hash_family(column_index, key, self.entries_per_bucket)
        slots = self.buckets[column_index][offset]
        for entry_index in slots:
            if self._entries[entry_index] is not None and self._entries[
                entry_index
            ].match_key(key):
                return self._entries[entry_index]
        return None

    def _maybe_get_key(
        self, key, entry_index: int, column_index: int
    ) -> Optional[tuple[Entry, int]]:
        offset = self._hash_family(column_index, key, self.entries_per_bucket)
        slots = self.buckets[column_index][offset]
        if len(slots) == self.slots_per_bucket:
            next_entry_index = slots[-1]
            slots[-1] = entry_index
            return self._entries[next_entry_index].key, next_entry_index
        else:
            slots.append(entry_index)
            return None

    def __delitem__(self, key: Hashable):
        for column_index, index in enumerate(self.buckets):
            offset = self._hash_family(column_index, key, self.entries_per_bucket)
            slots = index[offset]
            for entry_index in slots[:]:
                if self._entries[entry_index].match_key(key):
                    slots.remove(entry_index)
                    self._entries[entry_index] = None
                    self._num_entries -= 1
                    return True
        raise KeyError(f"{key} not found")
