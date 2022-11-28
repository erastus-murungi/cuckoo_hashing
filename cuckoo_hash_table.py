import random
from array import array
from collections.abc import MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from itertools import cycle
from typing import (
    Any,
    Callable,
    Final,
    Hashable,
    Optional,
    MutableSequence,
    SupportsIndex,
)

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

    Notes
    -----
        * ``self._entries`` is a list of ``Entry`` objects or ``None`` (to represent deleted entries).

        * ``self._tables`` is composed of (d=2) hash tables of each with ``self.buckets_per_table`` buckets.

        * ``self._tables[i]`` holds the indices of entries in self._entries

        * The size of each table is ``self.buckets_per_table``

        * ``table_index`` will be used as selector of hash_functions and tables

        * Type of each value in each table is varies on self.buckets_per_table
            * int8  for          self.buckets_per_table <= 128
            * int16 for 256   <= self.buckets_per_table <= 2**15
            * int32 for 2**16 <= self.buckets_per_table <= 2**31
            * int64 for 2**32 <= self.buckets_per_table <= 2**63
            * pyint for 2**64 <= ...


    """

    #  a constant representing the minimum size of an index
    MIN_BUCKETS_PER_TABLE: Final[int] = 8
    # a constant representing the lack of a value in the index table
    NO_ENTRY: Final[int] = -1
    # the assumed word-size in the platform architecture
    WORD_SIZE: Final[int] = 64

    __slots__ = (
        "_tables",
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
        hash_family: Optional[HashFamily] = None,
    ):
        """
        Parameters
        ----------
        allowed_rehash_attempts: int, optional
                                 The number of failed rehashing attempts before the tables are expanded. The
                                 default is 20 it must be a number >= 1
        usable_fraction: float, optional
                                 The maximum load factor of the table. When this is exceeded, the table is expanded
                                 This value has to be in the range (0, 1). The default is 0.5
        hash_family: HashFamily, optional
                                 A hash family. The number of tables is read from the ``hash_family.size()``
                                 method. This value is optional, by default it is initialized with a tabulation hashing
                                 family of size 2

        """
        self._hash_family: HashFamily = hash_family or HashFamilyTabulation(2)
        # the gen method must always be invoked before HashFamily objects can be used
        self._hash_family.gen()
        # a list of the entries in our hash table
        self._entries: list[Optional[Entry]] = []
        # the number of entries in our hash table, might not be necessarily equal to
        # len(self._entries) due to presence of ``None`` values in self._entries
        self._num_entries: int = 0
        # a counter to maintain the number of times rehashes have been triggered since the last
        # rehash caused expansion
        self._rehashes: int = 0
        self._allowed_rehash_attempts: int = allowed_rehash_attempts
        self._usable_fraction: float = usable_fraction
        self._gen_index(CuckooHashTable.MIN_BUCKETS_PER_TABLE)
        self._validate_attributes()

    def _validate_attributes(self):
        self._validate_usable_fraction()
        self._validate_number_of_tables()
        self._validate_allowed_rehash_attempts()

    def _validate_allowed_rehash_attempts(self):
        if self._allowed_rehash_attempts < 1:
            raise TypeError(
                f"Allowed rehash attempts must be >= 1: not {self._allowed_rehash_attempts}"
            )

    def _validate_usable_fraction(self):
        if self._usable_fraction <= 0 or self._usable_fraction >= 1:
            raise TypeError(
                f"The usable fraction should be a value in the range (0, 1): not {self._usable_fraction}"
            )

    def _validate_number_of_tables(self):
        if self._hash_family.size() < 2:
            raise TypeError(
                f"The size of the hash family must be >= 2: not {self._hash_family.size()}"
            )

    @property
    def buckets_per_table(self) -> int:
        return len(self._tables[0])

    def _gen_index(self, buckets_per_table=None, growth_factor: float = 2):
        if buckets_per_table is None:
            buckets_per_table = int(self.buckets_per_table * growth_factor)
        capacity = buckets_per_table * self._hash_family.size()

        if capacity <= 0x7F:  # signed 8-bit
            table = array("b", [CuckooHashTable.NO_ENTRY]) * buckets_per_table
        elif capacity <= 0x7FFF:  # signed 16-bit
            table = array("h", [CuckooHashTable.NO_ENTRY]) * buckets_per_table
        elif capacity <= 0x7FFFFFFF:  # signed 32-bit
            table = array("l", [CuckooHashTable.NO_ENTRY]) * buckets_per_table
        else:  # signed 64-bit
            table = array("q", [CuckooHashTable.NO_ENTRY]) * buckets_per_table
        self._tables: tuple[MutableSequence[SupportsIndex], ...] | tuple[
            list[MutableSequence[SupportsIndex]], ...
        ] = (table,) + tuple(table[:] for _ in range(self._hash_family.size() - 1))

    def _get_entry_specific_table(
        self, table_index: int, key: Hashable
    ) -> Optional[Entry]:
        bucket_index = self._hash_family(table_index, key, self.buckets_per_table)
        entry_index = self._tables[table_index][bucket_index]
        if (
            entry_index != CuckooHashTable.NO_ENTRY
            and self._entries[entry_index] is not None
            and self._entries[entry_index].match_key(key)
        ):
            return self._entries[entry_index]
        return None

    def _get_entry(self, key: Hashable) -> Optional[Entry]:
        for table_index in range(self._hash_family.size()):
            if (
                maybe_entry := self._get_entry_specific_table(table_index, key)
            ) is not None:
                return maybe_entry
        return None

    def __contains__(self, key: Hashable) -> bool:
        return self._get_entry(key) is not None

    def __getitem__(self, key: Hashable) -> Any:
        if (entry := self._get_entry(key)) is None:
            return None
        return entry.value

    def _gen_and_append_entry(self, key: Hashable, value: Any) -> None:
        self._num_entries += 1
        self._entries.append(Entry(key, value))

    def load_factor(self) -> int:
        return len(self._entries) / (self._hash_family.size() * self.buckets_per_table)

    def _should_grow_table(self) -> bool:
        return self.load_factor() >= self._usable_fraction

    def _insert_entry_index(self, key: Hashable, entry_index: int) -> bool:
        """
        Parameters
        ----------
        :param key: A key to insert into the buckets. It is needed to compute hashes
        :param entry_index: the entry index to insert
        :return: True if a rehash has been triggered
        """

        if self._should_grow_table():
            self._rehash(expand=True)
            return True
        else:
            return self._insert(key, entry_index)

    def __setitem__(self, key: Hashable, value: Any):
        if (entry := self._get_entry(key)) is not None:
            # overwrite old value
            entry.value = value
        else:
            self._gen_and_append_entry(key, value)
            self._insert_entry_index(key, len(self._entries) - 1)
            self._rehashes = 0

    def _maybe_get_key(
        self, key, entry_index: int, table_index: int
    ) -> Optional[tuple[Hashable, SupportsIndex]]:
        bucket_index: int = self._hash_family(table_index, key, self.buckets_per_table)
        buckets: MutableSequence[SupportsIndex] = self._tables[table_index]

        if buckets[bucket_index] == CuckooHashTable.NO_ENTRY:
            buckets[bucket_index] = entry_index
            return None

        evicted_entry_index = buckets[bucket_index]
        buckets[bucket_index] = entry_index
        return self._entries[evicted_entry_index].key, evicted_entry_index

    def _get_eviction_policy(self) -> Callable[[int], int]:
        c = cycle(range(self._hash_family.size()))
        return lambda table_index: next(c)

    def _insert(self, key: Hashable, entry_index: int) -> bool:
        table_index = 0
        find_next = self._get_eviction_policy()
        max_insert_failures = max(
            self._hash_family.size() * 2, 6 * self._num_entries.bit_length()
        )
        for _ in range(max_insert_failures):
            maybe_key_entry_index = self._maybe_get_key(key, entry_index, table_index)
            if maybe_key_entry_index is None:
                return False
            key, entry_index = maybe_key_entry_index
            table_index = find_next(table_index)

        if self._rehashes >= self._allowed_rehash_attempts:
            self._rehash(expand=True)
        else:
            self._rehash()
        return True

    def _rehash(self, *, expand: bool = False):
        # Rehash using a new hash function and recurse to try insert again.
        if expand:
            self._gen_index()
            # _rehashes counter must be reset here because this rehash will be caused by an expansion
            self._rehashes = 0
        else:
            self._gen_index(self.buckets_per_table)

        self._rehashes += 1
        self._hash_family.gen()

        for entry_index, entry in self._enumerate_entries():
            if self._insert_entry_index(entry.key, entry_index):
                return

    def __delitem__(self, key: Hashable):
        for table_index, buckets in enumerate(self._tables):
            bucket_index = self._hash_family(table_index, key, self.buckets_per_table)
            entry_index: SupportsIndex = buckets[bucket_index]
            if entry_index != CuckooHashTable.NO_ENTRY:
                entry = self._entries[entry_index]
                if entry is not None:
                    if entry.match_key(key):
                        buckets[
                            bucket_index
                        ] = CuckooHashTable.NO_ENTRY  # type : ignore
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

    def __len__(self) -> int:
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
    DEFAULT_D: Final[int] = 4

    def __init__(
        self, hash_family: Optional[HashFamily] = None, usable_fraction: float = 0.9
    ):
        super().__init__(
            hash_family=hash_family or HashFamilyTabulation(self.DEFAULT_D),
            usable_fraction=usable_fraction,
        )


class CuckooHashTableDAryRandomWalk(CuckooHashTableDAry):
    """
    A variation of cuckoo hashing where the table to evict and element from is chosen at random
    """

    def _get_eviction_policy(self) -> Callable[[int], int]:
        def random_walk_policy(current_table_index: int) -> int:
            index = random.randint(0, self._hash_family.size() - 1)
            while index == current_table_index:
                index = random.randint(0, self._hash_family.size() - 1)
            return index

        return random_walk_policy


class CuckooHashTableBucketed(CuckooHashTable):
    DEFAULT_SLOTS_PER_BUCKET = 2

    def __init__(self, slots_per_bucket: Optional[int] = None):
        self._slots_per_bucket = slots_per_bucket or self.DEFAULT_SLOTS_PER_BUCKET
        super().__init__()

    def _gen_index(
        self, buckets_per_table: Optional[int] = None, growth_factor: float = 2
    ):
        if buckets_per_table is None:
            buckets_per_table = int(self.buckets_per_table * growth_factor)
        capacity = buckets_per_table * self._hash_family.size()
        type_codes = ["b", "h", "l", "q"]
        item_sizes = [0x7F, 0x7FFF, 0x7FFFFFFF, 0x7FFFFFFFFFFFFFFF]
        index = 0
        while capacity >= item_sizes[index]:
            index += 1
        table = [array(type_codes[index]) for _ in range(buckets_per_table)]

        self.buckets = (table,) + tuple(
            deepcopy(table) for _ in range(self._hash_family.size() - 1)
        )

    def _get_entry_specific_table(
        self, table_index: int, key: Hashable
    ) -> Optional[Entry]:
        bucket_index = self._hash_family(table_index, key, self.buckets_per_table)
        bucket = self.buckets[table_index][bucket_index]
        for entry_index in bucket:
            entry = self._entries[entry_index]
            if entry is not None:
                if entry.match_key(key):
                    return entry
        return None

    def _maybe_get_key(
        self, key, entry_index: int, table_index: int
    ) -> Optional[tuple[Hashable, int]]:
        bucket_index = self._hash_family(table_index, key, self.buckets_per_table)
        bucket = self.buckets[table_index][bucket_index]
        if len(bucket) == self._slots_per_bucket:
            evicted_entry_index = bucket[-1]
            bucket[-1] = entry_index
            entry = self._entries[evicted_entry_index]
            if entry is not None:
                return entry.key, evicted_entry_index
            else:
                raise RuntimeError(f"self._entries[{entry_index}] is None")
        else:
            bucket.append(entry_index)
            return None

    def __delitem__(self, key: Hashable):
        for column_index, buckets in enumerate(self.buckets):
            bucket_index = self._hash_family(column_index, key, self.buckets_per_table)
            bucket = buckets[bucket_index]
            for entry_index in bucket[:]:
                if self._entries[entry_index] is not None and self._entries[  # type: ignore
                    entry_index
                ].match_key(
                    key
                ):
                    bucket.remove(entry_index)
                    self._entries[entry_index] = None
                    self._num_entries -= 1
                    return True
        raise KeyError(f"{key} not found")
