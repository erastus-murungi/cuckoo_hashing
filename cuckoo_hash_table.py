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
    MutableSequence,
    Optional,
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

    def key_matches(self, candidate_key: Hashable) -> bool:
        """
        This method matches some candidate key with the key belonging to self

        This method is an attempt to speed up the matching process using referential equality checking,
        when the keys could potentially complex objects whose hashes are expensive to compute.

        A key can only match an entry if
            1) this entry is alive
            2) the candidate key equals self.key either by reference or by value

        Parameters
        ----------
        :param candidate_key: a key to try and match
        :return: true if `key` matches `self.key` and false otherwise
        """
        return (
            self.is_alive() and candidate_key is self.key or candidate_key == self.key
        )

    def is_alive(self) -> bool:
        """
        Returns true if an entry is "alive"

        Dead entries should be skipped when checking whether (key, value) pairs exist in our dictionary

        In this implementation, an entry is alive if it is not equal to the tombstone constant
        Maybe to avoid the use of global constants, an extra bit could be attached to each entry
        to indicate whether an entry is alive

        """
        return self is not TOMBSTONE


# this is used to represent a deleted entry
TOMBSTONE: Final[Entry] = Entry("", None)

# these are used to check which int size to use for representing entry indices in our buckets
type_codes, item_sizes = ("b", "h", "l", "q"), (
    0x7F,
    0x7FFF,
    0x7FFFFFFF,
    0x7FFFFFFFFFFFFFFF,
)

Index = array
BucketIndex = list[Index]


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
        "_max_insert_failures_log_constant",
        "_buckets_per_table",
    )

    def __init__(
        self,
        *,
        allowed_rehash_attempts: int = 20,
        usable_fraction: float = 0.5,
        hash_family: Optional[HashFamily] = None,
        max_insert_failures_log_constant: int = 10,
    ):
        """
        Parameters
        ----------
        allowed_rehash_attempts : int, optional
            The number of failed rehashing attempts before the tables are expanded.
            The default is 20 it must be a number >= 1
        usable_fraction : float, optional
            The maximum load factor of the table. When this is exceeded, the table is expanded.
            This value has to be in the range (0, 1). The default is 0.5.
        hash_family : HashFamily, optional
            A hash family. The number of tables is read from the ``hash_family.size()``
            method. This value is optional, by default it is initialized with a tabulation hashing
            family of size 2
        max_insert_failures_log_constant : int, optional
            Used to control the number of to times before evictions due to inserts trigger a rehash.
            max_insert_failures = max_insert_failures_log_constant * log_2(num_entries)
            This implementation expects it to range from mis expected to range from 1 to 1000
            By default it is set to 10.
        """

        self._hash_family: HashFamily = hash_family or HashFamilyTabulation(2)
        # the gen method must always be invoked before HashFamily objects can be used
        self._hash_family.gen()
        # a list of the entries in our hash table
        self._entries: list[Entry] = []
        # the number of entries in our hash table, might not be necessarily equal to
        # len(self._entries) due to presence of ``None`` values in self._entries
        self._num_entries: int = 0
        # a counter to maintain the number of times rehashes have been triggered since the last
        # rehash caused expansion
        self._rehashes: int = 0
        # The number of buckets in each of the d tables we have
        self._buckets_per_table = CuckooHashTable.MIN_BUCKETS_PER_TABLE
        self._allowed_rehash_attempts: int = allowed_rehash_attempts
        self._usable_fraction: float = usable_fraction
        self._max_insert_failures_log_constant = max_insert_failures_log_constant
        self._gen_tables(CuckooHashTable.MIN_BUCKETS_PER_TABLE)
        self._validate_attributes()

    def _validate_attributes(self):
        """
        A central location to validate all our instantiated variables
        Overriding methods should always call this method
        """
        self._validate_usable_fraction()
        self._validate_number_of_tables()
        self._validate_allowed_rehash_attempts()
        self._validate_max_insert_failures_log_constant()

    def _validate_max_insert_failures_log_constant(self):
        if (
            self._max_insert_failures_log_constant < 1
            or self._max_insert_failures_log_constant > 1000
        ):
            raise ValueError(
                f"Expected max_insert_failures_log_constant to be in the inclusive range [1, 1000]: "
                f"not {self._max_insert_failures_log_constant}"
            )

    def _validate_allowed_rehash_attempts(self):
        if self._allowed_rehash_attempts < 1:
            raise ValueError(
                f"Allowed rehash attempts must be >= 1: not {self._allowed_rehash_attempts}"
            )

    def _validate_usable_fraction(self):
        if self._usable_fraction <= 0 or self._usable_fraction >= 1:
            raise ValueError(
                f"The usable fraction should be a value in the range (0, 1): not {self._usable_fraction}"
            )

    def _validate_number_of_tables(self):
        if self._hash_family.size() < 2:
            raise ValueError(
                f"The size of the hash family must be >= 2: not {self._hash_family.size()}"
            )

    @property
    def capacity(self):
        # the capacity (total number of slots) in our hashtable
        return self._buckets_per_table * self._hash_family.size()

    def _update_buckets_per_table(
        self, buckets_per_table: Optional[int], growth_factor: float
    ):
        # updates _buckets_per_table to be the correct value
        self._buckets_per_table = (
            int(self._buckets_per_table * growth_factor)
            if buckets_per_table is None
            else buckets_per_table
        )

    def _get_correct_type_code(self) -> str:
        # find the right word size to using a linear search
        # we only have 4 item-sizes so there is no need to use binary search
        index = 0
        while self.capacity >= item_sizes[index]:
            index += 1
        return type_codes[index]

    def _gen_tables(
        self, buckets_per_table: Optional[int] = None, growth_factor: float = 2
    ):
        """
        Generates (d=2) tables to store our entry indices and stores them in self._tables

        Parameters
        ----------
        buckets_per_table : int, optional
            The number of buckets in each table. This number should be a power of 2 when using some hash families
            such as multiply-shift scheme described by Dietzfelbinger et al. in 1997.
            Typically, this is parameter is expected to be used during initialization to a specific size
        growth_factor : float, optional
            The factor by which to expand each table, by default it is 2.

        Raises
        ------
        ValueError
            If both `buckets_per_table` and `growth_factor` are passed into the function

        Returns
        -------
        None to indicate success

        Notes
        -----
        This method stores entry indices in `array.array` and not list
        This is to take advantage of the fact most entry indices can be represented using integer sizes smaller than
        the one provided by python. The savings in space can be very important when using Cuckoo tables

        """

        self._update_buckets_per_table(buckets_per_table, growth_factor)
        type_code = self._get_correct_type_code()
        table = array(type_code, [CuckooHashTable.NO_ENTRY]) * self._buckets_per_table
        self._tables: tuple[Index, ...] = (table,) + tuple(
            table[:] for _ in range(self._hash_family.size() - 1)
        )

    def _entry_at_index_matches_key(
        self, entry_index: SupportsIndex, candidate_key: Hashable
    ) -> bool:
        """
        Try to match the entry at index `entry_index` with `candidate_key`

        Parameters
        ----------
        entry_index : SupportsIndex
            The index in self._entries of the entry we are inserting
        candidate_key : Hashable
            The key corresponding to ``entry_index``

        Returns
        -------
        True:
            If there is entry_index points to an entry that is alive and
            if the key attached to the entry at index `entry_index` matches candidate_key
        False:
            otherwise
        """

        if entry_index == CuckooHashTable.NO_ENTRY:
            return False
        return self._entries[entry_index].key_matches(candidate_key)

    def _remove_entry(self, entry_index: SupportsIndex):
        """
        Remove an entry from self._entries.

        Parameters
        ----------
        entry_index: SupportsIndex
            The index in self._entries of the entry we are inserting

        Returns
        -------
        None when the steps are completed

        Notes
        -----
        This composes of only 2 steps:

        1) "Killing" an entry by replacing it with the TOMBSTONE
        2) Decrementing the number of entries by 1

        """
        self._entries[entry_index] = TOMBSTONE
        self._num_entries -= 1

    def _get_entry_specific_table(
        self, table_index: int, key: Hashable
    ) -> Optional[Entry]:
        """
        Try to retrieve an entry corresponding to a given key from a specific table

        Parameters
        ----------
        table_index : int
            The index in self._tables of the table from which we are trying to
            retrieve the entry corresponding the given key

        key : Hashable
            A key whose corresponding entry we try and retrieve into self._tables[table_index]

        Returns
        -------
        Entry
            The entry if was found
        None
            If the entry wasn't found
        """
        bucket_index = self._hash_family(table_index, key, self._buckets_per_table)
        entry_index = self._tables[table_index][bucket_index]
        if self._entry_at_index_matches_key(entry_index, key):
            return self._entries[entry_index]
        return None

    def _get_entry(self, key: Hashable) -> Optional[Entry]:
        """
        Retrieve an entry corresponding to the given key from the dictionary

        Parameters
        ---------
        key : Hashable
            A key whose corresponding entry we try and retrieve into self._tables[table_index]

        Returns
        -------
        Entry
            The entry if was found
        None
            If the entry wasn't found

        Notes
        ----
        Unlike ``_get_entry_specific_table``, this method looks at the given tables
        """

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
        """
        Create and append an entry to self._entries
        """
        self._num_entries += 1
        self._entries.append(Entry(key, value))

    def load_factor(self) -> int:
        return len(self._entries) / (self._hash_family.size() * self._buckets_per_table)

    def _should_grow_table(self) -> bool:
        """
        Return True if we should grow our table

        We decide True if our load factor is >= the usable fraction

        Returns
        ------
        True
            If we should grow our table
        False:
            Otherwise
        """
        return self.load_factor() >= self._usable_fraction

    def _insert_entry_index(self, key: Hashable, entry_index: SupportsIndex) -> bool:
        """
        Insert an entry index corresponding to a given key into our hashtable

        Parameters
        ----------
        key : Hashable
            A key to insert into the buckets. It is needed to compute hashes
        entry_index : SupportsIndex
            The entry index to insert

        Returns
        -------
        True
            If a rehash has been triggered was triggered during insertion
        False
            Otherwise
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

    def try_insert_entry_index_into_specific_table(
        self, key: Hashable, entry_index: SupportsIndex, table_index: int
    ) -> Optional[SupportsIndex]:
        """
        Try and insert an entry_index for the given key into the table represented by entry index.
        If unsuccessful, then evict some other entry index and return it in a tuple along with it's associated
        key.

        Parameters
        ----------
        key : Hashable
            A key to try and insert into self._tables[table_index]
        entry_index : SupportsIndex
            The index in self._entries of the entry we are inserting
        table_index: int
            The index in self._tables of the table we are inserting `entry_index` into

        Returns
        -------
        None
            If insertion was successful
        SupportIndex
            representing a key and entry index which was evicted
                 if the insertion was unsuccessful. This entry index will be inserted in the next cycle
        """

        bucket_index: int = self._hash_family(table_index, key, self._buckets_per_table)
        buckets: MutableSequence[SupportsIndex] = self._tables[table_index]

        if buckets[bucket_index] == CuckooHashTable.NO_ENTRY:
            buckets[bucket_index] = entry_index
            return None

        evicted_entry_index = buckets[bucket_index]
        buckets[bucket_index] = entry_index
        return evicted_entry_index

    def _get_eviction_policy(self) -> Callable[[int], int]:
        """
        Return an eviction policy
        This is a function which receives the current table index and returns
        the next table index to try and insert an evicted entry index

        The default policy just cycles through all the tables, ignoring the current table index

        Returns
        -------
        Callable[[int], int]
            A eviction policy
        """
        c = cycle(range(self._hash_family.size()))
        return lambda table_index: next(c)

    def _get_max_insert_failures(self) -> int:
        return max(
            self._hash_family.size() * 2,
            self._max_insert_failures_log_constant * self._num_entries.bit_length(),
        )

    def _insert(self, key: Hashable, entry_index: SupportsIndex) -> bool:
        """
        Insert a key and an entry index into the hash table

        Parameters
        ----------
        key : Hashable
            The key corresponding to ``entry_index``
        entry_index : SupportsIndex
            The index in self._entries of the entry we are inserting

        Returns
        -------
        True
            If a rehash was triggered
        False
            Otherwise
        """

        find_next = self._get_eviction_policy()
        max_insert_failures: int = self._get_max_insert_failures()

        # we always start with the first table before moving to the others
        table_index = 0
        for _ in range(max_insert_failures):
            evicted_entry_index = self.try_insert_entry_index_into_specific_table(
                key, entry_index, table_index
            )
            # if we did not evict any entry index,
            # then we successfully inserted entry_index without rehashing
            if evicted_entry_index is None:
                return False

            # prepare for another cycle of insertion
            key, entry_index = (
                self._entries[evicted_entry_index].key,
                evicted_entry_index,
            )
            table_index = find_next(table_index)

        # we only expand the table if we have rehashed a certain number of times unsuccessfully
        self._rehash(expand=self._rehashes >= self._allowed_rehash_attempts)
        return True

    def _rehash(self, *, expand: bool = False):
        # Rehash using a new hash function and recurse to try insert again.
        if expand:
            self._gen_tables()
            # self._rehashes counter must be reset here because this rehash is caused by an expansion request
            self._rehashes = 0
        else:
            self._gen_tables(self._buckets_per_table)

        self._rehashes += 1
        self._hash_family.gen()

        for entry_index, entry in self._enumerate_entries():
            if self._insert(entry.key, entry_index):
                # some other rehash was triggered while this rehash isn't over
                # me must return to prevent erroneous values in our tables
                return

    def _shrink_table(self):

        pass

    def __delitem__(self, key: Hashable):
        for table_index, table in enumerate(self._tables):
            bucket_index = self._hash_family(table_index, key, self._buckets_per_table)
            entry_index: SupportsIndex = table[bucket_index]

            if self._entry_at_index_matches_key(entry_index, key):
                table[bucket_index] = CuckooHashTable.NO_ENTRY
                self._remove_entry(entry_index)
                return True

        raise KeyError(f"{key} not found")

    def _enumerate_entries(self):
        for entry_index, entry in enumerate(self._entries):
            if entry.is_alive():
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
    """
    This class represents a Dary CuckooHashTable where the number of tables is > 2
    """

    DEFAULT_D: Final[int] = 4

    def __init__(
        self, hash_family: Optional[HashFamily] = None, usable_fraction: float = 0.9
    ):
        super().__init__(
            hash_family=hash_family or HashFamilyTabulation(self.DEFAULT_D),
            usable_fraction=usable_fraction,
        )

    def _validate_number_of_tables(self):
        if self._hash_family.size() < 3:
            raise ValueError(
                f"The size of the hash family must be >= 3: not {self._hash_family.size()}"
            )


class CuckooHashTableDAryRandomWalk(CuckooHashTableDAry):
    """
    A variation of cuckoo hashing where the table to evict and element from is chosen at random
    """

    def _random_walk_policy(self, current_table_index: int) -> int:
        """
        This policy involves randomly and uniformly choosing some other table, not equal to
        the one we just inserted into to reduce the probability of cycles

        Parameters
        ----------
        current_table_index: the index of the table we have just failed at inserting into

        Returns
        -------
        A new table index not equal to ``current_table_index``

        References
        ---------
        Alan Frieze, Páll Melsted, and Michael Mitzenmacher. 2011.
        An Analysis of Random-Walk Cuckoo Hashing. SIAM J. Comput. 40, 2 (March 2011), 291–308.
        https://doi.org/10.1137/090770928

        """
        while (
            index := random.randint(0, self._hash_family.size() - 1)
        ) == current_table_index:
            continue
        return index

    def _get_eviction_policy(self) -> Callable[[int], int]:
        return self._random_walk_policy


class CuckooHashTableBucketed(CuckooHashTable):
    """
    A variation of a cuckoo hash table which uses multiple slots in each bucket

    Uses 2 slots per bucket by default
    """

    DEFAULT_SLOTS_PER_BUCKET = 3

    __slots__ = "_slots_per_bucket"

    def __init__(self, slots_per_bucket: int = DEFAULT_SLOTS_PER_BUCKET):
        self._slots_per_bucket = slots_per_bucket
        super().__init__()

    @property
    def capacity(self):
        return (
            self._buckets_per_table * self._hash_family.size() * self._slots_per_bucket
        )

    def _gen_tables(
        self, buckets_per_table: Optional[int] = None, growth_factor: float = 2
    ):
        self._update_buckets_per_table(buckets_per_table, growth_factor)
        type_code = self._get_correct_type_code()
        table = [array(type_code) for _ in range(self._buckets_per_table)]
        self._tables: tuple[BucketIndex, ...] = (table,) + tuple(  # type: ignore[assignment]
            deepcopy(table) for _ in range(self._hash_family.size() - 1)
        )

    def _get_entry_specific_table(
        self, table_index: int, key: Hashable
    ) -> Optional[Entry]:
        bucket_index = self._hash_family(table_index, key, self._buckets_per_table)
        bucket = self._tables[table_index][bucket_index]

        for entry_index in bucket:
            if self._entry_at_index_matches_key(entry_index, key):
                return self._entries[entry_index]

        return None

    def try_insert_entry_index_into_specific_table(
        self, key, entry_index: SupportsIndex, table_index: int
    ) -> Optional[SupportsIndex]:
        bucket_index = self._hash_family(table_index, key, self._buckets_per_table)
        bucket = self._tables[table_index][bucket_index]

        # all the slots in this bucket are occupied
        if len(bucket) == self._slots_per_bucket:
            # we place this object as the last one in the bucket
            # should we use LRU approach here to take advantage of temporal locality?
            evicted_entry_index = bucket[-1]
            bucket[-1] = entry_index

            if self._entries[evicted_entry_index].is_alive():
                return evicted_entry_index
            else:
                # we don't expect to be harboring None values in our buckets at this point
                # they should have been removed during the deletion process
                raise RuntimeError(f"self._entries[{entry_index}] is None")
        bucket.append(entry_index)
        return None

    def __delitem__(self, key: Hashable):
        """
        Delete key from the dictionary if key exists in the dictionary

        Parameters
        ----------
        key : Hashable
            The key to remove from the dictionary

        Raises
        ------
        KeyError
            If the key was not in the dictionary

        Notes
        -----

        Deletes the key from the dictionary in O(d) steps by looping through the possible tables:
        and deleting only if the key matches the key at the entry pointed to by entry index

        Deletion happens by
         1) removing the entry_index from the bucket
         2) replacing the entry with TOMBSTONE

        """
        for table_index, table in enumerate(self._tables):
            bucket_index = self._hash_family(table_index, key, self._buckets_per_table)
            bucket = table[bucket_index]

            for entry_index in bucket[:]:
                if self._entry_at_index_matches_key(entry_index, key):
                    bucket.remove(entry_index)
                    self._remove_entry(entry_index)
                    return True

        raise KeyError(f"{key} not found")


class CuckooHashTableStashed(CuckooHashTable):
    """
    https://link.springer.com/content/pdf/10.1007/s00453-013-9840-x.pdf


    """

    __slots__ = ("_stash", "_max_stash_size", "_should_try_insert_stashed_keys")

    def __init__(
        self,
        max_stash_size: int = 10,
        hash_family: HashFamily = HashFamilyTabulation(2),
    ):
        """

        Notes
        -----
        The reason for having ``_should_try_insert_stashed_items``

        Kirsch et al. [13] propose reinserting all stash keys at the beginning of the first
        insert operation following a delete operation. Each such insertion takes time
        maxloop = O(log n) in the worst-case. We know that
        the stash is empty with probability 1 − O(1/n). Thus, the contribution of the time
        needed to reinsert all stash keys to the expected insertion time is O((log n)/n) =
        o(1) using our class of hash functions.

        References
        ----------
        A. Kirsch, M. Mitzenmacher, and U. Wieder.
        More Robust Hashing: Cuckoo Hashing with a Stash.
        SIAM Journal on Computing, 39(4).

        """
        super().__init__(hash_family=hash_family)
        self._stash: list[SupportsIndex] = []
        self._max_stash_size: int = max_stash_size
        self._should_try_insert_stashed_keys: bool = False

    def _get_max_insert_failures(self):
        return (self._max_stash_size + 1) * self._num_entries.bit_length()

    def _try_insert_stashed_keys(self):
        if self._should_try_insert_stashed_keys:
            for entry_index in self._stash:
                self._insert_entry_index(self._entries[entry_index].key, entry_index)
        self._should_try_insert_stashed_keys = False

    # noinspection DuplicatedCode
    def _insert(self, key: Hashable, entry_index: SupportsIndex) -> bool:
        find_next = self._get_eviction_policy()
        max_insert_failures: int = self._get_max_insert_failures()

        table_index = 0
        for _ in range(max_insert_failures):
            evicted_entry_index = self.try_insert_entry_index_into_specific_table(
                key, entry_index, table_index
            )
            if evicted_entry_index is None:
                return False

            key, entry_index = (
                self._entries[evicted_entry_index].key,
                evicted_entry_index,
            )
            table_index = find_next(table_index)

        # this is really the only big difference
        if len(self._stash) < self._max_stash_size:
            self._stash.append(entry_index)
            return False

        self._rehash(expand=self._rehashes >= self._allowed_rehash_attempts)
        return True

    def _get_entry(self, key: Hashable) -> Optional[Entry]:
        if (entry := super(CuckooHashTableStashed, self)._get_entry(key)) is None:
            for entry_index in self._stash:
                if self._entry_at_index_matches_key(entry_index, key):
                    return self._entries[entry_index]
        return entry

    def __delitem__(self, key):
        try:
            super(CuckooHashTableStashed, self).__delitem__(key)
            self._should_try_insert_stashed_keys = True
        except KeyError as e:
            for entry_index in self._stash[:]:
                if self._entry_at_index_matches_key(entry_index, key):
                    self._should_try_insert_stashed_keys = True
                    return self._stash.remove(entry_index)
            raise e

    def __setitem__(self, key: Hashable, value: Any):
        if (entry := self._get_entry(key)) is not None:
            # overwrite old value
            entry.value = value
        else:
            self._gen_and_append_entry(key, value)
            self._insert_entry_index(key, len(self._entries) - 1)
            self._try_insert_stashed_keys()
            self._rehashes = 0


class CuckooHashTableStashedDAry(CuckooHashTableStashed):
    def __init__(self, d=CuckooHashTableDAry.DEFAULT_D):
        super().__init__(hash_family=HashFamilyTabulation(d))

    _validate_number_of_tables = CuckooHashTableDAry._validate_number_of_tables
