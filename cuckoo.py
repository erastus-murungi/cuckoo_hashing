from dataclasses import dataclass, astuple
from io import StringIO
from random import randint
from typing import Union
from random import getrandbits
from operator import attrgetter
from array import array
import struct
from collections import namedtuple
from itertools import groupby

__all__ = ["Cuckoo"]

is64 = (struct.calcsize("P") * 8) == 64


@dataclass
class Entry:
    __slots__ = ('hash_val', 'key', 'val')
    hash_val: int
    key: object
    val: object

    def __iter__(self):
        return astuple(self)


class Cuckoo:
    _max_rehash = 2
    _minsize = 8
    _no_entry = -1
    _max_loop = 100
    __slots__ = ('index', 'entries', 'm', 'used', 'hf', 'nt', '_num_rehash')
    w = 64   # word size
    p = 0x1fffffffffffffff  # Mersenne prime (2^61 - 1)
    data = namedtuple('data', ['a', 'b'])

    def __init__(self, nt=2):
        self.m = self._minsize
        self.nt = nt
        self.index = self.make_index()
        self.entries = []
        self.hf = self._get_universal_hash_functions()
        self.used = 0
        self._num_rehash = 0

    def make_index(self):
        if self.m <= 0x7f:  # signed 8-bit
            return [array('b', [self._no_entry]) * self.m for _ in range(self.nt)]
        elif self.m <= 0x7fff:   # signed 16-bit
            return [array('h', [self._no_entry]) * self.m for _ in range(self.nt)]
        elif self.m <= 0x7fffffff:   # signed 32-bit
            return [array('l', [self._no_entry]) * self.m for _ in range(self.nt)]
        elif is64 and self.m <= 0x7fffffffffffffff:   # signed 64-bit
                return [array('q', [self._no_entry]) * self.m for _ in range(self.nt)]
        else:
            return [[self._no_entry] * self.m for _ in range(self.nt)]

    @staticmethod
    def hash_mod(hash_x, a, b, p):
        return (a * hash_x + b) % p

    @staticmethod
    def is_power_of_two(x):
        return not (x & (x - 1))

    @staticmethod
    def usable_fraction(x):
        return x >> 1

    @staticmethod
    def get_rand_odd(w: int):
        x = getrandbits(w)
        while not (x & 1):
            x = getrandbits(w)
        return x

    @staticmethod
    def hash_shift(hash_x, a, b, w):
        mask = (1 << w) - 1
        return (a * hash_x + b) & mask

    def table_offset_hash_shift(self, hash_val):
        assert self.is_power_of_two(self.m)
        M = self.m.bit_length() - 1
        return hash_val >> (self.w - M)

    def table_offset_hash_mod(self, hash_val):
        return hash_val % self.m

    def table_offset(self, hash_val):
        return self.table_offset_hash_shift(hash_val)

    def get_hash_functions_shift(self, n):
        ab_pairs = set()
        while len(ab_pairs) != n:
            a = self.get_rand_odd(self.w)  # uniformly random odd w-bit integer a
            b = randint(0, self.p)
            ab_pairs.add((a, b))

        def f(_a, _b, w):
            def _f(item):
                x = self.hash_shift(hash(item), _a, _b, w)
                return x
            return _f
        return [f(a, b, self.w) for a, b, in ab_pairs]

    def get_hash_functions_mod(self, n):
        # 􏰓􏰐􏰔 H(m,p)= h_{a,b} [k]=(((ak+b) mod p) mod m) 􏰐􏰐 a,b ∈ {0,...,p−1} and a̸=0. p >= m
        ab_pairs = set()
        while len(ab_pairs) != n:
            a = randint(1, self.p)  # uniformly random odd w-bit integer a
            b = randint(0, self.p)
            ab_pairs.add((a, b))

        def f(_a, _b, p):
            def _f(item):
                x = self.hash_mod(hash(item), _a, _b, p)
                return x
            return _f
        return [f(a, b, self.p) for a, b in ab_pairs]

    def _get_universal_hash_functions(self):
        return self.get_hash_functions_shift(self.nt)

    @staticmethod
    def fast_match(key, hash_val, en: Entry):
        if key is en.key:
            return True
        if hash_val != en.hash_val:
            return False
        return key == en.key

    def __contains__(self, key):
        en = self._get_entry(key)
        return en is not None

    def __getitem__(self, key) -> Union[None, object]:
        """Gets a value"""
        en = self._get_entry(key)
        return en if en is None else en.val

    def _get_entry(self, key):
        for t, h in zip(self.index, self.hf):
            hash_val = h(key)
            i = self.table_offset(hash_val)
            ix = t[i]
            if ix != Cuckoo._no_entry and self.fast_match(key, hash_val, self.entries[ix]):
                return self.entries[ix]
        return None

    def _get_entry_from_table(self, hash_val, t=0):
        ix_t = self.index[t][self.table_offset(hash_val)]
        return None if ix_t == Cuckoo._no_entry else self.entries[ix_t]

    def __setitem__(self, new_key, value):
        if new_key in self:
            # replace old value
            en = self._get_entry(new_key)
            en.val = value
        else:
            self._resize()
            self.entries.append(Entry(0, new_key, value))
            self.used += 1
            should_rehash, hash_val = self.insert(new_key, self.used-1)
            i = 0
            while should_rehash:
                should_rehash, hash_val = self._rehash()
                print(i)
                i += 1
            self.entries[-1].hash_val = hash_val

    def insert(self, cur_key, ix):
        """Inserts the index `ix` of entry with cur_key into the hash_table.
        Runs the loop max_loop times."""

        for _ in range(self._max_loop):
            # each table has its own unique hash function
            for hashtable, hashfunc in zip(self.index, self.hf):
                hashval = hashfunc(cur_key)
                i = self.table_offset(hashval)
                next_ix = hashtable[i]

                if next_ix == self._no_entry:
                    hashtable[i] = ix
                    return False, hashval

                cur_key = self.entries[next_ix].key
                hashtable[i] = ix
                ix = next_ix

        # create new hash functions and rehash everything
        return True, self.entries[ix].hash_val

    def _rehash(self):
        self._num_rehash += 1
        if self._num_rehash > 2:
            self._grow_index()
            self._num_rehash = 0

        self._get_universal_hash_functions()
        self._compress_entries()
        self.reset_index()
        hash_val = None
        for i, entry in enumerate(self._iter_entries()):
            should_rehash, hash_val = self.insert(entry.key, i)
            if should_rehash:
                return True, None
        return False, hash_val

    def pop(self, key):
        if key in self:
            raise KeyError("key not in dict")
        else:
            for t, h in zip(self.index, self.hf):
                hash_val = h(key)
                ix = t[self.table_offset(hash_val)]
                if ix != Cuckoo._no_entry and self.fast_match(key, hash_val, self.entries[ix]):
                    t[ix] = Cuckoo._no_entry
                    self.entries[ix] = None
                    self.used -= 1
                    return True
            raise SystemExit()

    def __delitem__(self, key):
        return self.pop(key)

    def reset_index(self):
        for index in self.index:
            for i in range(len(index)):
                index[i] = self._no_entry

    def _resize(self):
        self._compress_entries()
        if len(self.entries) >= self.usable_fraction(self.m * self.nt):
            # we need to resize index
            # all tables must be the same size
            self._grow_index()

    def _compress_entries(self):
        if len(self.entries) != self.used:
            self.entries = [entry for entry in self._iter_entries()]

    def _grow_index(self):
        old_ws = (self.m >> 1) - 1
        new_ws = self.m - 1
        self.m <<= 1
        word_sizes = (0x7f, 0x7fff, 0x7fffffff, 0x7fffffffffffffff)
        grow_ws = False
        for i, word_size in enumerate(word_sizes):
            if old_ws < word_size:
                grow_ws = new_ws >= word_size
                break
        if grow_ws:
            self._grow_index_and_word_size()
        else:
            self._grow_index_only()

    def _grow_index_only(self):
        for t in self.index:
            t.extend([Cuckoo._no_entry] * (self.m - len(t)))

    def _grow_index_and_word_size(self):
        old_index = self.index
        self.index = self.make_index()
        for old_t, t in zip(old_index, self.index):
            # get the next size
            for i, elem in enumerate(old_t):
                t[i] = elem
            assert self.is_power_of_two(len(t))

    def _iter_entries(self):
        for entry in self.entries:
            if entry is not None:
                yield entry

    def __iter__(self):
        f_key = attrgetter('key')
        yield from map(lambda entry: f_key(entry), self._iter_entries())

    def itervalues(self):
        f_val = attrgetter('val')
        yield from map(lambda entry: f_val(entry), self._iter_entries())

    def iteritems(self):
        f_key, f_val = attrgetter('key'), attrgetter('val')
        yield from map(lambda entry: (f_key(entry), f_val(entry)), self._iter_entries())

    def __len__(self):
        return self.used

    def dict_structure(self):
        ret = ["\033[1m\033[32m--Dictionary Attributes--:\033[0m\n",
               "  allocated       : \033[0m\033[33m%d\033[0m\n" % (self.m * self.nt),
               "  used            : \033[0m\033[34m%d\033[0m\n" % self.used,
               "  load factor     : \033[1m\033[35m%.3f\033[0m\n" % (self.used / (self.m * self.nt))]
        return ''.join(ret)

    def __repr__(self):
        if not self.used:
            return '{}'
        else:
            ret = StringIO()
            m = self.used
            ret.write("{")
            for ent in self.entries:
                if ent:
                    ret.write(repr(ent.key))
                    ret.write(': ')
                    ret.write(repr(ent.val))
                    m -= 1
                    if m:
                        ret.write(', ')
            ret.write("}\n")
        return ret.getvalue()

    def test_hash(self, key):
        def all_equal(iterable):
            """Returns True if all the elements are equal to each other"""
            g = groupby(iterable)
            return next(g, True) and not next(g, False)

        hf = self.hf
        if all_equal(self.table_offset(h(key)) for h in hf):
            print(key)
        else:
            print("no collision")


if __name__ == "__main__":
    import numpy as np
    import string
    import random

    n = 8000
    rand_nums = np.random.randint(0, 10000, n)
    rand_strings = (''.join(random.choice(string.ascii_letters) for _ in range(10)) for _ in range(n))
    ck = Cuckoo(4)
    for _ in range(10):
        for k, val in zip(rand_nums, rand_strings):
            ck[k] = val
            assert k in ck
        assert (len(ck) == ck.used)
        print("Ok!")

