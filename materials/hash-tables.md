# Hash tables

A hash table stores key-value pairs in an array. A hash function turns a key into an index, which gives expected O(1) lookups, inserts and deletes when keys spread evenly.

## Collisions

A collision happens when two keys map to the same index. Chaining keeps a small list at each index. Open addressing probes for another free slot in the array.

## Load factor

The load factor is the number of entries divided by the number of slots. As it grows, collisions become more common and operations slow down, so tables usually resize and rehash every entry once it passes a threshold. One resize costs O(n), but its cost averages out across inserts.
