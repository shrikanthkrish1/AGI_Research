# pattern_memory.py
import time
from typing import Optional, Dict, Any
from collections import OrderedDict

class PatternMemory:
    def __init__(self, max_size: int = 4096):
        self.max_size = max_size
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def _prune(self):
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def store(self, input_grid_hash: str, output_grid: Any, transform: Optional[Any] = None, meta: Optional[Dict] = None):
        entry = {"output_grid": output_grid, "transform": transform, "meta": meta or {}, "timestamp": time.time()}
        if input_grid_hash in self._store:
            self._store.pop(input_grid_hash)
        self._store[input_grid_hash] = entry
        self._prune()

    def retrieve(self, input_grid_hash: str) -> Optional[Dict[str, Any]]:
        entry = self._store.get(input_grid_hash)
        if entry:
            self._store.pop(input_grid_hash)
            self._store[input_grid_hash] = entry
        return entry

    def contains(self, input_grid_hash: str) -> bool:
        return input_grid_hash in self._store

    def clear(self):
        self._store.clear()
