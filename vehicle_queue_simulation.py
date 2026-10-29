
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import random


EPS = 1e-9


@dataclass
class VehicleType:
    name: str
    length: float
    width: float
    longitudinal_gap: float
    lateral_clearance: float = 0.0

    @property
    def effective_width(self) -> float:
        return self.width + self.lateral_clearance


@dataclass
class Entry:
    width: float
    queue_length: float


class _Node:
    __slots__ = (
        "width", "queue_length", "priority",
        "left", "right", "size",
        "max_width", "min_queue_length",
    )

    def __init__(self, width: float, queue_length: float):
        self.width = width
        self.queue_length = queue_length
        self.priority = random.random()

        self.left: Optional[_Node] = None
        self.right: Optional[_Node] = None

        self.size = 1
        self.max_width = width
        self.min_queue_length = queue_length


def _size(node: Optional[_Node]) -> int:
    return node.size if node else 0


def _max_width(node: Optional[_Node]) -> float:
    return node.max_width if node else float("-inf")


def _min_queue(node: Optional[_Node]) -> float:
    return node.min_queue_length if node else float("inf")


def _pull(node: _Node) -> None:
    node.size = 1 + _size(node.left) + _size(node.right)

    node.max_width = max(
        node.width,
        _max_width(node.left),
        _max_width(node.right),
    )

    node.min_queue_length = min(
        node.queue_length,
        _min_queue(node.left),
        _min_queue(node.right),
    )


def _merge(
    a: Optional[_Node],
    b: Optional[_Node],
) -> Optional[_Node]:
    """Concatenate sequence a followed by sequence b."""
    if a is None:
        return b
    if b is None:
        return a

    if a.priority < b.priority:
        a.right = _merge(a.right, b)
        _pull(a)
        return a

    b.left = _merge(a, b.left)
    _pull(b)
    return b


def _split(
    root: Optional[_Node],
    count: int,
) -> tuple[Optional[_Node], Optional[_Node]]:
    """Split before index count."""
    if root is None:
        return None, None

    left_size = _size(root.left)

    if count <= left_size:
        a, b = _split(root.left, count)
        root.left = b
        _pull(root)
        return a, root

    a, b = _split(
        root.right,
        count - left_size - 1,
    )
    root.right = a
    _pull(root)
    return root, b


def _get_at(
    root: Optional[_Node],
    index: int,
) -> Optional[_Node]:
    """Return the node at a sequence index."""
    while root is not None:
        left_size = _size(root.left)

        if index < left_size:
            root = root.left
        elif index == left_size:
            return root
        else:
            index -= left_size + 1
            root = root.right

    return None


def _remove_at(
    root: Optional[_Node],
    index: int,
) -> tuple[Optional[_Node], Optional[_Node]]:
    """Remove and return one entry at index."""
    before, rest = _split(root, index)
    selected, after = _split(rest, 1)

    if selected is not None:
        selected.left = None
        selected.right = None
        _pull(selected)

    return _merge(before, after), selected


def _find_shortest_fitting(
    root: Optional[_Node],
    required_width: float,
) -> Optional[int]:
    """
    Find the entry with the smallest queue length among entries
    whose width is sufficient.

    Ties are resolved by choosing the leftmost entry.
    """
    best_index: Optional[int] = None
    best_length = float("inf")

    def search(
        node: Optional[_Node],
        offset: int,
    ) -> None:
        nonlocal best_index, best_length

        if node is None:
            return

        # No entry in this subtree can fit.
        if node.max_width + EPS < required_width:
            return

        # No entry can improve the best result found so far.
        if node.min_queue_length >= best_length:
            return

        left_size = _size(node.left)

        # Search in lateral order, left to right.
        search(node.left, offset)

        index = offset + left_size

        if (
            node.width + EPS >= required_width
            and node.queue_length < best_length
        ):
            best_length = node.queue_length
            best_index = index

        search(node.right, index + 1)

    search(root, 0)
    return best_index


def _find_shortest_index(
    root: Optional[_Node],
) -> Optional[int]:
    """Find the leftmost entry having the minimum queue length."""
    if root is None:
        return None

    target = root.min_queue_length

    def search(
        node: Optional[_Node],
        offset: int,
    ) -> Optional[int]:
        if node is None or node.min_queue_length > target:
            return None

        left_size = _size(node.left)

        result = search(node.left, offset)
        if result is not None:
            return result

        if abs(node.queue_length - target) <= EPS:
            return offset + left_size

        return search(node.right, offset + left_size + 1)

    return search(root, 0)

