
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

