
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

