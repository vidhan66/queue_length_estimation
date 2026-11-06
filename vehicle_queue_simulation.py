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


def _entries_in_order(
    root: Optional[_Node],
    output: list[Entry],
) -> None:
    if root is None:
        return

    _entries_in_order(root.left, output)
    output.append(Entry(root.width, root.queue_length))
    _entries_in_order(root.right, output)


class QueueSimulation:
    def __init__(
        self,
        road_width: float,
        wall_clearance: float = 0.0,
        seepage_ratio_T: float = 1.0,
    ):
        usable_width = road_width - 2 * wall_clearance

        if usable_width <= 0:
            raise ValueError("Usable road width must be positive.")
        if seepage_ratio_T < 0:
            raise ValueError("T must be non-negative.")

        self.usable_width = usable_width
        self.seepage_ratio_T = seepage_ratio_T

        self.root: Optional[_Node] = _Node(
            usable_width, 0.0
        )

        self.max_queue_length = 0.0
        self.placed_count = 0
        self.unplaced_count = 0

    def _insert_at(
        self,
        index: int,
        entry: _Node,
    ) -> None:
        before, after = _split(self.root, index)
        self.root = _merge(
            _merge(before, entry),
            after,
        )

    def add_vehicle(
        self,
        vehicle: VehicleType,
    ) -> Optional[dict]:
        required_width = vehicle.effective_width

        if required_width <= 0:
            raise ValueError(
                "Vehicle effective width must be positive."
            )

        # PASS 1:
        # Search without modifying the tree.
        # Select the shortest queue that fits the vehicle.
        index = _find_shortest_fitting(
            self.root,
            required_width,
        )

        if index is not None:
            # A suitable individual entry exists.
            self.root, selected = _remove_at(
                self.root, index
            )

            if selected is None:
                raise RuntimeError("Selected entry not found.")

            old_width = selected.width
            old_length = selected.queue_length

            # An empty region starts a new queue.
            if old_length <= EPS:
                new_length = vehicle.length
            else:
                new_length = (
                    old_length
                    + vehicle.longitudinal_gap
                    + vehicle.length
                )

            remainder_width = old_width - required_width

            # Insert the vehicle's occupied region.
            replacement = _Node(
                required_width,
                new_length,
            )

            # The remaining width keeps its old queue length.
            if remainder_width > EPS:
                replacement = _merge(
                    replacement,
                    _Node(remainder_width, old_length),
                )

            self._insert_at(index, replacement)

            self.max_queue_length = max(
                self.max_queue_length,
                new_length,
            )
            self.placed_count += 1

            return {
                "vehicle": vehicle.name,
                "placement": "individual_fit",
                "effective_width": required_width,
                "previous_width": old_width,
                "previous_queue_length": old_length,
                "new_queue_length": new_length,
                "remaining_width": max(
                    0.0, remainder_width
                ),
                "max_queue_length": self.max_queue_length,
                "entry_index": index,
            }

        # PASS 2:
        # No individual entry fits. Remove entries in
        # increasing queue-length order and aggregate widths.
        consumed: list[tuple[float, float, int]] = []
        total_width = 0.0

        while self.root is not None and total_width + EPS < required_width:
            shortest_index = _find_shortest_index(self.root)

            if shortest_index is None:
                break

            self.root, selected = _remove_at(
                self.root,
                shortest_index,
            )

            if selected is None:
                break

            consumed.append((
                selected.width,
                selected.queue_length,
                shortest_index,
            ))
            total_width += selected.width

        if total_width + EPS < required_width:
            # Not enough total width exists.
            # Restore every consumed entry.
            for width, length, original_index in sorted(
                consumed, key=lambda item: item[2]
            ):
                self._insert_at(
                    min(original_index, _size(self.root)),
                    _Node(width, length),
                )

            self.unplaced_count += 1
            return None

        # The entries are consumed in ascending queue-length order.
        # Use the longest consumed queue as the base length.
        base_length = max(length for _, length, _ in consumed)

        if base_length <= EPS:
            new_length = vehicle.length
        else:
            new_length = (
                base_length
                + vehicle.longitudinal_gap
                + vehicle.length
            )

        # The consumed entries are removed and replaced by the
        # vehicle-width region. Any excess combined width is retained.
        remainder_width = total_width - required_width

        replacement = _Node(
            required_width,
            new_length,
        )

        if remainder_width > EPS:
            replacement = _merge(
                replacement,
                _Node(remainder_width, base_length),
            )

        # The consumed entries may not have been adjacent laterally.
        # Insert the replacement at the leftmost consumed position.
        insertion_index = min(
            original_index
            for _, _, original_index in consumed
        )

        self._insert_at(
            min(insertion_index, _size(self.root)),
            replacement,
        )

        self.max_queue_length = max(
            self.max_queue_length,
            new_length,
        )
        self.placed_count += 1

        return {
            "vehicle": vehicle.name,
            "placement": "aggregated",
            "effective_width": required_width,
            "aggregated_width": total_width,
            "previous_queue_lengths": [
                length for _, length, _ in consumed
            ],
            "base_queue_length": base_length,
            "new_queue_length": new_length,
            "remaining_width": max(0.0, remainder_width),
            "max_queue_length": self.max_queue_length,
            "entries_consumed": len(consumed),
        }

    def entries(self) -> list[Entry]:
        result: list[Entry] = []
        _entries_in_order(self.root, result)
        return result

    def snapshot(self) -> list[dict]:
        return [
            {
                "width": round(entry.width, 3),
                "queue_length": round(entry.queue_length, 3),
            }
            for entry in self.entries()
        ]


if __name__ == "__main__":
    vehicle_types = {
        "car": VehicleType(
            "car", 3.72, 1.44, 1.00, 0.20
        ),
        "two_wheeler": VehicleType(
            "two_wheeler", 1.87, 0.64, 0.50, 0.10
        ),
        "lcv": VehicleType(
            "lcv", 6.10, 2.10, 1.00, 0.20
        ),
        "truck": VehicleType(
            "truck", 7.50, 2.35, 1.20, 0.20
        ),
        "bus": VehicleType(
            "bus", 10.10, 2.43, 1.20, 0.20
        ),
    }

    simulation = QueueSimulation(
        road_width=7.0,
        wall_clearance=0.2,
        seepage_ratio_T=1.0,
    )

    arrivals = [
        "truck",
        "car",
        "two_wheeler",
        "car",
        "bus",
        "two_wheeler",
    ]

    for name in arrivals:
        result = simulation.add_vehicle(vehicle_types[name])

        print(f"Arrival: {name}")
        print("Placement:", result or "Unplaced")
        print("Entries:", simulation.snapshot())
        print(
            "Maximum queue length:",
            round(simulation.max_queue_length, 3),
        )
        print("-" * 60)
