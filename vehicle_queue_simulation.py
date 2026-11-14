from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional
import heapq
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


def _entries_in_order(
    root: Optional[_Node],
    output: list[Entry],
) -> None:
    if root is None:
        return

    _entries_in_order(root.left, output)
    output.append(Entry(root.width, root.queue_length))
    _entries_in_order(root.right, output)


def _find_best_contiguous_window(
    entries: list[Entry],
    required_width: float,
) -> Optional[tuple[int, int, float, float]]:
    n = len(entries)
    if n == 0:
        return None

    best: Optional[tuple[float, float, int, int]] = None
    max_deque: deque[int] = deque()
    left = 0
    width_sum = 0.0

    for right in range(n):
        width_sum += entries[right].width

        while (
            max_deque
            and entries[max_deque[-1]].queue_length
            <= entries[right].queue_length
        ):
            max_deque.pop()
        max_deque.append(right)

        while (
            left <= right
            and width_sum - entries[left].width + EPS >= required_width
        ):
            width_sum -= entries[left].width
            if max_deque and max_deque[0] == left:
                max_deque.popleft()
            left += 1

        if width_sum + EPS >= required_width:
            max_queue = entries[max_deque[0]].queue_length
            candidate = (max_queue, width_sum, left, right)
            if (
                best is None
                or candidate[0] < best[0] - EPS
                or (
                    abs(candidate[0] - best[0]) <= EPS
                    and candidate[1] < best[1] - EPS
                )
            ):
                best = candidate

    if best is None:
        return None

    max_queue, total_width, start, end = best
    return start, end, total_width, max_queue


class QueueSimulation:
    def __init__(
        self,
        road_width: float,
        wall_clearance: float = 0.0,
        seepage_ratio_T: float = 1.0,
        track_positions: bool = False,
    ):
        usable_width = road_width - 2 * wall_clearance

        if usable_width <= 0:
            raise ValueError("Usable road width must be positive.")
        if seepage_ratio_T < 0:
            raise ValueError("T must be non-negative.")

        self.usable_width = usable_width
        self.seepage_ratio_T = seepage_ratio_T
        self.track_positions = track_positions

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

    def _remove_range(
        self,
        start: int,
        count: int,
    ) -> None:
        before, rest = _split(self.root, start)
        _middle, after = _split(rest, count)
        self.root = _merge(before, after)

    def add_vehicle(
        self,
        vehicle: VehicleType,
    ) -> Optional[dict]:
        required_width = vehicle.effective_width

        if required_width <= 0:
            raise ValueError(
                "Vehicle effective width must be positive."
            )

        entries_before = self.entries()

        window = _find_best_contiguous_window(
            entries_before, required_width
        )

        if window is None:
            self.unplaced_count += 1
            return None

        start, end, total_width, base_length = window
        count = end - start + 1

        offset = sum(e.width for e in entries_before[:start])

        self._remove_range(start, count)

        if base_length <= EPS:
            new_length = vehicle.length
        else:
            new_length = (
                base_length
                + vehicle.longitudinal_gap
                + vehicle.length
            )

        remainder_width = total_width - required_width

        replacement = _Node(required_width, new_length)

        if remainder_width > EPS:
            replacement = _merge(
                replacement,
                _Node(remainder_width, base_length),
            )

        self._insert_at(start, replacement)

        self.max_queue_length = max(
            self.max_queue_length,
            new_length,
        )
        self.placed_count += 1

        result = {
            "vehicle": vehicle.name,
            "placement": "individual_fit" if count == 1 else "aggregated",
            "effective_width": required_width,
            "aggregated_width": total_width,
            "previous_queue_lengths": [
                entries_before[i].queue_length
                for i in range(start, end + 1)
            ],
            "base_queue_length": base_length,
            "new_queue_length": new_length,
            "remaining_width": max(0.0, remainder_width),
            "max_queue_length": self.max_queue_length,
            "entries_consumed": count,
            "insertion_index": start,
        }

        if self.track_positions:
            result["rect_x_start"] = offset
            result["rect_x_width"] = required_width
            result["rect_y_start"] = new_length - vehicle.length
            result["rect_y_end"] = new_length

        return result

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


@dataclass
class ArrivalRecord:
    entry_time: float
    vehicle: VehicleType
    speed: float


class EventDrivenQueue:
    def __init__(
        self,
        queue_sim: QueueSimulation,
        intersection_length: float,
    ):
        self.queue_sim = queue_sim
        self.intersection_length = intersection_length
        self._heap: list[tuple[float, int, VehicleType]] = []
        self.current_time = 0.0
        self._counter = 0

    def schedule_arrival(
        self,
        vehicle: VehicleType,
        speed: float,
        entry_time: float,
    ) -> float:
        distance = max(
            self.intersection_length - self.queue_sim.max_queue_length,
            0.0,
        )
        arrival_time = entry_time + distance / speed
        self._counter += 1
        heapq.heappush(
            self._heap,
            (arrival_time, self._counter, vehicle),
        )
        return arrival_time

    def run(
        self,
        entry_stream: list[ArrivalRecord],
        on_event=None,
    ) -> None:
        entry_stream = sorted(
            entry_stream, key=lambda rec: rec.entry_time
        )
        entry_idx = 0

        while entry_idx < len(entry_stream) or self._heap:
            next_entry_time = (
                entry_stream[entry_idx].entry_time
                if entry_idx < len(entry_stream)
                else float("inf")
            )
            next_arrival_time = (
                self._heap[0][0] if self._heap else float("inf")
            )

            if next_entry_time <= next_arrival_time:
                record = entry_stream[entry_idx]
                entry_idx += 1
                self.current_time = record.entry_time
                arrival_time = self.schedule_arrival(
                    record.vehicle, record.speed, record.entry_time
                )
                if on_event:
                    on_event({
                        "type": "entry",
                        "time": self.current_time,
                        "vehicle": record.vehicle.name,
                        "speed": record.speed,
                        "scheduled_arrival_time": arrival_time,
                    })
            else:
                arrival_time, _, vehicle = heapq.heappop(self._heap)
                self.current_time = arrival_time
                result = self.queue_sim.add_vehicle(vehicle)
                if on_event:
                    on_event({
                        "type": "arrival",
                        "time": self.current_time,
                        "vehicle": vehicle.name,
                        "placement": result,
                        "entries": self.queue_sim.snapshot(),
                        "max_queue_length": self.queue_sim.max_queue_length,
                    })


def default_vehicle_types() -> dict[str, VehicleType]:
    return {
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


def default_arrivals() -> list[ArrivalRecord]:
    vehicle_types = default_vehicle_types()
    return [
        ArrivalRecord(0.0, vehicle_types["truck"], 8.0),
        ArrivalRecord(1.5, vehicle_types["car"], 15.0),
        ArrivalRecord(3.0, vehicle_types["two_wheeler"], 16.0),
        ArrivalRecord(4.0, vehicle_types["car"], 14.0),
        ArrivalRecord(6.0, vehicle_types["bus"], 9.0),
        ArrivalRecord(7.5, vehicle_types["two_wheeler"], 17.0),
    ]


def run_default_scenario(
    track_positions: bool = False,
    on_event=None,
) -> tuple[QueueSimulation, EventDrivenQueue]:
    simulation = QueueSimulation(
        road_width=7.0,
        wall_clearance=0.2,
        seepage_ratio_T=1.0,
        track_positions=track_positions,
    )

    event_queue = EventDrivenQueue(
        queue_sim=simulation,
        intersection_length=150.0,
    )

    event_queue.run(default_arrivals(), on_event=on_event)

    return simulation, event_queue


if __name__ == "__main__":
    def log_event(event: dict) -> None:
        if event["type"] == "entry":
            print(
                f"[t={event['time']:.2f}] ENTRY  "
                f"{event['vehicle']} speed={event['speed']} "
                f"-> scheduled arrival at t={event['scheduled_arrival_time']:.2f}"
            )
        else:
            print(
                f"[t={event['time']:.2f}] ARRIVE {event['vehicle']} "
                f"placement={event['placement'] or 'Unplaced'}"
            )
            print(f"    entries={event['entries']}")
            print(
                f"    max_queue_length={event['max_queue_length']:.3f}"
            )

    run_default_scenario(track_positions=False, on_event=log_event)