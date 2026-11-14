from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from vehicle_queue_simulation import run_default_scenario


VEHICLE_COLORS = {
    "car": "#4C72B0",
    "two_wheeler": "#DD8452",
    "lcv": "#55A868",
    "truck": "#C44E52",
    "bus": "#8172B2",
}


@dataclass
class PlacedRectangle:
    time: float
    vehicle: str
    x_start: float
    x_width: float
    y_start: float
    y_end: float


def collect_rectangles() -> tuple[list[PlacedRectangle], list[float], float]:
    rectangles: list[PlacedRectangle] = []
    frame_times: list[float] = []

    def on_event(event: dict) -> None:
        if event["type"] != "arrival":
            return

        frame_times.append(event["time"])

        placement = event["placement"]
        if placement is None:
            return

        rectangles.append(
            PlacedRectangle(
                time=event["time"],
                vehicle=event["vehicle"],
                x_start=placement["rect_x_start"],
                x_width=placement["rect_x_width"],
                y_start=placement["rect_y_start"],
                y_end=placement["rect_y_end"],
            )
        )

    simulation, _ = run_default_scenario(
        track_positions=True, on_event=on_event
    )

    return rectangles, frame_times, simulation.usable_width


def draw_frame(
    ax,
    rectangles: list[PlacedRectangle],
    up_to_time: float,
    usable_width: float,
    max_y: float,
) -> None:
    ax.clear()
    ax.set_xlim(0, usable_width)
    ax.set_ylim(0, max_y * 1.1 + 1)
    ax.set_xlabel("Lateral position (m)")
    ax.set_ylabel("Longitudinal distance from stop line (m)")
    ax.set_title(f"Queue state at t = {up_to_time:.2f} s")
    ax.invert_yaxis()

    ax.axhline(0, color="black", linewidth=2)

    seen_labels = set()
    for rect in rectangles:
        if rect.time > up_to_time + 1e-9:
            continue

        color = VEHICLE_COLORS.get(rect.vehicle, "#888888")
        label = rect.vehicle if rect.vehicle not in seen_labels else None
        seen_labels.add(rect.vehicle)

        patch = patches.Rectangle(
            (rect.x_start, rect.y_start),
            rect.x_width,
            rect.y_end - rect.y_start,
            facecolor=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
            label=label,
        )
        ax.add_patch(patch)

    if seen_labels:
        ax.legend(loc="upper right", fontsize=8)


def main() -> None:
    rectangles, frame_times, usable_width = collect_rectangles()

    if not rectangles:
        print("No vehicles were placed; nothing to visualize.")
        return

    max_y = max(rect.y_end for rect in rectangles)

    out_dir = "frames"
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 7))

    for i, t in enumerate(frame_times):
        draw_frame(ax, rectangles, t, usable_width, max_y)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"frame_{i:03d}.png"), dpi=130)

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 7))

    def animate(i):
        draw_frame(ax, rectangles, frame_times[i], usable_width, max_y)
        return []

    from matplotlib.animation import FuncAnimation, PillowWriter

    anim = FuncAnimation(
        fig, animate, frames=len(frame_times), interval=800
    )
    anim.save("queue_animation.gif", writer=PillowWriter(fps=1))
    plt.close(fig)

    print(f"Saved {len(frame_times)} frames to '{out_dir}/'")
    print("Saved animated GIF to 'queue_animation.gif'")


if __name__ == "__main__":
    main()