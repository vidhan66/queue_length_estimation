
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vehicle_queue_simulation import run_default_scenario


@dataclass
class QueueSample:
    time: float
    queue_length: float


def collect_queue_profile() -> list[QueueSample]:
    """Collect maximum queue length after every arrival event."""

    samples = [QueueSample(0.0, 0.0)]

    def on_event(event: dict) -> None:
        if event["type"] != "arrival":
            return

        samples.append(
            QueueSample(
                time=float(event["time"]),
                queue_length=float(event["max_queue_length"]),
            )
        )

    run_default_scenario(
        track_positions=False,
        on_event=on_event
    )

    return samples


def prepare_samples(
    samples: list[QueueSample],
) -> tuple[np.ndarray, np.ndarray]:

    if not samples:
        raise ValueError("No queue samples collected.")

    # Keep the last state if multiple events share a timestamp.
    by_time = {}

    for sample in samples:
        by_time[float(sample.time)] = float(sample.queue_length)

    times = np.array(list(by_time.keys()))
    lengths = np.array(list(by_time.values()))

    order = np.argsort(times)

    return times[order], lengths[order]


def plot_histogram(ax, times, lengths):

    # Determine the duration of each queue state.
    intervals = np.diff(times)

    positive_intervals = intervals[intervals > 0]

    # The last observation has no subsequent event.
    # Extend its bar by the median observed interval for display.
    final_width = (
        np.median(positive_intervals)
        if len(positive_intervals)
        else 1.0
    )

    edges = np.append(
        times,
        times[-1] + final_width
    )

    widths = np.diff(edges)

    ax.bar(
        edges[:-1],
        lengths,
        width=widths,
        align="edge",
        edgecolor="black",
        color="#4C72B0",
        alpha=0.85
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Maximum queue length (m)")
    ax.set_title("Queue length over time (interval histogram)")
    ax.grid(axis="y", alpha=0.25)

    ax.annotate(
        f"{lengths[-1]:.2f} m",
        xy=(times[-1], lengths[-1]),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        fontweight="bold"
    )

    return edges[-1]



def plot_kde(ax, starts, ends, heights):

    widths = ends - starts
    mids = starts + widths / 2.0

    if len(mids) < 2:
        ax.set_title("Insufficient data for KDE")
        return

    # Bandwidth using Silverman's rule
    std = np.std(mids)
    n = len(mids)

    bandwidth = 1.06 * std * n ** (-1 / 5)

    if bandwidth <= 1e-9:
        bandwidth = 1.0

    grid = np.linspace(
        starts.min(),
        ends.max(),
        500
    )

    # Gaussian kernel weights
    z = (grid[:, None] - mids[None, :]) / bandwidth

    kernel = np.exp(-0.5 * z ** 2)

    # Weight observations by their interval durations
    weighted_kernel = kernel * widths[None, :]

    numerator = weighted_kernel @ heights
    denominator = weighted_kernel.sum(axis=1)

    smoothed = numerator / np.maximum(denominator, 1e-12)

    ax.plot(
        grid,
        smoothed,
        color="#C44E52",
        linewidth=2,
        label="Duration-weighted kernel regression"
    )

    ax.scatter(
        mids,
        heights,
        color="black",
        s=25,
        label="Histogram midpoints",
        zorder=5
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue length (m)")
    ax.set_title("KDE-smoothed queue growth trend")
    ax.legend()
    ax.grid(True, alpha=0.25)


def main():

    samples = collect_queue_profile()
    times, lengths = prepare_samples(samples)

    print("\nRecorded queue profile:")

    for t, q in zip(times, lengths):
        print(f"t = {t:7.2f} s | Queue = {q:6.2f} m")

    # Construct intervals, including the final observation.
    intervals = np.diff(times)
    positive_intervals = intervals[intervals > 0]

    final_width = (
        np.median(positive_intervals)
        if len(positive_intervals)
        else 1.0
    )

    starts = times
    ends = np.append(times[1:], times[-1] + final_width)
    heights = lengths

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        sharex=True
    )

    right_limit = plot_histogram(
        axes[0],
        times,
        lengths
    )

    plot_kde(
        axes[1],
        starts,
        ends,
        heights
    )

    axes[0].set_xlim(times[0], right_limit)
    axes[1].set_xlim(times[0], right_limit)

    fig.tight_layout()

    fig.savefig(
        "queue_profile.png",
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved 'queue_profile.png'")


if __name__ == "__main__":
    main()
