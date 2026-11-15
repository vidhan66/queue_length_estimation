
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
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


def collect_queue_profile_from_csv(csv_path):
    """Calculate event-based queue-length proxy from CSV."""

    import pandas as pd

    df = pd.read_csv(csv_path)

    required = [
        "time",
        "lane",
        "length_m",
        "longitudinal_spacing_m"
    ]

    if not all(col in df.columns for col in required):
        raise ValueError("Required queue calculation columns missing.")

    df = df.sort_values("time").copy()

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["length_m"] = pd.to_numeric(df["length_m"], errors="coerce")
    df["longitudinal_spacing_m"] = pd.to_numeric(
        df["longitudinal_spacing_m"], errors="coerce"
    )

    df = df.dropna(subset=required)

    # Initial queue state
    samples = [QueueSample(0.0, 0.0)]

    # Maintain accumulated occupied length separately for each lane
    lane_lengths = {}

    # Process every vehicle arrival event
    for _, vehicle in df.iterrows():

        lane = vehicle["lane"]

        vehicle_length = float(vehicle["length_m"])
        spacing = float(vehicle["longitudinal_spacing_m"])

        if lane not in lane_lengths:
            lane_lengths[lane] = 0.0

        # Update occupied length of the respective lane
        lane_lengths[lane] += vehicle_length + spacing

        # Maximum occupied length among all lanes
        max_queue_length = max(lane_lengths.values())

        samples.append(
            QueueSample(
                time=float(vehicle["time"]),
                queue_length=max_queue_length
            )
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

def plot_vissim_csv(
    csv_path="VISSIM_Vehicle_Data(1).csv",
    output_path="vissim_vehicle_profile.png"
):
    """Generate vehicle data and queue profile plots."""

    df = pd.read_csv(csv_path)

    required_columns = [
        "time",
        "vehicle_id",
        "lane",
        "vehicle_type",
        "longitudinal_spacing_m",
        "total_vehicles_seen"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values("time").copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    if df.empty:
        raise ValueError("No valid vehicle data found.")

    # ==============================================
    # PART 1: CSV VEHICLE DATA PLOTS
    # ==============================================

    # ==============================================
    # PART 2: QUEUE LENGTH AND SMOOTHING PLOTS
    # ==============================================

    samples = collect_queue_profile_from_csv(csv_path)
    times, lengths = prepare_samples(samples)
    
    intervals = np.diff(times)
    positive_intervals = intervals[intervals > 0]

    final_width = (
        np.median(positive_intervals)
        if len(positive_intervals)
        else 1.0
    )

    starts = times
    ends = np.append(
        times[1:],
        times[-1] + final_width
    )

    heights = lengths

    # ==============================================
    # CREATE ALL SIX PLOTS
    # ==============================================

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 15)
    )

    # ----------------------------------------------
    # 1. Cumulative vehicle arrivals
    # ----------------------------------------------

    cumulative = np.arange(1, len(df) + 1)

    axes[0, 0].step(
        df["time"],
        cumulative,
        where="post",
        color="#4C72B0",
        linewidth=2
    )

    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Cumulative vehicles")
    axes[0, 0].set_title("Cumulative Vehicle Arrivals")
    axes[0, 0].grid(alpha=0.3)

    # ----------------------------------------------
    # 2. Vehicle arrivals per minute
    # ----------------------------------------------

    df["time_interval"] = (
        df["time"] // 60
    ) * 60

    arrivals = df.groupby("time_interval").size()

    axes[0, 1].bar(
        arrivals.index,
        arrivals.values,
        width=50,
        color="#55A868",
        edgecolor="black",
        alpha=0.85
    )

    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Number of vehicles")
    axes[0, 1].set_title("Vehicle Arrivals per Minute")
    axes[0, 1].grid(axis="y", alpha=0.3)

    # ----------------------------------------------
    # 3. Vehicle type distribution
    # ----------------------------------------------

    vehicle_counts = df["vehicle_type"].value_counts()

    axes[1, 0].bar(
        vehicle_counts.index,
        vehicle_counts.values,
        color="#C44E52",
        edgecolor="black",
        alpha=0.85
    )

    axes[1, 0].set_xlabel("Vehicle type")
    axes[1, 0].set_ylabel("Number of vehicles")
    axes[1, 0].set_title("Vehicle Type Distribution")
    axes[1, 0].grid(axis="y", alpha=0.3)

    # ----------------------------------------------
    # 4. Longitudinal spacing versus time
    # ----------------------------------------------

    axes[1, 1].scatter(
        df["time"],
        df["longitudinal_spacing_m"],
        color="#8172B3",
        s=35,
        alpha=0.8
    )

    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Longitudinal spacing (m)")
    axes[1, 1].set_title("Longitudinal Spacing over Time")
    axes[1, 1].grid(alpha=0.3)

    # ----------------------------------------------
    # 5. Queue length interval histogram
    # ----------------------------------------------

    right_limit = plot_histogram(
        axes[2, 0],
        times,
        lengths
    )

    # ----------------------------------------------
    # 6. KDE-smoothed queue growth trend
    # ----------------------------------------------

    plot_kde(
        axes[2, 1],
        starts,
        ends,
        heights
    )

    axes[2, 0].set_xlim(times[0], right_limit)
    axes[2, 1].set_xlim(times[0], right_limit)

    # ==============================================
    # FINAL FORMATTING AND SAVING
    # ==============================================

    fig.suptitle(
        "VISSIM Traffic and Queue Profile Analysis",
        fontsize=16,
        fontweight="bold"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ==============================================
    # PRINT SUMMARY
    # ==============================================

    print("\n========== VEHICLE DATA SUMMARY ==========")

    print(f"Total vehicles: {len(df)}")
    print(
        f"Simulation duration: "
        f"{df['time'].max():.2f} seconds"
    )

    print("\nVehicle type counts:")
    print(vehicle_counts)

    print("\nLane-wise vehicle counts:")
    print(df["lane"].value_counts())

    print("\n========== QUEUE PROFILE SUMMARY ==========")

    for t, q in zip(times, lengths):
        print(f"Time = {t:7.2f} s | Queue = {q:6.2f} m")

    print(f"\nSaved '{output_path}'")

def main():

    samples = collect_queue_profile()
    times, lengths = prepare_samples(samples)

    print("\nRecorded queue profile:")

    for t, q in zip(times, lengths):
        print(f"t = {t:7.2f} s | Queue = {q:6.2f} m")

    # Construct intervals, including the final observation.
    # intervals = np.diff(times)
    # positive_intervals = intervals[intervals > 0]

    # final_width = (
    #     np.median(positive_intervals)
    #     if len(positive_intervals)
    #     else 1.0
    # )

    # starts = times
    # ends = np.append(times[1:], times[-1] + final_width)
    # heights = lengths

    # fig, axes = plt.subplots(
    #     2,
    #     1,
    #     figsize=(11, 9),
    #     sharex=True
    # )

    # right_limit = plot_histogram(
    #     axes[0],
    #     times,
    #     lengths
    # )

    # plot_kde(
    #     axes[1],
    #     starts,
    #     ends,
    #     heights
    # )

    # axes[0].set_xlim(times[0], right_limit)
    # axes[1].set_xlim(times[0], right_limit)

    # fig.tight_layout()

    # fig.savefig(
    #     "queue_profile.png",
    #     dpi=160,
    #     bbox_inches="tight"
    # )

    # plt.close(fig)

    # print("\nSaved 'queue_profile.png'")
    plot_vissim_csv(
        csv_path="VISSIM_Vehicle_Data(1).csv",
        output_path="vissim_vehicle_profile.png"
    )

if __name__ == "__main__":
    main()
