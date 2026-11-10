
import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from simulation import CrowdSimulation


# =================================================
# RUN ONE EXPERIMENT
# =================================================

def run_one_experiment(
    exit_width,
    seed,
    number_of_particles,
    dt,
    max_seconds
):

    simulation = CrowdSimulation(
        width=12.0,
        height=8.0,
        number_of_particles=number_of_particles,
        exit_width=exit_width,
        random_seed=seed
    )

    max_steps = int(max_seconds / dt)

    for step in range(max_steps):

        if len(simulation.particles) == 0:
            break

        simulation.step(dt)

    # Check if all pedestrians exited
    completed = (
        len(simulation.particles) == 0
    )

    exited = len(
        simulation.finished_particles
    )

    remaining = len(
        simulation.particles
    )

    elapsed_time = simulation.elapsed_time

    if completed:
        evacuation_time = simulation.evacuation_time
    else:
        evacuation_time = None

    # Average exit flow over simulated duration
    if elapsed_time > 0:
        flow_rate = exited / elapsed_time
    else:
        flow_rate = 0.0

    return {
        "exit_width_m": exit_width,
        "seed": seed,
        "initial_people": number_of_particles,
        "exited_people": exited,
        "remaining_people": remaining,
        "completed": completed,
        "evacuation_time_s": (
            round(evacuation_time, 3)
            if evacuation_time is not None
            else ""
        ),
        "simulated_time_s": round(elapsed_time, 3),
        "avg_exit_flow_people_per_s": round(
            flow_rate, 4
        )
    }


# =================================================
# SAVE CSV
# =================================================

def save_csv(filename, records):

    if not records:
        return

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)


# =================================================
# SUMMARIZE RESULTS
# =================================================

def summarize_results(records, exit_widths):

    summaries = []

    for width in exit_widths:

        trials = [
            record
            for record in records
            if record["exit_width_m"] == width
        ]

        completed_trials = [
            record
            for record in trials
            if record["completed"]
        ]

        times = [
            float(record["evacuation_time_s"])
            for record in completed_trials
        ]

        exited_counts = [
            record["exited_people"]
            for record in trials
        ]

        # Only report mean evacuation time
        # when all trials for this width finish.
        all_completed = (
            len(completed_trials) == len(trials)
        )

        if all_completed and times:

            mean_time = round(
                float(np.mean(times)), 3
            )

            if len(times) > 1:
                std_time = round(
                    float(np.std(times, ddof=1)), 3
                )
            else:
                std_time = 0.0

        else:

            mean_time = ""
            std_time = ""

        summaries.append({
            "exit_width_m": width,
            "completed_runs": len(completed_trials),
            "total_runs": len(trials),
            "mean_evacuation_time_s": mean_time,
            "std_evacuation_time_s": std_time,
            "mean_exited_people": round(
                float(np.mean(exited_counts)), 2
            )
        })

    return summaries


# =================================================
# GENERATE COMPARISON GRAPH
# =================================================

def generate_graph(summaries, output_dir):

    valid_results = [
        result
        for result in summaries
        if result["mean_evacuation_time_s"] != ""
    ]

    if not valid_results:

        print(
            "No fully completed width groups. "
            "Evacuation-time graph was not generated."
        )

        return

    widths = [
        result["exit_width_m"]
        for result in valid_results
    ]

    times = [
        result["mean_evacuation_time_s"]
        for result in valid_results
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        widths,
        times,
        marker="o",
        linewidth=2
    )

    plt.xlabel("Exit Width (m)")
    plt.ylabel("Mean Evacuation Time (s)")

    plt.title(
        "Effect of Exit Width on Evacuation Time"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "exit_width_vs_time.png",
        dpi=160
    )

    plt.close()


# =================================================
# MAIN EXPERIMENT
# =================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--people",
        type=int,
        default=100
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=3
    )

    parser.add_argument(
        "--max-seconds",
        type=float,
        default=120
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.02
    )

    parser.add_argument(
        "--widths",
        type=float,
        nargs="+",
        default=[0.6, 1.2, 2.4]
    )

    args = parser.parse_args()

    if (
        args.people < 1
        or args.seeds < 1
        or args.dt <= 0
        or args.max_seconds <= 0
    ):
        parser.error(
            "People, seeds, dt and max-seconds "
            "must be positive."
        )

    if any(
        width <= 0 or width > 8.0
        for width in args.widths
    ):
        parser.error(
            "Exit widths must be between 0 and 8 meters."
        )

    # Output directory
    output_dir = (
        Path(__file__).resolve().parent
        / "experiment_results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    # Run all experiment configurations
    for seed in range(args.seeds):

        for width in args.widths:

            print()
            print("=" * 60)

            print(
                f"Experiment: Width={width:.2f}m, "
                f"Seed={seed}"
            )

            result = run_one_experiment(
                exit_width=width,
                seed=seed,
                number_of_particles=args.people,
                dt=args.dt,
                max_seconds=args.max_seconds
            )

            records.append(result)

            if result["completed"]:

                print(
                    f"Evacuation Time: "
                    f"{result['evacuation_time_s']}s"
                )

            else:

                print(
                    "Evacuation not completed "
                    "within time limit."
                )

            print(
                f"Exited: "
                f"{result['exited_people']}"
                f"/{args.people}"
            )

    # Save individual trials
    save_csv(
        output_dir / "trials.csv",
        records
    )

    # Calculate summaries
    summaries = summarize_results(
        records,
        args.widths
    )

    # Save summary table
    save_csv(
        output_dir / "summary.csv",
        summaries
    )

    # Generate graph
    generate_graph(
        summaries,
        output_dir
    )

    # Final results
    print()
    print("=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    for result in summaries:

        print(
            f"Exit Width: "
            f"{result['exit_width_m']:.2f} m"
        )

        print(
            f"Completed: "
            f"{result['completed_runs']}"
            f"/{result['total_runs']}"
        )

        print(
            f"Mean Evacuation Time: "
            f"{result['mean_evacuation_time_s']}"
        )

        print("-" * 60)

    print(
        f"Results saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
