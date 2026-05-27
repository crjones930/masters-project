#!/usr/bin/env python3

"""
Plots time series data from the `time_series.dat` file produced by lscale
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_tseries_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=" ", header=0, skipinitialspace=True)
    # clean the data frame a little more in case we end up with an unnamed column
    # columns with unnamed values
    drop_list = [j for j in df.columns if j.find("Unnamed") != -1]
    return df.drop(drop_list, axis=1).sort_values(by="time")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--files",
        type=str,
        nargs="+",
        help="paths to time series data files",
    )
    parser.add_argument(
        "-p",
        "--plot",
        nargs="+",
        required=True,
        type=str,
        help="Columns to plot",
    )
    parser.add_argument(
        "--log-time",
        default=False,
        action="store_true",
        help="Plot time on a log scale",
    )
    parser.add_argument(
        "--t0",
        type=float,
        default=0.0,
        help="time to start the plot at",
    )
    parser.add_argument(
        "--format",
        default="png",
        type=str,
        help="Save the figure in this format",
    )
    args = parser.parse_args()

    N = len(args.plot)
    fig, axes = plt.subplots(
        nrows=N, figsize=(5, 3 * N), squeeze=False, layout="constrained"
    )

    for file in args.files:
        fpath = Path(file)
        print(fpath.resolve().as_posix())
        tseries = read_tseries_file(fpath)
        tseries = tseries[tseries["time"] >= args.t0]
        if not all(col in tseries.columns for col in args.plot):
            print("Invalid column name provided. Available columns:")
            print("\n".join(tseries.columns.tolist()))
            exit(1)

        for col, ax in zip(args.plot, axes.flat):
            if "log" in col:
                ax.plot(tseries["time"], tseries[col], label=file)
                ax.set_ylabel(col)
            else:
                ax.plot(tseries["time"], np.log10(tseries[col]), label=file)
                ax.set_ylabel(f"log_{col}")
            if args.log_time:
                ax.set_xscale("log")

    for ax in axes.flat:
        ax.legend()
    for ax in axes.flat[:-1]:
        ax.set_xticklabels([])
    axes.flat[-1].set_xlabel("time [s]")

    fig.savefig("time_series.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
