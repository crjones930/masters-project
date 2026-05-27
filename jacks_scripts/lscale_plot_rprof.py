#!/usr/bin/env python3

"""
Plots radial profiles from lscale struc_* files
"""

from argparse import ArgumentParser, FileType
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_struc_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=" ", header=2, skipinitialspace=True, index_col=0)
    # clean the data frame a little more in case we end up with an unnamed column
    # columns with unnamed values
    drop_list = [j for j in df.columns if j.find("Unnamed") != -1]
    return df.drop(drop_list, axis=1)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        required=True,
        type=FileType(mode="r"),
        help="The structure files to read",
    )
    parser.add_argument(
        "-p",
        "--plot",
        nargs="+",
        required=False,
        type=str,
        help="Columns to plot, use a plus (+) sign to plot on the same set of axes, e.g. 'vel_1_rms+vel_2_rms'",
    )
    parser.add_argument(
        "-r",
        "--r-marks",
        nargs="*",
        default=[],
        type=float,
        help="radial positions to mark",
    )
    parser.add_argument(
        "--log",
        default=False,
        action="store_true",
        help="Plot with a logarithmic y axis scale",
    )
    parser.add_argument(
        "--format",
        default="png",
        type=str,
        help="Save the figure in this format",
    )
    args = parser.parse_args()

    fpaths = [Path(s.name) for s in args.files]
    strucs = [read_struc_file(f) for f in fpaths]

    def print_columns_and_exit():
        print("Available columns via -p/--plot:")
        print("\n".join(strucs[0].columns.tolist()))
        exit(1)

    if not args.plot:
        print_columns_and_exit()
    for cols in args.plot:
        if not all(col in strucs[0].columns for col in cols.split("+")):
            print("Invalid column name provided.")
            print_columns_and_exit()

    fignames = [f.with_suffix("." + args.format) for f in fpaths]

    N = len(args.plot)
    ncols = int(np.floor(np.sqrt(N)))
    nrows = int(np.ceil(N / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False, layout="constrained"
    )
    for struc, fname in zip(strucs, fignames):
        for cols_str, ax in zip(args.plot, axes.flat):

            cols = cols_str.split("+")
            for col in cols:
                ax.plot(struc["radius"], struc[col], label=col)

            for r in args.r_marks:
                ax.axvline(x=r, ls="--", c="k", lw="0.5")

            ax.set_ylabel(cols[0])
            ax.legend()
            if args.log:
                ax.set_yscale("log")

        for ax in axes.flat[:N]:
            ax.set_xlabel("radius [cm]")

        fig.savefig(fname)
        print(f"Saved figure to {fname.as_posix()}")
        plt.close(fig)
        fig, axes = plt.subplots(
            nrows=N, figsize=(5, 3 * N), squeeze=False, layout="constrained"
        )

    plt.close(fig)


if __name__ == "__main__":
    main()
