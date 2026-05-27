#!/usr/bin/env python3

"""
Compute time-averaged radial profiles using the lscale structure files
"""

from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_struc_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=" ", header=2, skipinitialspace=True, index_col=0)
    # clean the data frame a little more in case we end up with an unnamed column
    # columns with unnamed values
    drop_list = [j for j in df.columns if j.find("Unnamed") != -1]
    return df.drop(drop_list, axis=1)


def read_struc_header(path: Path) -> dict[str, float]:
    df = pd.read_csv(path, sep=" ", nrows=1, header=0, skipinitialspace=True)
    return {col: float(df.loc[0][col]) for col in df.columns}


def write_struc_file(
    path: Path, df: pd.DataFrame, header: dict[str, float] = dict(averaged=1.0)
):
    assert not path.exists()
    assert "radius" in df.columns
    with open(path, "w") as f:
        f.write(" ".join(k for k in header) + "\n")
        f.write(" ".join([f"{header[k]:E}" for k in header]) + "\n")
    df.to_csv(path, sep=" ", float_format="%E", header=True, index=True, mode="a")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "prefix",
        type=str,
        help="prefic for the structure files to read, e.g. 'lscale/struc_'",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file",
    )
    args = parser.parse_args()

    file_paths = sorted(glob(args.prefix + "".join(["[0-9]"] * 8)))
    assert len(file_paths) > 1

    # Number of time points to average over
    nt = len(file_paths)

    # Need to sum the squares of RMS quantities
    cols_to_sum = ["radius", "brunt_vaissala", "nabla", "nabla_adiab", "flux_heat_instant", "flux_rad"]
    cols_to_sum_sq = ["vel_rms", "vel_1_rms", "vel_2_rms", "e_kin"]

    # Initialise summed data and headers with zeros
    struc0 = read_struc_file(file_paths[0])
    head0 = read_struc_header(file_paths[0])
    summed: pd.DataFrame = struc0[cols_to_sum] * 0.0
    summed_sq: pd.DataFrame = struc0[cols_to_sum_sq] * 0.0
    summed_header = {k: 0.0 for k in head0}

    # Sum all the files together
    for fp in tqdm(file_paths, desc="Reading files..."):
        struc = read_struc_file(fp)
        head = read_struc_header(fp)
        summed += struc[cols_to_sum]
        summed_sq += struc[cols_to_sum_sq]**2
        for k in head:
            summed_header[k] += head[k]

    # Convert to averages
    summed /= nt
    summed_sq /= nt
    df = summed.join(np.sqrt(summed_sq), validate="one_to_one", how="inner")
    summed_header = {k: summed_header[k] / nt for k in summed_header if k != "time"}

    # Write the averaged file with the start and end times
    tstart = read_struc_header(file_paths[0])["time"]
    tend = read_struc_header(file_paths[-1])["time"]
    write_struc_file(
        Path(args.output), df, dict(time_start=tstart, time_end=tend, **summed_header)
    )


if __name__ == "__main__":
    main()
