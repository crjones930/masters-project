#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Sequence

import matplotlib.axes as mpla
import matplotlib.colors as mplc
import numpy as np
from numpy.typing import NDArray

import pymusic.big_array as pma
import pymusic.io as pmio
import pymusic.plotting as pmp


@dataclass(frozen=True)
class Spherical2DArrayPlot(pmp.Plot):
    array: pma.BigArray
    cmap: mplc.Colormap | None = None
    color_bounds: pmp.ArrayToBoundsFunc = pmp.BoundsFromMinMax()
    with_colorbar: bool = True
    with_labels: bool = True
    label: str = ""
    r_axis: str = "x1"
    theta_axis: str = "x2"
    log_scale: bool = False

    def draw_on(self, ax: mpla.Axes) -> None:
        assert self.array.axes == (self.r_axis, self.theta_axis)

        def centres_to_edges(centres: np.ndarray) -> np.ndarray:
            """Compute approximate cell edges from the `centres`"""
            dx = np.diff(centres)
            return np.concatenate(
                (centres[:-1] - 0.5 * dx, centres[-2:] + 0.5 * dx[-2:])
            )

        # Set up the grid and coordinates
        r_edge = centres_to_edges(np.array(self.array.labels_along_axis("x1")))
        theta_edge = centres_to_edges(np.array(self.array.labels_along_axis("x2")))
        R, Theta = np.meshgrid(r_edge, theta_edge)
        X = R * np.sin(Theta)
        Y = R * np.cos(Theta)

        # Set up the plot, hide axes
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for direc in ax.spines:
            ax.spines[direc].set_visible(False)

        # Inner and outer radii text labels
        if self.with_labels:
            ax.set_yticks([r_edge[0], r_edge[-1]])

        # Draw the colormesh
        arr = self.array.array()
        vmin, vmax = self.color_bounds(arr)
        norm = (
            mplc.LogNorm(vmin, vmax) if self.log_scale else mplc.Normalize(vmin, vmax)
        )
        mesh = ax.pcolormesh(X, Y, arr.T, cmap=self.cmap, norm=norm, rasterized=True)

        if self.with_colorbar:
            ax.figure.colorbar(mesh, ax=ax, label=self.label)

        np.savetxt("data.csv", arr.T, delimiter=",")
        


class VorticityFromData(pma.BigArray[np.floating]):
    def __init__(self, data: pma.BigArray[np.floating]) -> None:
        assert "x1" in data.axes and "x2" in data.axes and "var" in data.axes
        assert "vel_1" in data.labels_along_axis("var")
        assert "vel_2" in data.labels_along_axis("var")
        self._data = data

    def array(self) -> NDArray[np.floating]:
        r = np.array(self._data.labels_along_axis("x1"), dtype=np.float64)
        t = np.array(self._data.labels_along_axis("x2"), dtype=np.float64)
        vr, vt = np.moveaxis(
            self._data.take(["vel_1", "vel_2"], "var").array(),
            source=self._data.iaxis("var"),
            destination=0,
        )
        drvtdr = np.gradient(
            np.apply_along_axis(lambda vt_s: r * vt_s, axis=self.iaxis("x1"), arr=vt),
            r,
            axis=self.iaxis("x1"),
        )
        dvrdt = np.gradient(vr, t, axis=self.iaxis("x2"))
        return np.apply_along_axis(
            lambda s: 1.0 / r * s, axis=self.iaxis("x1"), arr=(drvtdr - dvrdt)
        )

    def _index(self) -> pma.IndexNd:
        return self._data.index.drop("var")

    def sum(self, axis: str) -> pma.BigArray[np.floating]:
        assert axis in self.axes
        if axis == "x1" or axis == "x2":
            return pma.SummedArray(self, axis)
        else:
            return VorticityFromData(self._data.sum(axis))

    def take(self, labels: Sequence[object], axis: str) -> pma.BigArray[np.floating]:
        assert axis in self.axes
        if axis == "x1" or axis == "x2":
            return pma.TakeArray(self._data, labels, axis)
        else:
            return VorticityFromData(self._data.take(labels, axis))


def AbsVelFromData(data: pma.BigArray[np.floating]) -> pma.BigArray[np.floating]:
    assert "var" in data.axes
    assert "vel_1" in data.labels_along_axis("var")
    assert "vel_2" in data.labels_along_axis("var")
    return pma.DerivedFieldArray(
        array=data,
        axis="var",
        inputs=["vel_1", "vel_2"],
        formula_func=lambda v1, v2: np.sqrt(v1**2 + v2**2),
    )


def add_fields_to_vars(
    data: pma.BigArray[np.floating],
    fields: Sequence[pma.BigArray[np.floating]],
    names: Sequence["str"],
) -> pma.BigArray[np.floating]:
    fields_array = pma.StackedArray(
        tuple(fields),
        pma.ItemsIndex1d("var", tuple(names)),
        data.iaxis("var"),
    )
    return pma.ConcatenatedArray([data, fields_array], "var")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dumps-glob",
        required=True,
        type=str,
        help="glob for MUSIC dump files",
    )
    parser.add_argument(
        "-f",
        "--field",
        required=True,
        type=str,
        help="MUSIC field to plot, from vel_*, vel_abs, e_int_spec, density, scalar_*, vorticity",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        type=str,
        help="colour map to use",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        required=True,
        help="maximum of the colour scale",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        required=True,
        help="minimum of the colour scale",
    )
    parser.add_argument(
        "--phi",
        type=float,
        required=False,
        help="if the simulation is 3D, provide this argument to select the phi angle",
    )
    parser.add_argument(
        "--boundary-conds",
        type=str,
        choices=("reflective", "periodic"),
        required=True,
        help="the horizontal (theta) boundary conditions",
    )
    parser.add_argument(
        "--format",
        default="png",
        type=str,
        help="Save the figure in this format",
    )
    args = parser.parse_args()

    dump_files: list[str] = sorted(glob(args.dumps_glob))
    print(f"Plotting {len(dump_files)} MUSIC dumps")
    ang_bc = (
        pmio.ReflectiveArrayBC()
        if args.boundary_conds == "reflective"
        else pmio.PeriodicArrayBC()
    )
    boundary_conds = [pmio.ReflectiveArrayBC()] + [ang_bc] * (
        1 if args.phi is None else 2
    )
    dumps = [
        pmio.MusicDump.from_file(
            pmio.MusicNewFormatDumpFile(fname),
            boundary_conds,
            pmio.KnownMusicVariables(),
        )
        for fname in dump_files
    ]
    dump_arrays = [pmio.MusicDumpArray(dump, verbose=False) for dump in dumps]
    field_arrays = [
        add_fields_to_vars(
            array,
            (AbsVelFromData(array), VorticityFromData(array)),
            ("vel_abs", "vorticity"),
        ).xs(args.field, "var")
        for array in dump_arrays
    ]

    for fp, arr in zip(dump_files, field_arrays):
        print(fp)
        out = Path(
            "figures", args.field + "_" + Path(fp).with_suffix("." + args.format).name
        )
        x3 = None
        if args.phi is not None:
            x3s = np.array(arr.labels_along_axis("x3"))
            x3 = x3s[np.argmin(np.abs(x3s - args.phi))]
        pmp.SinglePlotFigure(
            Spherical2DArrayPlot(
                array=arr if x3 is None else arr.xs(x3, "x3"),
                cmap=args.cmap,
                color_bounds=pmp.FixedBounds(args.vmin, args.vmax),
                with_labels=False,
                label=args.field,
            ),
        ).save_to(out)

    return 0


if __name__ == "__main__":
    exit(main())
