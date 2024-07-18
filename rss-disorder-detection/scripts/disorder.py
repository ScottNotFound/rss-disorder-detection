import argparse
import colorsys
import functools
import io
import itertools
import json
import math
import operator
import os
import pickle
import random
import re
import sys
import tempfile
import tarfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from multiprocessing import Pool, shared_memory
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.pool import AsyncResult
from pathlib import Path
from typing import (
    Any,
    Collection,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    TypeAlias,
    TypeVar,
    Union,
)

import ase.db
import ase.io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.signal
import seaborn as sns
import sklearn.decomposition
import sklearn.metrics
import spglib
from ase.atoms import Atoms
from ase.io.res import Res
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel, REMatchKernel
from joblib import Parallel, delayed, load, dump
from numpy.typing import ArrayLike, NDArray
from pymatgen.analysis.diffraction import xrd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.res import AirssProvider, ParseError, ResParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.utils import gen_even_slices

kB = 8.617333262e-5  # eV/T

_T = TypeVar("_T")


class c:
    """
    Column shortform names
    These are just easy aliases for column names in the dataframes.
    """

    x = "composition"
    y = "unit_formation_enthalpy"
    p = "pressure"
    v = "volume"
    vpa = "volume_per_atom"
    rp = "rounded_pressure"
    e = "element"
    ph = "phase"
    s = "seed"
    e1 = "E1"
    e2 = "E2"
    e_reassigned = False
    n1 = f"num_{e1}"
    n2 = f"num_{e2}"
    te = "total_enthalpy"
    ue = "unit_enthalpy"
    ufe = y
    n = "num_atoms"
    ufe_d = ufe + "_dia"
    ufe_g = ufe + "_gra"
    h = "hull_distance"
    sg = "spacegroup"
    sgno = "spg_int_no"
    a = "a"
    b = "b"
    c = "c"
    al = "alpha"
    be = "beta"
    ga = "gamma"
    isd = "integrated_spin_density"

    hdb = "hdbscan"
    dbs = "dbscan"
    opt = "optics"

    clc = "clust_cen"
    clm = "clust_met"
    cll = "clust_lab"
    clp = "clust_prb"
    cln = "clust_num"

    G = "GibbsProxy"
    hG = "GP_hdist"

    i = 'icat'


def set_params() -> None:
    """
    Parameters for matplotlib plots.
    """
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.size"] = 20
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.it"] = "Arial:italic"
    mpl.rcParams["mathtext.bf"] = "Arial:bold"
    mpl.rcParams["axes.labelweight"] = 400
    mpl.rcParams["axes.labelsize"] = 20
    mpl.rcParams["axes.labelpad"] = 14
    mpl.rcParams["axes.titleweight"] = 400
    mpl.rcParams["axes.titlesize"] = 20
    mpl.rcParams["axes.titlepad"] = 28
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.spines.left"] = True
    mpl.rcParams["axes.spines.bottom"] = True
    mpl.rcParams["axes.xmargin"] = 0
    mpl.rcParams["axes.ymargin"] = 0
    mpl.rcParams["axes.unicode_minus"] = True
    mpl.rcParams["axes.linewidth"] = 1.5
    mpl.rcParams["xtick.major.size"] = 10
    mpl.rcParams["xtick.major.width"] = 1.5
    mpl.rcParams["xtick.minor.size"] = 5
    mpl.rcParams["xtick.minor.width"] = 1.5
    mpl.rcParams["xtick.labelsize"] = 20
    mpl.rcParams["ytick.major.size"] = 10
    mpl.rcParams["ytick.major.width"] = 1.5
    mpl.rcParams["ytick.minor.size"] = 5
    mpl.rcParams["ytick.minor.width"] = 1.5
    mpl.rcParams["ytick.labelsize"] = 20
    mpl.rcParams["legend.fontsize"] = 20
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["legend.facecolor"] = "none"
    mpl.rcParams["figure.figsize"] = (8, 6)
    mpl.rcParams["figure.dpi"] = 72
    mpl.rcParams["lines.linewidth"] = 2
    mpl.rcParams["lines.color"] = "black"


structure_data_t: TypeAlias = dict[str, Union[str, int, float]]


def data_from_provider(provider: AirssProvider) -> structure_data_t:
    """
    Builds data from the pymatgen airss res reader.

    This takes some object that provides access to information in a file that represents
    a crystal structure. In this case, the provider provides access to res files that come
    from an AIRSS search. In order to build data from other file types, an alternative function
    similar to this one will need to be written that returns a dict of tha same information, but
    that gets that information from the file in its own way.

    Must return a dict with at least:
    seed
    pressure
    rounded pressure
    number of first element
    number of second element
    volume
    energy
    international spacegroup number
    """
    structure = provider.structure
    lattice = structure.lattice
    spg_analyzer = SpacegroupAnalyzer(structure)
    comp = structure.composition.element_composition
    if c.e_reassigned:
        e1, e2 = c.e1, c.e2
    else:
        elements = comp.elements
        e1 = elements[0].symbol
        e2 = elements[1].symbol
    comp1 = comp.get(e1)
    comp2 = comp.get(e2)
    return {
        c.s: provider.seed,
        c.p: provider.pressure,
        c.rp: round(provider.pressure),
        c.n1: comp1,
        c.n2: comp2,
        c.v: provider.volume,
        c.te: provider.energy,
        c.sgno: spg_analyzer.get_space_group_number(),
        c.a: lattice.a,
        c.b: lattice.b,
        c.c: lattice.c,
        c.al: lattice.alpha,
        c.be: lattice.beta,
        c.ga: lattice.gamma,
    }


def data_from_aseres(res: Res) -> structure_data_t:
    """
    Builds data from the ase res reader.
    """
    atoms: Atoms = res.atoms
    spglibdataset: dict[str, Any] | None = spglib.get_symmetry_dataset(
        (
            atoms.get_cell(),
            atoms.get_scaled_positions(),
            atoms.get_atomic_numbers(),
        ),  # type: ignore
        symprec=1e-5,
    )
    if spglibdataset is None:
        raise ValueError(f"Could not find the spacegroup of {res.name}")
    if c.e_reassigned:
        e1, e2 = c.e1, c.e2
    else:
        elements = list(atoms.symbols.species())
        e1, e2 = elements[:2]
    comp1 = atoms.symbols.count(e1)
    comp2 = atoms.symbols.count(e2)
    cell_lat_params = atoms.cell.cellpar()
    return {
        c.s: str(res.name),
        c.p: float(res.pressure),
        c.rp: round(res.pressure),
        c.n1: comp1,
        c.n2: comp2,
        c.v: atoms.get_volume(),
        c.te: float(res.energy or "nan"),
        c.sgno: spglibdataset["number"],
        c.a: cell_lat_params[0],
        c.b: cell_lat_params[1],
        c.c: cell_lat_params[2],
        c.al: cell_lat_params[3],
        c.be: cell_lat_params[4],
        c.ga: cell_lat_params[5],
    }


def data_from_file_pymatgen(file: str) -> structure_data_t:
    """
    Returns the data dict from a filename.
    """
    return data_from_provider(AirssProvider.from_file(file))


def data_from_txt_pymatgen(src: str) -> structure_data_t:
    """
    Returns the data dict from the text of a file.
    """
    return data_from_provider(AirssProvider.from_str(src))


def data_from_file_ase(file: str) -> structure_data_t:
    """
    Returns the data dict from a filename using the ase reader.
    """
    return data_from_aseres(Res.from_file(file))


def data_from_txt_ase(src: str) -> structure_data_t:
    """
    Returns the data dict from the text of a file using the ase reader.
    """
    return data_from_aseres(Res.from_string(src))


def iterate_tar_info(file: str) -> Generator[tuple[str, tarfile.TarInfo], Any, None]:
    """
    Iterates through a tar file and yields the text and info of each file.
    """
    with tarfile.open(file) as tf:
        try:
            tarinfo = tf.next()
            while True and tarinfo:
                tarextfile = tf.extractfile(tarinfo)
                if tarextfile:
                    src = tarextfile.read().decode()
                    yield src, tarinfo
                tarinfo = tf.next()
        except:
            return


def iterate_path_files(path: str, ext: str = ".res") -> Generator[str, Any, None]:
    """
    Iterates through files under a path and yields the filenames with full path after filtering for the extension.
    """
    for entry in os.scandir(path):
        if entry.name.endswith(ext):
            yield entry.path


def datas_from_tar(
    file: str, data_function: Callable[[str], structure_data_t] = data_from_txt_pymatgen
) -> Generator[structure_data_t, Any, None]:
    """
    Iterates through a tar file and yields the data of the structure.
    Needs a function that converts the text of the file into the data dict.
    """
    for src, tarinfo in iterate_tar_info(file):
        try:
            yield data_function(src)
        except (ValueError, ParseError):
            print(f"Error in file {tarinfo.name}, skipping.", file=sys.stderr)


def datas_from_path(
    path: str,
    data_function: Callable[[str], structure_data_t] = data_from_file_pymatgen,
    ext: str = ".res",
) -> Generator[structure_data_t, Any, None]:
    """
    Iterates through files under a path and yields the data of each .res file.
    The file extension may need to be changed for different file types.
    Needs a function that returns the data dict from the given file name.
    """
    for entrypath in iterate_path_files(path, ext=ext):
        try:
            yield data_function(entrypath)
        except (ValueError, ParseError):
            print(f"Error in file {entrypath}, skipping.", file=sys.stderr)


def generic_from_tar(
    file: str, data_function: Callable[[str], _T]
) -> Generator[_T, Any, None]:
    """
    Iterates through a tar file and yields the data of the structure.
    Needs a function that converts the text of the file into some type.
    """
    for src, tarinfo in iterate_tar_info(file):
        try:
            yield data_function(src)
        except (ValueError, ParseError):
            print(f"Error in file {tarinfo.name}, skipping.", file=sys.stderr)


def generic_from_path(
    path: str,
    data_function: Callable[[str], _T],
    ext: str = ".res",
) -> Generator[_T, Any, None]:
    """
    Iterates through files under a path and yields the data of each .res file.
    The file extension may need to be changed for different file types.
    Needs a function that returns the some type from the given file name.
    """
    for entrypath in iterate_path_files(path, ext=ext):
        try:
            yield data_function(entrypath)
        except (ValueError, ParseError):
            print(f"Error in file {entrypath}, skipping.", file=sys.stderr)


def generic_from_disk(
    path: str,
    provider_functions: dict[Literal["str", "file"], Callable[[str], _T]],
    ext: str = ".res",
) -> Generator[_T, Any, None]:
    """
    Iterates through a tar file or files under a path and yields the data of the structure.
    If the given path ends in '.tar' then it will treat the file as a tar file. Anything else
    will be treated as a path to search in. The given ext will be used to filter files searched for.
    The ext has no effect of the path is a tar file. Needs a function to map the source data.
    The provider_functions must be a dict that maps the description of the needed funtion to the function.
    """
    if re.match(r".*[.]tar[.]?(.*)", path):
        # if path.endswith(".tar"):
        return generic_from_tar(path, provider_functions["str"])
    else:
        return generic_from_path(path, provider_functions["file"], ext=ext)


def reciprocal_scatter_ase(
    atoms: Atoms, cutoff: float
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Calculates the hkls and scattering distances of the base lattice points of the ase atoms.
    Work in reciprocal space without the 2pi factor.
    """
    from ase.neighborlist import neighbor_list

    recip_atoms = Atoms(
        cell=atoms.cell.reciprocal().array,
        symbols=["X"],
        positions=[(0, 0, 0)],
        pbc=atoms.pbc,
    )
    i, g, hkl = neighbor_list("idS", recip_atoms, cutoff)
    return i, g, hkl


def reciprocal_scatter_pymatgen(
    structure: Structure, cutoff: float
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Calculates the hkls and scattering distances of the base lattice points of the pymatgen structure.
    Work in reciprocal space without the 2pi factor.
    """
    hkl, g, i, _ = (
        structure.lattice.reciprocal_lattice_crystallographic.get_points_in_sphere(
            [(0, 0, 0)], (0, 0, 0), cutoff, zip_results=False
        )
    )
    return i, g, hkl  # type: ignore


def reciprocal_scatter(
    structure: Union[Structure, Atoms], cutoff: float
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Calculates the hkls and scattering distances of the base lattice points. Generic version.
    Returns 3-tuple of:
        indicies (typically all 0)  type: int[]
        reciprocal distances        type: float[]
        HKLs of lattice planes      type: int[][3]

    This is independent of the atomic sites. We only need the lattice to get the
    scattering vectors. Then the scattering vectors are applied to each atomic site.

    Work in reciprocal space without the 2pi factor.
    """
    if isinstance(structure, Structure):
        return reciprocal_scatter_pymatgen(structure, cutoff)
    elif isinstance(structure, Atoms):
        return reciprocal_scatter_ase(structure, cutoff)
    else:
        raise ValueError(f"Diffraction of {type(structure)} type not supported.")


def get_scatter_coeffs(
    symbol: str, scatter_coeffs: Optional[dict[str, list[list[float]]]] = None
) -> list[list[float]]:
    """
    Finds the scattering coefficients for the given symbol.
    The scattering coefficients are represented as four pairs of floats.
    They are of the form: [[a1, b1], [a2, b2], [a3, b3], [a4, b4]].
    A dictionary to alter the coefficients used pay be provided.
    This allows the user to specify alternative scattering.
    The dict may refer to any length list as long as the last level has dimension of 2.
    The list will accordingly be padded by [0,0] or trimmed down to reach a length of 4.
    """
    try:
        if scatter_coeffs and symbol in scatter_coeffs:
            coef = scatter_coeffs[symbol]
            # must be 4 pairs so pad and trim accordingly
            coef_padded: list[list[float]] = coef + [0.0, 0.0] * (4 - len(coef))  # type: ignore
            coef_padded_trimmed = coef_padded[:4]
            coef = coef_padded_trimmed
        else:
            coef: list[list[float]] = xrd.ATOMIC_SCATTERING_PARAMS[symbol]
    except KeyError:
        raise ValueError(
            f"Unable to calculate XRD pattern as there is no scattering coefficients for {symbol}."
        )
    return coef


def scattering_centers_ase(
    atoms: Atoms, scatter_coeffs: Optional[dict[str, list[list[float]]]] = None
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Finds the scattering centers for an ase atoms.
    Does not support disorder or mixed sites.
    """
    # ase doesn't inherently handle mixed occupancies so we don't have to support it
    # just pretend they don't exist, return array of ones.
    Z_arr = np.asarray(atoms.numbers)
    fractional_coords_arr = np.ascontiguousarray(atoms.get_scaled_positions())
    coefficient_arr = np.ascontiguousarray(
        [
            get_scatter_coeffs(symbol, scatter_coeffs=scatter_coeffs)
            for symbol in atoms.get_chemical_symbols()
        ]
    )
    occupancy_arr = np.ones_like(Z_arr)
    return Z_arr, occupancy_arr, coefficient_arr, fractional_coords_arr


def scattering_centers_pymatgen(
    structure: Structure, scatter_coeffs: Optional[dict[str, list[list[float]]]] = None
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Finds the scattering centers for a pymatgen structure.
    Supports disordered or mixed sites.
    """
    # need to handle the different occupancies
    # can't average sites before applying scattering amplitudes
    coeffs_list = []
    specie_Z_list = []
    frac_coords_list = []
    occupancies_list = []
    for site in structure:
        for specie, occu in site.species.items():
            specie_Z_list.append(specie.Z)
            coeffs_list.append(
                get_scatter_coeffs(specie.symbol, scatter_coeffs=scatter_coeffs)
            )
            frac_coords_list.append(site.frac_coords)
            occupancies_list.append(occu)
    Z_arr = np.asarray(specie_Z_list)  # int[sites]
    occupancy_arr = np.asarray(occupancies_list)  # float[sites]
    coefficient_arr = np.ascontiguousarray(coeffs_list)  # float[sites][4][2]
    fractional_coords_arr = np.ascontiguousarray(frac_coords_list)  # float[sites][3]
    return Z_arr, occupancy_arr, coefficient_arr, fractional_coords_arr


def scattering_centers(
    structure: Union[Structure, Atoms], scatter_coeffs: Optional[dict] = None
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Returns a list of scattering centers, their occupancies, their scattering parameters, and their fractional coordinates.

    Returns 4-tuple of:
        Zs (atomic numbers)     type: int[sites]
        occupancies             type: float[sites]
        scattering coefficients type: float[sites][4][2]
        fractional coordinates  type: float[sites][3]

    The scattering coefficients are represented as four pairs of floats.
    They are of the form: [[a1, b1], [a2, b2], [a3, b3], [a4, b4]].
    A dictionary to alter the coefficients used pay be provided.
    This allows the user to specify alternative scattering.
    The dict may refer to any length list as long as the last level has dimension of 2.
    The list will accordingly be padded by [0,0] or trimmed down to reach a length of 4.

    An idea of some coeffs to set:
    small: (1, 30)
    medium: (3, 40)
    large: (5, 70)
    """
    if isinstance(structure, Structure):
        return scattering_centers_pymatgen(structure, scatter_coeffs)
    elif isinstance(structure, Atoms):
        return scattering_centers_ase(structure, scatter_coeffs)
    else:
        raise ValueError(f"Diffraction of {type(structure)} type not supported.")


def diffraction_intensities(
    structure: Union[Structure, Atoms],
    qrange: tuple[float, float] = (0, 2),
    scatter_coeffs: Optional[dict] = None,
    normalize: bool = True,
):
    """
    Calculates the diffraction indicies and intensities of the structure.
    Work in reciprocal space without the 2pi factor. So d = 1/g, q = g, s = 1/(2d) = q/2
    Operates in q-space where q = 2 * sin(th) / wavelength.

    Returns 3-tuple of:
        intensity   type: float[reflections]
        q (no 2pi)  type: float[reflections]
        d spacing   type: float[reflections]
    """
    # 'refl' will be used to refer to number of reflections, i.e. len(hkl)
    _, g_, hkl_ = reciprocal_scatter(structure, qrange[1])
    Z, occ, coef, frac = scattering_centers(structure, scatter_coeffs)
    idx = np.flatnonzero((g_ != 0) & (g_ >= qrange[0]))
    hkl = hkl_[idx]  # reflections
    g = g_[idx]  # reciprocal distances |G|
    s = g / 2
    s2 = s * s  # int[refl]

    # g_hkl . r_xyz = hkl . xyz = hx + ky + lz
    # T(int[refl][3]) = int[3][refl]
    # float[sites][3] @ int[3][refl] = float[sites][refl]
    gr = np.matmul(frac, hkl.T)  # float[sites][refl]
    # note this g_hkl here is different (but related) to the g used earlier

    # loop over the 4 sets of coefficients
    f_i = np.zeros((4, *gr.shape))  # float[4][refl][sites]
    for i in range(4):
        # terms inside sum_i ( a_i * exp(-b_i * s^2) )
        # make sure the inner product of the matrix mult. is not over any axis (we create another)
        f_i[i] = coef[:, i, 0, None] * np.exp(
            np.matmul(-coef[:, i, 1, None], s2[None, :])
        )  # float[i][sites][refl]

    # f(s) = Z - 41.78241 * s^2 * sum_i ( a_i * exp(-b_i * s^2) )
    f = Z[:, None] - 41.78241 * s2 * np.sum(f_i, axis=0)  # float[sites][refl]

    # sum f over the sites
    f_hkl = np.sum(f * occ[:, None] * np.exp(2j * np.pi * gr), axis=0)  # float[refl]

    # don't apply lorentz factors or debye waller corrections

    intensity_hkl = (f_hkl * f_hkl.conjugate()).real
    d = 1 / g

    if normalize:
        intensity_hkl = intensity_hkl / np.max(intensity_hkl) * 100

    return intensity_hkl, g, d


def lorentzian_scaled(
    x: NDArray[np.float_], y: float = 1, eps: float = 0.001
) -> NDArray[np.float_]:
    """
    Applies lorentzian over the supplied domain, centered at 0. Scaled by y.
    """
    return y * (1 / np.pi) * (eps / (x**2 + eps**2))


def lorentzian(x: NDArray[np.float_], eps: float = 0.001) -> NDArray[np.float_]:
    """
    Applies lorentzian over the supplied domain, centered at 0.
    """
    # return lorentzian_scaled(x, 1, eps)
    return (1 / np.pi) * (eps / (x**2 + eps**2))


def gaussian_scaled(
    x: NDArray[np.float_], y: float = 1, sigma: float = 0.005
) -> NDArray[np.float_]:
    """
    Applies gaussian over the supplied domain, centered at 0. Scaled by y.
    """
    return y * (1 / (sigma * math.sqrt(2 * np.pi))) * np.exp(-(x**2) / (2 * sigma**2))


def gaussian(x: NDArray[np.float_], sigma: float = 0.005) -> NDArray[np.float_]:
    """
    Applies gaussian over the supplied domain, centered at 0.
    """
    # return gaussian_scaled(x, 1, sigma)
    return (1 / (sigma * math.sqrt(2 * np.pi))) * np.exp(-(x**2) / (2 * sigma**2))


def distribution_sum(
    distribution: Callable[[NDArray[np.float_], float], NDArray[np.float_]],
    X: Iterable[float],
    domain: NDArray[np.float_],
    param: float,
) -> NDArray[np.float_]:
    """
    Generic applies the given distribution to the elements in X over the supplied domain and sums them.
    The results will be a sum of distributions at each X.
    """
    ret = np.zeros_like(domain)
    for x in X:
        ret += distribution(domain - x, param)
    return ret


def distribution_scaled_sum(
    distribution: Callable[[NDArray[np.float_], float], NDArray[np.float_]],
    X: Sequence[float],
    Y: Sequence[float],
    domain: NDArray[np.float_],
    param: float,
) -> NDArray[np.float_]:
    """
    Generic applies the given distribution to the elements in X over the supplied domain and sums them.
    The results will be a sum of distributions at each X. Scaled by values in Y. X and Y must be same length.
    """
    assert len(X) == len(Y)
    ret = np.zeros_like(domain)
    for i in range(len(X)):
        ret += Y[i] * distribution(domain - X[i], param)
    return ret


def lorentzian_scaled_sum(
    X: Sequence[float],
    Y: Sequence[float],
    domain: NDArray[np.float_],
    eps: float = 0.001,
) -> NDArray[np.float_]:
    """
    Applies lorentzians to the elements in X over the supplied domain and sums them.
    The result will be the sum of lorentzians at each X. Scaled by values in Y. X and Y must be same length.
    """
    # return distribution_scaled_sum(lorentzian, X, Y, domain, eps)
    assert len(X) == len(Y)
    ret = np.zeros_like(domain)
    for i in range(len(X)):
        ret += Y[i] * lorentzian(domain - X[i], eps)
    return ret


def gaussian_scaled_sum(
    X: Sequence[float],
    Y: Sequence[float],
    domain: NDArray[np.float_],
    sigma: float = 0.005,
) -> NDArray[np.float_]:
    """
    Applies gaussians to the elements in X over the supplied domain and sums them.
    The result will be the sum of gaussians at each X. Scaled by values in Y. X and Y must be same length.
    """
    # return distribution_scaled_sum(gaussian, X, Y, domain, sigma)
    assert len(X) == len(Y)
    ret = np.zeros_like(domain)
    for i in range(len(X)):
        ret += Y[i] * gaussian(domain - X[i], sigma)
    return ret


def lorentzian_sum(
    X: Iterable[float], domain: NDArray[np.float_], eps: float = 0.001
) -> NDArray[np.float_]:
    """
    Applies lorentzians to the elements in X over the supplied domain and sums them.
    The result will be the sum of lorentzians at each X. Scaled by values in Y. X and Y must be same length.
    """
    # return distribution_sum(lorentzian, X, domain, eps)
    ret = np.zeros_like(domain)
    for x in X:
        ret += lorentzian(domain - x, eps)
    return ret


def gaussian_sum(
    X: Iterable[float], domain: NDArray[np.float_], sigma: float = 0.005
) -> NDArray[np.float_]:
    """
    Applies gaussians to the elements in X over the supplied domain and sums them.
    The result will be the sum of a gaussians at each X. Scaled by values in Y. X and Y must be same length.
    """
    # return distribution_sum(gaussian, X, domain, sigma)
    ret = np.zeros_like(domain)
    for x in X:
        ret += gaussian(domain - x, sigma)
    return ret


def gaussian_scaled_sum_fast(
    X: NDArray[np.float_],
    Y: NDArray[np.float_],
    domain: NDArray[np.float_],
    sigma: float = 0.005,
) -> NDArray[np.float_]:
    """
    Applies lorentzians to the elements in X over the supplied domain and sums them.
    The result will be the sum of lorentzians at each X. Scaled by values in Y. X and Y must be same length.
    """
    assert X.shape == Y.shape
    sigma_u = -1 / (2 * sigma * sigma)
    sigma_f = 1 / (sigma * 4.44288293816)
    ret = Y[:, None] * sigma_f * np.exp(((domain[None, :] - X[:, None]) ** 2) * sigma_u)
    return np.sum(ret, axis=0)


def diffraction_signal(
    structure: Structure | Atoms,
    qrange: tuple[float, float] = (0, 2),
    sample_density: int = 5000,
    gauss_width=0.005,
    scatter_coeffs: Optional[dict] = None,
):
    """
    Calculate the diffraction signal over the given q-space range and sample density.
    The result is a gaussian smeared sum of peaks.
    """
    I, q, d = diffraction_intensities(
        structure, qrange=qrange, scatter_coeffs=scatter_coeffs, normalize=True
    )
    domain = np.linspace(*qrange, num=sample_density)
    signal = gaussian_scaled_sum_fast(q, I, domain, gauss_width)
    return signal


def normalize_xcorr_signal(x: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Normalizes the signal such that its maximum self correlation is 1.
    """
    return x / math.sqrt(scipy.signal.correlate(x, x, mode="same").max())


def normalize_xcorr_signals(
    X: Sequence[NDArray[np.float_]], n_jobs: int = 1
) -> Sequence[NDArray[np.float_]]:
    """
    Normalizes the signals such that their maximum self correlation is 1.
    Can run on multiple cores.
    """
    if n_jobs == 1:
        return [normalize_xcorr_signal(x) for x in X]
    else:
        return list(Parallel(n_jobs=n_jobs)(delayed(normalize_xcorr_signal)(x) for x in X))  # type: ignore


def signal_kernel(X: Sequence[NDArray[np.float_]]) -> NDArray[np.float_]:
    """
    Base serial code for computing the kernels of the signals. Assumes the input has
    not already been normed. Is not optimal, but does use scipy functions.
    """
    X_len = len(X)
    X_max_corr_self = np.zeros((X_len,))
    for i in range(X_len):
        X_max_corr_self[i] = scipy.signal.correlate(X[i], X[i]).max()
    X_xcorr_norm = np.zeros((X_len, X_len))
    for i in range(X_len):
        for j in range(X_len):
            if i <= j:
                X_xcorr_norm[i, j] = scipy.signal.correlate(
                    X[i], X[j]
                ).max() / math.sqrt(X_max_corr_self[i] * X_max_corr_self[j])
            else:
                X_xcorr_norm[i, j] = X_xcorr_norm[j, i]
    return X_xcorr_norm


def yield_UP_SQ(n: int) -> Generator[tuple[int, int], Any, None]:
    """
    Yields a generator for the indicies of the upper triangle of a square nxn matrix.
    """
    for i in range(n):
        for j in range(n):
            yield (i, j)


def self_corr_max(x):
    return scipy.signal.correlate(x, x).max()


def corr_max(x, y):
    return scipy.signal.correlate(x, y).max()


def corr_max_pairwise(X, Y):
    out = np.empty((len(X), len(Y)), dtype=np.float_)
    it = itertools.product(range(len(X)), range(len(Y)))
    for i, j in it:
        out[i, j] = corr_max(X[i], Y[i])
    return out


def psks_worker(work_packet):
    """Version of parallel worker that norms using the self correlations."""
    x, y, xs, ys = work_packet
    return scipy.signal.correlate(x, y).max() / math.sqrt(xs * ys)


def psks_worker2(work_packet):
    """Version of the parallel worker that uses pre-normed signals."""
    x, y = work_packet
    return scipy.signal.correlate(x, y).max()


def signal_kernel_parallel(
    X: Sequence[NDArray[np.float_]],
    n_jobs: int = 1,
):
    """
    Computes the kernel of the signals using cross-correlation. Performs normalization on the signals.
    """
    l = len(X)
    ks = np.zeros(l)
    km = np.zeros((l, l))
    with Pool(processes=n_jobs) as pool:
        ks = np.asarray(pool.map(self_corr_max, X))
        inds = list(yield_UP_SQ(l))
        indx = tuple(np.asarray(inds).T)
        work = ((X[i], X[j], ks[i], ks[j]) for i, j in inds)
        km[indx] = np.asarray(list(pool.imap(psks_worker, work, chunksize=500)))
        km[indx[::-1]] = km[indx]
    return km

def sliced_shm_func_wrapper(func, shape, dt, shm, slice_, *args):
    matrix = np.ndarray(shape=shape, dtype=dt, buffer=shm.buf)
    matrix[slice_, :] = func(*args)

def sliced_func_wrapper(func, matrix, slice_, *args):
    matrix[slice_, :] = func(*args)

def sliced_func_wrapper_F(func, matrix, slice_, *args):
    matrix[:, slice_] = func(*args)

def signal_kernel_parallel_normed(
    X: Sequence[NDArray[np.float_]],
    n_jobs: int = 1,
    manager: Literal["multi", "joblib", "joblib-mmap"] = "multi",
) -> NDArray[np.float_]:
    """
    Computes the kernel of the signals using cross-correlation. Assumes the input has been normed.
    Uses either multiprocessing module or joblib. Implementing with chunked imap shows slight improvement in small tests.
    Designed to use cores on one machine.
    """
    if manager == "joblib-mmap":
        print('Using mmap with joblib to assign kernel matrix in place.')
        #f = delayed(sliced_func_wrapper)
        #l = len(X)
        #temp_folder = os.path.join(tempfile.mkdtemp(), 'disorder-{int(time.time_ns() % 1e8)}')
        #try:
        #    os.mkdir(temp_folder)
        #except FileExistsError:
        #    pass
        #data = np.asarray(X)
        #dat_mmap_file = os.path.join(temp_folder, 'dat.mmap')
        #out_mmap_file = os.path.join(temp_folder, 'out.mmap')
        #dump(data, dat_mmap_file)
        #data = load(dat_mmap_file, mmap_mode='r')
        #output = np.memmap(out_mmap_file, dtype=np.float_, shape=(l, l), mode='w+')
        #print(f'Memory maps written/loaded.')
        #print(f'Matrix initialized with shape: {output.shape} and nbytes: {output.nbytes}.')
        #Parallel(n_jobs=n_jobs)(
        #    f(corr_max_pairwise, output, slice_, data[slice_], data)
        #    for slice_ in gen_even_slices(l, n_jobs*2)
        #)
        #print('Completed parallel computing of kernel.')
        #return output
        
        f = delayed(sliced_shm_func_wrapper)
        l = len(X)
        ret = np.empty((l,l), dtype=np.float_)
        shm = shared_memory.SharedMemory(create=True, size=ret.nbytes)
        output = np.ndarray(ret.shape, dtype=ret.dtype, buffer=shm.buf)
        Parallel(n_jobs=n_jobs)(
            f(corr_max_pairwise, output.shape, output.dtype, shm, slice_, X[slice_], X)
            for slice_ in gen_even_slices(l, n_jobs*2)
        )
        ret[:] = output[:]
        shm.close()
        shm.unlink()
        return ret
    else:
        l = len(X)
        km = np.zeros((l, l))
        inds = list(yield_UP_SQ(l))
        indx = tuple(np.asarray(inds).T)
        if manager == "joblib":
            result = np.asarray(
                Parallel(n_jobs=n_jobs)(delayed(corr_max)(X[i], X[j]) for i, j in inds)
            )
        else:
            with Pool(processes=n_jobs) as pool:
                work = ((X[i], X[j]) for i, j in inds)
                result = np.asarray(list(pool.imap(psks_worker2, work, chunksize=500)))
        km[indx] = result
        km[indx[::-1]] = km[indx]
        return km


def dist_from_s(S: NDArray[np.float_], gamma_u: float) -> NDArray[np.float_]:
    """
    Computes the distances from the given similarities.
    """
    return -np.log(S) * gamma_u


def distance_from_similarity(
    S: NDArray[np.float_], gamma_u: float
) -> NDArray[np.float_]:
    """
    Computes the distance matrix from the given similarity matrix.
    """
    D = np.empty_like(S, dtype=np.float_)
    for i in range(len(S)):
        D[i] = -np.log(S[i]) * gamma_u
    return D


def distance_from_similarity_parallel(
    S: NDArray[np.float_],
    gamma_u: float,
    n_jobs: int = 1,
):
    """
    Computes the distance matrix from the given similarity matrix. Parallel.
    """
    if n_jobs == 1:
        return np.asarray([dist_from_s(s, gamma_u) for s in S])
    else:
        return np.asarray(
            Parallel(n_jobs=n_jobs)(
                delayed(functools.partial(dist_from_s, gamma_u=gamma_u))(s) for s in S
            )
        )


def cull_subgroups(groups: Collection[SpaceGroup]) -> set[SpaceGroup]:
    """
    Reduces the set of groups to their minimal supergroups. Groups are removed such that no
    remaining group is a subgroup of another in the set. For an input set X, the output set Y
    is a set of groups such that y1 is not a subgroup of y2 for any pair y1, y2 in Y.
    """
    groups = set(groups)
    subs: set[SpaceGroup] = set()
    for group1 in groups:
        if group1 not in subs:
            for group2 in groups:
                if group2 not in subs:
                    if group1.is_subgroup(group2):
                        subs.add(group1)
    return groups - subs


def tally_subroot(
    roots: set[SpaceGroup], groups: Collection[SpaceGroup]
) -> dict[SpaceGroup, int]:
    """
    Counts how many in groups are subgroups of the roots. Returns a dict mapping from
    the groups in root to the corresponding number of subgroups in groups.
    For roots set R and groups set Q, returns M: R -> Z, such that for M(r) -> z,
    z = |{q subgroup of r for all q in Q}|.
    """
    groups = set(groups)
    tally: dict[SpaceGroup, int] = dict()
    for root in roots:
        tally[root] = 0
        for group in groups:
            if group.is_subgroup(root):
                tally[root] += 1
    return tally


def minimal_supergroup_mode(groups: Collection[SpaceGroup]) -> SpaceGroup:
    """
    Find the minimal supergroup with the largest amount of subgroups in groups.
    """
    roots = cull_subgroups(groups)
    tally = tally_subroot(roots, groups)
    return max(tally, key=tally.__getitem__)


def integrate_chunks(
    p: NDArray[np.float_], domain: NDArray[np.float_], chunk_width: float
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """
    Integrates p into chunks by chunking over domain. Discretizes a continuous distribution. Useful for
    functions that require a discrete domain or are otherwise unstable over a continuous domain.
    The integral is over p with respect to the domain.
    Returns a 2-tuple of:
        domain chunked
        integral over chunks of p
    """
    num_chunks: int = math.ceil((domain[-1] - domain[0]) / chunk_width)
    d_chunked = np.array_split(domain, num_chunks)
    p_chunked = np.array_split(p, num_chunks)
    d_avg_chunk = np.empty(num_chunks, dtype=float)
    p_int_chunk = np.empty(num_chunks, dtype=float)
    for i in range(num_chunks):
        p_int_chunk[i] = np.trapz(p_chunked[i], d_chunked[i])
        d_avg_chunk[i] = np.mean(d_chunked[i])
    return d_avg_chunk, p_int_chunk


def entropy_from_dos(
    energies: Collection[float],
    temperature: float,
    sig_size: int = 5000,
    lorentz_width: float = 0.001,
    energy_margin: tuple[float, float] = (0, 0.05),
    pre_int_chunk_width: float = 0.01,
) -> float:
    """
    Calculates the entropy based on the density of states in the collection of energies.
    sig_size is how many energy points to break the domain into. Larger value is more costly but more precise.
    lorentz_width is the peak width used for turning the discrete energies into lorentzians.
    energy_margin is how much to expand the energy range. The range is increased/decreased by range*margin. Of form (lower, upper).
    Returns the entropy.
    """
    if temperature == 0:
        return 0
    kB = 8.617333262e-5
    beta = 1 / (kB * temperature)
    E: NDArray[np.float_] = np.asarray(energies)
    E_range = E.max() - E.min()
    E_dom_lo = E.min() - E_range * energy_margin[0]
    E_dom_hi = E.max() + E_range * energy_margin[1]
    E_domain = np.linspace(E_dom_lo, E_dom_hi, num=sig_size)
    D = lorentzian_sum(E, E_domain, lorentz_width)  # density of states
    betaE = beta * E_domain  # E/(kT) across the domain
    p_ = D * np.exp(-betaE)
    Z = np.trapz(p_, E_domain)  # partition function
    p = p_ / Z
    _, p_i = integrate_chunks(p, E_domain, pre_int_chunk_width)
    entropy: np.float_ = -kB * np.sum(p_i * np.log(p_i))
    return float(entropy)


def pull_provider_from_disk(
    source_path: str,
    data_support: Literal["pymatgen", "ase"],
) -> List[AirssProvider | Res]:
    pymatgen_provider_dict: dict[
        Literal["str", "file"], Callable[[str], AirssProvider]
    ] = {
        "str": AirssProvider.from_str,
        "file": AirssProvider.from_file,
    }
    ase_provider_dict: dict[Literal["str", "file"], Callable[[str], Res]] = {
        "str": Res.from_string,
        "file": Res.from_file,
    }
    if data_support == "pymatgen":
        return list(generic_from_disk(source_path, pymatgen_provider_dict, ext=".res"))
    elif data_support == "ase":
        return list(generic_from_disk(source_path, ase_provider_dict, ext=".res"))
    else:
        raise ValueError(f"Only supports loading using pymatgen or ase.")


def iterate_tar_src(file: str) -> Generator[str, Any, None]:
    for src, _ in iterate_tar_info(file):
        yield src


def iterate_path_src(path: str, ext: str = ".res") -> Generator[str, Any, None]:
    for filepath in iterate_path_files(path, ext=ext):
        with open(filepath, "r") as f:
            src = f.read()
        yield src


def pull_src_from_disk(
    source_path: str,
) -> List[str]:
    if re.match(r".*[.]tar[.]?(.*)", source_path):
        # if source_path.endswith(".tar"):
        return list(iterate_tar_src(source_path))
    else:
        return list(iterate_path_src(source_path, ext=".res"))


def pull_src_from_disk_alt1(source_path: str) -> List[str]:
    src_provider_dict: dict[Literal["str", "file"], Callable[[str], str]] = {
        "str": str,
        "file": lambda x: open(x).read(),
    }
    return list(generic_from_disk(source_path, src_provider_dict, ext=".res"))


def _p2s(p: AirssProvider) -> Structure:
    return p.structure


def _r2a(r: Res) -> Atoms:
    return r.atoms


def filter_pressure(frame, pressures, tol=0.1):
    return functools.reduce(operator.ior, [np.abs(frame - p) <= tol for p in pressures])


def formation_enthalpy_groupby(
    unit_frame: pd.DataFrame,
) -> Callable[[pd.DataFrame], pd.Series]:
    def func(frame: pd.DataFrame):
        products = frame[c.te]
        reactants = sum(
            [
                frame[n] * unit_frame.loc[frame.name, e]
                for n, e in zip([c.n1, c.n2], [c.e1, c.e2])
            ]
        )
        return (products - reactants) / frame[c.n]

    return func


def hullpoints(frame):
    hull = ConvexHull(frame)
    verts = hull.points[hull.vertices]
    args = verts.T[0].argsort()
    x, y = verts[args].T
    return pd.DataFrame({"x": x, "y": y})


def hulldist(frame, points, xkey=c.x, ykey=c.y):
    hully = np.interp(frame[xkey], points.x, points.y)
    dist = frame[ykey] - hully
    return dist


def dos_entropy_frame(
    frame: pd.DataFrame,
    temperature: float,
    nkey=c.cln,
    mkey=c.h,
    lorentz_width=0.001,
    signal_size=5000,
    pre_int_chunk_width=0.01,
    full_frame=False,
):
    frame = frame[[c.n, c.cll, nkey, mkey]].reset_index().set_index(c.cll)
    func = functools.partial(
        entropy_from_dos,
        temperature=temperature,
        sig_size=signal_size,
        lorentz_width=lorentz_width,
        pre_int_chunk_width=pre_int_chunk_width,
    )
    frame["S"] = frame.groupby(level=0)[mkey].apply(func)
    frame.loc[-1, "S"] = 0
    if full_frame:
        return frame.set_index("idx")
    else:
        return frame.set_index("idx")["S"]


def recalc_hull_G(frame, entropy, temperature, ref_index):
    frame = frame[[c.ufe, c.n, c.n1, c.n2, c.e, c.x, c.ue, c.te]].copy()
    frame["tG"] = frame[c.te] - temperature * entropy * frame[c.n]
    frame["uG"] = frame[c.ue] - temperature * entropy
    rfuG = frame.loc[ref_index].groupby(c.e)["uG"].min()
    frame["G"] = (
        frame["tG"] - frame[c.n1] * rfuG[c.e1] - frame[c.n2] * rfuG[c.e2]
    ) / frame[c.n]
    G_le0 = frame["G"] <= 0
    hframe_G = frame.loc[G_le0, [c.x, "G"]]
    hpoints_G = hullpoints(hframe_G)
    hdist_G = hulldist(frame, hpoints_G, ykey="G")
    return frame["G"], hdist_G, hpoints_G


def main():

    # sorry for the length of this function, it was made from a long jupyter notebook

    supported = ["pymatgen", "ase"]

    namespace = pargs()

    space: str = namespace.chemical_space
    splitspace = space.split(",")
    if len(splitspace) != 2:
        raise ValueError(f"Only supports 2D chemical space.")
    else:
        c.e1, c.e2 = splitspace
        c.n1, c.n2 = f"num_{c.e1}", f"num_{c.e2}"
        c.e_reassigned = True

    tempspace: str = namespace.temperature_range
    splittempspace = tempspace.split(",")
    if len(splittempspace) != 3:
        raise ValueError(
            f"Could not detect 3 values! Temperature range must be in form `START,STOP,STEP`."
        )
    else:
        t_start, t_stop, t_step = tuple(map(int, splittempspace))

    ignore_elemental_search: bool = namespace.no_search_elem

    scatter_coeffs: dict[str, list[list[float]]] | None = None
    if scatter_coeffs:
        try:
            scatter_coeffs = json.load(namespace.scatter_coeffs)
        except:
            print(f"Could not decode json {namespace.scatter_coeffs}.")

    main_path: str = namespace.main_path
    auxd_path: str = namespace.auxilliary_path
    run_name: str = namespace.run_name

    reload_data: bool = namespace.reload_data
    recalc_kern: bool = namespace.recalc_kern

    save_dfm: str = f"{run_name}_save-dfm.csv"
    save_dfa: str = f"{run_name}_save-dfa.csv"
    save_atoms: str = f"{run_name}_save-atoms.pkl"
    save_pymtg: str = f"{run_name}_save-pymtg.pkl"
    save_km_xrd: str = f"{run_name}_save-km-xrd.npy"
    save_ds_soap: str = f"{run_name}_save-ds-soap.npy"
    save_dfc: str = f"{run_name}_save-dfc.csv"

    save_plots_fmt: str = "{run_name}_recalc_G_T{temp}.{fmt}"

    save_plot_exts: list[str] = namespace.plot_types.split(",")

    pressure: float = namespace.pressure
    pressure_tol: float = namespace.pressure_tol
    qspace_upper: float = namespace.max_q
    diffraction_peak_width: float = namespace.sigma
    dos_peak_width: float = namespace.eps
    diffraction_signal_size: int = namespace.signal_size
    dos_signal_size: int = namespace.signal_size
    pre_integration_chunk_width: float = namespace.energy_int_width

    sampled_data_size: int | None = namespace.sample_size
    reflimit_data_size: int | None = namespace.elemental_size
    formation_enthalpy_cutoff: float | None = namespace.formation_cutoff
    random_seed: int | None = namespace.random_seed

    using_ase_structures: bool = False
    using_pmt_structures: bool = False
    using_soap: bool = namespace.use_soap
    build_dataframe_from: str = namespace.df_source
    build_diffract_from: str = namespace.diff_source

    n_jobs: int = namespace.n_jobs

    using_ase_structures = (
        using_soap or build_dataframe_from == "ase" or build_diffract_from == "ase"
    )
    using_pmt_structures = (
        build_dataframe_from == "pymatgen" or build_diffract_from == "pymatgen"
    )

    if build_dataframe_from not in supported:
        raise ValueError(
            f"{build_dataframe_from} is not a supported library to build dataframe from."
        )
    if build_diffract_from not in supported:
        raise ValueError(
            f"{build_diffract_from} is not a supported library to build diffraction from."
        )

    ############################
    # LOADING AND READING DATA #
    ############################

    def mp_pool_dpull(pool, path):
        if path:
            return pool.apply_async(pull_src_from_disk, [path])
        else:
            return pool.apply_async(list, [])

    try:
        if reload_data:
            raise RuntimeError
        print("Trying to load any existing data.")
        df_m = pd.read_csv(save_dfm, index_col=0)
        df_a = pd.read_csv(save_dfa, index_col=0)
        print("Loaded dataframes.")

        if build_diffract_from == "pymatgen":
            print("Trying to load pymatgen structures.")
            with open(save_pymtg, "rb") as fp:
                st_pmt = pickle.load(fp)
            print("Loaded pymatgen structures.")
        elif build_diffract_from == "ase" or using_soap:
            print("Trying to load ase atoms.")
            with open(save_atoms, "rb") as fp:
                st_ase = pickle.load(fp)
            print("Loaded ase atoms.")
        else:
            print(f"No code path here for {build_diffract_from}", file=sys.stderr)
            sys.exit()

    except:
        print("Trying to load data from source files.")
        recalc_kern = True
        with Pool(processes=n_jobs) as pool:
            print("Entering multiprocessing pool.")
            main_src_as = mp_pool_dpull(pool, main_path)
            auxd_src_as = mp_pool_dpull(pool, auxd_path)
            print("Sources assigned.")
            print("Collecting providers from disk using map.")
            if using_pmt_structures:
                main_pmt_im = pool.map(AirssProvider.from_str, main_src_as.get(), 200)
                auxd_pmt_im = pool.map(AirssProvider.from_str, auxd_src_as.get(), 200)
            if using_ase_structures:
                main_ase_im = pool.map(Res.from_string, main_src_as.get(), 200)
                auxd_ase_im = pool.map(Res.from_string, auxd_src_as.get(), 200)
            print("Collected map for providers.")

            if build_dataframe_from == "pymatgen":
                assert using_pmt_structures
                df_main_im = pool.imap(data_from_provider, main_pmt_im, 200)
                df_auxd_im = pool.imap(data_from_provider, auxd_pmt_im, 200)
            elif build_dataframe_from == "ase":
                assert using_ase_structures
                df_main_im = pool.imap(data_from_aseres, main_ase_im, 200)
                df_auxd_im = pool.imap(data_from_aseres, auxd_ase_im, 200)
            else:
                print(f"No code path here for {build_dataframe_from}", file=sys.stderr)
                sys.exit()
            print("imap for data for frames set.")

            if build_diffract_from == "pymatgen":
                st_main_pmt_im = pool.imap(_p2s, main_pmt_im, 200)
                st_auxd_pmt_im = pool.imap(_p2s, auxd_pmt_im, 200)
            elif build_diffract_from == "ase" or using_soap:
                st_main_ase_im = pool.imap(_r2a, main_ase_im, 200)
                st_auxd_ase_im = pool.imap(_r2a, auxd_ase_im, 200)
            else:
                print(f"No code path here for {build_diffract_from}", file=sys.stderr)
                sys.exit()
            print("imap for structures set.")

            print("Building dataframes.")
            df_m = pd.DataFrame(df_main_im)
            df_a = pd.DataFrame(df_auxd_im)
            print("Dataframes built.")

            df_m.index += df_a.index.stop  # type: ignore

            print("Building structure lists.")
            if build_diffract_from == "pymatgen":
                st_pmt = list(st_auxd_pmt_im) + list(st_main_pmt_im)
            elif build_diffract_from == "ase" or using_soap:
                st_ase = list(st_auxd_ase_im) + list(st_main_ase_im)
            else:
                print(f"No code path here for {build_diffract_from}", file=sys.stderr)
                sys.exit()
            print("Structure lists built.")

        print("Saving structure lists to disk.")
        if build_diffract_from == "pymatgen":
            with open(save_pymtg, "wb") as fp:
                pickle.dump(st_pmt, fp)
        elif build_diffract_from == "ase" or using_soap:
            with open(save_atoms, "wb") as fp:
                pickle.dump(st_ase, fp)
        else:
            print(f"No code path here for {build_diffract_from}", file=sys.stderr)
            sys.exit()
        print("Structure lists saved to disk.")

        print("Saving dataframes to disk.")
        df_m.to_csv(save_dfm)
        df_a.to_csv(save_dfa)
        print("Dataframes saved to disk.")

    if build_diffract_from == "pymatgen":
        st_diffract = st_pmt
    elif build_diffract_from == "ase":
        st_diffract = st_ase
    else:
        print(f"No code path here for {build_diffract_from}", file=sys.stderr)
        sys.exit()

    if build_diffract_from == "pymatgen":
        print(f"Collected {len(st_pmt)} structures into the pymatgen structure list.")
    elif build_diffract_from == "ase" or using_soap:
        print(f"Collected {len(st_ase)} structures into the ase atoms list.")
    else:
        print(f"No code path here for {build_diffract_from}", file=sys.stderr)
        sys.exit()

    print(f"df_m shape: {df_m.shape}")
    print(f"df_a shape: {df_a.shape}")

    ############################
    # PROCESSING AND FILTERING #
    ############################

    true_m_i = pd.Series(True, index=df_m.index)
    false_m_i = pd.Series(False, index=df_m.index)
    false_a_i = pd.Series(False, index=df_a.index)
    # 'or' op with the false series ensures the boolean series covers the whole index range
    mmix_i_p = (df_m[c.n1] != 0) & (df_m[c.n2] != 0) | false_a_i
    amix_i_p = (
        (df_a[c.n1] != 0) & (df_a[c.n2] != 0) if not df_a.empty else pd.Series([])
    ) | false_m_i
    if ignore_elemental_search:
        elem_i_p = ~(amix_i_p | true_m_i)
    else:
        elem_i_p = ~(amix_i_p | mmix_i_p)
    # mmix has mixed phases found in the main frame
    # amix has mixed phases found in the aux frame
    # elem has elemental phases

    if not mmix_i_p.sum():
        print(
            "nothing mixed found in df, are the elements set correctly?",
            file=sys.stderr,
        )

    mmid = mmix_i_p
    amid = amix_i_p
    rfid = elem_i_p

    df_all = pd.concat([df_a, df_m])

    df = df_all
    print("pre-filter:")
    print(f"all         : {df.shape}")
    print(f"main mix    : {df.loc[mmid].shape}")
    print(f"aux mix     : {df.loc[amid].shape}")
    print(f"elemental   : {df.loc[rfid].shape}")

    df_align_pressure = df_all.loc[
        filter_pressure(df_all[c.p], [pressure], tol=pressure_tol)
    ]

    df = df_align_pressure.copy()
    print("post-pressure-alignment-filter:")
    print(f"all         : {df.shape}")
    print(f"main mix    : {df.loc[mmid].shape}")
    print(f"aux mix     : {df.loc[amid].shape}")
    print(f"elemental   : {df.loc[rfid].shape}")

    # process the data

    # df = df_align_pressure

    # sum total atoms
    df[c.n] = df[c.n1] + df[c.n2]

    # volume per atom
    df[c.vpa] = df[c.v] / df[c.n]

    # composition as e2_{1-x}e1_{x}
    df[c.x] = df[c.n1] / df[c.n]

    rf = df.loc[rfid]

    # total enthalpy per atom for reference phases
    df.loc[rfid, c.ue] = rf[c.te] / rf[c.n]
    # assign elements
    df.loc[rfid & (rf[c.n1] > 0), c.e] = c.e1
    df.loc[rfid & (rf[c.n2] > 0), c.e] = c.e2
    # get indicies of minimum for the element
    rfudf = df.groupby([c.rp, c.e])[c.ue].agg(["min", "idxmin"])
    rfu = rfudf["min"]
    rfui = rfudf["idxmin"]

    # compute formation enthalpy
    df[c.ufe] = (
        df.groupby(c.rp)[[c.n1, c.n2, c.te, c.n]]
        .apply(formation_enthalpy_groupby(rfu))
        .squeeze(0)
    )
    print("calculated formation data")

    # compute hull data
    frame = df.loc[df[c.ufe] <= 0, [c.x, c.y]].sort_values(c.x)
    hpoints = hullpoints(frame)
    hdist = hulldist(df, hpoints)
    df[c.h] = hdist
    print("calculated hull data")

    # filter out high formation enthalpy or high hulld data
    if formation_enthalpy_cutoff is not None:
        df_filter_enthalpy = df.loc[df[c.ufe] < formation_enthalpy_cutoff]
        df = df_filter_enthalpy

    if reflimit_data_size:
        i1 = (
            df.loc[rfid & (rf[c.n1] > 0), c.ue]
            .sort_values(ascending=True)
            .head(reflimit_data_size)
            .index
        )
        i2 = (
            df.loc[rfid & (rf[c.n2] > 0), c.ue]
            .sort_values(ascending=True)
            .head(reflimit_data_size)
            .index
        )
        rfid_limit = rfid & pd.Series(True, index=i1.union(i2))
        rfid = rfid_limit

    if sampled_data_size:
        i3 = df.loc[mmid].sample(min(sampled_data_size, len(df.loc[mmid])), random_state=random_seed).index
        mmid_sampled = mmid & pd.Series(True, index=i3)
        mmid = mmid_sampled

    if sampled_data_size or reflimit_data_size:
        print("post-downsize-filter:")
        print(f"all         : {df.shape}")
        print(f"main mix    : {df.loc[mmid].shape}")
        print(f"aux mix     : {df.loc[amid].shape}")
        print(f"elemental   : {df.loc[rfid].shape}")

    ##
    #
    # -ANY OTHER PREPROCESSING OF THE DATA FRAMES SHOULD BE DONE HERE AS LONG AS IT DOES NOT MODIFY THE INDEX
    # -THE INDEX LABELS WILL BE REALIGNED AT THE END OF THIS CELL
    ######

    print("post-all-filters:")
    print(f"all         : {df.shape}")
    print(f"main mix    : {df.loc[mmid].shape}")
    print(f"aux mix     : {df.loc[amid].shape}")
    print(f"elemental   : {df.loc[rfid].shape}")

    df_preindex = df

    df = df.loc[mmid | amid | rfid]

    # at this point, indicies in the structure lists correspond to index labels in the dataframes i.e. (df.loc[i] == st_pmt)
    indx = df.index.to_numpy()  # used to index the structure lists
    df = df.reset_index(names="pdx").reset_index(names="idx").set_index("pdx")
    rfi = pd.Index(df.loc[rfid, "idx"])
    ami = pd.Index(df.loc[amid, "idx"])
    mmi = pd.Index(df.loc[mmid, "idx"])
    df = df.reset_index().set_index("idx")
    # now pdx indexes the original structure list, idx indexes the structure list after indexing by pdx, idx is the index of the frames
    # st_pmt[pdx][idx][i] = df.loc[i]
    # st_pmt[indx][idx][i] = df.loc[i]

    df.loc[rfi, c.i] = 'ref'
    df.loc[ami, c.i] = 'aux'
    df.loc[mmi, c.i] = 'mix'

    df_reindex = df

    print("new index:")
    print(f"main mix    : {mmi.shape}")
    print(f"aux mix     : {ami.shape}")
    print(f"elemental   : {rfi.shape}")
    print(f"union       : {rfi.union(ami).union(mmi).shape}")

    print("post-reindexing:")
    print(f"all         : {df.shape}")
    print(f"main mix    : {df.loc[mmi].shape}")
    print(f"aux mix     : {df.loc[ami].shape}")
    print(f"elemental   : {df.loc[rfi].shape}")

    ###########################
    # DESCRIPTORS AND KERNELS #
    ###########################

    print("Preparing for similarity matrix")

    try:
        if recalc_kern:
            raise RuntimeError
        print("reading kernel matrix from file")
        km_xrd = np.load(save_km_xrd)
        if km_xrd.shape[0] != indx.shape[0]:
            print('kernel matrix not same size as structure index, maybe the dataframe options changed between runs.')
            del km_xrd
            raise RuntimeError
    except:
        print("Computing diffraction signals")
        structures = np.asarray(st_diffract, dtype="object")[indx]
        partial_diff_signal = functools.partial(
            diffraction_signal,
            qrange=(0, qspace_upper),
            sample_density=diffraction_signal_size,
            gauss_width=diffraction_peak_width,
            scatter_coeffs=scatter_coeffs,
        )
        signals: list[NDArray] = list(
            Parallel(n_jobs=n_jobs)(
                delayed(partial_diff_signal)(structure) for structure in structures
            )
        )  # type: ignore
        print("Normalizing signals via self-correlation.")
        normed_signals = normalize_xcorr_signals(signals, n_jobs=n_jobs)
        print("Computing the kernel matrix.")
        km_xrd = signal_kernel_parallel_normed(
            normed_signals, n_jobs=n_jobs, manager=namespace.parallel_kernel_backend,
        )
        print("Clipping matrix to [0,1].")
        np.clip(km_xrd, 0, 1, km_xrd)
        print("Saving matrix to disk.")
        np.save(save_km_xrd, km_xrd)
        print(f"kernel matrix shape: {km_xrd.shape}")

    if using_soap:
        try:
            if recalc_kern:
                raise RuntimeError
            print("reading soap kernel matrix from file")
            ds_soap = np.load(save_ds_soap)
        except:
            print("Computing soap descriptors.")
            soap = SOAP(
                species=[c.e1, c.e2],
                r_cut=4,
                n_max=2,
                l_max=3,
                sigma=0.5,
                periodic=True,
                average="outer",
            )
            descriptors: NDArray[np.float_] = np.asarray(
                Parallel(n_jobs=n_jobs)(delayed(soap.create)(atoms) for atoms in st_ase)
            )  # type: ignore
            ds_soap = descriptors
            print("Saving soap descriptors.")
            np.save(save_ds_soap, ds_soap)
            print(f"Soap descriptors shape: {ds_soap.shape}")

    print("Computing distance matrix from diffraction similarity matrix.")
    distance_matrix = distance_from_similarity_parallel(km_xrd, 30, n_jobs)

    print("Computing clusters from distance matrix.")
    cluster_db = DBSCAN(eps=0.2, min_samples=3, n_jobs=1, metric="precomputed").fit(
        distance_matrix
    )

    if using_soap:
        print("Computing soap clusters.")
        # note this is not yet properly implemented
        soap_cluster_db = DBSCAN(
            eps=0.2, min_samples=3, n_jobs=n_jobs, metric="euclidian"
        )

    ####################
    # CLUSTER ANALYSIS #
    ####################

    print("Performing cluster analysis.")

    df = df_reindex
    dfcl = df.copy()

    print("Assigning cluster labels and counting cluster members.")
    dfcl['idx'] = df.index
    dfcl[c.cll] = cluster_db.labels_
    dfcl = dfcl.set_index(c.cll, drop=False).sort_index()
    lab_vc = dfcl.loc[0:].index.value_counts()
    dfcl[c.cln] = lab_vc

    print("Cluster info:")
    print(f"{dfcl.shape[0]} total samples")
    print(f"{lab_vc.shape[0]} clusters")
    print(f"{lab_vc.max()} is largest cluster")
    print(f"{lab_vc.mean():.1f} is average cluster size")
    print(f"{lab_vc.median()} is median cluster size")
    print(f"{dfcl.index.value_counts().loc[-1] if -1 in dfcl.index else 0} without a cluster")

    # lab_vc.plot(kind='bar', logy=True, xlabel='', xticks=[], title='Cluster Sizes')

    dfclc = dfcl.loc[0:]
    byclsize = (
        dfclc.reset_index(drop=True)
        .groupby(c.cll)
        .first()[c.cln]
        .sort_values(ascending=False)
    )
    bycsi = byclsize.index

    print("Saving dataframe with cluster information to disk.")
    dfcl.set_index('idx').to_csv(save_dfc)
    print('Saved cluster dtaframe to disk.')

    #####################################
    ############################
    ### PLOTTING
    ###############

    lab_cmap = plt.cm.tab20  # type: ignore
    sub_palette = lab_cmap(dfcl.loc[0:, c.cll].unique() % lab_cmap.N)
    color_palette = np.insert(sub_palette, 0, [(0, 0, 0, 1)], axis=0) if -1 in dfcl.index else sub_palette
    colors = color_palette[dfcl.set_index("idx").sort_index()[c.cll] + 1 if -1 in dfcl.index else 0]

    dfh = dfcl.set_index("idx").copy()

    print(
        f"Running plots modified by T from {t_start} to {t_stop} in steps of {t_step}."
    )

    t_str = f'{{temp:0>{len(str(t_stop))}}}'
    if not any([t_start, t_stop, t_step]):
        print('Skipping plotting because T range is 0,0,0.')
        print('Run complete.')
        return
    for temperature in np.arange(start=t_start, stop=t_stop+1, step=t_step):
        print(f"Building plot for T={temperature}K.")
        print("Calculating entropy and rebuilding hull.")
        G, hdG, hpoints_G = recalc_hull_G(
            dfh,
            dos_entropy_frame(
                dfh,
                temperature,
                lorentz_width=dos_peak_width,
                signal_size=dos_signal_size,
                pre_int_chunk_width=pre_integration_chunk_width,
            ),
            temperature,
            rfi,
        )
        dfh[c.G] = G
        dfh[c.hG] = hdG
        dfclh = dfh.reset_index().set_index(c.cll)

        df_on = dfh.loc[dfh[c.hG] == 0]
        df_off = dfh.loc[dfh[c.hG] != 0]

        fig, ax = plt.subplots(figsize=(12, 9), dpi=144)
        ax.axhline(0, color="k", ls="-.", lw=1)
        ax.scatter(
            x=df_off[c.x],
            y=df_off[c.G],
            c=colors[df_off.index],
            s=20,
            marker="x",
            alpha=0.5,
        )
        ax.plot(hpoints_G.x, hpoints_G.y, color="k", lw=2, ls="--", alpha=0.6)
        ax.scatter(
            x=df_on[c.x],
            y=df_on[c.G],
            c=colors[df_on.index],
            s=40,
            marker="o",
            alpha=0.8,
        )

        print("Building hulls for clusters.")
        skipped_clusters = []
        added_clusters = []
        for cluster_i in byclsize.head(50).index:
            frame = dfclh.loc[cluster_i, [c.x, c.G]]
            if len(frame[c.x].unique()) < 2 or len(frame[c.G].unique()) < 2:
                skipped_clusters.append(cluster_i)
                continue
            hull = ConvexHull(frame)
            ax.fill(
                *hull.points[hull.vertices].T,
                color=sub_palette[cluster_i],
                lw=2,
                ls="-",
                alpha=0.4,
            )
            added_clusters.append(cluster_i)
        print(f"Skipped {len(skipped_clusters)} clusters for being too small.")
        print(f"Included {len(added_clusters)} clusters to the plot.")

        cmap = mpl.colors.ListedColormap(sub_palette[added_clusters][::-1])  # type: ignore
        norm = mpl.colors.BoundaryNorm(range(len(added_clusters) + 1), len(added_clusters))  # type: ignore
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),  # type: ignore
            ax=ax,
        )

        clust_spgs = (
            dfclc.groupby(level=0)[c.sgno]
            .agg(
                lambda series: minimal_supergroup_mode(
                    series.map(SpaceGroup.from_int_number)
                ).to_latex_string()
            )
            .loc[added_clusters[::-1]]
        )

        cbar.set_ticks(ticks=np.arange(len(added_clusters)) + 0.5, labels=clust_spgs)  # type: ignore
        cbar.minorticks_off()
        cbar.set_label("Mode of Minimal Supergroups")

        ax.set_ylabel("Proxy Formation Energy (eV/atom)")
        ax.set_xlabel(f"$\\mathregular{{{c.e2}_{{1-x}}{c.e1}_{{x}}}}$")
        ax.minorticks_on()

        ax.margins(0.01)
        for ext in save_plot_exts:
            fig.savefig(
                save_plots_fmt.format(run_name=run_name, temp=t_str.format(temp=temperature), fmt=ext),
                dpi=300,
            )
    print('Done generating and saving plots.')

    print('Run complete.')
    return


def pargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="disorder",
        description="""Handles data processing of AIRSS results and
        idenitifies regions or phases of disorder using cluster
        analysis on the density of structures in some metric space.""",
    )

    parser.add_argument(
        "-E",
        "--elements",
        type=str,
        help="The elemental species in the search space. In form 'E1,E2' e.g. 'C,Cr' plots as 'Cr_{1-x}C_{x}'.",
        dest="chemical_space",
        required=True,
    )
    parser.add_argument(
        "-T",
        "--temperature",
        type=str,
        help="The temperature to calculate the cluster entropy from. In form 'START,STOP,STEP' e.g. '0,2400,800'.",
        dest="temperature_range",
        default="0,3200,800",
    )
    parser.add_argument(
        "-I",
        "--ignore-e-search",
        help="Sets whether to ignore elemental phases found in the search and only use the ones found in the auxiliarry data.",
        dest="no_search_elem",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-d",
        "--data-path",
        type=str,
        help="Path to search for res files or an archive of res files.",
        required=True,
        dest="main_path",
    )
    parser.add_argument(
        "-a",
        "--aux-data-path",
        type=str,
        help="Path to search for additional res files or an archive. Useful for keeping known phases separate from the searched phases.",
        dest="auxilliary_path",
    )
    parser.add_argument(
        "--scatter-coeffs",
        help="Json file with coefficients to use for scattering parameters.",
        type=str,
        default=None,
        dest="scatter_coeffs",
    )
    parser.add_argument(
        "-r",
        "--run-name",
        type=str,
        help="What the name of this run should be. Used to name checkpoint files, saved plots, or other data specific to this run.",
        default=time.strftime("%Y%m%d-%H%M%S", time.localtime()),
        dest="run_name",
    )
    parser.add_argument(
        "-D",
        "--reload-data",
        help="Force a reload of the data from the data paths. Will ignore any checkpoints or saved data from previous runs.",
        action="store_true",
        default=False,
        dest="reload_data",
    )
    parser.add_argument(
        "-M",
        "--reload_matrix",
        help="force a reload of the kernel matrix. Will ignore saved kernel matrices but still use saved data frames.",
        action="store_true",
        default=False,
        dest="recalc_kern",
    )
    parser.add_argument(
        "-f",
        "--formats",
        type=str,
        help="Comma separated list of matplotlib compatible file extensions to save plots in. Common options are: png,pdf,svg,jpg.",
        default="pdf,png",
        dest="plot_types",
    )
    parser.add_argument(
        "-p",
        "--pressure",
        type=float,
        help="The pressure of the data in the search. The data will be filtered for this pressure.",
        default=0,
        dest="pressure",
    )
    parser.add_argument(
        "--pressure-tolerance",
        type=float,
        help="The tolerance for filtering the pressure. Will keep pressures within this tolerance of the set pressure.",
        default=0.1,
        dest="pressure_tol",
    )
    parser.add_argument(
        "-Q",
        "--qspace-max",
        type=float,
        help="The upper limit on the q-space to consider for the generation and comparison of diffraction. 2*sin(theta) / wavelength",
        default=3,
        dest="max_q",
    )
    parser.add_argument(
        "-s",
        "--sigma",
        "--gaussian-width",
        type=float,
        help="""The value for sigma in the gaussians used for diffraction generation.
        Correspondes to the width of the peak in q-space.""",
        default=0.01,
        dest="sigma",
    )
    parser.add_argument(
        "-l",
        "--eps",
        "--epsilon",
        "--lorentz-width",
        type=float,
        help="""The value for eps in the lorentzian used for density of states generation.
        Corresponds to the width of the peaks in energy-space.
        Larger values may be helpful to smooth out sampling error for small data sets.""",
        default=0.001,
        dest="eps",
    )
    parser.add_argument(
        "--signal-size",
        type=int,
        help="""The size of the signal counts used in the diffraction signal and density of states.
        Larger is more precise but more expensive.""",
        default=5000,
        dest="signal_size",
    )
    parser.add_argument(
        "-w",
        "--pre-integration-width",
        type=float,
        help="""The width of chunks used to integrate probability density into more discrete probabilities before using them in the plogp sum.""",
        default=0.01,
        dest="energy_int_width",
    )
    parser.add_argument(
        "-S",
        "--sample-size",
        type=int,
        help="How many structures to sample from the filtered data after filtering for pressure and formation energy.",
        dest="sample_size",
    )
    parser.add_argument(
        "--ref-limit",
        type=int,
        help="How many structures to limit found elemental phases to. Can be helpful avoiding bias in small samples or when searches find many duplicate elemental phases.",
        dest="elemental_size",
    )
    parser.add_argument(
        "-c",
        "--energy-cutoff",
        type=float,
        help="Formation energy used to filter out high energy phases. Structures with formation energy above this will be filtered out.",
        dest="formation_cutoff",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed used to set the random state for the random sampling used by --sample-size option.",
        dest="random_seed",
    )
    parser.add_argument(
        "--diffraction-source",
        type=str,
        choices=["pymatgen", "ase"],
        help="The library to build diffraction data from. Will use that library to read files and build the structure.",
        dest="diff_source",
        default="pymatgen",
    )
    parser.add_argument(
        "--dataframe-source",
        type=str,
        choices=["pymatgen", "ase"],
        help="The library to build dataframes from. Will use that library to read files and read structure data from.",
        dest="df_source",
        default="pymatgen",
    )
    parser.add_argument(
        "-n",
        "--njobs",
        type=int,
        help="The number of proceses or jobs to use with multiprocessing or joblib.",
        dest="n_jobs",
        default=1,
    )
    parser.add_argument(
        "--use-soap",
        action="store_true",
        default=False,
        dest="use_soap",
        help="Whether to use soap descriptors for analysis.",
    )
    parser.add_argument(
        "--parallel-kernel-backend",
        type=str,
        choices=["multi", "joblib", "joblib-mmap"],
        default="joblib",
        help="Which backend to use for the parallelization of computing the kernel matrix. 'multi' refers to the multiporcessing module.",
        dest="parallel_kernel_backend",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
