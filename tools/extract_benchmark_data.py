"""Reduce raw VASP output from the 2021 GAP-20 carbon test suite to extxyz archives."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from warnings import warn

from ase import Atoms
from ase.io import read, write

SRC = Path(
    "/data/pr_archive/source_drives/sdb2_Patrick_4Tb_BU/Happy_Electron_Backup"
    "/Research/Carbon_Potential/1_Potential_Development"
)
EXCLUDED = {"graphite_layer_sep"}
FORBIDDEN_NAMES = ("POTCAR",)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "ml_peg_benchmark_data"
STAGING_ROOT = DATA_ROOT / "staging"

def read_reference(vasprun_path: Path) -> Atoms:
    """
    Read the final ionic step of a VASP run as an ASE Atoms object.

    Parameters
    ----------
    vasprun_path
        Path to a vasprun.xml file.

    Returns
    -------
    Atoms
        Structure with REF_energy in info and REF_forces in arrays.
    """
    atoms = read(vasprun_path, index=-1)
    atoms.info["REF_energy"] = atoms.get_potential_energy()
    atoms.arrays["REF_forces"] = atoms.get_forces()
    atoms.calc = None
    return atoms


def _read_reference_or_warn(vasprun_path: Path) -> Atoms | None:
    """
    Read a vasprun.xml, returning None and warning instead of raising on failure.

    Parameters
    ----------
    vasprun_path
        Path to a vasprun.xml file.

    Returns
    -------
    Atoms or None
        Parsed structure, or None if the read failed.
    """
    try:
        return read_reference(vasprun_path)
    except Exception as exc:  # noqa: BLE001
        warn(f"Failed to read reference {vasprun_path}: {exc}", stacklevel=2)
        return None


def scf_converged(vasprun_path: Path, default_nelm: int = 60) -> bool:
    """
    Report whether a static VASP run reached electronic convergence.

    An unconverged run exhausts NELM electronic steps instead of meeting EDIFF.
    Valid only for single-ionic-step runs, which is what the barrier scans are.

    Parameters
    ----------
    vasprun_path
        Path to a vasprun.xml file.
    default_nelm
        Electronic step limit assumed when the INCAR does not set NELM.

    Returns
    -------
    bool
        True if the run used fewer than NELM electronic steps.
    """
    nelm = default_nelm
    incar_path = vasprun_path.parent / "INCAR"
    if incar_path.is_file():
        for line in incar_path.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip().upper() == "NELM":
                nelm = int(value.split()[0])
                break
    steps = vasprun_path.read_text().count("<scstep>")
    return steps < nelm


def assert_no_forbidden_files(staging_dir: Path) -> None:
    """
    Fail if any licensed VASP pseudopotential file has been staged.

    Parameters
    ----------
    staging_dir
        Directory about to be archived.

    Raises
    ------
    RuntimeError
        If a POTCAR file is present.
    """
    offending = [
        path
        for path in staging_dir.rglob("*")
        if path.is_file() and path.name.startswith(FORBIDDEN_NAMES)
    ]
    if offending:
        raise RuntimeError(f"Refusing to archive licensed VASP files: {offending}")


def assert_no_excluded_paths(staging_dir: Path) -> None:
    """
    Fail if any path under an excluded source directory has been staged.

    Parameters
    ----------
    staging_dir
        Directory about to be archived.

    Raises
    ------
    RuntimeError
        If a path originating from an excluded source directory is present.
    """
    offending = [
        path for path in staging_dir.rglob("*") if EXCLUDED & set(path.parts)
    ]
    if offending:
        raise RuntimeError(f"Refusing to archive excluded paths: {offending}")


def _self_check_potcar_guard() -> None:
    """
    Exercise assert_no_forbidden_files against a clean and a contaminated directory.

    Raises
    ------
    AssertionError
        If the guard fails to raise on a contaminated directory, or raises on
        a clean one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "clean.xyz").write_text("clean")
        assert_no_forbidden_files(tmp_path)
        (tmp_path / "POTCAR").write_text("licensed")
        try:
            assert_no_forbidden_files(tmp_path)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "assert_no_forbidden_files failed to raise on a POTCAR file"
            )


def write_system(system_dir: Path, frames: list[Atoms]) -> None:
    """
    Write one or more reference frames for a single system.

    Parameters
    ----------
    system_dir
        Directory to create, named after the system.
    frames
        Structures to write, in order, to system_dir/reference.xyz.
    """
    if not frames:
        warn(f"No frames extracted for {system_dir}, skipping", stacklevel=2)
        return
    system_dir.mkdir(parents=True, exist_ok=True)
    write(system_dir / "reference.xyz", frames, format="extxyz")


def write_list(benchmark_dir: Path, names: list[str]) -> None:
    """
    Write the newline-separated list of system names defining iteration order.

    Parameters
    ----------
    benchmark_dir
        Top-level directory for one benchmark.
    names
        System names, in iteration order.
    """
    (benchmark_dir / "list").write_text("\n".join(names) + "\n")


def zip_benchmark(benchmark: str) -> None:
    """
    Zip a staged benchmark directory so it unpacks to a directory of the same name.

    Parameters
    ----------
    benchmark
        Name of the benchmark, matching STAGING_ROOT/<benchmark>.
    """
    benchmark_dir = STAGING_ROOT / benchmark
    zip_path = DATA_ROOT / f"{benchmark}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in sorted(benchmark_dir.rglob("*")):
            if path.is_file():
                zip_file.write(path, path.relative_to(STAGING_ROOT))


def extract_lattice_parameters(staging_root: Path) -> None:
    """
    Extract the eight bulk/molecular reference structures for lattice_parameters.

    Parameters
    ----------
    staging_root
        Root staging directory holding one subdirectory per benchmark.
    """
    systems = {
        "Graphite": "Graphite",
        "Graphene": "Graphene",
        "Diamond": "Diamond",
        "Lonsdaleite": "Lonsdaleite",
        "Nanotube_9_0": "Nanotube-(9,0)",
        "Nanotube_9_9": "Nanotube-(9,9)",
        "C60": "C60",
        "C100": "C100",
    }
    benchmark_dir = staging_root / "lattice_parameters"
    if benchmark_dir.exists():
        shutil.rmtree(benchmark_dir)

    bulk_root = SRC / "Tests" / "lattice_parameters" / "Bulk_Structures"

    # The Bulk_Structures (9,9) cell is a defect-generation working copy: 174 atoms
    # with 6 two-coordinated sites, where an intact 5-cell armchair tube needs 180.
    # The formation-energy reference is an intact 36-atom unit cell at the same level
    # of theory (optB88-vdW, ENCUT 500, ISPIN 1, ISIF 2).
    overrides = {
        "Nanotube_9_9": SRC
        / "Tests"
        / "nanotubes_formation_energy"
        / "Reference"
        / "Armchair"
        / "Nanotube_9_9"
        / "vasprun.xml",
    }

    names = []
    for source_name, system_name in systems.items():
        source = overrides.get(source_name, bulk_root / source_name / "vasprun.xml")
        atoms = _read_reference_or_warn(source)
        if atoms is None:
            continue
        write_system(benchmark_dir / system_name, [atoms])
        names.append(system_name)
    write_list(benchmark_dir, names)


def extract_surface_energies(staging_root: Path) -> None:
    """
    Extract bulk/as-cut/relaxed triples for four facets plus the amorphous ensemble.

    Parameters
    ----------
    staging_root
        Root staging directory holding one subdirectory per benchmark.
    """
    benchmark_dir = staging_root / "surface_energies"
    if benchmark_dir.exists():
        shutil.rmtree(benchmark_dir)

    dft_root = SRC / "Surfaces" / "DFT_Reference"
    # Diamond {111} is omitted: its as_cut and relaxed vasprun.xml are truncated
    # mid-SCF with no closing </calculation> or </modeling>, no final structure and no
    # forces. The scheduler log Not_A_VASP_Calculation_.o729670 shows the job never ran.
    # Only the bulk survived, which alone cannot give a surface energy.
    facet_systems = {
        "Diamond_100": ("Diamond/100", ("bulk", "as_cut", "relaxed")),
        "Diamond_110": ("Diamond/110", ("bulk", "as_cut", "relaxed")),
        "Graphite_0001": ("Graphite/0001", ("bulk", "as_cut", "actual_relaxed")),
    }

    frame_roles = ("bulk", "as_cut", "relaxed")
    names = []
    for system_name, (relative_path, frame_names) in facet_systems.items():
        system_root = dft_root / relative_path
        frames = []
        for frame_name, frame_role in zip(frame_names, frame_roles, strict=True):
            atoms = _read_reference_or_warn(system_root / frame_name / "vasprun.xml")
            if atoms is not None:
                atoms.info["structure_role"] = frame_role
                frames.append(atoms)
        write_system(benchmark_dir / system_name, frames)
        names.append(system_name)

    amorphous_root = dft_root / "Amorphous" / "Bulk_Unrelaxed_2"
    amorphous_frames = []
    for config_index in range(1, 11):
        config = f"{config_index:02d}"
        bulk_atoms = _read_reference_or_warn(
            amorphous_root / config / "bulk" / "vasprun.xml"
        )
        if bulk_atoms is not None:
            bulk_atoms.info["amorphous_config"] = config_index
            bulk_atoms.info["structure_role"] = "bulk"
            amorphous_frames.append(bulk_atoms)

        slab_root = amorphous_root / config / "surfaces_unrelaxed"
        for slab_dir in sorted(p for p in slab_root.iterdir() if p.is_dir()):
            slab_atoms = _read_reference_or_warn(slab_dir / "vasprun.xml")
            if slab_atoms is None:
                continue
            slab_atoms.info["amorphous_config"] = config_index
            slab_atoms.info["structure_role"] = "slab"
            slab_atoms.info["slab_name"] = slab_dir.name
            amorphous_frames.append(slab_atoms)

    write_system(benchmark_dir / "Amorphous_Bulk_Unrelaxed_2", amorphous_frames)
    names.append("Amorphous_Bulk_Unrelaxed_2")

    write_list(benchmark_dir, names)


def extract_defect_energies(staging_root: Path) -> None:
    """
    Extract pristine-host/defect pairs for diamond, graphite, graphene and nanotubes.

    Parameters
    ----------
    staging_root
        Root staging directory holding one subdirectory per benchmark.
    """
    benchmark_dir = staging_root / "defect_energies"
    if benchmark_dir.exists():
        shutil.rmtree(benchmark_dir)

    pristine_root = SRC / "Defect_Structure" / "Reference_Structures"
    defect_root = (
        SRC
        / "Tests"
        / "diamond_defect_energies"
        / "Reference_Configs"
        / "Defect_Structures"
    )

    hosts = {
        "Diamond_Large": {
            "Diamond_Split_Interstitial": (
                "Defective_Diamond/Diamond_Split_Interstitial"
            ),
            "Diamond_Monovacancy": "Diamond_Large/Diamond_Monovacancy",
            "Diamond_Divacancy": "Diamond_Large/Diamond_Divacancy",
        },
        "Graphite": {
            "Graphite_Stone_Wales": "Defective_Graphite/Graphite_Stone_Wales",
            "Graphite_Monovacancy": "Defective_Graphite/Graphite_Monovacancy",
            "Graphite_Divacancy": "Defective_Graphite/Graphite_Divacancy",
            "Graphite_Interstitial_Bridging": (
                "Defective_Graphite/Graphite_Interstitial_Bridging"
            ),
            "Graphite_Dumbell_Adatom": "Defective_Graphite/Graphite_Dumbell_Adatom",
        },
        "Graphene": {
            "Graphene_Stone_Wales": "Defective_Graphene/Stone_Wales",
            "Graphene_Monovacancy": "Defective_Graphene/Monovacancy",
            "Graphene_Divacancy": "Defective_Graphene/Divacancy",
            "Graphene_Divacancy_555-777": "Defective_Graphene/Divacancy_555-777",
            "Graphene_Divacancy_5555-6-7777": (
                "Defective_Graphene/Divacancy_5555-6-7777"
            ),
            "Graphene_Bridging_Adatom": "Defective_Graphene/Bridging_Adatom",
            "Graphene_Dumbell_Adatom": "Defective_Graphene/Dumbell_Adatom",
        },
        "Nanotube_9_0": {
            "Nanotube_9_0_Stone_Wales_Aligned": (
                "Defective_9_0_Nanotube/Nanotube_Stone_Wales_Aligned"
            ),
            "Nanotube_9_0_Stone_Wales_Unaligned": (
                "Defective_9_0_Nanotube/Nanotube_Stone_Wales_Unaligned"
            ),
            "Nanotube_9_0_Monovacancy": "Defective_9_0_Nanotube/Nanotube_Monovacancy",
            "Nanotube_9_0_Divacancy": "Defective_9_0_Nanotube/Nanotube_Divacancy",
        },
        "Nanotube_9_9": {
            "Nanotube_9_9_Stone_Wales_Aligned": (
                "Defective_9_9_Nanotube/Nanotube_Stone_Wales_Aligned"
            ),
            "Nanotube_9_9_Stone_Wales_Unaligned": (
                "Defective_9_9_Nanotube/Nanotube_Stone_Wales_Unaligned"
            ),
            "Nanotube_9_9_Monovacancy": "Defective_9_9_Nanotube/Nanotube_Monovacancy",
            "Nanotube_9_9_Divacancy": "Defective_9_9_Nanotube/Nanotube_Divacancy",
        },
    }

    names = []
    for pristine_name, defects in hosts.items():
        pristine_atoms = _read_reference_or_warn(
            pristine_root / pristine_name / "vasprun.xml"
        )
        if pristine_atoms is not None:
            pristine_atoms.info["structure_role"] = "pristine_host"
        for system_name, relative_defect_path in defects.items():
            defect_atoms = _read_reference_or_warn(
                defect_root / relative_defect_path / "vasprun.xml"
            )
            if defect_atoms is not None:
                defect_atoms.info["structure_role"] = "defect"
            frames = [
                atoms
                for atoms in (pristine_atoms, defect_atoms)
                if atoms is not None
            ]
            if len(frames) != 2:
                warn(
                    f"Incomplete pristine/defect pair for {system_name}, skipping",
                    stacklevel=2,
                )
                continue
            write_system(benchmark_dir / system_name, frames)
            names.append(system_name)

    write_list(benchmark_dir, names)


def extract_barriers(staging_root: Path) -> None:
    """
    Extract the four reaction-coordinate scans making up the barriers benchmark.

    Parameters
    ----------
    staging_root
        Root staging directory holding one subdirectory per benchmark.
    """
    benchmark_dir = staging_root / "barriers"
    if benchmark_dir.exists():
        shutil.rmtree(benchmark_dir)

    barriers_root = SRC / "Barriers"
    names = []

    def extract_scan(system_name: str) -> None:
        system_root = barriers_root / system_name
        step_dirs = sorted(
            (p for p in system_root.iterdir() if p.is_dir()),
            key=lambda p: int(p.name.split("_")[0]),
        )
        frames = []
        for step_dir in step_dirs:
            vasprun_path = step_dir / "vasprun.xml"
            atoms = _read_reference_or_warn(vasprun_path)
            if atoms is not None:
                atoms.info["scan_point"] = f"step_{step_dir.name}"
                atoms.info["scf_converged"] = scf_converged(vasprun_path)
                frames.append(atoms)
        write_system(benchmark_dir / system_name, frames)
        names.append(system_name)

    extract_scan("Fullerene_Sw")

    transition_root = (
        barriers_root
        / "Fullerene_Sw_Transition_Optim"
        / "Test_Path_1_C60"
        / "Ref_Check"
    )
    frame_dirs = sorted(
        (p for p in transition_root.iterdir() if p.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    frames = []
    for frame_dir in frame_dirs:
        vasprun_path = frame_dir / "vasprun.xml"
        atoms = _read_reference_or_warn(vasprun_path)
        if atoms is not None:
            atoms.info["scan_point"] = f"step_{frame_dir.name}"
            atoms.info["scf_converged"] = scf_converged(vasprun_path)
            frames.append(atoms)
    write_system(benchmark_dir / "Fullerene_Sw_Transition_Optim", frames)
    names.append("Fullerene_Sw_Transition_Optim")

    extract_scan("Graphene_Monovacancy")
    extract_scan("Graphene_SW")

    write_list(benchmark_dir, names)


def extract_nanotube_formation_energies(staging_root: Path) -> None:
    """
    Extract single-point nanotube energies plus the graphene reference.

    Parameters
    ----------
    staging_root
        Root staging directory holding one subdirectory per benchmark.
    """
    benchmark_dir = staging_root / "nanotube_formation_energies"
    if benchmark_dir.exists():
        shutil.rmtree(benchmark_dir)

    reference_root = SRC / "Tests" / "nanotubes_formation_energy" / "Reference"

    names = []
    for chirality_index in range(5, 15):
        system_name = f"Nanotube_{chirality_index}_0"
        atoms = _read_reference_or_warn(
            reference_root / "Zigzag" / system_name / "vasprun.xml"
        )
        if atoms is None:
            continue
        write_system(benchmark_dir / system_name, [atoms])
        names.append(system_name)

    for chirality_index in range(5, 15):
        system_name = f"Nanotube_{chirality_index}_{chirality_index}"
        atoms = _read_reference_or_warn(
            reference_root / "Armchair" / system_name / "vasprun.xml"
        )
        if atoms is None:
            continue
        write_system(benchmark_dir / system_name, [atoms])
        names.append(system_name)

    # Tube energies are reported relative to graphene, so the flat-sheet limit
    # ships alongside them. Same structure as the lattice_parameters reference.
    graphene = _read_reference_or_warn(
        SRC / "Tests" / "lattice_parameters" / "Bulk_Structures" / "Graphene"
        / "vasprun.xml"
    )
    if graphene is not None:
        write_system(benchmark_dir / "Graphene", [graphene])
        names.append("Graphene")

    write_list(benchmark_dir, names)


def main() -> None:
    """Run all five extractions, guard the output, then zip each benchmark."""
    _self_check_potcar_guard()

    STAGING_ROOT.mkdir(parents=True, exist_ok=True)

    extract_lattice_parameters(STAGING_ROOT)
    extract_surface_energies(STAGING_ROOT)
    extract_defect_energies(STAGING_ROOT)
    extract_barriers(STAGING_ROOT)
    extract_nanotube_formation_energies(STAGING_ROOT)

    assert_no_forbidden_files(STAGING_ROOT)
    assert_no_excluded_paths(STAGING_ROOT)

    for benchmark in (
        "lattice_parameters",
        "surface_energies",
        "defect_energies",
        "barriers",
        "nanotube_formation_energies",
    ):
        zip_benchmark(benchmark)


if __name__ == "__main__":
    main()
