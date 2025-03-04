import os, json
from pathlib import Path
import logging
from typing import List
import numpy as np
from dataclasses import dataclass
from pint import UnitRegistry
import h5py
import sparse
import glob


@dataclass
class MatchingData:
    session_name: str = None
    plane_index: int = None
    plane_name: str = None
    dims: List[int] = None
    rois: np.array = None
    um_per_px: float = None
    image: np.array = None


def load_extraction_h5(matching, projection, extraction_file):
    with h5py.File(extraction_file) as f:
        matching.rois = sparse.COO(
            f["rois/coords"], f["rois/data"], f["rois/shape"]
        ).todense()
        matching.image = f[projection][:]


def load_suite2p(matching, projection, stat_file, ops_file):
    s2p_projection_lookup = {"meanImg": "meanImg", "maxImg": "max_proj"}
    s2p_projection = s2p_projection_lookup[projection]

    stat = np.load(stat_file, allow_pickle=True)
    rois = np.empty((len(stat),) + matching.dims, dtype=np.float32)
    for i, s in enumerate(stat):
        rois[i, s["ypix"], s["xpix"]] = s["lam"]

    matching.rois = rois

    ops = np.load(ops_file, allow_pickle=True).item()
    matching.image = ops[s2p_projection]
    return rois


def load_session_planes(session_dir, projection, default_fov_scale_factor=0.78):
    session_dir = Path(session_dir)
    planes = []
    for plane_index, plane_name in enumerate(sorted(os.listdir(session_dir))):
        plane_dir = session_dir / plane_name
        if plane_dir.is_dir() and plane_dir.parts[-1] != "nextflow":

            matching = MatchingData(
                plane_name=plane_name,
                plane_index=plane_index,
                session_name=session_dir.parts[-1],
            )

            try:
                session_file = next(session_dir.glob("session.json"))
                matching.um_per_px, matching.dims = get_plane_metadata(session_file)
            except StopIteration as e:
                logging.warning(f"could not find session.json for {str(session_dir)}")
                matching.um_per_px = default_fov_scale_factor

            try:
                extraction_file = next(plane_dir.glob("**/*extraction.h5"))
                load_extraction_h5(matching, projection, extraction_file)

            except StopIteration as e:
                try:
                    stat_file = next(plane_dir.glob("**/stat.npy"))
                    ops_file = next(plane_dir.glob("**/ops.npy"))
                    load_suite2p(matching, projection, stat_file, ops_file)
                except StopIteration as e:
                    logging.error(f"could not find rois for {plane_dir}")
                    continue

            planes.append(matching)

    return planes


def get_plane_metadata(session_file):
    """get um_per_pixel and dims (FOV size) from session.json"""

    with open(session_file, "r") as j:
        session_data = json.load(j)

    for data_stream in session_data["data_streams"]:
        if data_stream.get("ophys_fovs"):
            fov = data_stream["ophys_fovs"][0]
            um_per_pixel = (
                float(fov["fov_scale_factor"])
                * (UnitRegistry().parse_expression(fov["fov_scale_factor_unit"]))
                .to("um/pixel")
                .magnitude
            )
            dims = (fov["fov_height"], fov["fov_width"])
    return um_per_pixel, dims


def load_planes(data_dir, projection, default_fov_scale_factor=None):
    if default_fov_scale_factor:
        logging.info(
            f"running with default fov scale factor: {default_fov_scale_factor}"
        )

    sessions = []
    for session_path in os.listdir(data_dir):
        session = load_session_planes(
            session_dir=Path(data_dir) / Path(session_path),
            projection=projection,
            default_fov_scale_factor=default_fov_scale_factor,
        )
        sessions.append(session)

    planes = {}
    for session in sessions:
        for plane in session:
            if plane.plane_index in planes:
                planes[plane.plane_index].append(plane)
            else:
                planes[plane.plane_index] = [plane]

    return planes
