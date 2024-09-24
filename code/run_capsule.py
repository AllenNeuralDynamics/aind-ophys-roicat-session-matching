""" top level run script """

import logging
import datetime
import argparse

from metadata import build_processing, find_data_descriptions, build_data_description
from tracking import align_plane, build_roi_table
from path_io import load_planes

def run():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-fov-scale-factor", default=None, type=float)
    args = parser.parse_args()

    planes = load_planes('/data/', default_fov_scale_factor=args.default_fov_scale_factor)

    outputs = []
    for name, plane in planes.items():
        print(f"running plane: {name}")
        t_start = datetime.datetime.now()
        results = align_plane(plane, '/results', str(name))
        t_end = datetime.datetime.now()
        outputs.append(dict(
            t_start=t_start,
            t_end=t_end,
            results=results,
            plane_name=name
        ))
        
    dds = list(find_data_descriptions('/data/'))
    p = build_processing(outputs)
    p.write_standard_file(output_directory='/results')
    d = build_data_description(dds, outputs)
    d.write_standard_file(output_directory='/results')

if __name__ == "__main__": run()