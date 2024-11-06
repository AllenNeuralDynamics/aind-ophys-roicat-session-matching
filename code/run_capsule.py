""" top level run script """

import logging
import datetime
import argparse

from metadata import build_processing, find_data_descriptions, build_data_description
from tracking import align_plane, build_roi_table
from path_io import load_planes

GEOMETRIC_METHOD_DEFAULTS = {
    'RoMa': {  ## Accuracy: Best, Speed: Very slow (can be fast with a GPU).
            'model_type': 'outdoor',
            'n_points': 10000,  ## Higher values mean more points are used for the registration. Useful for larger FOV_images. Larger means slower.
            'batch_size': 1000,
    },
    'DISK_LightGlue': {  ## Accuracy: Good, Speed: Fast.
        'num_features': 3000,  ## Number of features to extract and match. I've seen best results around 2048 despite higher values typically being better.
        'threshold_confidence': 0.0,  ## Higher values means fewer but better matches.
        'window_nms': 7,  ## Non-maximum suppression window size. Larger values mean fewer non-suppressed points.
    },
    'LoFTR': {  ## Accuracy: Okay. Speed: Medium.
        'model_type': 'indoor_new',
        'threshold_confidence': 0.2,  ## Higher values means fewer but better matches.
    },
    'ECC_cv2': {  ## Accuracy: Okay. Speed: Medium.
        'mode_transform': 'euclidean',  ## Must be one of {'translation', 'affine', 'euclidean', 'homography'}. See cv2 documentation on findTransformECC for more details.
        'n_iter': 200,
        'termination_eps': 1e-09,  ## Termination criteria for the registration algorithm. See documentation for more details.
        'gaussFiltSize': 1,  ## Size of the gaussian filter used to smooth the FOV_image before registration. Larger values mean more smoothing.
        'auto_fix_gaussFilt_step': 10,  ## If the registration fails, then the gaussian filter size is reduced by this amount and the registration is tried again.
    },
    'PhaseCorrelation': {  ## Accuracy: Poor. Speed: Very fast. Notes: Only applicable for translations, not rotations or scaling.
        'bandpass_freqs': [1, 30],
        'order': 5,
    },
}

NONRIGID_METHOD_DEFAULTS = {
    'DeepFlow': {},  ## Accuracy: Good (good in middle, poor on edges), Speed: Fast (CPU only)
    'RoMa': {  ## Accuracy: Okay (decent in middle, poor on edges), Speed: Slow (can be fast with a GPU), Notes: This method can work on the raw images without pre-registering using geometric methods.
        'model_type': 'outdoor',
    },
    'OpticalFlowFarneback': {  ## Accuracy: Varies (can sometimes be tuned to be the best as there are no edge artifacts), Speed: Medium (CPU only)
        'pyr_scale': 0.7,
        'levels': 5,
        'winsize': 256,
        'iterations': 15,
        'poly_n': 5,
        'poly_sigma': 1.5,            
    },
}

def run():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-fov-scale-factor", default=None, type=float)
    parser.add_argument("--geometric-method", default="RoMa", type=str)
    parser.add_argument("--nonrigid-method", default="RoMa", type=str)
    parser.add_argument("--all-to-all", default="on", type=str)
    parser.add_argument("--debug", default="off", type=str)
    parser.add_argument("--projection", default="maxImg", type=str)
    args = parser.parse_args()

    args.nonrigid_method = None if args.nonrigid_method == "Off" else args.nonrigid_method
    args.all_to_all = args.all_to_all == "on"

    logging.info(f"geometric method: {args.geometric_method}, {GEOMETRIC_METHOD_DEFAULTS.get(args.geometric_method)}")
    logging.info(f"nonrigid method: {args.nonrigid_method}, {NONRIGID_METHOD_DEFAULTS.get(args.nonrigid_method)}")

    planes = load_planes(
        data_dir='/data/',
        projection=args.projection, 
        default_fov_scale_factor=args.default_fov_scale_factor
    )

    outputs = []
    for name, plane in planes.items():
        logging.info(f"running plane: {name}")
        t_start = datetime.datetime.now()
        
        results = align_plane(
            plane=plane,
            geometric_method=args.geometric_method,
            geometric_method_params=GEOMETRIC_METHOD_DEFAULTS,
            nonrigid_method=args.nonrigid_method,
            nonrigid_method_params=NONRIGID_METHOD_DEFAULTS,
            all_to_all=args.all_to_all,
            out_dir='/results', 
            out_name=str(name))
        
        t_end = datetime.datetime.now()
        outputs.append(dict(
            t_start=t_start,
            t_end=t_end,
            results=results,
            plane_name=name
        ))

        if args.debug == "on":
            break
            
    p = build_processing(outputs)
    p.write_standard_file(output_directory='/results')
    dds = list(find_data_descriptions('/data/'))
    if dds:
        d = build_data_description(dds, outputs)
        d.write_standard_file(output_directory='/results')

if __name__ == "__main__": run()