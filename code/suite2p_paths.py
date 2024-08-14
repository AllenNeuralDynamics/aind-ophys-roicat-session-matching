import os
from pathlib import Path

def load_plane_paths(data_dir='../../data/'):
    planes={}
    for session_name in sorted(os.listdir(data_dir)):
        if session_name.startswith('multiplane'):            
            session_dir = Path(data_dir) / session_name
            for plane_idx, plane_name in enumerate(sorted(os.listdir(session_dir))):
                plane_dir = session_dir / plane_name
                if plane_dir.is_dir() and plane_dir.parts[-1] != 'nextflow': 
                    
                    projection_image_path = f"{plane_dir}/motion_correction/{plane_name}_maximum_projection.png"
                    stat_path=f"{plane_dir}/segmentation/suite2p/plane0/stat.npy"
                    ops_path=f"{plane_dir}/segmentation/suite2p/plane0/ops.npy"
                    
                    if os.path.exists(projection_image_path) and os.path.exists(stat_path) and os.path.exists(ops_path):
                        if plane_idx not in planes:
                            planes[plane_idx] = []
    
                        planes[plane_idx].append(dict(
                            plane_name=plane_name,
                            projection_image_path=projection_image_path,
                            stat_path=stat_path,
                            ops_path=ops_path,
                        ))
    return planes