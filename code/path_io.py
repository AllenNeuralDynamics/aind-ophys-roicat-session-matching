import os, json
from pathlib import Path

def load_session(session_dir, default_fov_scale_factor=None):
    planes={}
    
    for plane_idx, plane_name in enumerate(sorted(os.listdir(session_dir))):
        plane_dir = session_dir / plane_name
        if plane_dir.is_dir() and plane_dir.parts[-1] != 'nextflow': 
            
            projection_image_path = f"{plane_dir}/motion_correction/{plane_name}_maximum_projection.png"
            stat_path=f"{plane_dir}/segmentation/suite2p/plane0/stat.npy"
            ops_path=f"{plane_dir}/segmentation/suite2p/plane0/ops.npy"
            
            if os.path.exists(projection_image_path) and os.path.exists(stat_path) and os.path.exists(ops_path):
                planes[plane_idx] = dict(
                    plane_name=plane_name,
                    projection_image_path=projection_image_path,
                    stat_path=stat_path,
                    ops_path=ops_path,
                )
        
    session_file = session_dir / "session.json"
 
    if session_file.exists():
        with open(session_file, 'r') as f:
            session = json.load(f)
        
        for data_stream in session['data_streams']:
            ophys_fovs = data_stream['ophys_fovs']
            
            for i,ophys_fov in enumerate(ophys_fovs):
                plane_index = i # ophys_fov['index'] - TODO, bug! `index` was set wrong for awhile
                fov_scale_factor = ophys_fov.get('fov_scale_factor', None)
                if fov_scale_factor is not None:
                    planes[plane_index]['fov_scale_factor'] = float(fov_scale_factor)

    if default_fov_scale_factor is not None:
        for plane_index, plane in planes.items():
            if not 'fov_scale_factor' in plane:
                plane['fov_scale_factor'] = default_fov_scale_factor
                    
    return planes

def load_planes(data_dir, default_fov_scale_factor=None):
    sessions = []
    for session_path in os.listdir(data_dir):
        session = load_session(Path(data_dir) / Path(session_path))
        sessions.append(session)

    planes = {}
    for session in sessions:
        for plane_index, plane in session.items():
            if plane_index in planes:
                planes[plane_index].append(plane)
            else:
                planes[plane_index] = [plane]
        
    return planes
    