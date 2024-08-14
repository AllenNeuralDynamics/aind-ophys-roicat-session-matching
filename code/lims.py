## spatial footprints

def lims2dense(rois, dims):
    N = len(rois)
    masks = np.zeros((N,) + dims)
    for i, roi in enumerate(rois):
        x,y,h,w = roi['x'], roi['y'], roi['height'], roi['width']
        masks[i, y:y+h,x:x+w] = roi['mask_matrix']
    return masks

all_rois = []
for segmentation_output_path, fov in zip([p['segmentation_output_path'] for p in plane], fovs):    
    with open(segmentation_output_path, 'r') as file:
        rois = json.load(file)
        all_rois.append(lims2dense(rois, fov.size))
        
data.set_spatialFootprints(all_rois, um_per_pixel=.78)
data.set_ROI_images(all_rois, um_per_pixel=.78)
# ## Transform spatial footprints to ROI images
# data._transform_spatialFootprints_to_ROIImages(out_height_width=out_height_width)