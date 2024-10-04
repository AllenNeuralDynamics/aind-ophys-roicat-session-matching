from pathlib import Path
import copy
import multiprocessing as mp
import tempfile
import logging
import matplotlib.pyplot as plt
import numpy as np

import roicat
import pandas as pd

import uuid

def build_roi_table(plane, results, quality_metrics):
    all_rois = []
    
    ucids = set(results['clusters']['labels_dict'].keys())
    uuid_lookup = { ucid: uuid.uuid4() for ucid in ucids if int(ucid) >= 0 }    
    roi_global_index=0
    for session_index, session_roi_ids in enumerate(results['clusters']['labels_bySession']):
        for roi_session_index, ucid in enumerate(session_roi_ids):
            matched = ucid >= 0
        
            hdbscan = quality_metrics.get('hdbscan', None)
            sample_probabilities = hdbscan['sample_probabilities'][roi_global_index] if hdbscan else None
            all_rois.append(dict(
                fov_name=plane[session_index].plane_name,
                session_name=plane[session_index].session_name,
                roi_session_index=int(roi_session_index), 
                ucid=int(ucid),
                roi_id=str(uuid_lookup.get(ucid, uuid.uuid4())),
                silhouette=quality_metrics['sample_silhouette'][roi_global_index],
                hdbscan_probability=sample_probabilities,
                roi_index=roi_global_index,
                matched=bool(matched)))
            
            roi_global_index += 1        
    
    return pd.DataFrame.from_records(all_rois, index='roi_index')


def align_plane(plane, mode_transform, out_dir, out_name):
    data = roicat.data_importing.Data_roicat()
    data.set_spatialFootprints([p.rois for p in plane], um_per_pixel=plane[0].um_per_px)
    data.transform_spatialFootprints_to_ROIImages(out_height_width=(36, 36))
    data._make_session_bool()
    data.set_FOV_images([p.image for p in plane])

    assert data.check_completeness(verbose=False)['tracking'], f"Data object is missing attributes necessary for tracking."
    
    aligner = roicat.tracking.alignment.Aligner(verbose=True)

    FOV_images = aligner.augment_FOV_images(
        FOV_images=data.FOV_images,
        spatialFootprints=data.spatialFootprints,
        normalize_FOV_intensities=True,
        roi_FOV_mixing_factor=0.5,
        use_CLAHE=True,
        CLAHE_grid_size=100,
        CLAHE_clipLimit=1,
        CLAHE_normalize=True,
    )
    
    aligner.fit_geometric(
    #     template=FOV_images[4],
        template=0.5,  ## specifies which image to use as the template. Either array (image), integer (ims_moving index), or float (ims_moving fractional index)
        ims_moving=FOV_images,  ## input images
        template_method='image',  ## 'sequential': align images to neighboring images (good for drifting data). 'image': align to a single image
        mode_transform=mode_transform,  ## type of geometric transformation. See openCV's cv2.findTransformECC for details
        mask_borders=(10,10,10,10),  ## number of pixels to mask off the edges (top, bottom, left, right)
        n_iter=100,  ## number of iterations for optimization
        termination_eps=1e-09,  ## convergence tolerance
        gaussFiltSize=31,  ## size of gaussian blurring filter applied to all images
        auto_fix_gaussFilt_step=10,  ## increment in gaussFiltSize after a failed optimization
    )

    aligner.transform_images_geometric(FOV_images);
    
    aligner.fit_nonrigid(
    #     template=FOV_images[1],
        template=0.5,  ## specifies which image to use as the template. Either array (image), integer (ims_moving index), or float (ims_moving fractional index)
        ims_moving=aligner.ims_registered_geo,  ## Input images. Typically the geometrically registered images
        remappingIdx_init=aligner.remappingIdx_geo,  ## The remappingIdx between the original images (and ROIs) and ims_moving
        template_method='image',  ## 'sequential': align images to neighboring images. 'image': align to a single image, good if using geometric registration first
        mode_transform='createOptFlow_DeepFlow',  ## algorithm for non-rigid transformation. Either 'createOptFlow_DeepFlow' or 'calcOpticalFlowFarneback'. See openCV docs for each. 
        kwargs_mode_transform=None,  ## kwargs for `mode_transform`
    )

    aligner.transform_images_nonrigid(FOV_images);

    aligner.transform_ROIs(
        ROIs=data.spatialFootprints, 
        remappingIdx=aligner.remappingIdx_nonrigid,
        normalize=True,
    );
    
    blurrer = roicat.tracking.blurring.ROI_Blurrer(
        frame_shape=(data.FOV_height, data.FOV_width),  ## FOV height and width
        kernel_halfWidth=6,  ## The half width of the 2D gaussian used to blur the ROI masks
        plot_kernel=False,  ## Whether to visualize the 2D gaussian
    )

    blurrer.blur_ROIs(
        spatialFootprints=aligner.ROIs_aligned[:],
    );
    
    DEVICE = roicat.helpers.set_device(use_GPU=True, verbose=True)
    dir_temp = tempfile.gettempdir()

    roinet = roicat.ROInet.ROInet_embedder(
        device=DEVICE,  ## Which torch device to use ('cpu', 'cuda', etc.)
        dir_networkFiles=dir_temp,  ## Directory to download the pretrained network to
        download_method='check_local_first',  ## Check to see if a model has already been downloaded to the location (will skip if hash matches)
        download_url='https://osf.io/x3fd2/download',  ## URL of the model
        download_hash='7a5fb8ad94b110037785a46b9463ea94',  ## Hash of the model file
        forward_pass_version='latent',  ## How the data is passed through the network
        verbose=True,  ## Whether to print updates
    )
    
    roinet.generate_dataloader(
        ROI_images=data.ROI_images,  ## Input images of ROIs
        um_per_pixel=data.um_per_pixel,  ## Resolution of FOV
        pref_plot=False,  ## Whether or not to plot the ROI sizes

        jit_script_transforms=False,  ## (advanced) Whether or not to use torch.jit.script to speed things up

        batchSize_dataloader=8,  ## (advanced) PyTorch dataloader batch_size
        pinMemory_dataloader=True,  ## (advanced) PyTorch dataloader pin_memory
        numWorkers_dataloader=mp.cpu_count(),  ## (advanced) PyTorch dataloader num_workers
        persistentWorkers_dataloader=True,  ## (advanced) PyTorch dataloader persistent_workers
        prefetchFactor_dataloader=2,  ## (advanced) PyTorch dataloader prefetch_factor
    );
    
    roinet.generate_latents()
    
    swt = roicat.tracking.scatteringWaveletTransformer.SWT(
        kwargs_Scattering2D={'J': 2, 'L': 12},  ## 'J' is the number of convolutional layers. 'L' is the number of wavelet angles.
        image_shape=data.ROI_images[0].shape[1:3],  ## size of a cropped ROI image
        device=DEVICE,  ## PyTorch device
    )

    swt.transform(
        ROI_images=roinet.ROI_images_rs,  ## All the cropped and resized ROI images
        batch_size=100,  ## Batch size for each iteration (smaller is less memory but slower)
    );
    sim = roicat.tracking.similarity_graph.ROI_graph(
        n_workers=-1,  ## Number of CPU cores to use. -1 for all.
        frame_height=data.FOV_height,
        frame_width=data.FOV_width,
        block_height=128,  ## size of a block
        block_width=128,  ## size of a block
        algorithm_nearestNeigbors_spatialFootprints='brute',  ## algorithm used to find the pairwise similarity for s_sf. ('brute' is slow but exact. See docs for others.)
        verbose=True,  ## Whether to print outputs
    )

    

    s_sf, s_NN, s_SWT, s_sesh = sim.compute_similarity_blockwise(
        spatialFootprints=blurrer.ROIs_blurred,  ## Mask spatial footprints
        features_NN=roinet.latents,  ## ROInet output latents
        features_SWT=swt.latents,  ## Scattering wavelet transform output latents
        ROI_session_bool=data.session_bool,  ## Boolean array of which ROIs belong to which sessions
        spatialFootprint_maskPower=1.0,  ##  An exponent to raise the spatial footprints to to care more or less about bright pixels
    );
    
    sim.make_normalized_similarities(
        centers_of_mass=data.centroids,  ## ROI centroid positions
        features_NN=roinet.latents,  ## ROInet latents
        features_SWT=swt.latents,  ## SWT latents
        k_max=data.n_sessions*100,  ## Maximum number of nearest neighbors to consider for the normalizing distribution
        k_min=data.n_sessions*10,  ## Minimum number of nearest neighbors to consider for the normalizing distribution
        algo_NN='kd_tree',  ## Nearest neighbors algorithm to use
        device=DEVICE,
    )
    
    ## Initialize the clusterer object by passing the similarity matrices in
    clusterer = roicat.tracking.clustering.Clusterer(
        s_sf=sim.s_sf,
        s_NN_z=sim.s_NN_z,
        s_SWT_z=sim.s_SWT_z,
        s_sesh=sim.s_sesh,
        verbose=1,
    )
    
    kwargs_makeConjunctiveDistanceMatrix_best = clusterer.find_optimal_parameters_for_pruning(
        n_bins=None,  ## Number of bins to use for the histograms of the distributions. If None, then a heuristic is used.
        smoothing_window_bins=None,  ## Number of bins to use to smooth the distributions. If None, then a heuristic is used.
        kwargs_findParameters={
            'n_patience': 300,  ## Number of optimization epoch to wait for tol_frac to converge
            'tol_frac': 0.001,  ## Fractional change below which optimization will conclude
            'max_trials': 1200,  ## Max number of optimization epochs
            'max_duration': 60*10,  ## Max amount of time (in seconds) to allow optimization to proceed for
            'value_stop': 0.0,  ## Goal value. If value equals or goes below value_stop, optimization is stopped.
        },
        bounds_findParameters={
            'power_NN': (0.0, 2.),  ## Bounds for the exponent applied to s_NN
            'power_SWT': (0.0, 2.),  ## Bounds for the exponent applied to s_SWT
            'p_norm': (-5, -0.1),  ## Bounds for the p-norm p value (Minkowski) applied to mix the matrices
            'sig_NN_kwargs_mu': (0., 1.0),  ## Bounds for the sigmoid center for s_NN
            'sig_NN_kwargs_b': (0.1, 1.5),  ## Bounds for the sigmoid slope for s_NN
            'sig_SWT_kwargs_mu': (0., 1.0),  ## Bounds for the sigmoid center for s_SWT
            'sig_SWT_kwargs_b': (0.1, 1.5),  ## Bounds for the sigmoid slope for s_SWT
        },
        n_jobs_findParameters=-1,  ## Number of CPU cores to use (-1 is all cores)
    )
    
    clusterer.make_pruned_similarity_graphs(
        d_cutoff=None,  ## Optionally manually specify a distance cutoff
        kwargs_makeConjunctiveDistanceMatrix=kwargs_makeConjunctiveDistanceMatrix_best,
        stringency=1.0,  ## Modifies the threshold for pruning the distance matrix. Higher values result in LESS pruning. New d_cutoff = stringency * truncated d_cutoff.
        convert_to_probability=False,    
    )
    
    if data.n_sessions >= 6:
        labels = clusterer.fit(
            d_conj=clusterer.dConj_pruned,  ## Input distance matrix
            session_bool=data.session_bool,  ## Boolean array of which ROIs belong to which sessions
            min_cluster_size=2,  ## Minimum number of ROIs that can be considered a 'cluster'
            n_iter_violationCorrection=6,  ## Number of times to redo clustering sweep after removing violations
            split_intraSession_clusters=True,  ## Whether or not to split clusters with ROIs from the same session
            cluster_selection_method='leaf',  ## (advanced) Method of cluster selection for HDBSCAN (see hdbscan documentation)
            d_clusterMerge=None,  ## Distance below which all ROIs are merged into a cluster
            alpha=0.999,  ## (advanced) Scalar applied to distance matrix in HDBSCAN (see hdbscan documentation)
            discard_failed_pruning=True,  ## (advanced) Whether or not to set all ROIs that could be separated from clusters with ROIs from the same sessions to label=-1
            n_steps_clusterSplit=100,  ## (advanced) How finely to step through distances to remove violations
        )

    else:
        labels = clusterer.fit_sequentialHungarian(
            d_conj=clusterer.dConj_pruned,  ## Input distance matrix
            session_bool=data.session_bool,  ## Boolean array of which ROIs belong to which sessions
            thresh_cost=0.8,  ## Threshold. Higher values result in more permissive clustering. Specifically, the pairwise metric distance between ROIs above which two ROIs cannot be clustered together.
        )
        
    quality_metrics = clusterer.compute_quality_metrics();
    
    labels_squeezed, labels_bySession, labels_bool, labels_bool_bySession, labels_dict = roicat.tracking.clustering.make_label_variants(labels=labels, n_roi_bySession=data.n_roi)

    results_clusters = {
        'labels': labels_squeezed,
        'labels_bySession': labels_bySession,
        'labels_dict': labels_dict,
        'quality_metrics': quality_metrics,
    }

    results_all = {
        "clusters":{
            "labels": roicat.util.JSON_List(labels_squeezed),
            "labels_bySession": roicat.util.JSON_List(labels_bySession),
            "labels_bool": labels_bool,
            "labels_bool_bySession": labels_bool_bySession,
            "labels_dict": roicat.util.JSON_Dict(labels_dict),
            "quality_metrics": roicat.util.JSON_Dict(clusterer.quality_metrics) if hasattr(clusterer, 'quality_metrics') else None,
        },
        "ROIs": {
            "ROIs_aligned": aligner.ROIs_aligned,
            "ROIs_raw": data.spatialFootprints,
            "frame_height": data.FOV_height,
            "frame_width": data.FOV_width,
            "idx_roi_session": np.where(data.session_bool)[1],
            "n_sessions": data.n_sessions,
        },
        #"input_data": {
        #    "paths_stat": data.paths_stat,
        #    "paths_ops": data.paths_ops,
        #},
    }

    run_data = {
        'data': data.__dict__,
        'aligner': aligner.__dict__,
        'blurrer': blurrer.__dict__,
        'roinet': roinet.__dict__,
        'swt': swt.__dict__,
        'sim': sim.__dict__,
        'clusterer': clusterer.__dict__,
    }

    params_used = {name: mod['params'] for name, mod in run_data.items()}

    
    logging.info(f'Number of clusters: {len(np.unique(results_clusters["labels"]))}')
    logging.info(f'Number of discarded ROIs: {(np.array(results_clusters["labels"])==-1).sum()}')
    
    FOV_clusters = roicat.visualization.compute_colored_FOV(
        spatialFootprints=[r.power(1.0) for r in results_all['ROIs']['ROIs_aligned']],  ## Spatial footprint sparse arrays
        FOV_height=results_all['ROIs']['frame_height'],
        FOV_width=results_all['ROIs']['frame_width'],
        labels=results_all["clusters"]["labels_bySession"],  ## cluster labels
    #     labels=(np.array(results["clusters"]["labels"])!=-1).astype(np.int64),  ## cluster labels
        # alphas_labels=confidence*1.5,  ## Set brightness of each cluster based on some 1-D array
    #     alphas_labels=(clusterer.quality_metrics['cluster_silhouette'] > 0) * (clusterer.quality_metrics['cluster_intra_means'] > 0.4),
    #     alphas_sf=clusterer.quality_metrics['sample_silhouette'],  ## Set brightness of each ROI based on some 1-D array
    )

    dir_save = (Path(out_dir) / out_name).resolve() 
    dir_save.mkdir(parents=True, exist_ok=True) 
    
    paths_save = {
        'results_clusters': str(Path(dir_save) / 'tracking.results_clusters.json'),
        'params_used':      str(Path(dir_save) / 'tracking.params_used.json'),
        'results_all':      str(Path(dir_save) / 'tracking.results_all.richfile'),
        'run_data':         str(Path(dir_save) / 'tracking.run_data.richfile'),
    }

    roicat.helpers.json_save(obj=results_clusters, filepath=paths_save['results_clusters'])
    roicat.helpers.json_save(obj=params_used, filepath=paths_save['params_used'])
    roicat.util.RichFile_ROICaT(path=paths_save['results_all']).save(obj=results_all, overwrite=True)
    roicat.util.RichFile_ROICaT(path=paths_save['run_data']).save(obj=run_data, overwrite=True)
    
    roicat.helpers.save_gif(
        array=roicat.helpers.add_text_to_images(
            images=[((f/np.max(f)) * 255).astype(np.uint8) for f in FOV_clusters], 
            text=[[f"{ii}",] for ii in range(len(FOV_clusters))], 
            font_size=3,
            line_width=10,
            position=(30, 90),
        ), 
        path=str(Path(dir_save).resolve() / 'FOV_clusters.gif'),
        frameRate=3.0,
        loop=0,
    )

    roicat.helpers.save_gif(
        array=roicat.helpers.add_text_to_images(
            images=[(f * 255).astype(np.uint8) for f in aligner.ims_registered_nonrigid], 
            text=[[f"{ii}",] for ii in range(len(aligner.ims_registered_nonrigid))], 
            font_size=3,
            line_width=10,
            position=(30, 90),
        ), 
        path=str(Path(dir_save).resolve() / 'FOV_images.gif'),
        frameRate=3.0,
        loop=0,
    )
    
    csv_path = str(dir_save / ('ROICaT.tracking.results' + '.csv'))
    roi_table = build_roi_table(plane, results_all, quality_metrics)
    roi_table.to_csv(csv_path)
    
    return results_all