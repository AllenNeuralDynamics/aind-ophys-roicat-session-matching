# ROICaT Cross-session Matching

Given a session of sessions processed with suite2p, run ROICaT to align the FOVs of those sessions to each other and assign unique ROI identifiers. 

This is meant to work on both single- and multi-plane imaging configurations.

## Input files

```plaintext
📦 data
┣ 📦 session_0
   ┣ 📦 plane_0
   ┣ 📦 ...
   ┗ 📦 plane_N
┣ 📦 ...
┗ 📦 session_N

```
Session and plane directories are expected to be organized like [aind-single-plane-ophys-pipeline](https://github.com/AllenNeuralDynamics/aind-single-plane-ophys-pipeline?tab=readme-ov-file#output) outputs are organized.

## Output files

```plaintext
📦 results
┣ data_description.json  # generated if the input data has data_description.json
┣ processing.json        # summary of software used for this run
┣ 📦 0                  # one directory per plane in the inputs
   ┣ tracking.results_all.richfile   # file containing detailed clustering and tracking results
   ┣ tracking.run_data.richfile      # file containing metadata about parameters used 
   ┣ FOV_clusters.gif                # GIF showing ROIs colored by cluster identity across sessions
   ┣ FOV_images.gif                  # GIF showing registered tracking image across sessions
   ┣ ROICaT.tracking.results.csv     # spreadsheet containing various ROI metrics and identifiers
   ┣ tracking.params_used.json       # JSON representation of run parameters
   ┗ tracking.results_clusters.json  # JSON representation of ROI clusters
┣ 📦 ...
┗ 📦 N
```

### ROICaT.tracking.results.csv

A spreadsheet one row per ROI with the following columns:
- `roi_index`: cross-session global sequential index of the ROI based on the order sessions were loaded
- `fov_name`: name of the ROI's field of view (AKA plane), pulled from directory name containing FOV data
- `session_name`: name of the ROI's session, pulled from the FOV's parent directory name
- `roi_session_index`: within-session sequential index of the ROI 
- `ucid`: "universal cluster ID" as reported by ROICaT. If two ROIs are matched across sessions, they have the same `ucid`.
- `roi_id`: universally unique ID for the ROI cluster. Equivalent to `ucid` except can be combined without ID collision with data from other recordings/mice/etc.
- `silhouette`: ROI cluster silhouette score
- `hdbscan_probability`: ROI cluster probability assigned by hdbscan
- `matched`: boolean indicating whether the ROI matched any other sessions 


