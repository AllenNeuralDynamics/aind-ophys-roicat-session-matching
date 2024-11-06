from aind_data_schema.core.processing import (
    AnalysisProcess,
    DataProcess,
    PipelineProcess,
    Processing,
    ProcessName,
)
from aind_data_schema.core.data_description import (
    AnalysisDescription,
    Modality,
    DataLevel,
)
from aind_data_schema_models.platforms import Platform
import glob, json
import aind_metadata_upgrader
import copy
from aind_metadata_upgrader.data_description_upgrade import DataDescriptionUpgrade


def build_processing(outputs):
    p = Processing(
        processing_pipeline=PipelineProcess(
            processor_full_name="AIND",
            pipeline_url="https://github.com/allenNeuralDynamics/aind-roicat-session-matching",
            pipeline_version="0.1.0",
            data_processes=[build_plane_data_process(o) for o in outputs],
        )
    )
    return p


def build_plane_data_process(outputs):
    return DataProcess(
        name=ProcessName.VIDEO_ROI_CROSS_SESSION_MATCHING,
        software_version="1.4.4",
        input_location="/data/",
        output_location=f"/results/{outputs.get('plane_name')}",
        start_date_time=outputs["t_start"],
        end_date_time=outputs["t_end"],
        code_url="https://github.com/RichieHakim/ROICaT",
        parameters={},
    )


def build_data_description(input_data_descriptions, outputs):
    dd = copy.deepcopy(input_data_descriptions[0])
    dd["modality"] = [Modality.POPHYS]  # hardcoded until bug is fixed
    dd = DataDescriptionUpgrade(old_data_description_dict=dd).upgrade()
    if not dd.project_name:
        dd.project_name = "unknown"
    dd.data_level = DataLevel.DERIVED
    # TODO: AnalysisDescription is setting the name using project, in this case we should name by subject
    return AnalysisDescription(**dd.dict(), analysis_name="session-matching")


def find_data_descriptions(data_dir):
    for dd in glob.glob(f"{data_dir}/*/data_description.json"):
        with open(dd, "r") as f:
            yield json.load(f)
