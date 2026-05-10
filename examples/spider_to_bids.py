import argparse
import json
from pathlib import Path
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

dataset_description = {
    "Name": "SPIDER",
    "BIDSVersion": "1.10.0",
    "HEDVersion": "8.2.0",
    "DatasetType": "raw",
    "License": "",
    "Authors": ["", "", ""],
    "Acknowledgements": "",
    "HowToAcknowledge": "",
    "Funding": ["", "", ""],
    "EthicsApprovals": [""],
    "ReferencesAndLinks": ["", "", ""],
    "DatasetDOI": "doi:",
}

participants_description = {
    "sex": {
        "Description": "Sex of the participant",
        "Levels": {"M": "Male", "F": "Female"},
    },
}

fields_in_sidecart = {
    "AngioFlag",
    "BodyPartExamined",
    "DeviceSerialNumber",
    "EchoNumbers",
    "EchoTime",
    "EchoTrainLength",
    "FlipAngle",
    "ImagedNucleus",
    "ImagingFrequency",
    "InPlanePhaseEncodingDirection",
    "MRAcquisitionType",
    "MagneticFieldStrength",
    "Manufacturer",
    "ManufacturerModelName",
    "NumberOfPhaseEncodingSteps",
    "PercentPhaseFieldOfView",
    "PercentSampling",
    "PhotometricInterpretation",
    "PixelBandwidth",
    "PixelSpacing",
    "RepetitionTime",
    "SAR",
    "SamplesPerPixel",
    "ScanningSequence",
    "SequenceName",
    "SeriesDescription",
    "SliceThickness",
    "SoftwareVersions",
    "SpacingBetweenSlices",
    "SpecificCharacterSet",
    "TransmitCoilName",
    "WindowCenter",
    "WindowWidth",
}

ivd_id_to_str = {
    0: "disc_L5_S",
    1: "disc_L4_L5",
    2: "disc_L3_L4",
    3: "disc_L2_L3",
    4: "disc_L1_L2",
    5: "disc_T12_L1",
    6: "disc_T11_T12",
    7: "disc_T10_T11",
    8: "disc_T9_T10",
    9: "disc_T8_T9",
}


def make_dataset_description(output_bids_root_dir: Path):
    dataset_description_path = output_bids_root_dir / "dataset_description.json"
    dataset_description_path.write_text(json.dumps(dataset_description, indent=4))


def make_participants_tsv(
    output_bids_root_dir: Path,
    overview_df: pd.DataFrame,
    radiological_grading_df: pd.DataFrame,
):
    # Load from overview file
    overview_df["participant_id"] = overview_df.apply(
        lambda row: int(row["new_file_name"].split("_")[0]), axis=1
    )
    participants_df = overview_df[["participant_id", "sex"]].copy()
    participants_df["sex"] = participants_df["sex"].str.strip()
    participants_df.drop_duplicates(inplace=True)

    # Load from radiological grading
    radiological_grading_df = radiological_grading_df.pivot(
        index="Patient",
        columns="IVD label",
        values=[
            "Modic",
            "UP endplate",
            "LOW endplate",
            "Spondylolisthesis",
            "Disc herniation",
            "Disc narrowing",
            "Disc bulging",
            "Pfirrman grade",
        ],
    )
    radiological_grading_df.columns = [
        f"{metric.replace(' ', '_').lower()}_{ivd_id_to_str[ivd_id]}"
        for metric, ivd_id in radiological_grading_df.columns
    ]
    radiological_grading_df.reset_index(inplace=True)
    radiological_grading_df.rename(columns={"Patient": "participant_id"}, inplace=True)

    float_cols = radiological_grading_df.select_dtypes(include="float").columns
    radiological_grading_df[float_cols] = radiological_grading_df[float_cols].astype(
        "Int64"
    )

    participants_df = participants_df.merge(
        radiological_grading_df, on="participant_id", how="left"
    )
    participants_df.sort_values("participant_id", inplace=True)
    participants_df["participant_id"] = participants_df.apply(
        lambda row: f"sub-{row['participant_id']:0>3}", axis=1
    )

    participants_path = output_bids_root_dir / "participants.tsv"
    participants_df.to_csv(participants_path, sep="\t", index=False, na_rep="")


def make_participants_description(output_bids_root_dir: Path):
    participants_description_path = output_bids_root_dir / "participants.json"
    participants_description_path.write_text(
        json.dumps(participants_description, indent=4)
    )


def convert_spider_to_bids(spider_root_dir: Path, output_bids_root_dir: Path):
    output_bids_root_dir.mkdir(parents=True, exist_ok=True)

    # Load overview file
    overview_file_path = spider_root_dir / "overview.csv"
    overview_df = pd.read_csv(overview_file_path)

    # Load radiological gradings
    radiological_gradings_file_path = spider_root_dir / "radiological_gradings.csv"
    radiological_gradings_df = pd.read_csv(radiological_gradings_file_path)

    make_dataset_description(output_bids_root_dir)
    make_participants_tsv(output_bids_root_dir, overview_df, radiological_gradings_df)
    make_participants_description(output_bids_root_dir)

    # Loop through all images and write images metadata
    for index, row in tqdm(overview_df.iterrows(), desc="Converting images..."):
        file_name = row["new_file_name"]

        subject_id = int(file_name.split("_")[0])
        weight = "T1w" if "t1" in file_name.split("_")[1] else "T2w"
        subtype = "" if len(file_name.split("_")) == 2 else file_name.split("_")[2]

        original_image_path = spider_root_dir / "images" / (file_name + ".mha")

        # Write output image
        acq = "" if not subtype else f"_acq-{subtype}"
        new_image_name = f"sub-{subject_id:0>3}{acq}_{weight}.nii.gz"

        output_image_path = (
            output_bids_root_dir / f"sub-{subject_id:0>3}" / "anat" / new_image_name
        )
        output_image_path.parent.mkdir(parents=True, exist_ok=True)

        image = sitk.ReadImage(original_image_path)
        image = sitk.DICOMOrient(image, "LPS")
        sitk.WriteImage(image, output_image_path)

        # Write output image side cart
        output_image_side_cart_path = output_image_path.parent / (
            output_image_path.name.split(".")[0] + ".json"
        )

        side_cart_data = {
            field: row[field] for field in fields_in_sidecart if pd.notna(row[field])
        }
        output_image_side_cart_path.write_text(json.dumps(side_cart_data, indent=4))

        # Write segmentation
        original_segmentation_path = spider_root_dir / "masks" / (file_name + ".mha")
        new_segmentation_name = f"sub-{subject_id:0>3}{acq}_{weight}_dseg.nii.gz"
        output_segmentation_path = (
            output_bids_root_dir
            / "derivatives"
            / "segmentation"
            / f"sub-{subject_id:0>3}"
            / "anat"
            / new_segmentation_name
        )
        output_segmentation_path.parent.mkdir(parents=True, exist_ok=True)

        segmentation = sitk.ReadImage(original_segmentation_path, sitk.sitkUInt8)
        segmentation = sitk.DICOMOrient(segmentation, "LPS")
        sitk.WriteImage(segmentation, output_segmentation_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="SpiderToBids",
        description="Converts a SPIDER source dataset into a BIDS compliant dataset",
    )

    parser.add_argument("input_folder")
    parser.add_argument("output_folder")

    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    convert_spider_to_bids(input_folder, output_folder)
