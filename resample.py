from pathlib import Path
import click
import numpy as np
import SimpleITK as sitk


path_input_images = "data/imagesTr"
path_input_labels = "data/labelsTr"
path_output_images = "data/labelsTr_resampled"
path_output_labels = "data/labelsTr_resampled"


@click.command()
@click.argument('input_image_folder',
                type=click.Path(exists=True),
                default=path_input_images)
@click.argument('input_label_folder',
                type=click.Path(exists=True),
                default=path_input_labels)
@click.argument('output_image_folder', default=path_output_images)
@click.argument('output_label_folder', default=path_output_labels)
@click.option('--cores',
              type=click.INT,
              default=12)
@click.option('--resampling',
              type=click.FLOAT,
              nargs=3,
              default=(1, 1, 1))
def main(input_image_folder, input_label_folder, output_image_folder,
         output_label_folder, cores, resampling):

    input_image_folder = Path(input_image_folder).resolve()
    input_label_folder = Path(input_label_folder).resolve()
    output_image_folder = Path(output_image_folder).resolve()
    output_label_folder = Path(output_label_folder).resolve()

    output_image_folder.mkdir(exist_ok=True)
    output_label_folder.mkdir(exist_ok=True)
    print('resampling is {}'.format(str(resampling)))

    patient_list = [
        f.name.split("__")[0] for f in input_image_folder.rglob("*_CT*")
    ]
    if len(patient_list) == 0:
        raise ValueError("No patient found in the input folder")

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(resampling)

    def resample_one_patient(p):
        ct = sitk.ReadImage(
            str([f for f in input_image_folder.rglob(p + "__CT*")][0]))
        pt = sitk.ReadImage(
            str([f for f in input_image_folder.rglob(p + "__PT*")][0]))
        labels = [(sitk.ReadImage(str(f)), f.name)
                  for f in input_label_folder.glob(p + "*")]
        bb = get_bouding_boxes(ct, pt)
        size = np.round((bb[3:] - bb[:3]) / resampling).astype(int)
        resampler.SetOutputOrigin(bb[:3])
        resampler.SetSize([int(k) for k in size])  # sitk is so stupid
        resampler.SetInterpolator(sitk.sitkBSpline)
        ct = resampler.Execute(ct)
        pt = resampler.Execute(pt)
        sitk.WriteImage(ct, str((output_image_folder / (p + "__CT.nii.gz"))))
        sitk.WriteImage(pt, str((output_image_folder / (p + "__PT.nii.gz"))))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        for label, name in labels:
            label = resampler.Execute(label)
            sitk.WriteImage(label, str((output_label_folder / name)))

    for p in patient_list:
        resample_one_patient(p)


def get_bouding_boxes(ct, pt):

    ct_origin = np.array(ct.GetOrigin())
    pt_origin = np.array(pt.GetOrigin())

    ct_position_max = ct_origin + np.array(ct.GetSize()) * np.array(
        ct.GetSpacing())
    pt_position_max = pt_origin + np.array(pt.GetSize()) * np.array(
        pt.GetSpacing())
    return np.concatenate(
        [
            np.maximum(ct_origin, pt_origin),
            np.minimum(ct_position_max, pt_position_max),
        ],
        axis=0,
    )


if __name__ == '__main__':
    main()
