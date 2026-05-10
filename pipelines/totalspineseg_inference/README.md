# TotalSpineSeg Inference

TotalSpineSeg Inference is a pipeline for segmentation of the spine using TotalSpineSeg [^1].

## Example

The configuration file [segment_example_dataset.yaml](./segment_example_dataset.yaml) provides an example configuration file for segmenting all images in the example dataset. The pipeline can be executed with:

```bash
poetry run python ./totalspineseg_inference.py --config ./segment_example_dataset.yaml --image-weight T1w --totalspineseg-data-dir ./../../totalspineseg-data/
```

[^1]: Warszawer, Yehuda & Molinier, Nathan & Valosek, Jan & Benveniste, Pierre-Louis & Bédard, Sandrine & Shirbint, Emanuel & Mohamed, Feroze & Tsagkas, Charidimos & Kolind, Shannon & Lynd, Larry & Oh, Jiwon & Prat, Alexandre & Tam, Roger & Traboulsee, Anthony & Patten, Scott & Lee, Lisa Eunyoung & Achiron, Anat & Cohen-Adad, Julien. (2025). TotalSpineSeg: Robust Spine Segmentation with Landmark-Based Labeling in MRI. [10.13140/RG.2.2.31318.56649](https://doi.org/10.13140/RG.2.2.31318.56649).
