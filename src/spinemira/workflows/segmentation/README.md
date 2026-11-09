# Segmentation Workflow

The Segmentation Workflow segments images in a MIDS dataset using TotalSpineSeg [^1] and MuscleMap [^2]. When both tools run, their labels are merged with TotalSpineSeg taking priority. The workflow saves a `dseg` image and a JSON label sidecar as derivatives.

## Example

The configuration file [example.yaml](./example.yaml) provides an example configuration. Run the workflow from the repository root with:

```bash
uv run python -m spinemira.workflows.segmentation --config ./src/spinemira/workflows/segmentation/example.yaml
```

## Features

- Skips disabled or unavailable tools; TotalSpineSeg requires a data directory when used.
- Optionally segments sagittal images first and uses them to localize related axial images with `--sagittal-first`.
- Reuses existing segmentations unless `--overwrite` is set. Use `--overwrite` when changing tool selection.
- Supports GPU execution with `--device cuda`.


[^1]: Warszawer, Yehuda & Molinier, Nathan & Valosek, Jan & Benveniste, Pierre-Louis & Bédard, Sandrine & Shirbint, Emanuel & Mohamed, Feroze & Tsagkas, Charidimos & Kolind, Shannon & Lynd, Larry & Oh, Jiwon & Prat, Alexandre & Tam, Roger & Traboulsee, Anthony & Patten, Scott & Lee, Lisa Eunyoung & Achiron, Anat & Cohen-Adad, Julien. (2025). TotalSpineSeg: Robust Spine Segmentation with Landmark-Based Labeling in MRI. [10.13140/RG.2.2.31318.56649](https://doi.org/10.13140/RG.2.2.31318.56649).

[^2]: McKay MJ, Weber KA 2nd, Wesselink EO, Smith ZA, Abbott R, Anderson DB, Ashton-James CE, Atyeo J, Beach AJ, Burns J, Clarke S, Collins NJ, Coppieters MW, Cornwall J, Crawford RJ, De Martino E, Dunn AG, Eyles JP, Feng HJ, Fortin M, Franettovich Smith MM, Galloway G, Gandomkar Z, Glastras S, Henderson LA, Hides JA, Hiller CE, Hilmer SN, Hoggarth MA, Kim B, Lal N, LaPorta L, Magnussen JS, Maloney S, March L, Nackley AG, O'Leary SP, Peolsson A, Perraton Z, Pool-Goudzwaard AL, Schnitzler M, Seitz AL, Semciw AI, Sheard PW, Smith AC, Snodgrass SJ, Sullivan J, Tran V, Valentin S, Walton DM, Wishart LR, Elliott JM. MuscleMap: An Open-Source, Community-Supported Consortium for Whole-Body Quantitative MRI of Muscle. J Imaging. 2024;10(11):262. [10.3390/jimaging10110262](https://doi.org/10.3390/jimaging10110262)
