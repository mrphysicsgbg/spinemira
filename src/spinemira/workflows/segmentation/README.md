# Segment workflows

The workflows segment a MIDS dataset into separate derivatives. All publish a `dseg` image and a JSON label sidecar. Existing outputs are reused unless `--overwrite` is supplied. Use `--device cuda` for GPU inference. All commands accept YAML configuration with `--config`.

TotalSpineSeg [^1]:

```bash
python -m spinemira.workflows.segmentation.totalspineseg --dataset-root /path/to/dataset \
  --image-query '`source` == "raw"' --totalspineseg-data-dir /path/to/models
```

The workflow passes `--no-stalling` to TotalSpineSeg by default. Use `--no-no-stalling` to disable it, or set `no_stalling: false` in a YAML configuration.

Add `--sagittal-first` to segment sagittal scans before related axial scans, using the sagittal segmentations as localizers. `--sagittal-query` and `--axial-query` select orientations. Axial images without exactly one matching sagittal image in the same subject and session are skipped.

MuscleMap [^2]:

```bash
python -m spinemira.workflows.segmentation.musclemap --dataset-root /path/to/dataset \
  --image-query '`source` == "raw"'
```

The default derivative names are `totalspineseg` and `musclemap`; use `--output-derivative-name` to override them. The installed commands are `spinemira-segment-totalspineseg` and `spinemira-segment-musclemap`.

SCT spinal canal segmentation (T2-weighted images):

Uses the model provided by [model-canal-seg](https://github.com/ivadomed/model-canal-seg).

```bash
python -m spinemira.workflows.segmentation.sct_canal --dataset-root /path/to/dataset \
  --image-query '`source` == "raw" and `suffix` == "T2w"'
```

Install Spinal Cord Toolbox and its `sc_canal_t2` model first (`sct_deepseg sc_canal_t2 -install`); the workflow checks the executable but does not install or verify the model. Use `--sct-deepseg-executable` to specify its path. The default derivative is `sct_canal_t2`; the installed command is `spinemira-segment-sct-canal`. The model outputs a binary mask of the spinal canal (dural sac). See the [SCT model documentation](https://spinalcordtoolbox.com/stable/user_section/command-line/deepseg/sc_canal_t2.html).

## References

[^1]: Warszawer, Yehuda & Molinier, Nathan & Valosek, Jan & Benveniste, Pierre-Louis & Bédard, Sandrine & Shirbint, Emanuel & Mohamed, Feroze & Tsagkas, Charidimos & Kolind, Shannon & Lynd, Larry & Oh, Jiwon & Prat, Alexandre & Tam, Roger & Traboulsee, Anthony & Patten, Scott & Lee, Lisa Eunyoung & Achiron, Anat & Cohen-Adad, Julien. (2025). TotalSpineSeg: Robust Spine Segmentation with Landmark-Based Labeling in MRI. [10.13140/RG.2.2.31318.56649](https://doi.org/10.13140/RG.2.2.31318.56649).

[^2]: McKay MJ, Weber KA 2nd, Wesselink EO, Smith ZA, Abbott R, Anderson DB, Ashton-James CE, Atyeo J, Beach AJ, Burns J, Clarke S, Collins NJ, Coppieters MW, Cornwall J, Crawford RJ, De Martino E, Dunn AG, Eyles JP, Feng HJ, Fortin M, Franettovich Smith MM, Galloway G, Gandomkar Z, Glastras S, Henderson LA, Hides JA, Hiller CE, Hilmer SN, Hoggarth MA, Kim B, Lal N, LaPorta L, Magnussen JS, Maloney S, March L, Nackley AG, O'Leary SP, Peolsson A, Perraton Z, Pool-Goudzwaard AL, Schnitzler M, Seitz AL, Semciw AI, Sheard PW, Smith AC, Snodgrass SJ, Sullivan J, Tran V, Valentin S, Walton DM, Wishart LR, Elliott JM. MuscleMap: An Open-Source, Community-Supported Consortium for Whole-Body Quantitative MRI of Muscle. J Imaging. 2024;10(11):262. [10.3390/jimaging10110262](https://doi.org/10.3390/jimaging10110262)
