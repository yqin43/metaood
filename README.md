# MetaOOD

### File Descriptions

- `stats_mf.py`: Script to generate statistical meta features.
- `earth_mover.py`: Script to compute the Earth Mover's Distance feature.
- `landmarker.py`: Script to load the data and generate landmarker features.
- `metaood.py`: Script to run MetaOOD.

## Meta Features Generation

The meta features generation code is organized in the `./meta_feature` folder. Here is a breakdown of the different scripts and the features they generate:

- **Statistical Meta Features**: Use `stats_mf.py` to generate various statistical meta features.
- **Earth Mover's Distance Feature**: Use `earth_mover.py` to generate the Earth Mover's Distance feature.
- **Landmarker Features**: Use `run_landmarker.py` to generate landmarker features.

To generate the respective features, run the corresponding script from the `./meta_feature` directory. For example:

```sh
import meta_feature.stats_mf
meta_feature.stats_mf.extract_meta_features(loader) # data loader is passed in

import meta_feature.earth_mover
meta_feature.earth_mover.run(loader1, loader2) # data loader od ID data and OOD data are passed in
```

`run_landmarker.py` includs data loading, generation of landmarker meta features, and saving of landmarker features
```sh
python run_landmarker.py
```

## MetaOOD Model Selection
To run the MetaOOD model, do:

```sh
import metaood
metaood.run(train_idx, test_idx)
```
 Input are the column indices of the meta-train samples and meta-test samples. Output is the corresponding performance (AUROC scores) of the meta-test samples. 
