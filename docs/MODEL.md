# Model and reproducibility

Runtime V49 is paired with student-v37, a 30,238-byte format-4 model:

`34b97b9443e376e983840468f897385ec752d343e068afc8e5690d3bfc7ff309`

The model binary, embedded C header and **editable JSON** carry the same table data. The model contains numeric operations, block recipes, primary entries and exception rules; it contains no image frames. Recent runtime revisions are C++ geometry changes, so model version alone does not select the algorithm.

```sh
python3 native/lut/tools/model_json.py unpack native/lut/models/student-v37.bin /tmp/model.json
python3 native/lut/tools/model_json.py pack /tmp/model.json /tmp/model.bin
cmp native/lut/models/student-v37.bin /tmp/model.bin
python3 native/lut/tools/embed_model.py /tmp/model.bin /tmp/model.h
```

Use fresh output paths. `table_model.py` also exposes lossless table editing through its TableModel class. `lut-fit` and `lut-fit5` are optional offline fitting tools. They read lists of caller-provided binary frame records; the native readers in fit_common.hpp and fit5.cpp define the format (four uint32 fields, RGB source, 3× RGB target, then source rectangles). They are research utilities, not hardened untrusted-file importers. The historical trainer emits an earlier format before subsequent format-4 table edits; it is not a one-command reconstruction of the included model.

No private training images or full historical experiment pipeline are supplied. You can rebuild the runtime, reproduce the public tests and inspect/change every model table; reproducing the original fitted model from training images is outside this release's reproducibility claim. Test your own legally usable inputs before changing a released model.
