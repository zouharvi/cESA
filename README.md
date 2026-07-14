# Contrastive ESA: Human Evaluation of Multiple Translations at Once

> **Abstract:**
> Current human evaluation of machine translation typically assesses single outputs in isolation, a paradigm that suffers from high annotator noise and cost.
> We introduce Contrastive Error Span Annotation (cESA), a protocol that presents *multiple* document-level translations alongside the source text.
> By allowing annotators to access the shared context across multiple outputs to mark error spans and assign absolute scores, cESA facilitates more consistent and efficient judgments.
> We validate cESA using a large-scale human evaluation of English->Japanese translations of 12 models, demonstrating reductions in annotation time and noise compared to standard pointwise evaluation.
> Unlike existing contrastive ranking methods, cESA yields absolute quality judgments that enable simple, interpretable non-parametric model rankings without the need for post-hoc corrections.

This repository contains the code for the paper [Contrastive ESA: Human Evaluation of Multiple Translations at Once](TODO).

<img alt="Pearmut cESA screenshot" src="https://github.com/user-attachments/assets/0d8efec2-08be-4d0e-830f-7accc295f5a4" />

You can browse an example the collected cESA annotations from the annotator perspective in [English->Japanese here](https://zouharvi.github.io/cESA/?baked=&campaign_id=main_cESA_phase1_v2_enja&bakedItemI=9).

All the data are stored in the [Release tab](https://github.com/zouharvi/cESA/releases/):
- [Campaign sources](https://github.com/zouharvi/cESA/releases/download/data_a0/campaigns.zip)
- [Campaign annotations](https://github.com/zouharvi/cESA/releases/download/data_a0/annotations.zip)
- [Campaign metadata](https://github.com/zouharvi/cESA/releases/download/data_a0/progress.zip)
- [Campaign assets](https://github.com/zouharvi/cESA/releases/download/data_a0/assets_pruned.zip) (pruned based on WMT25)

## Running Contrastive Error Span Annotation (cESA)

> [!NOTE]  
> This repository does not contain the Pearmut annotation interface, which can be found at [github.com/zouharvi/pearmut](https://github.com/zouharvi/pearmut).
> This repository only contains scripts to generate and analyze data for the cESA paper.
> To use cESA, specifiy `"protocol": "cESA"` in Pearmut.

For a quick example of cESA, you can run minimal annotations with:
```bash
cat << EOF > campaign.json
{
"info": {
  "assignment": "single-stream",
  "protocol": "cESA"
},
"campaign_id": "my_campaign",
"data": [[
  {
    "src": "Die sehr hungrige Raupe schlüpfte...",
    "tgt": {
      "modelA": "The famished larva hatched...",
      "modelB": "The hungry caterpillar hatched...",
      "modelC": "The very hungry caterpillar ate..."
    }
  }
]]
}
EOF

# or pip install "pearmut==1.1.6"
pip install pearmut
pearmut add campaign.json
pearmut run
```

## Reproducing Human Evaluation

To reproduce the human evaluation setup, you first need campaign source files.
You can either recreate them from scratch:
```bash
python3 scripts/10-prepare_humeval_main_enja.py
python3 scripts/10-prepare_humeval_main_enit.py
python3 scripts/10-prepare_humeval_preablation.py
```

Or download already prepared ones:
```bash
wget https://github.com/zouharvi/cESA/releases/download/data_a0/campaigns.zip
unzip campaigns.zip -d campaigns
```

Then these can be loaded into pearmut with:
```bash
cd humeval
pearmut add campaigns/main_*.json campaigns/preablation_*.json
pearmut run
```

## Analysis

For analysis, we store the collected annotations in `humeval/annotations/`.
The paper figures and data can be reproduced with:
```bash
python3 scripts/20-analyze_esa_cesa.py
python3 scripts/21-analyze_cesa.py
```

## Citation

If you use this work, please cite it as:
```bibtex
@misc{zouhar2026cesa,
  title = {Contrastive {ESA}: Human Evaluation of Multiple Translations at Once},
  author = {Zouhar, Vilém and Grundkiewicz, Roman and Rajaee, Sara and Riley, Parker and Popel, Martin and Bawden, Rachel and Koehn, Philipp and Carpuat, Marine and Sachan, Mrinmaya and Kocmi, Tom},
  year = {2026},
  url = {https://github.com/zouharvi/cESA},
}
```