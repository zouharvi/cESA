# Contrastive Human Evaluation

Contrastive human evaluation of multilingual tasks, such as translation.

<!-- warning -->


Documentation WIP.

## Running Human Evaluation experiments

> [!NOTE]  
> This repository does not contain the Pearmut annotation interface, which can be found at [github.com/zouharvi/pearmut](https://github.com/zouharvi/pearmut).
> This repository only contains scripts to generate and analyze data for the cESA paper.
> To use cESA, specifiy `"protocol": "cESA"` in Pearmut.

To reproduce the human evaluation setup, run:
```bash
cd scripts
python3 10-prepare_humeval_main.py
python3 10-prepare_humeval_preablation.py
```

Then these can be loaded with:
```bash
cd humeval
pearmut add main_*.json preablation_*.json
pearmut run
```

For analysis, we store the collected annotations in `humeval/collected/`, which can be analyzed with:
```bash
cd scripts
python3 20-analyze_preablation.py
```

## Running LLM-as-a-judge experiments

The following assumes `COHERE_API_KEY` environment variable is set.

TODO
