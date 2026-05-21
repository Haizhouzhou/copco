# EyeBench Command Inventory

## Official Task Names
- `CopCo_TYP`
- `CopCo_RCS`

## Official Data Command
- `bash src/data/preprocessing/get_data.sh CopCo` is supported by `get_data.sh`.

## Files Inspected
- `eyebench/README.md`
- `eyebench/environment.yml`
- `eyebench/pyproject.toml`
- `eyebench/src/data/preprocessing/get_data.sh`
- `eyebench/src/configs/data.py`
- `eyebench/run_commands/CopCo_TYP.md`

## Run/Config Files
- `eyebench/run_commands/CopCo_RCS.md`
- `eyebench/run_commands/CopCo_TYP.md`
- `eyebench/run_commands/IITBHGC_CV.md`
- `eyebench/run_commands/MECOL2_LEX.md`
- `eyebench/run_commands/OneStop_RC.md`
- `eyebench/run_commands/PoTeC_DE.md`
- `eyebench/run_commands/PoTeC_RC.md`
- `eyebench/run_commands/SBSAT_RC.md`
- `eyebench/run_commands/SBSAT_STD.md`
- `eyebench/run_commands/utils/README_DGX.md`
- `eyebench/run_commands/utils/gpu_servers_status.sh`
- `eyebench/run_commands/utils/model_checker.sh`
- `eyebench/run_commands/utils/post_training_results_aggregation.sh`
- `eyebench/run_commands/utils/sweep_wrapper.sh`
- `eyebench/run_commands/utils/sync_data_between_servers.sh`
- `eyebench/run_commands/utils/sync_data_to_dgx.sh`
- `eyebench/run_commands/utils/sync_outputs_between_servers.sh`
- `eyebench/run_commands/utils/sync_outputs_between_servers_dgx.sh`
- `eyebench/run_commands/utils/test_wrapper_creator.sh`
- `eyebench/src/configs/__init__.py`
- `eyebench/src/configs/__pycache__/__init__.cpython-311.opt-beartype0v20v2.pyc`
- `eyebench/src/configs/__pycache__/__init__.cpython-312.opt-beartype0v20v2.pyc`
- `eyebench/src/configs/__pycache__/constants.cpython-311.opt-beartype0v20v2.pyc`
- `eyebench/src/configs/__pycache__/constants.cpython-312.opt-beartype0v20v2.pyc`
- `eyebench/src/configs/__pycache__/data.cpython-311.opt-beartype0v20v2.pyc`
- `eyebench/src/configs/__pycache__/data.cpython-312.opt-beartype0v20v2.pyc`
- `eyebench/src/configs/__pycache__/utils.cpython-311.opt-beartype0v20v2.pyc`
- `eyebench/src/configs/__pycache__/utils.cpython-312.opt-beartype0v20v2.pyc`
- `eyebench/src/configs/constants.py`
- `eyebench/src/configs/data.py`
- `eyebench/src/configs/main_config.py`
- `eyebench/src/configs/models/__init__.py`
- `eyebench/src/configs/models/__pycache__/__init__.cpython-312.opt-beartype0v20v2.pyc`
- `eyebench/src/configs/models/__pycache__/base_model.cpython-312.opt-beartype0v20v2.pyc`
- `eyebench/src/configs/models/base_model.py`
- `eyebench/src/configs/models/dl/Ahn.py`
- `eyebench/src/configs/models/dl/BEyeLSTM.py`
- `eyebench/src/configs/models/dl/MAG.py`
- `eyebench/src/configs/models/dl/PLMAS.py`
- `eyebench/src/configs/models/dl/PLMASF.py`
- `eyebench/src/configs/models/dl/PostFusion.py`
- `eyebench/src/configs/models/dl/RoBERTeye.py`
- `eyebench/src/configs/models/dl/__init__.py`
- `eyebench/src/configs/models/dl/__pycache__/BEyeLSTM.cpython-312.opt-beartype0v20v2.pyc`
- `eyebench/src/configs/models/dl/__pycache__/PLMASF.cpython-312.opt-beartype0v20v2.pyc`
- `eyebench/src/configs/models/dl/__pycache__/__init__.cpython-312.opt-beartype0v20v2.pyc`
- `eyebench/src/configs/models/ml/DummyClassifier.py`
- `eyebench/src/configs/models/ml/LogisticRegression.py`
- `eyebench/src/configs/models/ml/RandomForest.py`
- `eyebench/src/configs/models/ml/SVM.py`
- `eyebench/src/configs/models/ml/XGBoost.py`
- `eyebench/src/configs/models/ml/__init__.py`
- `eyebench/src/configs/trainers.py`
- `eyebench/src/configs/utils.py`
- `eyebench/src/run/multi_run/__init__.py`
- `eyebench/src/run/multi_run/cleanup_models.py`
- `eyebench/src/run/multi_run/csv_to_latex.py`
- `eyebench/src/run/multi_run/raw_to_processed_results.py`
- `eyebench/src/run/multi_run/search_spaces.py`
- `eyebench/src/run/multi_run/sweep_creator.py`
- `eyebench/src/run/multi_run/utils.py`
- `eyebench/src/run/single_run/test_dl.py`
- `eyebench/src/run/single_run/test_ml.py`
- `eyebench/src/run/single_run/train.py`
- `eyebench/src/run/single_run/utils.py`

## Key Excerpts

### README
```text
 🧠 Adding a New Model

1. Create a file under `src/models/YourModel.py` inheriting from `BaseModel`.
   Implement `forward()` and `shared_step()` methods.
2. Register it in:

    - `src/configs/constants.py` → `DLModelNames` (or `MLModelNames` for ML models)
    - `src/configs/models/dl/YourModel.py` → model config class decorated with `@register_model_config` (use `src/configs/models/ml/` for ML models)

3. Define its default parameters and search space in `src/run/multi_run/search_spaces.py`.
4. Verify integration:

```bash
bash run_commands/utils/model_checker.sh
```

---

## 📊 Adding a New Dataset

1. Store raw or preprocessed data in `data/YOUR_DATASET/`.
2. Define its loading logic in `src/data/datasets/YOUR_DATASET.py` (inherits from `ETDataset`).
3. Add preprocessing logic under `src/data/preprocessing/dataset_preprocessing/YOUR_DATASET.py`.
4. Register the dataset in `src/configs/data.py` and `src/configs/constants.py`.
5. Add a corresponding task configuration class if it supports multiple tasks.

Datasets must comply with EyeBench’s selection criteria:

- Passage-level texts
- ≥ 500 Hz sampling rate
- Publicly available raw or fixation-level data
- Released texts and gaze–text alignment

---

## 📘 Documentation

To build the local documentation site:

```bash
pip install mkdocs mkdocs-material 'mkdocstrings[python]' mkdocs-gen-files mkdocs-literate-nav
mkdocs serve
```

---

## 📄 Citation

If you use EyeBench in your research, please cite:

> Omer Shubi, David R. Reich, Keren Gruteke Klein, Yuval Angel, Paul Prasse, Lena Jäger, Yevgeni Berzak.
> **EyeBench: Predictive Modeling from Eye Movements in Reading.**
> *NeurIPS 2025.*

```bibtex
@inproceedings{shubireich2025eyebench,
  title={{EyeBench}: {P}redictive Modeling from Eye Movements in Reading},
  author={Shubi, Omer and Reich, David Robert and Gruteke Klein, Keren and Angel, Yuval and Prasse, Paul and J{\"a}ger, Lena Ann and Berzak, Yevgeni},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2025},
  url={https://openreview.net/forum?id=LhbYJJ3MFd}
}
```

---

## 🤝 Acknowledgments

EyeBench development is supported by:

- **COST Action MultiplEYE (CA21131)**
- **Swiss National Science Foundation (EyeNLG, IZCOZ0 _220330)**
- **Israel Science Foundation (grant 1499/22)**

---

## 🧩 License

All datasets included in EyeBench follow their respective original licenses.
Code released under the [MIT License](LICENSE).


```

### environment
```text
name: eyebench
channels:
  - pytorch
  - huggingface
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.12.10
  - pandas=2.2.3
  - seaborn=0.13.2
  - matplotlib=3.10.1
  - numpy=2.2.4
  - transformers=4.47.1
  - pytorch=2.5.1
  - pytorch-cuda=12.4
  - pyarrow=19.0.1
  - pyyaml=6.0.2
  - datasets=3.5.0
  - lightning=2.5.1
  - scikit-learn=1.6.1
  - pip=25.0.1
  - wordfreq=3.1.1
  - wandb=0.23.1
  - pre-commit=4.2.0
  - ruff=0.11.6
  - hydra-core=1.3.2
  - ipykernel=6.29.5
  - beartype=0.20.2
  - mypy=1.15.0
  - pandas-stubs=2.2.3
  - types-tqdm=4.67.0
  - py-xgboost=3.0.0
  - gpustat=1.1.1
  - ipywidgets=8.1.6
  - accelerate=1.5.2
  - typed-argument-parser=1.10.1
  - loguru=0.7.2
  - peft=0.15.2
  - spacy=3.8.5
  - pytorch-metric-learning=2.9.0
  - rdata=1.0.0
  - pip:
      - -e . # For development purposes
      - pymovements==0.25.0
      - git+https://github.com/lacclab/text-metrics.git
      - en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
      - da_core_news_sm@https://github.com/explosion/spacy-models/releases/download/da_core_news_sm-3.8.0/da_core_news_sm-3.8.0-py3-none-any.whl
      - de_core_news_sm@https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl

```

### pyproject
```text
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "eyebench"
version = "1.0.0"
description = "EyeBench: Predictive Modeling from Eye Movements in Reading"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT License" }

authors = [
  { name = "Omer Shubi", email = "shubi@campus.technion.ac.il" },
  { name = "David R. Reich", email = "davidrobert.reich@uzh.ch" },
  { name = "Keren Gruteke Klein", email = "gkeren@campus.technion.ac.il" },
  { name = "Yuval Angel", email = "yuval.angel@campus.technion.ac.il" },
  { name = "Paul Prasse", email = "paul.prasse@uni-potsdam.de" },
  { name = "Lena Jäger", email = "lenaann.jaeger@uzh.ch" },
  { name = "Yevgeni Berzak", email = "berzak@technion.ac.il" }
]

keywords = [
  "eye-tracking",
  "reading",
  "machine-learning",
  "benchmark",
  "multimodal-ai",
  "cognitive-science",
  "neural-networks"
  ]

classifiers = [
  "Intended Audience :: Science/Research",
   "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
]

[project.urls]
Homepage = "https://github.com/EyeBench/eyebench"
Documentation = "https://eyebench.github.io"
Repository = "https://github.com/EyeBench/eyebench"
"Bug Tracker" = "https://github.com/EyeBench/eyebench/issues"
"Paper (NeurIPS 2025)" = "https://arxiv.org/abs/TODO"
"DOI" = "https://doi.org/TODO"

[tool.hatch.metadata]
allow-direct-references = true

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.ruff]
extend-include = ["*.ipynb"]

[tool.ruff.format]
quote-style = "single"

```

### get_data
```text
set -euxo pipefail
DATASET_LIST="${1:-}"

if [ -n "$DATASET_LIST" ]; then
  python src/data/preprocessing/download_data.py --dataset "$DATASET_LIST"
  python src/data/preprocessing/union_raw_files.py --dataset "$DATASET_LIST"
  python src/data/preprocessing/preprocess_data.py --dataset "$DATASET_LIST"
  python src/data/preprocessing/create_folds.py --dataset "$DATASET_LIST" --do_not_recreate_trial_folds --do_not_recreate_item_subject_folds
  python src/data/preprocessing/stats.py --dataset "$DATASET_LIST"
else
  # No datasets passed — run without dataset argument
  python src/data/preprocessing/download_data.py
  python src/data/preprocessing/union_raw_files.py
  python src/data/preprocessing/preprocess_data.py
  python src/data/preprocessing/create_folds.py --do_not_recreate_trial_folds --do_not_recreate_item_subject_folds
  python src/data/preprocessing/stats.py
fi


```

### data_config
```text
edMode.RC
    target_column: str = 'RC'
    class_names: list[str] = field(default_factory=lambda: ['Incorrect', 'Correct'])
    max_q_len: int = 40
    # max_seq_len: int = 350
    max_tokens_in_word: int = 12


@register_data
@dataclass
class SBSAT(DataArgs):
    """
    SBSAT data.
    """

    text_source: str = 'SAT Reading Passages'
    text_language: str = DatasetLanguage.ENGLISH
    text_domain: str = 'Education'
    text_type: str = 'paragraph'
    stratify: str = 'RC'
    tasks: dict[str, str] = field(
        default_factory=lambda: {
            PredMode.RC: 'RC',
            PredMode.STD: 'difficulty',
        }
    )
    max_scanpath_length: int = 1240
    n_questions_per_item: int = 5
    max_seq_len: int = 740

    def __post_init__(self) -> None:
        super().__post_init__()
        self.raw_ia_dir: Path = Path(self.base_path / 'stimuli')
        self.raw_ia_path: Path = Path(
            self.base_path / 'stimuli/' / 'combined_stimulus.csv'
        )
        self.raw_fixations_path: Path = (
            self.base_path / 'precomputed_events/18sat_fixfinal.csv'
        )


@register_data
@dataclass
class SBSAT_RC(SBSAT):
    """
    SBSAT Text Reading Comprehension
    """

    task: PredMode = PredMode.RC
    target_column: str = 'RC'
    class_names: list[str] = field(default_factory=lambda: ['Incorrect', 'Correct'])
    max_q_len: int = 55
    max_tokens_in_word: int = 12


@register_data
@dataclass
class SBSAT_STD(SBSAT):
    """
    SBSAT Subjective Difficulty
    """

    task: PredMode = PredMode.STD
    target_column: str = 'difficulty'
    class_names: list[str] = field(default_factory=lambda: ['difficulty'])
    max_tokens_in_word: int = 12


def get_data_args(class_name: str) -> DataArgs | None:
    """
    Get the data path arguments class by its name.

    Args:
        class_name (str): The name of the class.

    Returns:
        DataArgs: An instance of the requested class.

    Raises:
        ValueError: If the class name is not found.
    """
    try:
        return globals()[class_name]()
    except KeyError:
        logger.error(f"Class '{class_name}' not found in src/configs/data.py.")
        return None


# Map each data_task to its config
DATA_CONFIGS_MAPPING = {
    'CopCo_TYP': CopCo_TYP,
    'CopCo_RCS': CopCo_RCS,
    'MECOL2_LEX': MECOL2_LEX,
    'SBSAT_STD': SBSAT_STD,
    'SBSAT_RC': SBSAT_RC,
    'PoTeC_DE': PoTeC_DE,
    'PoTeC_RC': PoTeC_RC,
    'IITBHGC_CV': IITBHGC_CV,
    'OneStop_RC': OneStop_RC,
}

```

### CopCo_TYP_commands
```text
_2_3basic.job
sbatch sweeps/CopCo_TYP_20251104/slurm/PLMASArgs/PLMASArgs_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/PLMASArgs/PLMASArgs_CopCo_TYP_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_TYP_20251104/slurm/PLMASfArgs/PLMASfArgs_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/PLMASfArgs/PLMASfArgs_CopCo_TYP_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_TYP_20251104/slurm/RoberteyeWord/RoberteyeWord_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/RoberteyeWord/RoberteyeWord_CopCo_TYP_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_TYP_20251104/slurm/Roberta/Roberta_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/Roberta/Roberta_CopCo_TYP_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_TYP_20251104/slurm/RoberteyeFixation/RoberteyeFixation_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/RoberteyeFixation/RoberteyeFixation_CopCo_TYP_folds_0_1_2_3basic.job
sbatch sweeps/CopCo_TYP_20251104/slurm/PostFusion/PostFusion_CopCo_TYP_folds_0_1_2_3normal.job
sbatch sweeps/CopCo_TYP_20251104/slurm/PostFusion/PostFusion_CopCo_TYP_folds_0_1_2_3basic.job
```

```bash
bash sweeps/CopCo_TYP_20251104/bash/lacc/DummyClassifierMLArgs/DummyClassifierMLArgs_CopCo_TYP_folds_0_1_2_3.sh
bash sweeps/CopCo_TYP_20251104/bash/lacc/SupportVectorMachineMLArgs/SupportVectorMachineMLArgs_CopCo_TYP_folds_0_1_2_3.sh
bash sweeps/CopCo_TYP_20251104/bash/lacc/LogisticRegressionMLArgs/LogisticRegressionMLArgs_CopCo_TYP_folds_0_1_2_3.sh
bash sweeps/CopCo_TYP_20251104/bash/lacc/LogisticMeziereArgs/LogisticMeziereArgs_CopCo_TYP_folds_0_1_2_3.sh
bash sweeps/CopCo_TYP_20251104/bash/lacc/RandomForestMLArgs/RandomForestMLArgs_CopCo_TYP_folds_0_1_2_3.sh
```

### 6. Post-Training Evaluation

Run on laccl-srv1 after training is complete:

```bash
# Sync outputs
tmux new-session -d -s sync_output "bash run_commands/utils/sync_outputs_between_servers.sh" ; tmux set-option remain-on-exit off
tmux new-session -d -s sync_output_dgx "bash run_commands/utils/sync_outputs_between_servers_dgx.sh" ; tmux set-option remain-on-exit off

# Evaluate DL models
tmux new-session -d -s eval_copco_typ "CUDA_VISIBLE_DEVICES=2 bash sweeps/CopCo_TYP_20251104/test_dl_wrapper.sh"

# Evaluate ML models
tmux new-session -d -s eval_copco_typ_ml 'python src/run/single_run/test_ml.py --data_task CopCo_TYP --wandb_project CopCo_TYP_20251104'
```

### 7. Final Step

**⚠️ Important:** Push all generated output to GitHub.

```
