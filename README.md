# Homework 1: Softmax Regression for MNIST

This is my Homework 1 project for **Introduction to Deep Learning**. It implements a ten-class linear classifier for handwritten digits using NumPy. The notebook includes implementation checks, three training experiments, learning curves, final test evaluation, and result export.

The completed eight-epoch experiments selected the baseline configuration by final validation accuracy: learning rate **0.10** and L2 coefficient **0.0001**. Its validation accuracy was **90.56%**, and its test accuracy was **91.43%**. The mathematical derivation and detailed result analysis are in [the report](report/HW1_Report.md).

## Repository Layout

```text
GitHub Version/
├── README.md
├── requirements.txt
├── .gitignore
├── codes/
│   ├── HW1_Softmax_MNIST.ipynb
│   └── mnist_data_loader.py
├── data/
│   └── mnist_data/
│       ├── README.md
│       ├── train-images-idx3-ubyte.gz
│       ├── train-labels-idx1-ubyte.gz
│       ├── t10k-images-idx3-ubyte.gz
│       └── t10k-labels-idx1-ubyte.gz
├── images/
│   ├── mnist_samples.png
│   └── learning_curves.png
├── references/
│   ├── README.md
│   ├── HW1_Softmax_MNIST_original.ipynb
│   ├── mnist_data_loader_original.py
│   ├── assignment_instructions.pdf
│   └── Lecture_3.pdf
└── report/
    ├── HW1_Report.md
    ├── HW1_Report.tex
    ├── HW1_Report.pdf
    ├── results.csv
    └── experiment_record.json
```

The four `.gz` files shown above are local inputs. They are excluded from Git. `data/mnist_data/README.md` is a short instruction file that can be committed so that the data directory remains visible in the repository.

| File or folder | Purpose |
| --- | --- |
| `codes/HW1_Softmax_MNIST.ipynb` | Completed notebook, including implementation checks, training, figures, evaluation, and the final export cell. |
| `codes/mnist_data_loader.py` | Supplied script for reading the compressed MNIST IDX files. |
| `data/mnist_data/` | Local dataset files, using their original names. |
| `images/mnist_samples.png` | One combined figure containing eight example images and labels. |
| `images/learning_curves.png` | Training loss and training/validation accuracy curves. |
| `report/HW1_Report.md` | Report with the Honor Code, derivation, results, and discussion. |
| `report/HW1_Report.tex` | LaTeX report source. |
| `report/HW1_Report.pdf` | Compiled report. |
| `report/results.csv` | One summary row for each training configuration. |
| `report/experiment_record.json` | Experiment settings, full per-epoch histories, selected configuration, final test result, and environment versions at export time. |
| `requirements.txt` | Recommended Python dependencies. |
| `.gitignore` | Excludes local data, environments, and temporary files. |

## Reference Materials

`references/` contains original materials for comparison. The completed notebook under `codes/` is the main entry point; do not run the original starter notebook to reproduce the completed experiments.

Recommended contents:

- `HW1_Softmax_MNIST_original.ipynb`: an untouched copy of the supplied starter notebook, including its original TODO blocks.
- `mnist_data_loader_original.py`: an untouched copy of the supplied loader, useful for checking changes.
- `README.md`: describe where these materials came from, their original filenames, and which files are included or kept locally.
- `assignment_instructions.pdf` and `Lecture_3.pdf`: optional local copies for reference. These are descriptive suggested filenames; retain the actual filenames if preferred. The suggested `.gitignore` excludes PDFs under `references/` by default.

Copy the original supplied files into this folder rather than creating “original” files from the completed notebook. Do not put MNIST data, virtual environments, or generated experiment records in `references/`. The directory tree describes the intended organization; create these reference files locally before publishing.

## Requirements and Installation

The reviewed notebook records **Python 3.11.9**. Recommended contents of `requirements.txt` for Python 3.11 are:

```text
numpy==1.26.4
matplotlib==3.8.4
ipykernel==6.29.5
```

These are recommended pinned versions, not a claim about the package versions originally used for the recorded experiment. The final export cell writes the currently installed versions to `experiment_record.json`. If the environment has changed since training, those versions describe the export environment only.

NumPy provides the model calculations, Matplotlib provides plots, and ipykernel supports notebook execution. Standard-library modules such as `os`, `sys`, `gzip`, `struct`, `csv`, and `json` do not need separate installation. The classifier does not use PyTorch, TensorFlow, automatic differentiation, or prebuilt softmax/cross-entropy functions.

Open this project folder in VS Code and install the Microsoft Python and Jupyter extensions. From the project root, install dependencies:

```bash
python -m pip install -r requirements.txt
```

On Windows, `py -3.11 -m pip install -r requirements.txt` explicitly targets Python 3.11. Select the same environment as the notebook kernel.

A virtual environment is optional if the existing environment can already run the notebook. To create a separate environment on Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then select `.venv\Scripts\python.exe` as the VS Code notebook kernel. Do not include the virtual environment in the submission or Git repository.

## Prepare the Data

Place the four course-supplied files in `data/mnist_data/`:

| Filename | Contents |
| --- | --- |
| `train-images-idx3-ubyte.gz` | 60,000 original training images |
| `train-labels-idx1-ubyte.gz` | Their labels |
| `t10k-images-idx3-ubyte.gz` | 10,000 test images |
| `t10k-labels-idx1-ubyte.gz` | Their labels |

Keep the files compressed and retain their exact names. The loader reads gzip files directly. The notebook uses local data and does not download MNIST automatically. When the dataset is omitted from the shared package, the reader must provide these four files before running the notebook.

## Two Supported Data Layouts

| Layout | Notebook and loader | Dataset folder | Output folders |
| --- | --- | --- | --- |
| Project layout | Both under `codes/` | Root `data/mnist_data/` | Root `images/` and `report/` |
| Original assignment layout | Both directly in one folder | `mnist_data/` beside them | `images/` and `report/` in that folder |

The path cell starts from `os.getcwd()`, the Python kernel's current working directory. If its last folder name is `codes`, the cell uses its parent as the candidate project root. It checks the project layout first and then the original layout. It does not search all higher-level directories, avoiding accidental use of files belonging to another version under the shared `Homework 1/` folder.

For the project layout, the kernel working directory must be this project root or its `codes/` folder. For the original layout, it must be the folder containing the notebook, loader, and `mnist_data/`. Inspect the directory with:

```python
import os
print(os.getcwd())
```

If necessary, use `os.chdir(r"YOUR_ACTUAL_PROJECT_ROOT")` before the path cell for local diagnosis. Replace the placeholder with your real directory and remove the machine-specific line before sharing.

The supplied loader is called with an explicit directory:

```python
images_all, labels_all, images_test, labels_test = load_mnist(DATA_DIR)
```

The loader file itself remains unchanged.

## Import Order

Import the loader after the path cell has added `CODE_DIR` to `sys.path`:

```python
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from mnist_data_loader import load_mnist
```

Remove any earlier `from mnist_data_loader import load_mnist` from the first import cell. An early import may work when the kernel starts in `codes/` but fail when it starts at the project root. This adjustment changes module lookup only. After checking it, remove the marked editing note above.

## Run the Notebook

1. Open this version's project root in VS Code, rather than the shared parent containing multiple versions.
2. Open `codes/HW1_Softmax_MNIST.ipynb` and select the environment with the installed dependencies using **Select Kernel**.
3. Confirm that all four data files are present and that the working directory matches one of the supported layouts.
4. Run the notebook from top to bottom in a fresh kernel, with the import order described above.
5. Confirm that the printed project/data paths belong to this version and that the split shapes are `(50000, 784)`, `(10000, 784)`, and `(10000, 784)`.
6. Check that the implementation-check cell prints `All checks passed.`.
7. Let the three configurations finish training. Selection uses their final validation accuracies.
8. The plotting cells save the figures, and the final test-evaluation cell reports the selected model's test accuracy.
9. Run the **final export cell**, placed after test evaluation. It saves `report/results.csv` and `report/experiment_record.json` and prints their absolute paths.
10. Save the executed notebook with its outputs, and check that the two record files exist.

If training and evaluation have already completed and their variables are still in memory, run only the added export cell. It reads the existing results and does not repeat training or test evaluation. If the kernel was restarted, saved notebook outputs alone do not restore those variables; run the preceding cells first.

## Saved Experiment Records

Both record files are saved under `REPORT_DIR`, which is the `report/` folder of the detected project root. For this version, the paths are:

```text
GitHub Version/report/results.csv
GitHub Version/report/experiment_record.json
```

`results.csv` contains the configuration name, learning rate, L2 coefficient, actual epoch count, batch size, final training loss, final training accuracy, final validation accuracy, selection flag, and test accuracy. Only the selected configuration has a test-accuracy value; other rows leave that field blank.

`experiment_record.json` contains the same results and the per-epoch training loss, training accuracy, and validation accuracy for every configuration. It also records seed, split sizes, preprocessing, initialization, selection rule, selected model, test accuracy, and Python/package versions at export time. Its timestamp is the export time in UTC, not the training time.

Accuracies are stored as fractions: `0.9143` means `91.43%`. Epoch counts come from the actual histories. The export cell's batch-size field is set to `256`, matching the current training calls; update it if future experiments use a different batch size.

Running the export cell again **overwrites the two same-named files**. It does not save model weights, download data, compile the report, or create another experiment. Keep these records together with the matching executed notebook and figures. They make the reported results easier to inspect but do not replace the notebook or report.

## Experiment Settings and Reproducibility

| Configuration | Learning rate | L2 coefficient |
| --- | ---: | ---: |
| baseline | 0.10 | 0.0001 |
| smaller learning rate | 0.03 | 0.0001 |
| stronger regularization | 0.10 | 0.01 |

Each run uses seed **2026**, zero weights and biases, **eight epochs**, and **batch size 256**. The original training set is split once using the seeded permutation into 50,000 training examples and 10,000 validation examples. Each configuration starts a new model with the same shuffling seed. Selection uses the last epoch's validation accuracy; the first configured run wins a tie.

The test set is evaluated for the selected model and must not be used to tune settings. Small floating-point differences may occur across environments. Use the actual notebook results and their matching records when preparing the report.

## Changes to the Supplied Notebook

- Completed the four required TODO blocks: stable softmax, loss and gradients, prediction, and parameter updates.
- Added `os`-based path handling for the two supported layouts and module lookup through `CODE_DIR`.
- Changed the loader call to `load_mnist(DATA_DIR)` while retaining the supplied loader script.
- Created root `images/` and `report/` output folders.
- Added saving of `mnist_samples.png` and `learning_curves.png` using explicit paths.
- Added a final export cell for `results.csv` and `experiment_record.json`, using results already in memory.
- Kept the supplied checks, seed, split, three configurations, epoch count, and batch size.

## Report and Figure Paths

The layout above stores the report files in `report/` and the figures in the sibling `images/` folder. Image references must resolve from the report source's location and, for LaTeX, its compilation working directory. From `report/`, the corresponding relative paths are `../images/mnist_samples.png` and `../images/learning_curves.png`. A Markdown source kept at the project root instead uses `images/...`.

Keep the image paths that match your existing report source and compilation workflow; the export cell does not change them. Complete the report's name, student ID, AI-use disclosure, and any remaining marked fields. Markdown/LaTeX conversion and PDF compilation remain separate steps from notebook execution.

## Git Ignore Rules

Suggested `.gitignore` contents:

```gitignore
.venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.DS_Store
Thumbs.db
data/mnist_data/*.gz
mnist_data/*.gz
references/*.pdf
*.aux
*.log
*.out
*.synctex.gz
```

Keep the executed notebook, figures, report, `report/results.csv`, and `report/experiment_record.json` in the shared project. Do not ignore the record files. Git does not track empty directories, so include the small data-location README described above. Ignore rules do not remove files already tracked by Git; inspect `git status` before committing.

## Prepare the GitHub Repository

1. Use `GitHub Version/` as the repository root; there is no need to commit the sibling Submission Version folder or their common parent.
2. Copy the completed notebook, loader, dependencies, figures, report, and the matching exported records into the directories above.
3. Add the untouched originals and a provenance README under `references/`.
4. Add `data/mnist_data/README.md` with the four required filenames and the data-preparation instructions above. Keep the actual dataset files local.
5. Check the import order, remove personal absolute paths and unresolved editing notes, and confirm that the selected kernel can run the project.
6. Review `git status`, then commit the project files. Keep the executed notebook and records so readers can inspect the reported experiment before running it.

If you copy outputs from Submission Version without rerunning training, keep the notebook, figures, and records together as one experiment. Running the export cell in GitHub Version writes to its own `report/` directory when `PROJECT_ROOT` and `REPORT_DIR` point to that version. Correct and rerun the path cell first if the kernel still holds paths from another version.

## Sources and AI Assistance

The assignment instructions, starter notebook, loader, and Lecture 3 were provided by the course. ChatGPT helped with explanations, debugging, file and data-path handling, code review, result export, and report preparation. The report's Honor Code provides the detailed disclosure, including LaTeX/PDF assistance.
