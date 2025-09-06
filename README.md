# Power Graph Lab

This is an academic project developed for Group Theory laboratories, designed to facilitate the study and visualization of power graphs of finite groups—through interactive computational tools and detailed graphical representations.

---

## Quick Start

Follow these steps to get started with the project.

### 1. Clone the repository

First, clone this repository to your local machine and navigate into the project folder:

```bash
git clone https://github.com/your-username/GroupTheory.git
cd GroupTheory
```
### 2. Create and activate the Conda environment

The project includes a `environment.yml` file with all required dependencies. Run the following commands to create and activate the Conda environment:


```bash
conda env create -f environment.yml
conda activate lab_powergraph
```

## 3. Install the package in editable mode

To be able to edit and test the code directly, install the package in editable mode:

```bash
pip install -e .
```

### 4. Initialize the Jupyter kernel

This step registers a custom kernel so that Jupyter Lab can use the environment:

```bash
python -m power_graph.utils.kernel
```

### 5. Open Jupyter Lab

In this stage, the project can be used by creating a main script. However, it is designed to be visualized and interacted with primarily through Jupyter notebooks.

```bash
jupyter lab
```

Inside Jupyter Lab, select Kernel → Change Kernel → "Power Graph Kernel" to ensure you are using the correct environment.

## Project Structure

```bash
GroupTheory/
├─ src/
│  └─ power_graph/
│     ├─ core/
│     │  ├─ groups/
│     │  └─ graphs/
│     ├─ utils/
│     └─ __init__.py
├─ notebook/
├─ environment.yml
├─ pyproject.toml
└─ README.md
```

## License

Academic License – Contact the author for usage permissions.

