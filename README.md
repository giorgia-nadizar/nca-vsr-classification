# Neural Cellular Automata for Voxel-based Soft Robot shape classification.

This repository contains the code for training a Neural Cellular Automata (NCA) for the shape classification of
Voxel-based Soft Robots (VSRs).

## Installation

To install this software you need to:

1. Clone the repository
2. Install the requirements

```
git clone https://github.com/giorgia-nadizar/nca-vsr-classification.git
cd nca-vsr-classification
pip install -r requirements.txt
```

## Classification

To perform a classification you can run the `nca_run.py` file.
It accepts some arguments, which need to be specified in the form `arg_name=arg_value`.

- `n_steps` indicates the amount of steps for the simulation (default is 50).
- `display_transient` indicates whether to print the entire evolution of the NCA or only the final state (default is
  false).
- `target_set` indicates which of the considered set of shapes to use (see them in the 'shapes' folder). Takes values 1
  to 3 (default is 1).
- `target_shape` indicates either the number of the shape in the set it belongs to (0 indexed), or a custom shape. To
  define a custom shape you can enumerate a voxel grid by rows, starting from top left, using the character '-' to
  separate lines. A '1' corresponds to a voxel, a '0' corresponds to an empty space.
- `deterministic` indicates whether each update step should be deterministic (synchronously updating all the nodes) or
  stochastic (update random nodes sequentially). Default value is true.
- `pretty_print` indicates whether the classification should be printed as a VSR (where each voxel displays its
  predicted value) or as a list of $(x,y,c)$, where $x,y$ are the coordinates of the voxels, $(0,0)$ being at the top
  left, and $c$ is the predicted class. Default value is true.

An example run would be:

```
python nca_run.py n_steps=20 display_transient=true target_set=1 deterministic=true pretty_print=true target_shape=1111111-1010101
```
