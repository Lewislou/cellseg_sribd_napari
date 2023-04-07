# cellseg_sribd_napari

[![License Apache Software License 2.0](https://img.shields.io/pypi/l/cellseg-sribd.svg?color=green)](https://github.com/githubuser/cellseg-sribd/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cellseg-sribd.svg?color=green)](https://pypi.org/project/cellseg-sribd)

A plugin for cell segmentation
![cellseg-napari_plugin](imgs/cellseg_sribd_napari.gif)
----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template. Most of the UI design is following the codes of [cellpose-napari](https://github.com/MouseLand/cellpose-napari/).

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `cellseg-sribd` via [pip]:
```shell
conda create -y -n napari-env -c conda-forge python=3.9
conda activate napari-env
pip install "napari[all]"
cd cellseg_sribd_napari
pip install -r requirements.txt
pip install -e .
```

## PLugin UI 
## Running the software

```shell
napari -w cellseg-sribd
```

There is sample data in the [imgs], or get started with your own images!



## Source codes and training
The source codes and training process are in [cell-seg](https://github.com/Lewislou/cell-seg/).

## License

Distributed under the terms of the [Apache Software License 2.0] license,
"cellseg-sribd" is free and open source software

## Dependencies

cellpose-napari relies on the following excellent packages (which are automatically installed with conda/pip if missing):
- [napari](https://napari.org)
- [magicgui](https://napari.org/magicgui/)

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

