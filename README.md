# speaker-change-detection
A full tour of the entire project is located in **Solution Analysis.ipynb**. Below I list the modules that actually implement this solution.

## Table of Contents
- **contrastive.py** - entrypoint for training a contrastive embedding model
- **energy.py** - entrypoint for training a pairwise energy model on top of an embedding model
- **datasets.py** - definitions of various datasets and dataloaders
- **architectures.py** - definitions of some building blocks for models
- **losses.py** - definitions of some loss functions
- **pipeline.py** - definition of an audio pipeline for training and inference
- **trainer.py** - a training class for training arbitrary models
- **inferencer.py** - a class to inference through the entire framework with unseen data
- **utils.py** - various utilities
- **visualize.py** - a GUI visualization/microphone interaction tool that never amounted to anything :(
- **old_stuff/** - code graveyard that is not representative of anything I would hand in or consider final!
