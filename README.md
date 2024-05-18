# BioSoundSegBench

Code for the BioSoundSegBench dataset

## Set-up

### Pre-requisites

You will need:

1. git for version control

(you can install git from [Github](https://help.github.com/en/github/getting-started-with-github/set-up-git),
with your operating system package manager, or using conda.)

### Set up environment and install code

Experiments were run on [Pop!_OS 22.04](https://pop.system76.com/).
It will be easiest to set up in a similar Linux environment (e.g., Ubuntu).

1. Clone this repository with git:

```
git clone git@github.com:vocalpy/BioSoundSegBench.git
cd BioSoundSegBench
```

2. Set up the virtual environment with the code installed into it:

```
virtualenv .venv
. .venv/bin/activate
pip install -e .
```

You will then want to run all code inside the activated virtual environment:
```console
. .venv/bin/activate
```
