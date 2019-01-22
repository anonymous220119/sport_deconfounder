# Sport deconfounder: Python code
Python version.
Copyright (c) 2019 Anonymous author

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Dependencies:
Needs the following main Python modules to be installed:

* `numpy` 
* `sklearn` 
* `pandas` 
* `argparse`
* `pickle`
* `path`
* `sktensor`

## What's included:
- `causal_noncausal_validation.py` : Example application of the algorithm (`data` directory is not included)
- `factor_model_validation.py` : Example application of the factor model validation (`data` directory is not included)
- `cv_fold.py` : Class to perform cross-validation for factor model validation
- `MultiTensor.py` : Contains the class definition of a Poisson tensor factorization with uniform priors (MLE)
- `bptf.py` : Contains the class definition of a Bayesian Poisson tensor factorization with Gamma priors (MAP)
- `utils.py` : Contains utility functions for bptf
- `FactorLineups.py` : Class containing factor model
- `OutcomeModel.py` : Class containing the outcome model
- `SportDeconfounder.py` : Class containing the whole model's implementation (uses both `FactorLineups.py` and `OutcomeModel.py`)
- `run_sport_deconfounder.py`,`run_factor_model.sh` : bash scripts to run the code for several parameters in series

## Requirements:
Need to make a local directory containing your data outside the folder where the `python` folder is stored. E.g.  type from the command line, inside that folder: 
* `mkdir data`

## How to compile the code:
The Python code does not need to be compiled.

Example implementation: see the bash scripts `run_sport_deconfounder.py`,`run_factor_model.sh`

