[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nickvgils/hMPC/master)
<!-- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lschoe/mpyc/master)
[![Travis CI](https://app.travis-ci.com/lschoe/mpyc.svg)](https://app.travis-ci.com/lschoe/mpyc)
[![codecov](https://codecov.io/gh/lschoe/mpyc/branch/master/graph/badge.svg)](https://codecov.io/gh/lschoe/mpyc)
[![Read the Docs](https://readthedocs.org/projects/mpyc/badge/)](https://mpyc.readthedocs.io)
[![PyPI](https://img.shields.io/pypi/v/mpyc.svg)](https://pypi.org/project/mpyc/) -->

# hMPC Multiparty Computation in Haskell

This hMPC library, written in the functional language Haskell, serves as a counterpart to the original [MPyC](https://github.com/lschoe/mpyc) library, written in the imperative language Python and developed by Berry Schoenmakers.

hMPC supports secure *m*-party computation tolerating a dishonest minority of up to *t* passively corrupt parties,
where *m &ge; 1* and *0 &le; t &lt; m/2*. The underlying cryptographic protocols are based on threshold secret sharing over finite
fields (using Shamir's threshold scheme and optionally pseudorandom secret sharing).

The details of the secure computation protocols are mostly transparent due to the use of sophisticated operator overloading
combined with asynchronous evaluation of the associated protocols.

## Documentation

See `demos` for Haskell programs with lots of example code. See `docs/basics.rst` for a basic secure computation example in Haskell. Click the "launch binder" badge above to view the entire
repository and try out the Jupyter notebooks from the `demos` directory in the cloud, without any install.

The initial reseach is part of a master's graduation project. For further reading, refer to the complementary master's thesis: [Multiparty Computation in Haskell: From MPyC to hMPC](https://research.tue.nl/en/studentTheses/multiparty-computation-in-haskell).


Original Python MPyC documentation:

[Read the Docs](https://mpyc.readthedocs.io/) for `Sphinx`-based documentation, including an overview of the `demos`.

The [MPyC homepage](https://www.win.tue.nl/~berry/mpyc/) has some more info and background.
<!-- [GitHub Pages](https://lschoe.github.io/mpyc/) for `pydoc`-based documentation. -->



## Installation

You can install this package using `cabal`:

```bash
cabal install hMPC
```

, or `cabal install` in the root directory.

<!-- Pure Python, no dependencies. Python 3.9+ (following [NumPy's deprecation policy](https://numpy.org/neps/nep-0029-deprecation_policy.html#support-table)).

Run `pip install .` in the root directory (containing file `setup.py`).\
Or, run `pip install -e .`, if you want to edit the MPyC source files.

Use `pip install numpy` to enable support for secure NumPy arrays in MPyC, along with vectorized implementations.

Use `pip install gmpy2` to run MPyC with the package [gmpy2](https://pypi.org/project/gmpy2/) for considerably better performance.

Use `pip install uvloop` (or `pip install winloop` on Windows) to replace Python's default asyncio event loop in MPyC for generally improved performance. -->

### Some Tips

- Try `run-all.sh` or `run-all.bat` in the `demos` directory to have a quick look at all Haskell demos.
<!-- Demos `bnnmnist.py` and `cnnmnist.py` require [NumPy](https://www.numpy.org/), demo `kmsurvival.py` requires
[pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), and [lifelines](https://pypi.org/project/lifelines/),
and demo `ridgeregression.py` (and therefore demo `multilateration.py`) even require [Scikit-learn](https://scikit-learn.org/).\
Try `np-run-all.sh` or `np-run-all.bat` in the `demos` directory to run all Python demos employing MPyC's secure arrays.
Major speedups are achieved due to the reduced overhead of secure arrays and vectorized processing throughout the
protocols. -->

<!-- - To use the [Jupyter](https://jupyter.org/) notebooks `demos\*.ipynb`, you need to have Jupyter installed,
e.g., using `pip install jupyter`. An interesting feature of Jupyter is the support of top-level `await`.
For example, instead of `mpc.run(mpc.start())` you can simply use `await mpc.start()` anywhere in
a notebook cell, even outside a coroutine.\
For Python, you also get top-level `await` by running `python -m asyncio` to launch a natively async REPL.
By running `python -m mpyc` instead you even get this REPL with the MPyC runtime preloaded! -->

<!-- - Directory `demos\.config` contains configuration info used to run MPyC with multiple parties.
The file `gen.bat` shows how to generate fresh key material for SSL. To generate SSL key material of your own, first run
`pip install cryptography` (alternatively, run `pip install pyOpenSSL`). -->

Copyright &copy; 2024 Nick van Gils