.. #############################################################################
.. README.rst
.. ==========
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. #############################################################################

#####
pyFFS
#####

*pyFFS* is a collection of efficient algorithms to compute Fourier Series and
related transforms.


Installation
------------

::

    $ pip install pyFFS


Developer Install
-----------------

Recommended setup using Anaconda, for optimized numerical libraries:

::

    # Create Anaconda environment
    $ conda create --name pyffs python=3
    $ conda activate pyffs

    # Clone repository
    $ git clone https://github.com/imagingofthings/pyFFS.git
    $ cd pyFFS
    $ # git checkout <commit>

    # Install requirements with conda
    $ conda install --file requirements.txt

    # Optionally install CuPy for GPU support
    $ conda install -c conda-forge cupy

    # Install pyFFS
    $ pip install -e .[dev]
    $ pytest                        # Run test suite
    $ python setup.py build_sphinx  # Generate documentation

More information about CuPy setup can be found `here <https://docs.cupy.dev/en/stable/install.html#installation)>`_.

New release
-----------
From master branch of original repo:

::

    # Create tag and upload
    $ git tag -a vX.X.X -m "Description."
    $ git push origin vX.X.X

    # Create package and upload to Pypi
    $ python setup.py sdist
    $ python -m twine upload  dist/pyFFS-X.X.X.tar.gz

You will need a username and password for uploading to PyPi.

Finally, `on GitHub <https://github.com/imagingofthings/pyFFS/releases>`_ set the new tag as the latest release by
pressing on it, at top right selecting "Edit tag", and at the bottom pressing "Publish release".

Remarks
-------

pyFFS is developed and tested on x86_64 systems running Linux and macOS
Catalina.

Citing this work
----------------

If you use this package in your own research, please cite `our paper <https://arxiv.org/abs/2110.00262>`_.

::

    @article{10.1137/21M1448641,
        author = {Bezzam, Eric and Kashani, Sepand and Hurley, Paul and Vetterli, Martin and Simeoni, Matthieu},
        title = {PyFFS: A Python Library for Fast Fourier Series Computation and Interpolation with GPU Acceleration},
        year = {2022},
        issue_date = {Aug 2022},
        publisher = {Society for Industrial and Applied Mathematics},
        address = {USA},
        volume = {44},
        number = {4},
        issn = {1064-8275},
        url = {https://doi.org/10.1137/21M1448641},
        doi = {10.1137/21M1448641},
        journal = {SIAM J. Sci. Comput.},
        month = {jan},
        pages = {C346–C366},
        numpages = {21},
        keywords = {GPU, chirp Z-transform, numerical library, 97N80, fast Fourier series, bandlimited interpolation, 65T40, 42B05, 97N50, Python}
    }
