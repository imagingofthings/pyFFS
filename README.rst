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
pressing on it and at the bottom pressing "Create release".

Remarks
-------

pyFFS is developed and tested on x86_64 systems running Linux and macOS
Catalina.
