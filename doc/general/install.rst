.. ############################################################################
.. install.rst
.. ===========
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. ############################################################################


Installation
============

::

    $ cd <pyFFS_dir>/
    $ python3 setup.py develop


Documentation / Tests
---------------------

::

    $ conda install sphinx=='2.1.*'            \
                    sphinx_rtd_theme=='0.4.*'
    $ python3 test.py                # Run test suite (optional, recommended)
    $ python3 setup.py build_sphinx  # Generate documentation (optional)


Remarks
-------

* pyFFS is developed and tested on x86_64 systems running Linux.

* It is recommended to install dependencies using `Miniconda <https://conda.io/miniconda.html>`_ or
  `Anaconda <https://www.anaconda.com/download/#linux>`_::

    $ conda install --channel=defaults    \
                    --channel=conda-forge \
                    --file=requirements.txt
