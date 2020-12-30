.. ############################################################################
.. index.rst
.. =========
.. Authors :
.. Sepand KASHANI [kashani.sepand@gmail.com]
.. Eric BEZZAM [ebezzam@gmail.com]
.. ############################################################################


###################
pyFFS documentation
###################

*pyFFS* is a collection of efficient algorithms to compute Fourier Series and related transforms.


Installation
------------

::

    $ pip install pyFFS


Developer Install
-----------------

::

    $ git clone https://github.com/imagingofthings/pyFFS.git
    $ cd pyFFS/
    $ # git checkout <commit>

    $ pip install --user -e .[dev]
    $ python3 test.py                # Run test suite
    $ python3 setup.py build_sphinx  # Generate documentation


Remarks
-------

pyFFS is developed and tested on x86_64 systems running Linux and macOS Catalina.


.. toctree::
   :caption: Contents
   :hidden:

   theory/index
   api/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
