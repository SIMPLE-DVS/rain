.. highlight:: shell

============
Installation
============


Stable release
--------------

To install Rain library, run this command in your terminal:

.. code-block:: console

    $ pip install git+https://github.com/SIMPLE-DVS/rain.git

This is the preferred method to install Rain, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for SIMPLE Repository can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/SIMPLE-DVS/rain.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/SIMPLE-DVS/rain/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/SIMPLE-DVS/rain
.. _tarball: https://github.com/SIMPLE-DVS/rain/tarball/master

Finally, you can use Rain as a standard Python library, so importing it in your script:

.. code-block:: python

    import rain
