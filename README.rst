.. image:: https://img.shields.io/codecov/c/github/SIMPLE-DVS/rain?flag=rain&style=for-the-badge&token=FVANEYLT21
   :alt: Codecov
   :target: https://app.codecov.io/gh/SIMPLE-DVS/rain

====
Rain
====

.. this is a comment, insert badge here
    .. image:: https://img.shields.io/pypi/v/simple_repo.svg
        :target: https://pypi.python.org/pypi/simple_repo
    .. image:: https://img.shields.io/travis/DazeDC/simple_repo.svg
        :target: https://travis-ci.com/DazeDC/simple_repo

What is it?
-----------

Rain is a Python library that supports the data scientist during the development of data pipelines,
here called Dataflow, in a rapid and easy way following a declarative approach.
In particular helps in data preparation/engineering where data are processed,
and in data analysis, consisting in the definition of the most suitable learning algorithm.

RAIN contains a collection of nodes that abstract functions of the main Python's ML
libraries as Scikit-learn, Pandas and PySpark. The capability to combine multiple Python libraries
and the possibility to define more nodes or adding support for other libraries are the main Rain's
strengths. Currently the library contains several nodes regarding Anomaly Detection strategies.

Dataflow
--------

A DataFlow represents a Directed Acyclic Graph. Since a DataFlow
must be executed in a remote machine, then the acyclicity of the DAG must be
ensured to avoid deadlocks.

Nodes can be added to the DataFlow and connected one to each other by edges.
A node can be seen as a meta-function, a combination of several methods of a particular ML library
embedded in Rain, that provides one or more functionalities (for instance a Pandas node/meta-function
could compute the mean of a column and then round it up to some given decimals).

Edges connect meta-functions outputs to meta-functions inputs using a specific semantic.
In general we can say that an output can be connected to an input if and only if their types match
(semantic verification). Moreover an output can have one or more outgoing edges while an input
can have at most one ingoing edge.

The library contains also the so-called executors to run the Dataflow. Currently there are the
Local executor, where the computation is performed in a single local machine, and the Spark
executor to harness an Apache Spark cluster. A DataFlow is run in a single device because data that
are transformed by nodes are directly passed to the following ones.

Installation
------------

The library can be accessed in a stand-alone way using Python simply by installing it.

To install Rain, run this command in your terminal (preferred way to install the most recent stable release):

.. code-block:: console

    $ pip install git+https://github.com/SIMPLE-DVS/rain.git

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Furthermore the tool comes with a back-end that lever-
ages the library and exposes its functionalities to a GUI which eases the usage of
the library itself.

QuickStart
----------

Here we provide a simple Python script in which Rain is used and a Dataflow is configured::

    import rain

    df = rain.DataFlow("df1", executor=rain.LocalExecutor())

    csv_loader = rain.PandasCSVLoader("load", path="./iris.csv")
    filter_col = rain.PandasColumnsFiltering("filter", column_indexes=[0, 1])
    writer = rain.PandasCSVWriter("write", path="./new_iris.csv")

    df.add_edges(
        csv_loader @ "dataset" > filter_col @ "dataset",
        filter_col @ "transformed_dataset" > writer @ "dataset"
    )

    df.execute()

In the above script we:

- first import the library;
- instantiate a Dataflow (with Id *"df1"* and referenced as *df*) passing a Local Executor, meaning
  that the Dataflow will be executed in the local machine that runs the script;
- instantiate 3 nodes (*csv_loader, filter_col, writer*):

 - the first one loads the *"iris.csv"* file stored in the root directory containing the Iris dataset,
   using the node PandasCSVLoader;
 - the second node filters some columns using a PandasColumnFiltering with its parameter
   *column_indexes*;
 - the last one saves the transformed dataset in a new file called *"new_iris.csv"* using the node
   PandasCSVWriter;

- create 2 edges to link the 3 nodes:

 - the *dataset* output variable of the node *csv_loader* is sent to the *dataset* input
   variable of the node *filter_col*;
 - the output *transformed_dataset* of the *filter_col* is then sent to the input of the
   node *writer* (*dataset*);

- finally call the *execute* method of the Dataflow *df*. In this way, when the script is run
  we get the expected result.

In general to use the library you have to perform the following steps:

- create a Dataflow specifying the type of executor;
- define all the nodes with the desired parameters to achieve your ML task;
- define the edges to link the nodes using the specific semantic:

 - **>**  is the symbol used to create an edge, where on the left you must specify the output of
   the source node while on the right you must specify the input of the destination node;
 - **@** is the symbol used to access an input/output variable of a node, where on the left you
   must specify the variable name of the node while on the right you must specify the name of
   the output/input variable of the source/destination node;

- execute the Dataflow and run the script.

More information about Rain usage, edges' semantic and all the possible executors are available `here`_.
A complete description of all the available nodes with their
behavior, accepted parameters, inputs and outputs is available at this `link`_.

.. _link: ./rain.nodes.html;
.. _here: ./usage.html;


Full Documentation
------------------

To load all the documentation follow the steps:

From the main directory cd to the 'docs' directory.

.. code-block:: console

    $ cd docs

If you are on Windows then run the 'make.bat' file.

Otherwise download sphinx and the sphinx theme specified in the requirements_dev.txt file.
Then run the command:

.. code-block:: console

    $ sphinx-build . ./_build

The _build directory will contain the html files, open the index.html file to read the full documentation.

Authors
-------

* Alessandro Antinori, Rosario Cappruccia, Riccardo Coltrinari, Flavio Corradini, Marco Piangerelli, Barbara Re, Marco Scarpetta

Copyright
-------

* Università degli Studi di Camerino and Sigma S.p.A
