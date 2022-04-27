==============
Extending Rain
==============

As we already said, the capability to combine multiple Python libraries and the possibility
to define more nodes or adding support for other libraries are the main Rain's strengths.

Node structure
--------------

Each node is represented by a Python class with a precise structure.

First of all it must be a child of one of these three parent classes:

- **InputNode**, an initial node of the Dataflow that should retrieve data from a source and pass them to
  the next node. It define the dictionary called *_output_vars* and has to be populated by those nodes
  that extend this class with the names and types of the variables used as outputs;
- **OutputNode**, a final node of the Dataflow that should return/store the transformed data.
  It define the dictionary called *_input_vars* and has to be populated by those nodes that extend
  this class with the names and types of the variables used as inputs;
- **ComputationalNode**, an intermediate node of the Dataflow that should transform the inputs received
  by the previous node and present the outputs to the following node. For this reason a node that
  extend this class should define both *_input_vars* and *_output_vars*.

These inputs and outputs will be used and manipulated during the execution of the node.

An example could be the following::

    class PandasSelectRows(PandasNode):
        _input_vars = {"dataset": pandas.DataFrame}
        _output_vars = {"selection": pandas.Series}

This means that the node PandasSelectRows takes as input a Pandas Dataframe, saved in a variable called
*"dataset"* and returns as output a Pandas Series via its *"selection"* variable.
Note that a PandasNode is simply an extensions of a Computational node, meaning that you can create your
hierarchy to better manage and abstract the nodes.

Constructor
-----------

The constructor of a node takes always the string representing its unique identifier as first parameter.
The Id should be then propagated to the parent class.

The other parameters of the *__init__* function are the ones that the node will use during the
execution and they should be saved in a variable called *parameters* through the Parameter class.
This class accept a list of key-value pairs where the key represents the name of the single parameter
and the value is an object child of the class *SimpleParameter*.
The variable *parameters* allow you to leverage several methods in order to access and pass the right
values to the functions used by the node. The documentation and usage of the classes *Parameter*,
*SimpleParameters* and related ones is available at this `link`_.

.. _link: ./rain.core.html#module-rain.core.parameter

For instance the constructor of the PandasSelectRows is::

    def __init__(self, node_id: str, select_nan: bool = False, conditions: List[str] = None):
        super(PandasSelectRows, self).__init__(node_id)
        self.parameters = Parameters(
            select_nan=KeyValueParameter("select_nan", bool, value=select_nan),
            conditions=KeyValueParameter("conditions", List[str], value=conditions),
        )

In this example PandasSelectRows takes two further parameters (*select_nan* and *condition*)
used in the variable *parameters* to create the corresponding *keyValueParameter*. The latter
stores the name, the type and the value taken from the constructor.

For a better understanding each constructor parameters should also define the type
and the default value.

Methods
-------

Each node should have two main methods:

- **_get_tags()**: it's a class method that is used to identify the library that the node use in order to
  achieve the task, and to classify the task itself. It should return an instance of the class *Tag* that
  is composed by two parts:

 - `LibTag`_, that is an enumeration to represent the used library;
 - `TypeTag`_, that is an enumeration to represent the type of task.

  Follows an example of this method::

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.PANDAS, TypeTag.TRANSFORMER)

.. _LibTag: ./rain.core.html#rain.core.base.LibTag
.. _TypeTag: ./rain.core.html#rain.core.base.TypeTag

This method is not mandatory if the parent class already implements it with the two right tags.
Indeed in the above example we can see the method referred to the node PandasSelectRows but it is
implemented in the PandasNode superclass and inherited by all the child classes.

- **execute()**: it is the main method the node must implement. This method contains the core of the
  computation, where the functions of the Python libraries are exploited with the parameters that
  the user set in the constructor (they can be accessed via the methods of the *Parameter* class).

Follows the execute method of the node PandasDropNan that removes the Nan values from a Pandas Dataframe::

    def execute(self):
        self.dataset = self.dataset.dropna(**self.parameters.get_dict())

We can see that the method simply uses the Pandas *dropna*
functions in which all the parameters are passed as keyword arguments through the *get_dict* method
implemented by the class Parameter (*self.parameters* stores all the parameters passed in the *__init__*).

Of course we can have more complex *execute* method which can combine several functions of the same
library (in this case the groups of parameters are useful, see the class Parameter for the usage).

Comments
--------

The nodes, and in general all the classes and methods, must be commented following the `numpydoc`_ style,
in this way the documentation will be automatically generated by Sphinx.

Comments must be written for:

- Class description;
- Parameters (name, type and description);
- Input (name, type and description);
- Output Inputs (name, type and description);

Follows an example of the documentation::

    class MongoCSVWriter(OutputNode):
        """Write a Pandas Dataframe into a MongoDB collection.

        Parameters
        ----------
        node_id : str
            The unique id of the node.
        connection : str
            Hostname or IP address or Unix domain socket path of a single MongoDB instance to connect to, or a mongodb URI
        db : str
            Name of the database to connect to.
        coll : str
            Name of the collection to connect to.

        Input
        -----
        dataset : pd.Dataframe
            The dataset that should be save in the MongoDB collection
        """

        _input_vars = {"dataset": pd.DataFrame}

        def __init__(self, node_id: str, connection: str, db: str, coll: str):
            ...


.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html

Adding a node
-------------

Do you want to help increase the potential of Rain?

- Think of one or more functions of a Python library that you would like to implement;
- Find the package corresponding to that library in *rain.nodes*. Is the library or package
  still not implemented? Go to `Adding a library`_ section;
- Find the appropriate module in the package where you will add your node/class or create a new one;
- Start to implement your node following the structure and guidelines described above. Pay attention!
  Don't forget the parent class, inputs/outputs variables, constructor, parameters, tags, execute and documentation).
- Enjoy and use your new node!

.. _Adding a library: ./extending.html#id1

Follows a full implementation of a node to better understand the structure::

    class MongoCSVWriter(OutputNode):
        """Write a Pandas Dataframe into a MongoDB collection.

        Parameters
        ----------
        node_id : str
            The unique id of the node.
        connection : str
            Hostname or IP address or Unix domain socket path of a single MongoDB instance to connect to, or a mongodb URI
        db : str
            Name of the database to connect to.
        coll : str
            Name of the collection to connect to.

        Input
        -----
        dataset : pd.Dataframe
            The dataset that should be save in the MongoDB collection
        """

        _input_vars = {"dataset": pd.DataFrame}

        def __init__(self, node_id: str, connection: str, db: str, coll: str):
            self.parameters = Parameters(
                connection=KeyValueParameter("connection", str, connection),
                db=KeyValueParameter("db", str, db),
                coll=KeyValueParameter("coll", str, coll),
            )
            super(MongoCSVWriter, self).__init__(node_id)

        def execute(self):
            params = self.parameters.get_dict()
            client = pymongo.MongoClient(
                params.get("connection"), ssl=True, ssl_cert_reqs=ssl.CERT_NONE
            )
            collection = client[params.get("db")][params.get("coll")]
            collection.insert_many(self.dataset.to_dict("records"))
            return collection

        @classmethod
        def _get_tags(cls):
            return Tags(LibTag.MONGODB, TypeTag.OUTPUT)


Adding a library
----------------

Would you like to implement the functionality of a library that is not yet implemented?

- Think of one or more functions of a new Python library that you would like to implement;
- Create a new package in *rain.nodes* with the name of the library;
- Create the appropriate modules in that package where you will add your nodes/classes;
- If necessary, create your hierarchy of nodes to better manage them;
- Update the enumeration *LibTag* and *TypeTag* in module *rain.core.base.py*;
- Start to implement your nodes following the structure and guidelines described above. Pay attention!
  Don't forget the parent class, inputs/outputs variables, constructor, parameters, tags, execute and documentation).
- Enjoy and use your new library and nodes!
