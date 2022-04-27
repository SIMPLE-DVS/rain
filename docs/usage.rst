===============
Getting started
===============

Once installed, to use Rain in a project you just have to import it in your Python script::

    import rain

The first thing is then to define a Dataflow variable::

    dataflowName = rain.DataFlow("dataflowId", executor=rain.ExecutorName())

Where:

- *dataFlowName* is the name of the variable associated to the instantiated Dataflow;
- *dataflowId* is the identified of the Dataflow;
- *ExecutorName* is the class of the chosen executor (more information about all the possible
  executors are available `here`_).

.. _here: ./usage.html

The second step is to define the nodes with their appropriate parameters that you want
use to achieve your task::

    node1Name = rain.Node1("node1Id", param1="value1")
    node2Name = rain.Node2("node2Id", param21="value21", param22="value22")
    node3Name = rain.Node3("node3Id", param3="value3)

Where:

- *nodeXName* is the name of the variable associated to the chosen nodeX;
- *NodeX* is the constructor used to instantiate the desired nodeX;
- *nodeXId* is the unique identifier of the nodeX. The id is always the first parameter of a node
  and it is mandatory;
- *paramX* is a generic parameter that you can pass to the constructor of the nodeX in order to
  model its behavior. Of course, depending on the node you can pass multiple parameters;
- *valueX* is the value you set for the corresponding paramX.

The third step is to create the edges that link the nodes and their inputs/outputs. In this way you
should create a Directed Acyclic Graph (cycle are not allowed!)::

    dataflowName.add_edges(
        node1Name @ "node1Output" > node2Name @ "node2Input",
        node2Name @ "node2Output" > node3Name @ ["node3Input1", "node3Input2"]
    )

Where:

- *>*  is the symbol used to create an edge:

 - on the left side you must specify the output of the source node (e.g. node1Name @ "node1Output");
 - on the right side you must specify the input of the destination node (e.g. node2Name @ "node2Input");

- *@* is the symbol used to access an input/output variable of a node:

 - on the left part you must specify the variable name of the node (e.g. node1Name)
 - on the right part you must specify the name of the output/input variable of the source/destination
   node (e.g. "node1Output"). The input of a destination node can also be a list, meaning that the
   output of the source node must be sent to all specified variables (e.g. ["node3Input1", "node3Input2"]).

A complete description of all the available nodes with their behavior, accepted parameters,
inputs and outputs is available at this `link`_.

.. _link: ./rain.nodes.html

The last step is to call execute method, so when you run the script the Dataflow will be executed::

    dataflowName.execute()

where *dataflowName* is the variable associated to the Dataflow that you created in the first step.

Follows an example of a possible Dataflow configuration::

    import rain

    df = rain.DataFlow("df1", executor=rain.LocalExecutor())

    csv_loader = rain.PandasCSVLoader("load", path="./iris.csv")
    filter_col = rain.PandasColumnsFiltering("filter", column_indexes=[0, 1])
    kmeans = rain.SimpleKMeans("km", execute=["fit", "predict"], n_clusters=3)
    writer = rain.PandasCSVWriter("write", path="./iris_predictions.csv")

    df.add_edges(
        csv_loader @ "dataset" > filter_col @ "dataset",
        filter_col @ "dataset" > kmeans @ ["fit_dataset", "predict_dataset"],
        kmeans @ "predictions" > writer @ "dataset"
    )

    df.execute()

In the above script we:

- first import the library;
- instantiate a Dataflow (with Id *df1* and referenced as *df*) passing a Local Executor, meaning
  that the Dataflow will be executed in the local machine that runs the script;
- instantiate 4 nodes (*csv_loader, filter_col, kmeans, writer*):

 - the first one loads a CSV file stored in the root directory and containing the Iris dataset,
   using the node PandasCSVLoader;
 - the second node filters some columns using a PandasColumnFiltering with its parameter
   *column_indexes*;
 - the third node predicts the cluster the flowers belong to, using the K-means algorithm with 3 cluster
   (the *execute* parameter is used to specify which methods of the node SimpleKMeans should be executed);
 - the last one saves the predictions in the root directory in a new file called "iris_predictions.csv";

- create 3 edges to link the 4 nodes:

 - the *dataset* output variable of the *csv_loader* is sent to the *dataset* input
   variable of the node *filter_col*;
 - the output of the *filter_col* (*dataset*) is sent to both inputs *fit_dataset* and
   *predict_dataset* of the node *kmeans*;
 - the output *predictions* of the *kmeans* is then sent to the input of the node *writer* (*dataset*);

- finally execute the Dataflow *df*.
