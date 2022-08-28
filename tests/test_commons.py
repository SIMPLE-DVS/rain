import pytest

from rain import (
    PandasColumnsFiltering,
    PandasPivot,
    PandasRenameColumn,
    SklearnLinearSVC,
    SimpleKMeans,
    SparkCSVLoader,
    SparkModelLoader,
    SparkSaveModel,
    SparkSaveDataset,
    SparkColumnSelector,
    SparkSplitDataset,
    Tokenizer,
    HashingTF,
    LogisticRegression,
    SparkPipelineNode,
    MongoCSVReader,
    MongoCSVWriter,
    PandasAddColumn,
    PandasReplaceColumn,
    PandasFilterRows,
    PandasSelectRows,
    PandasDropNan,
    DaviesBouldinScore,
    SklearnPCA,
    IrisDatasetLoader,
    PandasGroupBy,
    ZScoreTrainer,
    ZScorePredictor,
    TPOTClassificationTrainer,
    TPOTClassificationPredictor,
    TPOTRegressionTrainer,
    TPOTRegressionPredictor,
    PickleModelLoader,
    PickleModelWriter,
)
from rain.core.base import TypeTag, LibTag, Tags, SimpleNode, InputNode, OutputNode
from rain.nodes.custom import CustomNode
from rain.nodes.pandas.pandas_io import (
    PandasInputNode,
    PandasOutputNode,
    PandasCSVLoader,
    PandasCSVWriter,
)
from rain.nodes.pandas.node_structure import PandasTransformer
from rain.nodes.sklearn.functions import (
    TrainTestDatasetSplit,
    TrainTestSampleTargetSplit,
)
from rain.nodes.sklearn.node_structure import (
    SklearnClassifier,
    SklearnEstimator,
    SklearnClusterer,
    SklearnNode,
    SklearnFunction,
)

# TODO per come è impostato ora questo test non servono più i singoli test sui metodi nei moduli test degli engine. Detronizzarli!

# for each class write the expected list of inputs, output and methods (if present) respectively
from rain.nodes.spark.node_structure import (
    SparkInputNode,
    SparkOutputNode,
    SparkNode,
    Transformer,
    Estimator,
)

classes = [
    (SimpleNode, None, None, None, None),
    # Sklearn Nodes
    (SklearnNode, [], [], [], None),
    (SklearnFunction, [], [], [], Tags(LibTag.SKLEARN, TypeTag.TRANSFORMER)),
    (
        TrainTestDatasetSplit,
        ["dataset"],
        ["train_dataset", "test_dataset"],
        [],
        Tags(LibTag.SKLEARN, TypeTag.TRANSFORMER),
    ),
    (
        TrainTestSampleTargetSplit,
        ["sample_dataset", "target_dataset"],
        [
            "sample_train_dataset",
            "sample_test_dataset",
            "target_train_dataset",
            "target_test_dataset",
        ],
        [],
        Tags(LibTag.SKLEARN, TypeTag.TRANSFORMER),
    ),
    (
        SklearnEstimator,
        ["fit_dataset"],
        ["fitted_model"],
        ["fit"],
        Tags(LibTag.SKLEARN, TypeTag.ESTIMATOR),
    ),
    (
        DaviesBouldinScore,
        ["samples_dataset", "labels"],
        ["score"],
        [],
        Tags(LibTag.SKLEARN, TypeTag.METRICS),
    ),
    (
        SklearnPCA,
        ["fit_dataset", "transform_dataset", "score_dataset"],
        ["fitted_model", "transformed_dataset", "score_value"],
        ["fit", "transform", "score"],
        Tags(LibTag.SKLEARN, TypeTag.ESTIMATOR),
    ),
    (
        SklearnClassifier,
        [
            "fit_dataset",
            "fit_targets",
            "predict_dataset",
            "score_dataset",
            "score_targets",
        ],
        ["fitted_model", "predictions", "score_value"],
        ["fit", "predict", "score"],
        Tags(LibTag.SKLEARN, TypeTag.CLASSIFIER),
    ),
    (
        SklearnClusterer,
        ["fit_dataset", "predict_dataset", "score_dataset", "transform_dataset"],
        ["fitted_model", "predictions", "score_value", "transformed_dataset"],
        ["fit", "predict", "score", "transform"],
        Tags(LibTag.SKLEARN, TypeTag.CLUSTERER),
    ),
    (
        SimpleKMeans,
        ["fit_dataset", "predict_dataset", "score_dataset", "transform_dataset"],
        ["fitted_model", "predictions", "score_value", "transformed_dataset", "labels"],
        ["fit", "predict", "score", "transform"],
        Tags(LibTag.SKLEARN, TypeTag.CLUSTERER),
    ),
    (
        SklearnLinearSVC,
        [
            "fit_dataset",
            "fit_targets",
            "predict_dataset",
            "score_dataset",
            "score_targets",
        ],
        ["fitted_model", "predictions", "score_value"],
        ["fit", "predict", "score"],
        Tags(LibTag.SKLEARN, TypeTag.CLASSIFIER),
    ),
    (
        IrisDatasetLoader,
        None,
        [
            "dataset",
            "target",
        ],
        None,
        Tags(LibTag.SKLEARN, TypeTag.INPUT),
    ),  # Pandas Nodes
    (PandasInputNode, None, ["dataset"], None, Tags(LibTag.PANDAS, TypeTag.INPUT)),
    (PandasCSVLoader, None, ["dataset"], None, Tags(LibTag.PANDAS, TypeTag.INPUT)),
    (PandasOutputNode, ["dataset"], None, None, Tags(LibTag.PANDAS, TypeTag.OUTPUT)),
    (PandasCSVWriter, ["dataset"], None, None, Tags(LibTag.PANDAS, TypeTag.OUTPUT)),
    (
        PandasTransformer,
        ["dataset"],
        ["dataset"],
        None,
        Tags(LibTag.PANDAS, TypeTag.TRANSFORMER),
    ),
    (
        PandasColumnsFiltering,
        ["dataset"],
        ["dataset"],
        None,
        Tags(LibTag.PANDAS, TypeTag.TRANSFORMER),
    ),
    (
        PandasPivot,
        ["dataset"],
        ["dataset"],
        None,
        Tags(LibTag.PANDAS, TypeTag.TRANSFORMER),
    ),
    (
        PandasRenameColumn,
        ["dataset"],
        ["dataset"],
        None,
        Tags(LibTag.PANDAS, TypeTag.TRANSFORMER),
    ),
    (
        PandasAddColumn,
        ["dataset", "column"],
        ["dataset"],
        None,
        Tags(LibTag.PANDAS, TypeTag.TRANSFORMER),
    ),
    (
        PandasReplaceColumn,
        ["column"],
        ["column"],
        None,
        Tags(LibTag.PANDAS, TypeTag.TRANSFORMER),
    ),
    (
        PandasFilterRows,
        ["dataset", "selected_rows"],
        ["dataset"],
        None,
        Tags(LibTag.PANDAS, TypeTag.TRANSFORMER),
    ),
    (
        PandasSelectRows,
        ["dataset"],
        ["selection", "dataset"],
        None,
        Tags(LibTag.PANDAS, TypeTag.TRANSFORMER),
    ),
    (
        PandasDropNan,
        ["dataset"],
        ["dataset"],
        None,
        Tags(LibTag.PANDAS, TypeTag.TRANSFORMER),
    ),
    (
        PandasGroupBy,
        ["dataset"],
        ["dataset"],
        None,
        Tags(LibTag.PANDAS, TypeTag.TRANSFORMER),
    ),  # Spark Nodes
    (SparkInputNode, None, [], None, Tags(LibTag.SPARK, TypeTag.INPUT)),
    (SparkCSVLoader, None, ["dataset"], None, Tags(LibTag.SPARK, TypeTag.INPUT)),
    (SparkModelLoader, None, ["model"], None, Tags(LibTag.SPARK, TypeTag.INPUT)),
    (SparkOutputNode, [], None, None, Tags(LibTag.SPARK, TypeTag.OUTPUT)),
    (SparkSaveModel, ["model"], None, None, Tags(LibTag.SPARK, TypeTag.OUTPUT)),
    (SparkSaveDataset, ["dataset"], None, None, Tags(LibTag.SPARK, TypeTag.OUTPUT)),
    (SparkNode, ["dataset"], [], None, None),
    (
        Transformer,
        ["dataset"],
        ["dataset"],
        None,
        Tags(LibTag.SPARK, TypeTag.TRANSFORMER),
    ),
    (
        SparkColumnSelector,
        ["dataset"],
        ["dataset"],
        None,
        Tags(LibTag.SPARK, TypeTag.TRANSFORMER),
    ),
    (
        SparkSplitDataset,
        ["dataset"],
        ["dataset", "train_dataset", "test_dataset"],
        None,
        Tags(LibTag.SPARK, TypeTag.TRANSFORMER),
    ),
    (
        Tokenizer,
        ["dataset"],
        ["dataset"],
        None,
        Tags(LibTag.SPARK, TypeTag.TRANSFORMER),
    ),
    (
        HashingTF,
        ["dataset"],
        ["dataset"],
        None,
        Tags(LibTag.SPARK, TypeTag.TRANSFORMER),
    ),
    (Estimator, ["dataset"], ["model"], None, Tags(LibTag.SPARK, TypeTag.ESTIMATOR)),
    (
        LogisticRegression,
        ["dataset"],
        ["model"],
        None,
        Tags(LibTag.SPARK, TypeTag.ESTIMATOR),
    ),
    (
        SparkPipelineNode,
        ["dataset"],
        ["model"],
        None,
        Tags(LibTag.SPARK, TypeTag.ESTIMATOR),
    ),  # IO
    (MongoCSVReader, None, ["dataset"], None, Tags(LibTag.MONGODB, TypeTag.INPUT)),
    (MongoCSVWriter, ["dataset"], None, None, Tags(LibTag.MONGODB, TypeTag.OUTPUT)),
    (ZScoreTrainer, ["dataset"], ["model"], None, Tags(LibTag.PANDAS, TypeTag.TRAINER)),
    (ZScorePredictor, ["dataset", "model"], ["predictions"], None, Tags(LibTag.PANDAS, TypeTag.PREDICTOR)),
    (TPOTClassificationTrainer, ["dataset"], ["code", "model"], None, Tags(LibTag.TPOT, TypeTag.TRAINER)),
    (TPOTClassificationPredictor, ["dataset", "model"], ["predictions"], None, Tags(LibTag.TPOT, TypeTag.PREDICTOR)),
    (TPOTRegressionTrainer, ["dataset"], ["code", "model"], None, Tags(LibTag.TPOT, TypeTag.TRAINER)),
    (TPOTRegressionPredictor, ["dataset", "model"], ["predictions"], None, Tags(LibTag.TPOT, TypeTag.PREDICTOR)),
    (PickleModelLoader, None, ["model"], None, Tags(LibTag.PANDAS, TypeTag.INPUT)),
    (PickleModelWriter, ["model"], None, None, Tags(LibTag.PANDAS, TypeTag.OUTPUT)),
    (CustomNode, [], [], None, Tags(LibTag.BASE, TypeTag.CUSTOM)),
    (InputNode, None, [], None, Tags(LibTag.OTHER, TypeTag.INPUT)),
    (OutputNode, [], None, None, Tags(LibTag.OTHER, TypeTag.OUTPUT)),
]


@pytest.mark.parametrize("class_, in_vars, out_vars, methods_vars, tags", classes)
def test_class_integrity(class_, in_vars, out_vars, methods_vars, tags):
    input_string = "_input_vars"
    output_string = "_output_vars"
    methods_string = "_methods"
    tags_methods_string = "_get_tags"

    if in_vars is None:
        assert not hasattr(class_, input_string)
    else:
        assert set(in_vars) == set(class_._input_vars.keys()) and all(
            hasattr(class_, param_name) for param_name in in_vars
        )

    if out_vars is None:
        assert not hasattr(class_, output_string)
    else:
        assert set(out_vars) == set(class_._output_vars.keys()) and all(
            hasattr(class_, param_name) for param_name in out_vars
        )

    if methods_vars is None:
        assert not hasattr(class_, methods_string)
    else:
        assert set(methods_vars) == set(class_._methods.keys())

    if tags is not None:
        assert hasattr(class_, tags_methods_string)
        assert callable(getattr(class_, tags_methods_string))
        assert class_._get_tags().__eq__(tags)


def check_param_not_found(class_, **kwargs):
    """Checks whether the class raises a TypeError exception when instantiating,
    meaning that an invalid argument has been passed."""
    with pytest.raises(TypeError):
        class_("s1", **kwargs)
