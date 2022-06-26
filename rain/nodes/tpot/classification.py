import pickle

import pandas as pd
from tpot import TPOTClassifier
from rain import Tags, LibTag, TypeTag, ComputationalNode
from rain.core.parameter import Parameters, KeyValueParameter


class TPOTClassificationTrainer(ComputationalNode):
    """Node that returns the classification model trained with the TPOT library.

    Input
    -----
    dataset : pandas.DataFrame
        The dataset for training.

    Output
    ------
    code : str
        The Python code corresponding to the model.
    model : pickle
        The TPOT model in pickle format.

    Parameters
    ----------
    target_feature : str
        Name of the target feature.
    export_script : bool, default=False
        Whether to export the resulting Python script.
    generations : int, default=100
        Number of iterations to the run pipeline optimization process. It must be
        a positive number. If not set, the parameter max_time_mins must be defined
        as the runtime limit. Generally, TPOT will work better when you give it more
        generations (and therefore time) to optimize the pipeline. TPOT will evaluate
        POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.
    population_size : int, default=100
        Number of individuals to retain in the GP population every generation.
        Generally, TPOT will work better when you give it more individuals
        (and therefore time) to optimize the pipeline. TPOT will evaluate
        POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.
    offspring_size : int
        Number of offspring to produce in each GP generation.
        By default, offspring_size = population_size.
    mutation_rate : float, default=0.9
        Mutation rate for the genetic programming algorithm in the range [0.0, 1.0].
        This parameter tells the GP algorithm how many pipelines to apply random
        changes to every generation. We recommend using the default parameter unless
        you understand how the mutation rate affects GP algorithms.
    crossover_rate : float, default=0.1
        Crossover rate for the genetic programming algorithm in the range [0.0, 1.0].
        This parameter tells the genetic programming algorithm how many pipelines to
        "breed" every generation. We recommend using the default parameter unless you
        understand how the mutation rate affects GP algorithms.
    scoring : str
        Function used to evaluate the quality of a given pipeline for the
        problem. By default, accuracy is used for classification problems.
        Offers the same options as sklearn.model_selection.cross_val_score as well as
        a built-in score 'balanced_accuracy'. Classification metrics:
        ['accuracy', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy',
        'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
        'precision', 'precision_macro', 'precision_micro', 'precision_samples',
        'precision_weighted', 'recall', 'recall_macro', 'recall_micro',
        'recall_samples', 'recall_weighted', 'roc_auc']
    cv : int, default=5
        The number of folds to evaluate each pipeline over in k-fold cross-validation
        during the TPOT optimization process.
    subsample : float, default=1.0
        Subsample ratio of the training instance. Setting it to 0.5 means that TPOT
        randomly collects half of training samples for pipeline optimization process.
    n_jobs : int, default=1
        Number of CPUs for evaluating pipelines in parallel during the TPOT
        optimization process. Assigning this to -1 will use as many cores as available
        on the computer. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
        Thus for n_jobs = -2, all CPUs but one are used.
    max_time_mins : int
        How many minutes TPOT has to optimize the pipeline.
        If not None, this setting will allow TPOT to run until max_time_mins minutes
        elapsed and then stop. TPOT will stop earlier if generationsis set and all
        generations are already evaluated.
    max_eval_time_mins : float, default=5
        How many minutes TPOT has to optimize a single pipeline.
        Setting this parameter to higher values will allow TPOT to explore more
        complex pipelines, but will also allow TPOT to run longer.
    random_state : int
        Random number generator seed for TPOT. Use this parameter to make sure
        that TPOT will give you the same results each time you run it against the
        same data set with that seed.
    config_dict : {'TPOT light', 'TPOT MDR', 'TPOT sparse', 'TPOT NN'}, default=None
        String 'TPOT light':
            TPOT uses a light version of operator configuration dictionary instead of
            the default one.
        String 'TPOT MDR':
            TPOT uses a list of TPOT-MDR operator configuration dictionary instead of
            the default one.
        String 'TPOT sparse':
            TPOT uses a configuration dictionary with a one-hot-encoder and the
            operators normally included in TPOT that also support sparse matrices.
        String 'TPOT NN':
            TPOT uses a configuration dictionary for PyTorch neural network classifiers
            included in `tpot.nn`.
    template : str, default=None
        Template of predefined pipeline structure. The option is for specifying a desired structure
        for the machine learning pipeline evaluated in TPOT. So far this option only supports
        linear pipeline structure. Each step in the pipeline should be a main class of operators
        (Selector, Transformer, Classifier or Regressor) or a specific operator
        (e.g. SelectPercentile) defined in TPOT operator configuration. If one step is a main class,
        TPOT will randomly assign all subclass operators (subclasses of SelectorMixin,
        TransformerMixin, ClassifierMixin or RegressorMixin in scikit-learn) to that step.
        Steps in the template are delimited by "-", e.g. "SelectPercentile-Transformer-Classifier".
        By default value of template is None, TPOT generates tree-based pipeline randomly.
    warm_start : bool, default=False
        Flag indicating whether the TPOT instance will reuse the population from
        previous calls to fit().
    memory : str, default=None
        If supplied, pipeline will cache each transformer after calling fit. This feature
        is used to avoid computing the fit transformers within a pipeline if the parameters
        and input data are identical with another fitted pipeline during optimization process.
        String 'auto':
            TPOT uses memory caching with a temporary directory and cleans it up upon shutdown.
        String path of a caching directory
            TPOT uses memory caching with the provided directory and TPOT does NOT clean
            the caching directory up upon shutdown. If the directory does not exist, TPOT will
            create it.
        None:
            TPOT does not use memory caching.
    use_dask : bool, default=False
        Whether to use Dask-ML's pipeline optimizations. This avoid re-fitting
        the same estimator on the same split of data multiple times. It
        will also provide more detailed diagnostics when using Dask's
        distributed scheduler.
    periodic_checkpoint_folder : str, default=None
        If supplied, a folder in which tpot will periodically save pipelines in pareto front so far while optimizing.
        Currently once per generation but not more often than once per 30 seconds.
        Useful in multiple cases:
            Sudden death before tpot could save optimized pipeline
            Track its progress
            Grab pipelines while it's still optimizing
    early_stop : int, default=None
        How many generations TPOT checks whether there is no improvement in optimization process.
        End optimization process if there is no improvement in the set number of generations.
    verbosity : int, default=0
        How much information TPOT communicates while it's running.
        0 = none, 1 = minimal, 2 = high, 3 = all.
        A setting of 2 or higher will add a progress bar during the optimization procedure.
    log_file : str
        Save progress content to a file.
    """

    _input_vars = {"dataset": pd.DataFrame}

    _output_vars = {"code": "str", "model": "pickle"}

    def __init__(self, node_id: str, target_feature: str = None, export_script: bool = False, generations: int = 100,
                 population_size: int = 100, offspring_size: int = None, mutation_rate: float = 0.9,
                 crossover_rate: float = 0.1, scoring: str = 'accuracy', cv: int = 5, subsample: float = 1.0,
                 n_jobs: int = 1, max_time_mins: int = None, max_eval_time_mins: float = 5, random_state: int = None,
                 config_dict: str = None, template: str = None, warm_start: bool = False, memory: str = None,
                 use_dask: bool = False, periodic_checkpoint_folder: str = None, early_stop: int = None,
                 verbosity: int = 0, log_file: str = None):
        super(TPOTClassificationTrainer, self).__init__(node_id)

        self.parameters = Parameters(
            target_feature=KeyValueParameter("target_feature", str, target_feature, True),
            export_script=KeyValueParameter("export_script", bool, export_script),
            generations=KeyValueParameter("generations", int, generations),
            population_size=KeyValueParameter("population_size", int, population_size),
            offspring_size=KeyValueParameter("offspring_size", int, offspring_size),
            mutation_rate=KeyValueParameter("mutation_rate", float, mutation_rate),
            crossover_rate=KeyValueParameter("crossover_rate", float, crossover_rate),
            scoring=KeyValueParameter("scoring", str, scoring),
            cv=KeyValueParameter("cv", int, cv),
            subsample=KeyValueParameter("subsample", float, subsample),
            n_jobs=KeyValueParameter("n_jobs", int, n_jobs),
            max_time_mins=KeyValueParameter("max_time_mins", int, max_time_mins),
            max_eval_time_mins=KeyValueParameter("max_eval_time_mins", float, max_eval_time_mins),
            random_state=KeyValueParameter("random_state", int, random_state),
            config_dict=KeyValueParameter("config_dict", str, config_dict),
            template=KeyValueParameter("template", str, template),
            warm_start=KeyValueParameter("warm_start", bool, warm_start),
            memory=KeyValueParameter("memory", str, memory),
            use_dask=KeyValueParameter("use_dask", bool, use_dask),
            periodic_checkpoint_folder=KeyValueParameter("periodic_checkpoint_folder", str, periodic_checkpoint_folder),
            early_stop=KeyValueParameter("early_stop", int, early_stop),
            verbosity=KeyValueParameter("verbosity", int, verbosity),
            log_file=KeyValueParameter("log_file", str, log_file),
        )

    def execute(self):
        x_train = self.dataset.drop(self.parameters.target_feature.value, axis=1)
        y_train = self.dataset[self.parameters.target_feature.value]
        params = self.parameters.get_dict()
        del params['target_feature']
        del params['export_script']
        params['disable_update_check'] = True
        tpot = TPOTClassifier(**params)
        tpot.fit(x_train, y_train)
        self.code = tpot.export()
        self.model = pickle.dumps(tpot.fitted_pipeline_)

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.TPOT, TypeTag.TRAINER)


class TPOTClassificationPredictor(ComputationalNode):
    """Node that returns the predictions performed with a TPOT Classification model on the columns of a dataset
    without the target feature column.

    Input
    -----
    dataset : pandas.DataFrame
        The pandas DataFrame.

    model : pickle
        The TPOT Classification model in pickle format.

    Output
    ------
    predictions : pandas.DataFrame
        The DataFrame containing the predictions.
    """

    _input_vars = {"dataset": pd.DataFrame, "model": "pickle"}

    _output_vars = {"predictions": pd.DataFrame}

    def __init__(self, node_id: str):
        super(TPOTClassificationPredictor, self).__init__(node_id)
        self.predictions = {}

    def execute(self):
        tpot = pickle.loads(self.model)
        res = tpot.predict(self.dataset)
        self.predictions = pd.DataFrame(res)

    @classmethod
    def _get_tags(cls):
        return Tags(LibTag.TPOT, TypeTag.PREDICTOR)
