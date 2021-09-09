import pymongo

from simple_repo.exception import BadSimpleNodeClass, BadConfigurationKeyType, BadSimpleParameter, \
    MissingMandatoryParameter, UnexpectedParameter, MissingSimpleNodeKey, UnexpectedKey
from simple_repo.simple_sklearn.node_structure import SklearnEstimator

conf = {
    "pipeline_uid": "U-54654649",
    "pandas": [
        {
            "node_id": "loader",
            "node": "simple_repo.simple_pandas.load_nodes.PandasCSVLoader",
            "parameters": {
                "path": "C:/Users/RICCARDO/Desktop/iris_ds.csv"
            },
            "then": [
                {
                    "send_to": "km",
                    "dataset": "fit_dataset"
                },
                {
                    "send_to": "km",
                    "dataset": "predict_dataset"
                }
            ]
        },
        {
            "node_id": "addcol",
            "node": "simple_repo.simple_pandas.transform_nodes.PandasAddColumn",
            "parameters": {
                "columns": [
                    "predictions"
                ]
            },
            "then": [
                {
                    "send_to": "save",
                    "dataset": "dataset"
                }
            ]
        },
        {
            "node_id": "save",
            "node": "simple_repo.simple_pandas.load_nodes.PandasCSVWriter",
            "parameters": {
                "path": "C:/Users/RICCARDO/Desktop/iris_ds_elbafasdf.csv",
                "include_rows": True,
                "rows_column_label": "id"
            }
        }
    ],
    "sklearn": [
        {
            "node_id": "km",
            "node": "simple_repo.simple_sklearn.cluster.SimpleKMeans",
            "parameters": {},
            "execute": [
                "fit",
                "predict"
            ],
            "then": [
                {
                    "send_to": "addcol",
                    "predictions": "dataset"
                }
            ]
        }
    ]
}

client = pymongo.MongoClient("mongodb+srv://admin:admin@cluster0.yhcxc.mongodb.net/simple?retryWrites=true&w=majority")
simple_db = client["simple"]
node_info_collection = simple_db["nodes_info"]
all_nodes_structure = list(node_info_collection.find({}))


def check_parameters(clazz, config_params: dict):
    clazz_params = clazz.get("node_parameter")
    if not isinstance(config_params, dict):
        raise BadConfigurationKeyType(
            "The structure of the 'parameters' is not a 'dict' in node {}".format(clazz.get("node_class")))

    for key, value in config_params.items():
        if key not in clazz_params.keys():
            raise BadSimpleParameter(
                "Parameter '{}' is not available for class '{}'".format(key, clazz.get("node_class")))

        param = clazz_params.get(key)
        if param.get("param_type") == "KeyValueParameter":
            if not isinstance(value, eval(param.get("type"))):
                raise BadConfigurationKeyType(
                    "Wrong type '{}' for parameter '{}' in class {}".format(type(value), key, clazz.get("node_class")))

        elif param.get("param_type") == "StructuredParameterList":
            check_structured_param_list(clazz.get("node_class"), key, param, value)

        elif param.get("param_type") == "SimpleHyperParameter":
            pass


def check_structured_param_list(clazz, key, param, value):
    if not isinstance(value, list):
        raise BadConfigurationKeyType("Wrong type '{}' for parameter '{}' in class {}".format(type(value), key, clazz))

    mandatory_keys, optional_keys = split_params_list(param.get("structure"))
    for parameter in value:
        check_mandatory_keys(clazz, mandatory_keys, parameter)
        check_optional_keys(clazz, mandatory_keys, optional_keys, parameter)


def split_params_list(param_list):
    mandatory_keys = list(filter(lambda x: x.get("is_mandatory"), param_list))
    optional_keys = list(filter(lambda x: not x.get("is_mandatory"), param_list))
    mandatory_keys = {x.get("name"): x.get("type") for x in mandatory_keys}
    optional_keys = {x.get("name"): x.get("type") for x in optional_keys}
    return mandatory_keys, optional_keys


def check_optional_keys(clazz, mandatory_keys, optional_keys, parameter):
    for name, p_type in parameter.items():
        if name in mandatory_keys.keys():
            continue
        if name not in optional_keys.keys():
            raise UnexpectedParameter("Unexpected parameter '{}' in class {}".format(name, clazz))
        if not type(parameter.get(name)).__name__ == optional_keys.get(name):
            raise BadConfigurationKeyType(
                "Wrong type {} for parameter '{}' in class {}".format(type(parameter.get(name)), name, clazz))


def check_mandatory_keys(clazz, mandatory_keys, parameter):
    for name, p_type in mandatory_keys.items():
        if name not in parameter.keys():
            raise MissingMandatoryParameter("Mandatory parameter '{}' is missing in class {}".format(name, clazz))
        if not type(parameter.get(name)).__name__ == p_type:
            raise BadConfigurationKeyType(
                "Wrong type {} for parameter '{}' in class {}".format(type(parameter.get(name)), name, clazz))


def check_then(id_class: dict, id_then: dict):
    for idd, then_list in id_then.items():
        for then in then_list:
            if "send_to" not in then.keys():
                raise MissingSimpleNodeKey("Key 'send_to' is missing in 'then' field for node with id '{}'".format(idd))
            if then["send_to"] not in id_class.keys():
                raise BadSimpleParameter("Node '{}' is not not a valid ID to send the result".format(then["send_to"]))
            if len(then) < 2:
                raise BadSimpleParameter("'then' field in node '{}' should also contains the items "
                                         "'output_sender_var': 'input_receiver_var'".format(idd))
            check_sender_receiver(id_class, idd, then)


def check_sender_receiver(id_class, idd, then):
    for sender_var, receiver_var in then.items():
        if sender_var == "send_to":
            continue
        keys = list(id_class.get(idd).get("node_output"))
        if sender_var not in keys:
            raise BadSimpleParameter("Variable {} is not present in class {}".format(
                sender_var, id_class.get(idd).get("node_class")))
        keys = list(id_class.get(then["send_to"]).get("node_input").keys())
        if receiver_var not in keys:
            raise BadSimpleParameter("Variable {} is not present in class {}".format(receiver_var, id_class.get(
                then["send_to"]).get("node_class")))


def check_node_key_type(node, nodes_id_then):
    if not isinstance(node["node_id"], str):
        raise BadConfigurationKeyType("'node_id' key should be a string in node {}".format(node["node"]))
    if not isinstance(node["node"], str):
        raise BadConfigurationKeyType("'node' key should be a string in node {}".format(node["node_id"]))
    if "then" in node:
        if not isinstance(node["then"], list):
            raise BadConfigurationKeyType("'then' key should be a list in node {}".format(node["node_id"]))
        nodes_id_then[node["node_id"]] = node["then"]


def check_main_key(key, config):
    if key not in ["pandas", "sklearn", "spark", "pipeline_uid"]:
        raise UnexpectedKey("Unexpected key {} in config file".format(key))
    if key != "pipeline_uid" and not isinstance(config[key], list):
        raise BadConfigurationKeyType("Bad '{}' field, should be a 'list'".format(key))


def check_node_keys(key, mand_keys, node):
    for k in mand_keys:
        if k not in node.keys():
            raise MissingSimpleNodeKey("Mandatory Key '{}' is missing in one '{}' node".format(k, key))
    for k in node.keys():
        if k not in mand_keys + ["then"]:
            raise UnexpectedKey("Unexpected key {} in node {}".format(k, node["node_id"]))


def check_node_class(node: str):
    node_info = list(filter(lambda x: x.get("node_package") == node, all_nodes_structure))
    if len(node_info) == 0:
        full_name_parts = node.split(".")
        package_name = ".".join(full_name_parts[:-1])
        class_name = full_name_parts[-1]
        raise BadSimpleNodeClass("Class '{}' doesn't exist in module '{}'".format(class_name, package_name))
    return node_info[0]


def check_execute(node):
    if not isinstance(node["execute"], list):
        raise BadConfigurationKeyType("'execute' key should be a list in node {}".format(node["node_id"]))
    for method in node["execute"]:
        if method not in SklearnEstimator._methods.keys():
            raise UnexpectedParameter("Unexpected method '{}' in node {}".format(method, node["node_id"]))


class ConfigurationParser:
    def __init__(self, config: dict):
        super(ConfigurationParser, self).__init__()
        self.config = config
        self.nodes_id_class = {}
        self.nodes_id_then = {}

    def parse_configuration(self):
        for key in self.config:
            check_main_key(key, self.config)
            if key == "pipeline_uid":
                continue
            mandatory_keys = ['node', 'node_id', 'parameters', 'execute'] if key == "sklearn" else ['node', 'node_id',
                                                                                                    'parameters']
            for node in self.config[key]:
                check_node_keys(key, mandatory_keys, node)
                check_node_key_type(node, self.nodes_id_then)
                cls = check_node_class(node["node"])
                check_parameters(cls, node["parameters"])
                self.nodes_id_class[node["node_id"]] = cls
                if key == "sklearn":
                    check_execute(node)
        check_then(self.nodes_id_class, self.nodes_id_then)


if __name__ == '__main__':
    c = ConfigurationParser(conf)
    c.parse_configuration()
    print("Configuration ok")
