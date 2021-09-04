import importlib

from simple_repo.exception import BadSimpleNodeClass, BadSimpleModule, BadSimpleParameterType, BadSimpleParameterName, \
    MissingMandatoryParameter, UnexpectedParameter, MissingSimpleNodeKey
from simple_repo.parameter import KeyValueParameter, StructuredParameterList, SimpleHyperParameter

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


def check_parameters(clazz, config_params: dict):
    clazz_params = clazz._parameters
    if not isinstance(config_params, dict):
        raise BadSimpleParameterType("Config Paramters are not dict")

    for key, value in config_params.items():
        if key not in clazz_params.keys():
            raise BadSimpleParameterName("Parameter {} is not available for class {}".format(key, clazz.__name__))

        param = clazz_params.get(key)
        if isinstance(param, KeyValueParameter):
            if not isinstance(value, param.type):
                raise BadSimpleParameterType(
                    "Wrong type {} for parameter {} in class {}".format(type(value), key, clazz.__name__))

        elif isinstance(param, StructuredParameterList):
            if not isinstance(value, list):
                raise BadSimpleParameterType(
                    "Wrong type {} for parameter {} in class {}".format(type(value), key, clazz.__name__))

            param_list = clazz_params.get(key).get_structure()
            mandatory_keys = list(filter(lambda x: x.get("is_mandatory"), param_list))
            optional_keys = list(filter(lambda x: not x.get("is_mandatory"), param_list))
            mandatory_keys = {x.get("name"): x.get("type") for x in mandatory_keys}
            optional_keys = {x.get("name"): x.get("type") for x in optional_keys}

            for parameter in value:
                for name, p_type in mandatory_keys.items():
                    if name not in parameter.keys():
                        raise MissingMandatoryParameter(
                            "Mandatory parameter '{}' is missing in class {}".format(name, clazz.__name__))
                    if not type(parameter.get(name)).__name__ == p_type:
                        raise BadSimpleParameterType(
                            "Wrong type {} for parameter '{}' in class {}".format(type(parameter.get(name)), name,
                                                                                  clazz.__name__))
                for name, p_type in parameter.items():
                    if name in mandatory_keys.keys():
                        continue
                    if name not in optional_keys.keys():
                        raise UnexpectedParameter("Unexpected parameter '{}' in class {}".format(name, clazz.__name__))
                    if not type(parameter.get(name)).__name__ == optional_keys.get(name):
                        raise BadSimpleParameterType(
                            "Wrong type {} for parameter '{}' in class {}".format(type(parameter.get(name)), name,
                                                                                  clazz.__name__))
        elif isinstance(value, SimpleHyperParameter):
            pass


def check_node(node: str):
    full_name_parts = node.split(".")
    package_name = ".".join(full_name_parts[:-1])
    class_name = full_name_parts[-1]
    try:
        module = importlib.import_module(package_name)
        clazz = getattr(module, class_name)
    except AttributeError:
        raise BadSimpleNodeClass("Class {} does not exist in module {}".format(class_name, package_name))
    except ModuleNotFoundError:
        raise BadSimpleModule("Module {} does not exist in simple library".format(package_name))
    return clazz


def check_then(id_class: dict, id_then: dict):
    for idd, then_list in id_then.items():
        for then in then_list:
            if "send_to" not in then.keys():
                raise Exception("Key 'send_to' is missing in 'then' field for node with id {}".format(idd))
            if then["send_to"] not in id_class.keys():
                raise Exception("Node {} is not not a valid ID to send the result".format(then["send_to"]))
            if len(then) < 2:
                raise Exception("Each 'then' field should contains the items 'output_sender_var': 'input_receiver_var'")
            for sender_var, receiver_var in then.items():
                if sender_var == "send_to":
                    continue
                keys = list(id_class.get(idd)._output_vars.keys())
                if sender_var not in keys:
                    raise Exception("Variable {} is not present in class {}".format(
                        sender_var, id_class.get(idd).__name__))
                keys = list(id_class.get(then["send_to"])._input_vars.keys())
                if receiver_var not in keys:
                    raise Exception("Variable {} is not present in class {}".format(receiver_var, id_class.get(
                        then["send_to"]).__name__))


class ConfigurationParser:
    def __init__(self, config: dict):
        super(ConfigurationParser, self).__init__()
        self.config = config

    def parser(self):
        nodes_id_class = {}
        nodes_id_then = {}

        for key in self.config:
            if key not in ["pandas", "sklearn", "spark", "pipeline_uid"]:
                raise Exception("Unexpected key {} in config file".format(key))
            if key == "pipeline_uid":
                continue
            if not isinstance(self.config[key], list):
                raise Exception("Bad '{}' field, should be a 'list'".format(key))

            for node in self.config[key]:
                if not all(k in node.keys() for k in ['node', 'node_id', 'parameters']):
                    raise MissingSimpleNodeKey("Mandatory Key 'node' or 'node_id' or 'parameters' are missing in one "
                                               "{} node".format(key))
                if not isinstance(node["node_id"], str):
                    raise BadSimpleParameterType("'node_id' field should be a string")
                if not isinstance(node["node"], str):
                    raise BadSimpleParameterType("'node' field should be a string in node {}".format(node["node_id"]))
                cls = check_node(node["node"])
                check_parameters(cls, node["parameters"])
                nodes_id_class[node["node_id"]] = cls
                if "then" in node:
                    if not isinstance(node["then"], list):
                        raise BadSimpleParameterType("'then' field should be a list in node {}".format(node["node_id"]))
                    nodes_id_then[node["node_id"]] = node["then"]
                # check_sklearn_execute
        check_then(nodes_id_class, nodes_id_then)


if __name__ == '__main__':
    c = ConfigurationParser(conf)
    c.parser()
    print("ok")
