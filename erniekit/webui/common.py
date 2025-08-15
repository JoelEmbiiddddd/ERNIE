# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Configuration, general method handling
"""

import ast
import base64
import copy
import json
import os
import re
import signal
import time
from datetime import datetime
from pathlib import Path

import GPUtil
import yaml
from psutil import NoSuchProcess, Process

WEBUI_PATH = os.path.dirname(os.path.abspath(__file__))
ERNIEKIT_PATH = os.path.dirname(WEBUI_PATH)
ROOT_PATH = os.path.dirname(ERNIEKIT_PATH)
CONFIG_PATH = os.path.join(WEBUI_PATH, "config")
DEFAULT_DATASET_PATH = os.path.join(CONFIG_PATH, "dataset.json")
EXECUTE_PATH = os.path.join(CONFIG_PATH, "execute")


class ConfigManager:
    def __init__(self):
        self.user_dict = {}
        self.paddle_png_path = os.path.join(CONFIG_PATH, "paddle.png")
        self._default_path_config = {
            "default_yaml": os.path.join(CONFIG_PATH, "default.yaml"),
            "dataset_info_json": os.path.join(CONFIG_PATH, "dataset_info.json"),
            "save_checkpoint_dir": os.path.join(ERNIEKIT_PATH, "save"),
            "output_dir": "./output",
            "logging_dir": "vdl_log",
            "paddle_log_dir": "paddle_dist_log",
        }

        self._commands_cli = {
            "train": "erniekit train",
            "export": "erniekit export",
            "eval": "erniekit eval",
            "server": "erniekit server",
            "split": "erniekit split",
            "chat": "erniekit chat",
            "version": "erniekit version",
            "help": "erniekit help",
        }

        self._user_default_config = self._init_user_dict()
        self._execute_yaml_path = self._init_execute_yaml_path()
        self._dataset_info = self._init_dataset_info()
        self._thought_models = ["ERNIE-X1-300B-A47B"]
        self._choices_kwargs = {
            "model_name": [
                "Customization",
            ],
            "model_source_ernie": ["Local"],
            "model_source_custom": ["Local"],
            "fine_tuning": ["LoRA", "Full"],
            "existed_dataset_list": list(self._dataset_info.keys()),
            "stage": ["SFT", "DPO"],
            "compute_type_Full": ["bf16", "fp16", "fp8"],
            "compute_type_LoRA": ["bf16", "fp16", "fp8", "wint8", "wint4/8"],
            "best_config": ["SFT", "DPO"],
            "stages": ["SFT", "DPO"],
            "language": ["zh", "en"],
            "boolean_choice": ["True", "False"],
            "moe_group": ["dummy", "mp"],
            "dataset_type": ["erniekit", "alpaca"],
            "strategy": ["epoch", "steps"],
        }

        self._init_dataset_elem = [
            {
                "type": "dropdown",
                "options": list(self._dataset_info.keys()) + ["Customization"],
                "scale": 3,
            },
            {"type": "dropdown", "options": ["alpaca", "file", "erniekit"], "scale": 2},
            {"type": "textbox", "scale": 7},
            {"type": "textbox", "scale": 2},
        ]

        self._default_dataset_name = "demo_sft_train"

    def get_init_dataset_elem(self):
        return self._init_dataset_elem

    def get_default_dataset_name(self):
        return self._default_dataset_name

    def get_dataset_info(self):
        return self._dataset_info

    def get_compute_type_by_fine_tuning(self, fine_tuning):
        """
        Determine compute type based on fine-tuning configuration

        Args:
            self (object): Instance of the class
            fine_tuning (str): fine-tuning

        """
        result = self._choices_kwargs["compute_type_" + fine_tuning].keys()
        if result is None:
            return self._choices_kwargs["compute_type_LoRA"]
        return result

    def get_cuda_visible_devices(self):
        """
        Retrieves the formatted CUDA_VISIBLE_DEVICES environment variable string.

        Args:
            self (instance method, implicitly uses self)

        Returns:
            str: Formatted CUDA_VISIBLE_DEVICES environment variable string
                 Example: "CUDA_VISIBLE_DEVICES='0,1,2,3'"
                 Returns "CUDA_VISIBLE_DEVICES=''" if no GPUs are available
        """

        num_gpus = len(GPUtil.getGPUs())
        default_gpus = ",".join(map(str, range(0, num_gpus)))
        cuda_visible_gpu = os.getenv("CUDA_VISIBLE_DEVICES", default_gpus)
        result = f"CUDA_VISIBLE_DEVICES='{cuda_visible_gpu}'"

        return result

    def update_user_dict(self, module, value, update_value):
        """
        Updates a specified value in the user configuration dictionary with robust error handling.

        Args:
            module (str): The name of the configuration module.
            value (str): The name of the configuration item to update.
            update_value: The new configuration value.
        """

        if module not in self._user_default_config:
            return
        module_config = self._user_default_config[module]
        module_config[value] = update_value

    def _init_user_dict(self):
        """
        Load user configuration from YAML file.

        Args:
            self (object): Instance of the class

        Returns:
            dict: Configuration dictionary
        """
        try:
            with open(self.get_path_config("default_yaml"), "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            if "train_sft" in data:
                data["train"] = copy.deepcopy(data["train_sft"])
            return data
        except FileNotFoundError:
            print(f"Error: File not found: {self.get_path_config('default_yaml')}")
            return None
        except yaml.YAMLError as e:
            print(f"YAML parsing error：{e}")
            return None

    def get_paddle_png(self):
        """
        Retrieve Paddle.png resource base64

        Args:
            self (object): Instance of the class

        Returns:
            str: Paddle.png resource base64
        """
        try:
            with open(self.paddle_png_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except FileNotFoundError:
            return None

    def is_thought_model(self, model_name: str) -> bool:
        """
        Check if the model supports thought process generation

        Args:
            self (object): Instance of the class
            model_name (str): Name of the model

        Returns:
            bool: True if the model supports thought process
        """
        if not isinstance(model_name, str):
            return False

        return model_name in self._thought_models

    def get_model_name_or_path(self, model_name, model_source="Local"):
        """_summary_
        评估
        Returns:
            _type_: _description_
        """
        return self._choices_kwargs["model_name_or_path" + "_" + model_source].get(
            model_name
        )

    def get_execute_command(self, name):
        """
        Generate executable command string

        Args:
            self (object): Instance of the class
            name (str): Command or module name

        Returns:
            str: Formatted command string
        """
        execute_command = {
            "export": self.get_commands_cli("export")
            + " "
            + self.get_execute_yaml_path("export_yaml_path"),
            "split": self.get_commands_cli("split")
            + " "
            + self.get_execute_yaml_path("export_yaml_path"),
            "eval": self.get_commands_cli("eval")
            + " "
            + self.get_execute_yaml_path("eval_yaml_path"),
            "chat": self.get_commands_cli("server")
            + " "
            + self.get_execute_yaml_path("chat_yaml_path"),
            "train_sft": self.get_commands_cli("train")
            + " "
            + self.get_execute_yaml_path("train_sft_yaml_path"),
            "train_dpo": self.get_commands_cli("train")
            + " "
            + self.get_execute_yaml_path("train_dpo_yaml_path"),
        }

        return execute_command.get(name)

    def get_execute_yaml_path(self, name: str):
        """
        Retrieve the YAML configuration file path for a given execution name

        Args:
            self (object): Instance of the class
            name (str): Execution configuration name

        Returns:
            str: Path to the corresponding YAML configuration file
        """
        return self._execute_yaml_path[name]

    def get_default_user_dict(self, module, name):
        """
        Retrieve default user configuration from dictionary

        Args:
            self (object): Instance of the class
            module (str): Configuration module
            name (str): Configuration item name

        Returns:
            dict: Default configuration dictionary or None if not found
        """
        try:
            return self._user_default_config[module][name]
        except KeyError:
            return None

    def get_default_dict_module(self, module):
        """
        Retrieve default configuration dictionary for a specific module

        Args:
            self (object): Instance of the class
            module (str): Configuration module name

        Returns:
            dict: Module configuration dictionary or None if not found
        """
        try:
            return self._user_default_config[module]
        except KeyError:
            return None

    def _init_execute_yaml_path(self):
        """
        Initialize execution YAML configuration file paths

        Args:
            self (object): Instance of the class

        Returns:
            dict: Dictionary mapping configuration names to their file paths
        """
        execute_path_list = self.get_default_dict_module("execute")
        return {
            "chat_yaml_path": os.path.join(
                EXECUTE_PATH, execute_path_list["chat_yaml_path"]
            ),
            "export_yaml_path": os.path.join(
                EXECUTE_PATH, execute_path_list["export_yaml_path"]
            ),
            "eval_yaml_path": os.path.join(
                EXECUTE_PATH, execute_path_list["eval_yaml_path"]
            ),
            "train_sft_yaml_path": os.path.join(
                EXECUTE_PATH, execute_path_list["train_sft_yaml_path"]
            ),
            "train_dpo_yaml_path": os.path.join(
                EXECUTE_PATH, execute_path_list["train_dpo_yaml_path"]
            ),
        }

    def get_gpu_count(self):
        """
        Retrieve the number of available GPUs

        Args:
            self (object): Instance of the class

        Returns:
            int: Number of detected GPUs, or 0 if detection fails
        """
        try:
            return len(GPUtil.getGPUs())
        except Exception as e:
            print(f"Failed to retrieve GPU information: {e}")
            return 0

    def get_path_config(self, name: str):
        """
        Retrieve path configuration by name

        Args:
            self (object): Instance of the class
            name (str): Configuration key name

        Returns:
            str: Path configuration value
        """
        return self._default_path_config[name]

    def get_choices_kwargs(self, name: str):
        """
        Retrieve keyword arguments choices configuration

        Args:
            self (object): Instance of the class
            name (str): Configuration key name

        Returns:
            Valid configuration value or None if invalid type
        """
        value = self._choices_kwargs[name]

        if not isinstance(value, (list, tuple, dict)):
            return None

        return self._choices_kwargs[name]

    def get_dataset_info_kwagrs(self, name: str):
        """
        Retrieve dataset information keyword arguments

        Args:
            self (object): Instance of the class
            name (str): Dataset identifier

        Returns:
           Dataset configuration or None if not found
        """
        try:
            value = self._dataset_info[name]
        except KeyError:
            return None
        return value

    def get_commands_cli(self, name: str):
        """
        Retrieve CLI command configuration by name

        Args:
            self (object): Instance of the class
            name (str): Command identifier

        Returns:
            str: CLI command string
        """
        return self._commands_cli[name]

    def _init_dataset_info(self):
        """
        Load dataset information from JSON configuration

        Returns:
            dict: Dataset information dictionary
        """
        try:
            with open(self.get_path_config("dataset_info_json"), "r") as f:
                data = dict(json.load(f))
                return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[WARN] Failed to load dataset: {e}")
            return []


config = ConfigManager()


def yaml_to_args(yaml_path, erniekit_execute):
    """
    Read YAML configuration file and convert to command line argument string.

    Args:
        yaml_path (str): Path to YAML file
        erniekit_execute (str): Execution command prefix

    Returns:
        str: Converted command line argument string
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_value = yaml.safe_load(f)

    args_list = []
    indentation = "    "

    for key, value in yaml_value.items():
        arg_name = f"--{key}"

        if isinstance(value, bool):
            args_list.append(f"{indentation}{arg_name} {str(value).lower()} \\")
        elif isinstance(value, list):
            list_str = ",".join(map(str, value))
            args_list.append(f'{indentation}{arg_name} "{list_str}" \\')
        elif isinstance(value, str):
            args_list.append(f'{indentation}{arg_name} "{value}" \\')
        else:
            args_list.append(f"{indentation}{arg_name} {value} \\")

    if args_list:
        args_list[-1] = args_list[-1].rstrip(" \\")

    return (
        config.get_cuda_visible_devices()
        + " "
        + erniekit_execute
        + " \\\n"
        + "\n".join(args_list)
    )


def abort_process(pid: int) -> None:
    """
    Recursively aborts a process and all its child processes in a bottom-up manner.

    Args:
        pid (int): Process ID of the parent process to abort
    """

    try:
        parent = Process(pid)
        children = parent.children(recursive=True)
        pids_to_kill = [child.pid for child in children]
        pids_to_kill.append(pid)
    except NoSuchProcess:
        return

    for pid in pids_to_kill:
        try:
            os.kill(pid, signal.SIGABRT)
        except OSError:
            pass

    time.sleep(0.5)

    still_alive = []
    for pid in pids_to_kill:
        try:
            Process(pid).status()
            still_alive.append(pid)
        except NoSuchProcess:
            pass

    for pid in still_alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def flatten_dict(nested_dict, parent_key="", separator=".", exclude_keys=None):
    """
    Flatten a nested dictionary into a single-level dictionary, with optional key exclusion.

    Args:
        nested_dict (dict): The nested dictionary to flatten
        parent_key (str): Prefix for parent keys (default: "")
        separator (str): Separator between nested key levels (default: ".")
        exclude_keys (list): List of keys to exclude from flattening (default: None)

    Returns:
        dict: Flattened dictionary with combined keys
    """
    if exclude_keys is None:
        exclude_keys = []

    items = {}
    for key, value in nested_dict.items():
        if key in exclude_keys:
            continue

        if isinstance(value, dict):
            items.update(flatten_dict(value, "", separator, exclude_keys))
        else:
            items[key] = value
    return items


def parse_string_to_list(value):
    """
    Parse a string representation of a list into an actual list.

    Args:
        value (str): String representation of a list, e.g., "['a', 'b']" or "[1.0]"

    Returns:
        list: Parsed list object if successful, otherwise returns the original value
    """
    if not isinstance(value, str):
        return value

    if value.strip().startswith("[") and value.strip().endswith("]"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    return value


def convert_boolean_strings(value):
    """
    Convert a string representation of a boolean to a Python boolean value.

    Args:
        value (str): Value to convert (e.g., "true", "false", "1", "0")

    Returns:
        bool: Converted boolean value if successful, otherwise returns the original value
    """
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
    return value


def merge_dict_to_yaml(
    manager,
    dict_data,
    yaml_file_path,
    first_level_keys=None,
    exclude_keys=None,
    is_preview=False,
):
    """
    Update a YAML file with flattened dictionary data, excluding specific keys
    and applying special list formatting to specified keys.

    Args:
        dict_data (dict): Source dictionary data to update from
        yaml_file_path (str): Path to the target YAML file
        first_level_keys (list): List of first-level keys to process (None processes all)
        exclude_keys (list): List of keys to exclude from updating
        (list): List of keys requiring special list formatting
    """

    all_components = manager.get_all_specific_component_values()
    filtered_dict = {
        k.replace("specific_", ""): v
        for k, v in all_components.items()
        if k.replace("specific_", "") in set(first_level_keys)
    }

    merged_dict = deep_merge(filtered_dict, dict_data.copy())

    if first_level_keys:
        merged_dict = {key: merged_dict.get(key, {}) for key in first_level_keys}
        merged_dict = update_dataset_paths(merged_dict, manager, is_preview)

    flattened_dict = flatten_dict(merged_dict, exclude_keys=exclude_keys)

    for key, value in flattened_dict.items():
        parsed_value = parse_string_to_list(value)
        if isinstance(parsed_value, list):
            if len(parsed_value) == 1:
                flattened_dict[key] = str(parsed_value[0])
            else:
                flattened_dict[key] = ",".join(map(str, parsed_value))
        else:
            converted_value = convert_boolean_strings(value)
            if isinstance(converted_value, str) and is_numeric_string(converted_value):
                try:
                    if "." in converted_value:
                        flattened_dict[key] = float(converted_value)
                    else:
                        flattened_dict[key] = int(converted_value)
                except ValueError:
                    flattened_dict[key] = converted_value
            else:
                flattened_dict[key] = converted_value

    if os.path.exists(yaml_file_path):
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
    else:
        yaml_data = {}

    yaml_data.update(flattened_dict)

    with open(yaml_file_path, "w", encoding="utf-8") as f:
        yaml.dump(
            yaml_data, f, default_flow_style=False, allow_unicode=True, width=1000
        )

    with open(yaml_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r"'__NOQUOTE_START__(.*?)__NOQUOTE_END__'", r"\1", content)
    content = re.sub(r'"__NOQUOTE_START__(.*?)__NOQUOTE_END__"', r"\1", content)
    content = content.replace("__NOQUOTE_START__", "").replace("__NOQUOTE_END__", "")

    with open(yaml_file_path, "w", encoding="utf-8") as f:
        f.write(content)


def is_numeric_string(s):
    """
    Check if a string can be converted to a numeric value (integer or float).

    Args:
        s (str): String to check

    Returns:
        bool: True if the string represents a numeric value, False otherwise
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def deep_merge(source, destination):
    """
    Recursively merge two dictionaries:
    - If a key exists in both and both values are dictionaries, merge inner fields recursively
    - Otherwise, overwrite the source value with the destination value

    Args:
        source (dict): Source dictionary to merge from
        destination (dict): Destination dictionary to merge into

    Returns:
        dict: Merged dictionary (destination updated in-place and returned)
    """
    for key, value in source.items():
        if (
            key in destination
            and isinstance(destination[key], dict)
            and isinstance(value, dict)
        ):
            destination[key] = deep_merge(value, destination[key])
        else:
            destination[key] = value
    return destination


def update_dataset_paths(config_dict, manager, is_preview=False):
    """
    Find and replace values of all keys ending with "dataset" in a configuration dictionary
    with default file paths.

    Args:
        config_dict (dict): Configuration dictionary to process
        manager (Manager): Manager instance containing dataset information
    Returns:
        dict: New configuration dictionary with replaced values
    """

    def update_output_dir(manager_input, config_dict_input):
        if config_dict_input["output_dir_view"]:
            config_dict_input["output_dir"] = config_dict_input["output_dir_view"]
        else:
            config_dict_input["output_dir"] = mkdir_output_dir(
                manager_input, is_preview
            )

    def update_logging_dir(config_dict_input, basic_config_input, module):
        config_dict_input["logging_dir"] = os.path.join(
            basic_config_input["output_dir"], config.get_path_config("logging_dir")
        )
        config.update_user_dict(module, "logging_dir", config_dict_input["logging_dir"])

    def update_dataset_info(module_config, train_eval_type):
        dataset_group = module_config[train_eval_type + "_dataset_group"]
        module_config[train_eval_type + "_dataset_type"] = extract_dataset_and_join(
            dataset_group, "col1"
        )
        module_config[train_eval_type + "_dataset_path"] = extract_dataset_and_join(
            dataset_group, "col2"
        )
        module_config[train_eval_type + "_dataset_prob"] = extract_dataset_and_join(
            dataset_group, "col3"
        )

    basic_config = config_dict.get("basic", {})
    train_config = config_dict.get("train", {})
    eval_config = config_dict.get("eval", {})
    export_config = config_dict.get("export", {})
    chat_config = config_dict.get("chat", {})

    if train_config != {}:
        update_output_dir(manager, basic_config)
        update_logging_dir(train_config, basic_config, "train")
        update_dataset_info(train_config, "train")
        update_dataset_info(train_config, "eval")

    if eval_config != {}:
        update_output_dir(manager, basic_config)
        update_logging_dir(eval_config, basic_config, "eval")
        update_dataset_info(eval_config, "eval")

    if export_config != {}:
        update_output_dir(manager, basic_config)
        update_logging_dir(export_config, basic_config, "export")

    if chat_config != {}:
        update_output_dir(manager, basic_config)
        update_logging_dir(chat_config, basic_config, "chat")

    if basic_config != {}:
        export_paddle_log(basic_config["output_dir"])

    return config_dict


def mkdir_output_dir(manager, is_preview):
    """
    Create a standardized output directory for model training/evaluation results.

    Args:
        manager (object): Component manager providing configuration access

    Returns:
        str: Path to the created output directory
    """

    model_name = manager.get_component_value("basic", "model_name")
    stage = manager.get_component_value("train", "stage")
    fine_tuning = manager.get_component_value("basic", "fine_tuning")
    current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dir_name = f"{model_name}_{stage}_{fine_tuning}_{current_date}"

    output_dir_default = config.get_path_config("output_dir")
    output_dir_view = manager.get_component_value("basic", "output_dir_view")
    base_output_dir = output_dir_view if output_dir_view else output_dir_default

    if os.path.isabs(base_output_dir):
        full_path = Path(base_output_dir) / dir_name
    else:
        full_path = Path(ROOT_PATH) / base_output_dir / dir_name

    if not is_preview:
        full_path.mkdir(parents=True, exist_ok=True)

    return os.path.join(base_output_dir, dir_name)


def extract_dataset_and_join(json_str, col_key):
    """
    Extract values from a JSON string based on a specified column key and join them into a single string.

    Args:
        json_str (str): A JSON string containing dataset information
        col_key (str): The key of the column to extract from each dataset item

    Returns:
        str: A comma-separated string of values extracted from the specified column. Returns empty string on failure.
    """
    try:
        json_data = json.loads(json_str)
        column_values = [str(item.get(col_key, "")) for item in json_data.values()]
        return ",".join(column_values)
    except Exception:
        return ""


def export_paddle_log(output_dir):
    """
    Configure PaddlePaddle logging directory for distributed training.

    Args:
        output_dir (str): Base directory where Paddle logs will be stored
    """

    paddle_log_dir = os.path.join(output_dir, config.get_path_config("paddle_log_dir"))
    os.environ["ERNIEKIT_DIST_LOG"] = paddle_log_dir
