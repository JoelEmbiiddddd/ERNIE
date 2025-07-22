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
Event response mechanism handling
"""

import asyncio
import json
import os
import time

import aiohttp
import gradio as gr

import erniekit.webui.lang as la
from erniekit.webui import common
from erniekit.webui.alert import alert
from erniekit.webui.chatbot import chatbot as chat_generator
from erniekit.webui.common import config
from erniekit.webui.runner import CommandRunner
from erniekit.webui.view.style import html_progress

has_shown_info = False


def basic_reaction(manager):
    """
    Perform basic configuration reactions for model setup.

    Args:
        manager (manager): Configuration manager for component values

    """
    setup_model_name_or_path_update(manager)
    reflash_compute_type_by_fine_tuning(manager)


def eval_reaction(manager, runner, module):
    """
    Perform eval configuration reactions for model setup.

    Args:
        manager (manager): Configuration manager for component values
        runner (CommandRunner): Runner instance
        module (str): Module name

    """

    setup_command_buttons(manager, runner, module)
    handle_dataset_preview(manager, module, "eval")
    setup_save_dataset_btn_update(manager, module, "eval")


def export_reaction(manager, runner, module):
    """
    Perform export configuration reactions for model setup.

    Args:
        manager (manager): Configuration manager for component values
        runner (CommandRunner): Runner instance
        module (str): Module name

    """
    setup_command_buttons(manager, runner, module)


def chat_reaction(manager, runner):
    """
    Perform chat configuration reactions for model setup.

    Args:
        manager (manager): Configuration manager for component values
        runner (CommandRunner): Runner instance

    """
    chat_load_model_button(manager, runner)
    setup_chatbot_response(manager)
    chat_upload_model_button(manager, runner)
    chat_status_button_handler(manager, CommandRunner())
    chat_update_max_new_len_max(manager)
    chat_role_setting_system_prompt_handler(manager)


def train_reaction(manager, runner, module):
    """
    Perform train configuration reactions for model setup.

    Args:
        manager (manager): Configuration manager for component values
        runner (CommandRunner): Runner instance
        module (str): Module name

    """
    setup_command_buttons(manager, runner, module)
    setup_update_stage(manager)
    train_epochs_change(manager)
    handle_dataset_preview(manager, module, "train")
    setup_save_dataset_btn_update(manager, module, "train")
    handle_dataset_preview(manager, module, "eval")
    setup_save_dataset_btn_update(manager, module, "eval")
    train_plot_reaction(manager)
    train_progress_display(manager, runner)


def train_plot_reaction(manager):
    """
    Perform plotting configuration reactions for model setup.

    Args:
        manager (manager): Configuration manager for component values

    """
    open_close_plot_btn = manager.get_elem_by_id("train", "open_close_plot_btn")
    start_btn = manager.get_elem_by_id("train", "start_btn")
    output_plot_column = manager.get_elem_by_id("train", "output_plot_column")

    def open_btn_and_plot():
        output_plot_column.visible = True
        return gr.update(visible=True), gr.update(visible=True)

    def toggle_plot_visibility():

        if output_plot_column.visible == False:
            output_plot_column.visible = True
            return gr.update(visible=True)

        output_plot_column.visible = False
        return gr.update(visible=False)

    start_btn.click(fn=open_btn_and_plot,
                    outputs=[output_plot_column, open_close_plot_btn])

    open_close_plot_btn.click(fn=toggle_plot_visibility, outputs=[output_plot_column])


def train_epochs_change(manager):
    """
    Handle interactions between training epochs and max steps parameters.

    Args:
        manager (manager): component manager
    """

    num_train_epochs = manager.get_elem_by_id("train", "num_train_epochs")
    max_steps = manager.get_elem_by_id("train", "max_steps")

    def on_num_train_epochs_change(max_steps_value):
        global has_shown_info
        if max_steps_value > 0 and not has_shown_info:
            gr.Info(alert.get("max_steps_notice", "info"))
            has_shown_info = True

        pass

    num_train_epochs.change(fn=on_num_train_epochs_change, inputs=[max_steps], outputs=[])


def train_progress_display(manager, runner):
    progress_display = manager.get_elem_by_id("train", "progress_display")
    max_steps = manager.get_elem_by_id("train", "max_steps")
    start_btn = manager.get_elem_by_id("train", "start_btn")

    def train_progress_compute(max_steps_value):
        vdl_data = runner.get_plot()
        percentage = runner.compute_percentage(len(vdl_data), max_steps_value)
        return gr.update(value=html_progress.format(percentage, f"{percentage:.1f}"),
                                        visible=True)

    start_btn.click(
        fn=train_progress_compute,
        inputs=[max_steps],
        outputs=[progress_display]
    )




def react_preview_dataset_button(manager, preview_button, module, elem_type):
    """
    Render dataset preview buttons with support for multiple datasets, pagination,
    and dataset switching.

    Args:
        elem_type:
        manager (Manager): Component manager instance
        module (str): Module identifier (e.g., "train", "eval")
        elem_type (str): Element identifier for the dataset configuration
    """

    language = manager.get_elem_by_id("basic", "language")

    current_page = gr.State(value=1)
    current_dataset_index = gr.State(value=0)

    with gr.Column(visible=False, elem_classes="modal-overlay") as overlay:
        pass

    with gr.Column(visible=False, elem_classes="modal-box") as popup:
        with gr.Row():
            with gr.Column(scale=1):
                dataset_preview_title = gr.Markdown()
            with gr.Column(scale=1):
                page_info = gr.Markdown(elem_classes="page-info")
            with gr.Row():
                dataset_info = gr.Markdown()

        json_preview = gr.Code(language="json", show_label=False)

        with gr.Row(elem_classes="pagination-controls"):
            prev_btn = gr.Button(size="lg", visible=False)
            manager.add_elem(module, "prev_btn", prev_btn)
            next_btn = gr.Button(size="lg", visible=False)
            close_button = gr.Button(size="lg")

        with gr.Row(elem_classes="pagination-controls"):
            prev_dataset_btn = gr.Button(size="lg", visible=False)
            next_dataset_btn = gr.Button(size="lg", visible=False)

    def get_data_paths():
        user_data_path_value = manager.get_component_value(module_id=module, elem_id=f"{elem_type}_dataset_path")

        if user_data_path_value:
            data_paths = []
            for path in user_data_path_value.split(","):
                path = path.strip()
                if not path:
                    continue

                if os.path.isabs(path):
                    full_path = os.path.normpath(path)
                else:
                    full_path = os.path.normpath(os.path.join(common.ROOT_PATH, path))

                if os.path.exists(full_path):
                    data_paths.append(full_path)
                else:
                    preview_data_non_existent = alert.get("preview_data_non_existent", "error").format(path)
                    print(preview_data_non_existent)
                    gr.Warning(preview_data_non_existent)

            if data_paths:
                return data_paths

        return []

    def load_data(data_paths, dataset_index=0, page=1, page_size=10, max_pages=100):
        if not data_paths or dataset_index >= len(data_paths):
            return [], 0, 0, dataset_index, len(data_paths), ""

        path = data_paths[dataset_index]
        all_data = []

        try:
            if path.lower().endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        all_data.append(json.loads(line))
            else:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_data.extend(data)
        except FileNotFoundError as fe:
            preview_data_error = alert.get("preview_data_error", "error").format(path, fe)
            print(preview_data_error)
            raise gr.Error(preview_data_error)
        except json.JSONDecodeError:
            preview_data_non_json = alert.get("preview_data_non_json", "error").format(path)
            print(preview_data_non_json)
            gr.Warning(preview_data_non_json)
        except Exception as e:
            preview_data_error = alert.get("preview_data_error", "error").format(path, e)
            print(preview_data_error)
            raise gr.Error(preview_data_error)

        total = len(all_data)
        cutoff_page = False
        if total == 0:
            return [], 0, 0, dataset_index, len(data_paths), path

        total_pages = (total + page_size - 1) // page_size
        effective_max_page = min(max_pages, total_pages)

        if total_pages > max_pages:
            cutoff_page = True

        safe_page = min(max(1, page), effective_max_page)

        start_idx = (safe_page - 1) * page_size
        end_idx = start_idx + page_size

        return all_data[start_idx:end_idx], total, cutoff_page, dataset_index, len(data_paths), path

    def keep_last_five_levels(path_str):
        parts = path_str.strip('/').split('/')
        if len(parts) <= 5:
            return '/' + path_str.strip('/')
        else:
            return '/' + '/'.join(parts[-5:])

    def show_popup(language, page_number, dataset_index):
        page = int(page_number) if isinstance(page_number, str) else page_number
        data_paths = get_data_paths()
        data, total_records, cutoff_page, current_dataset, total_datasets, current_path = load_data(
            data_paths, dataset_index=dataset_index, page=page, page_size=10
        )

        total_pages = max(1, (total_records + 9) // 10)

        dataset_text = la.get("dataset_info", language, "value").format(
            current_dataset + 1, total_datasets, keep_last_five_levels(current_path)
        )

        has_multiple_datasets = total_datasets > 1
        show_prev_dataset = has_multiple_datasets and current_dataset > 0
        show_next_dataset = has_multiple_datasets and current_dataset < total_datasets - 1
        page_info_text = la.get("page_info", language, "value").format(page, total_pages)

        return (
            gr.Column(visible=True),
            gr.Column(visible=True),
            json.dumps(data, ensure_ascii=False, indent=2),
            str(page),
            page_info_text,
            dataset_text,
            gr.Button(visible=page > 1),
            gr.Button(visible=page < total_pages),
            gr.Button(visible=show_prev_dataset),
            gr.Button(visible=show_next_dataset),
            current_dataset,
        )

    def hide_popup():
        return (
            gr.Column(visible=False),
            gr.Column(visible=False),
            None,
            "1",
            "第 1 页，共 1 页",
            "",
            gr.Button(visible=False),
            gr.Button(visible=False),
            gr.Button(visible=False),
            gr.Button(visible=False),
            0,
        )

    def update_page(language, page_num, dataset_index):
        return show_popup(language, page_num, dataset_index)

    def on_preview_check(language, current_page, current_dataset_index):
        dataset_path_value = manager.get_component_value(module_id=module, elem_id=f"{module}_dataset_path")
        if dataset_path_value == "" or dataset_path_value is None:
            gr.Warning(alert.get("preview_data_path_none", "warning"))
            return hide_popup()
        else:
            data_paths = get_data_paths()
            if not data_paths:
                return hide_popup()
            return show_popup(language, current_page, current_dataset_index)

    def navigate_dataset(dataset_index, direction):
        data_paths = get_data_paths()
        if not data_paths:
            return 0, gr.Button(visible=False), gr.Button(visible=False)

        new_index = dataset_index + direction
        new_index = max(0, min(new_index, len(data_paths) - 1))

        show_prev = len(data_paths) > 1 and new_index > 0
        show_next = len(data_paths) > 1 and new_index < len(data_paths) - 1

        return new_index, gr.Button(visible=show_prev), gr.Button(visible=show_next)

    def go_to_prev_dataset(dataset_index):
        new_index, prev_btn_vis, next_btn_vis = navigate_dataset(dataset_index, -1)
        return (
            new_index,
            prev_btn_vis,
            next_btn_vis,
            "1",
        )

    def go_to_next_dataset(dataset_index):
        new_index, prev_btn_vis, next_btn_vis = navigate_dataset(dataset_index, 1)
        return (
            new_index,
            prev_btn_vis,
            next_btn_vis,
            "1",
        )

    def update_language(language):
        dataset_preview_title_la = la.get(key="dataset_preview_title", lang=language, prop="value")
        page_info_la = la.get(key="page_info", lang=language, prop="value")
        dataset_info_la = la.get(key="dataset_info", lang=language, prop="value")
        next_btn_la = la.get(key="next_btn", lang=language, prop="value")
        prev_btn_la = la.get(key="prev_btn", lang=language, prop="value")
        close_button_la = la.get(key="close_button", lang=language, prop="value")
        prev_dataset_btn_la = la.get(key="prev_dataset_btn", lang=language, prop="value")
        next_dataset_btn_la = la.get(key="next_dataset_btn", lang=language, prop="value")

        return (
            dataset_preview_title_la,
            page_info_la,
            dataset_info_la,
            next_btn_la,
            prev_btn_la,
            close_button_la,
            prev_dataset_btn_la,
            next_dataset_btn_la,
        )

    preview_button.click(
        on_preview_check,
        inputs=[language, current_page, current_dataset_index],
        outputs=[
            popup,
            overlay,
            json_preview,
            current_page,
            page_info,
            dataset_info,
            prev_btn,
            next_btn,
            prev_dataset_btn,
            next_dataset_btn,
            current_dataset_index,
        ],
    )

    prev_btn.click(
        lambda page: int(page) - 1 if isinstance(page, str) else page - 1,
        inputs=[current_page],
        outputs=[current_page],
    ).then(
        update_page,
        inputs=[language, current_page, current_dataset_index],
        outputs=[
            popup,
            overlay,
            json_preview,
            current_page,
            page_info,
            dataset_info,
            prev_btn,
            next_btn,
            prev_dataset_btn,
            next_dataset_btn,
            current_dataset_index,
        ],
    )

    next_btn.click(
        lambda page: int(page) + 1 if isinstance(page, str) else page + 1,
        inputs=[current_page],
        outputs=[current_page],
    ).then(
        update_page,
        inputs=[language, current_page, current_dataset_index],
        outputs=[
            popup,
            overlay,
            json_preview,
            current_page,
            page_info,
            dataset_info,
            prev_btn,
            next_btn,
            prev_dataset_btn,
            next_dataset_btn,
            current_dataset_index,
        ],
    )

    close_button.click(
        hide_popup,
        outputs=[
            popup,
            overlay,
            json_preview,
            current_page,
            page_info,
            dataset_info,
            prev_btn,
            next_btn,
            prev_dataset_btn,
            next_dataset_btn,
            current_dataset_index,
        ],
    )

    prev_dataset_btn.click(
        go_to_prev_dataset,
        inputs=[current_dataset_index],
        outputs=[
            current_dataset_index,
            prev_dataset_btn,
            next_dataset_btn,
            current_page,
        ],
    ).then(
        update_page,
        inputs=[language, current_page, current_dataset_index],
        outputs=[
            popup,
            overlay,
            json_preview,
            current_page,
            page_info,
            dataset_info,
            prev_btn,
            next_btn,
            prev_dataset_btn,
            next_dataset_btn,
            current_dataset_index,
        ],
    )

    next_dataset_btn.click(
        go_to_next_dataset,
        inputs=[current_dataset_index],
        outputs=[
            current_dataset_index,
            prev_dataset_btn,
            next_dataset_btn,
            current_page,
        ],
    ).then(
        update_page,
        inputs=[language, current_page, current_dataset_index],
        outputs=[
            popup,
            overlay,
            json_preview,
            current_page,
            page_info,
            dataset_info,
            prev_btn,
            next_btn,
            prev_dataset_btn,
            next_dataset_btn,
            current_dataset_index,
        ],
    )

    language.change(
        fn=update_language,
        inputs=[language],
        outputs=[
            dataset_preview_title,
            page_info,
            dataset_info,
            next_btn,
            prev_btn,
            close_button,
            prev_dataset_btn,
            next_dataset_btn,
        ],
    )

    manager.demo.load(
        fn=update_language,
        inputs=[language],
        outputs=[
            dataset_preview_title,
            page_info,
            dataset_info,
            next_btn,
            prev_btn,
            close_button,
            prev_dataset_btn,
            next_dataset_btn,
        ],
    )


def setup_command_buttons(manager, runner, module):
    """
    Configure command execution buttons for a specific module.

    Args:
        manager (manager): the component manager
        runner (runner): the execution runner
        module (str): Name of the target module (e.g., "training", "inference")

    Returns:
        None
    """
    setup_preview_button(manager, module)
    setup_start_button(manager, runner, module)
    setup_stop_button(manager, runner, module)
    setup_clean_button(manager, runner, module)


def setup_model_name_or_path_update(manager):
    """
    Configure model name or path update functionality.

    Args:
        manager (object): Manager instance handling model configurations
    """

    model_name = manager.get_elem_by_id("basic", "model_name")
    model_name_or_path = manager.get_elem_by_id("basic", "model_name_or_path")
    model_source = manager.get_elem_by_id("basic", "model_source")

    def update_path(selected_model, select):

        if selected_model == "Customization":
            gr.Info(alert.get("custom_model_notice", "info"))
            return (
                gr.update(interactive=True, value=""),
                gr.update(value="Local", choices=config.get_choices_kwargs("model_source_custom")),
            )

        if config.is_thought_model(selected_model):
            gr.Info(alert.get("thought_model_notice", "info"))

        path = config.get_model_name_or_path(selected_model, select.replace(" ", "_").lower())

        manager._component_values["basic"]["model_name_or_path"] = path
        return path, gr.update(value="Local", choices=config.get_choices_kwargs("model_source_ernie"))

    def update_path_selector(selected_model, model_source_value):

        path = config.get_model_name_or_path(selected_model, model_source_value.replace(" ", "_").lower())

        manager._component_values["basic"]["model_name_or_path"] = path
        return path

    model_name.change(
        fn=update_path,
        inputs=[model_name, model_source],
        outputs=[model_name_or_path, model_source],
    )

    model_source.change(fn=update_path_selector, inputs=[model_name, model_source], outputs=[model_name_or_path])


def reflash_compute_type_by_fine_tuning(manager):
    """
    Refresh compute type based on fine-tuning configuration.

    Args:
        manager (object): Manager instance handling compute resources

    """

    fine_tuning = manager.get_elem_by_id("basic", "fine_tuning")
    compute_type = manager.get_elem_by_id("basic", "compute_type")

    last_update_time = [0]

    def update_choices(fine_tuning_value):

        current_time = time.time()
        last_update_time[0] = current_time

        time.sleep(0.1)

        if current_time != last_update_time[0]:
            return gr.update()

        try:
            if fine_tuning_value == "LoRA":
                choices = config.get_choices_kwargs("compute_type_LoRA")
            else:
                choices = config.get_choices_kwargs("compute_type_Full")

            return gr.update(choices=choices, value="bf16")
        except Exception as e:
            print(f"update fail: {e}")
            return gr.update()

    def alert_compute_type_fp8(compute_type_value):
        if compute_type_value == "fp8":
            gr.Warning(alert.get("compute_type_fp8_notice", "warning"))

    fine_tuning.change(fn=update_choices, inputs=[fine_tuning], outputs=[compute_type])
    compute_type.change(fn=alert_compute_type_fp8, inputs=[compute_type], outputs=[])


def setup_preview_button(manager, module):
    """
    Initialize preview button configuration.

    Args:
        manager (object): Component manager instance
        module (str): Target module identifier

    """

    preview_command_btn = manager.get_elem_by_id(module, "preview_command_btn")
    command_preview = manager.get_elem_by_id(module, "command_preview")
    output_text = manager.get_elem_by_id(module, "output_text")
    output_container = manager.get_elem_by_id(module, "output_container")
    stage = manager.get_elem_by_id("train", "stage")

    def show_command_preview():
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            "",
        )

    def module_command_preview(stage):

        if module == "train":
            execute_path = f"train_{stage.lower()}_yaml_path"
        else:
            execute_path = module + "_yaml_path"

        update_config_yaml(manager, execute_path, module, True)

        return common.yaml_to_args(config.get_execute_yaml_path(execute_path), config.get_commands_cli(module))

    if preview_command_btn and command_preview and output_text and output_container:
        preview_command_btn.click(
            fn=show_command_preview,
            inputs=[],
            outputs=[command_preview, output_text, output_container, command_preview],
        ).then(fn=module_command_preview, inputs=[stage], outputs=command_preview)


async def execute_command(runner, command):
    """
    Asynchronously execute a shell command.

    Args:
        runner (object): Execution context or runner instance
        command (str): Command string to be executed
    """
    async for output, percentage in runner.execute(command):
        yield output, percentage


def chat_load_model_button(manager, runner):
    """
    Configure the chat interface's model loading button.

    Args:
        manager (object): Manager for UI component coordination
        runner (object): Execution handler for model loading

    """
    load_model_btn = manager.get_elem_by_id("chat", "load_model_btn")
    output_text = manager.get_elem_by_id("chat", "output_text")
    port = manager.get_elem_by_id("chat", "port")
    save_port = manager.get_elem_by_id("chat", "save_port")
    progress_display = manager.get_elem_by_id("chat", "progress_display")

    async def chat_start_execution(port):

        update_config_yaml(manager, "chat_yaml_path", "chat")

        command = config.get_execute_command("chat")
        async for output, percentage in execute_command(runner, command):
            yield output, gr.update(value=port), gr.update(value=html_progress.format(percentage, f"{percentage:.1f}"),
                                        visible=True)

    load_model_btn.click(fn=chat_start_execution, inputs=[port], outputs=[output_text, save_port, progress_display])


def chat_upload_model_button(manager, runner):
    """
    Initialize the chat interface's model upload button.

    Args:
        manager (object): Manager for handling UI component states
        runner (object): Handler for model upload execution
    """
    unload_model_btn = manager.get_elem_by_id("chat", "unload_model_btn")
    output_text = manager.get_elem_by_id("chat", "output_text")

    async def chat_stop_current_process():
        result = await runner.stop()
        return result

    if unload_model_btn and output_text:
        unload_model_btn.click(fn=chat_stop_current_process, inputs=[], outputs=output_text)


def setup_start_button(manager, runner, module):
    """
    Configure the start button for a specific module.

    Args:
        manager (object): Manager for UI component coordination
        runner (object): Execution handler for module operations
        module (str): target module
    """
    start_btn = manager.get_elem_by_id(module, "start_btn")
    command_preview = manager.get_elem_by_id(module, "command_preview")
    output_text = manager.get_elem_by_id(module, "output_text")
    output_container = manager.get_elem_by_id(module, "output_container")
    start_merge_btn = manager.get_elem_by_id("export", "start_merge_btn")
    start_split_btn = manager.get_elem_by_id("export", "start_split_btn")
    stage = manager.get_elem_by_id("train", "stage")
    progress_display = manager.get_elem_by_id(module, "progress_display")
    train_loss_plot = manager.get_elem_by_id("train", "train_loss_plot")

    def show_output_text():
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            "",
        )

    async def start_execution(stage_value):
        if module == "train":
            execute_path = f"train_{stage_value.lower()}_yaml_path"
            command_name = f"train_{stage_value.lower()}"
        else:
            execute_path = module + "_yaml_path"
            command_name = module

        update_config_yaml(manager, execute_path, module)
        command = config.get_execute_command(command_name)
        gr.Info(alert.get("allow_switch_button", "info"))

        if progress_display:
            async for output, _ in execute_command(runner, command):
                yield output
        else:
            async for output, _ in execute_command(runner, command):
                yield output

    async def loss_plot():
        await asyncio.sleep(2)
        while runner.is_loss_monitoring_active():
            await asyncio.sleep(2)
            yield runner.get_plot()

        runner.loss_tracker.clear_history_data()
        yield runner.get_plot()

    async def start_export_merge_execution():
        if runner.is_running():
            gr.Warning(alert.get("split_is_running", "warning"))
            return

        gr.Info(alert.get("allow_switch_button", "info"))

        execute_path = "export_yaml_path"
        update_config_yaml(manager, execute_path, "export")
        command = config.get_execute_command("export")

        async for output in execute_command(runner, command):
            yield output

    async def start_export_split_execution():
        if runner.is_running():
            gr.Warning(alert.get("merge_is_running", "warning"))
            return

        gr.Info(alert.get("allow_switch_button", "info"))

        execute_path = "export_yaml_path"
        update_config_yaml(manager, execute_path, "export")
        command = config.get_execute_command("split")

        async for output in execute_command(runner, command):
            yield output

    if start_btn and output_text and progress_display:
        start_btn.click(
            fn=show_output_text,
            inputs=[],
            outputs=[command_preview, output_text, output_container, output_text],
        ).then(fn=start_execution, inputs=[stage], outputs=[output_text])

        start_btn.click(
            fn=loss_plot,
            outputs=[train_loss_plot]
        )
    elif start_btn and output_text:
        start_btn.click(
            fn=show_output_text,
            inputs=[],
            outputs=[command_preview, output_text, output_container, output_text],
        ).then(fn=start_execution, inputs=[stage], outputs=output_text)

    if start_merge_btn and start_split_btn:
        start_merge_btn.click(
            fn=show_output_text,
            inputs=[],
            outputs=[command_preview, output_text, output_container, output_text],
        ).then(fn=start_export_merge_execution, inputs=[], outputs=output_text)

        start_split_btn.click(
            fn=show_output_text,
            inputs=[],
            outputs=[command_preview, output_text, output_container, output_text],
        ).then(fn=start_export_split_execution, inputs=[], outputs=output_text)


def setup_stop_button(manager, runner, module):
    """
    Configure the stop button for a specific module.

    Args:
        manager (object): Manager for UI component coordination
        runner (object): Execution handler for module operations
        module (str): Name of the target module
    """
    stop_btn = manager.get_elem_by_id(module, "stop_btn")
    output_text = manager.get_elem_by_id(module, "output_text")

    async def stop_current_process():
        result = await runner.stop()
        return result

    if stop_btn and output_text:
        stop_btn.click(fn=stop_current_process, inputs=[], outputs=output_text)


def handle_dataset_preview(manager, module, elem_type):

    dataset_path = manager.get_elem_by_id(module, f"{elem_type}_dataset_path")

    def init_dataset_path():
        user_config = config.get_default_user_dict(module, f"{elem_type}_dataset")
        dataset_info = config.get_dataset_info()

        path = []
        for key in user_config:
            if key in dataset_info:
                path.append(dataset_info[key].get('path', ''))

        path_str = ", ".join(path)
        return gr.update(value=path_str)

    manager.demo.load(
        fn=init_dataset_path,
        outputs=[dataset_path],
    )


def setup_save_dataset_btn_update(manager, module, elem_type):
    dataset_save_btn = manager.get_elem_by_id(module, f"{elem_type}_dataset_save_btn")
    dataset_group = manager.get_elem_by_id(module, f"{elem_type}_dataset_group")

    dataset_path = manager.get_elem_by_id(module, f"{elem_type}_dataset_path")
    dataset_prob = manager.get_elem_by_id(module, f"{elem_type}_dataset_prob")
    dataset_type = manager.get_elem_by_id(module, f"{elem_type}_dataset_type")

    def update_path_by_group(dataset_group_value):
        return (
            common.extract_dataset_and_join(dataset_group_value, "col2"),
            common.extract_dataset_and_join(dataset_group_value, "col3"),
            common.extract_dataset_and_join(dataset_group_value, "col1"),
        )

    dataset_save_btn.click(
        fn=update_path_by_group,
        inputs=[dataset_group],
        outputs=[dataset_path, dataset_prob, dataset_type]
    )


def setup_clean_button(manager, runner, module):
    """
    Configure the clean button for resource cleanup.

    Args:
        manager (object): Manager for UI component coordination
        runner (object): Handler for cleanup operations
        module (str):   Name of the target module
    """
    clean_btn = manager.get_elem_by_id(module, "clean_btn")
    output_text = manager.get_elem_by_id(module, "output_text")

    def clear_output():
        return runner.clear_output()

    if clean_btn and output_text:
        clean_btn.click(fn=clear_output, inputs=[], outputs=output_text)


def model_update_callback(manager, stage):
    """
    Handle model update events at specified stages.

    Args:
        manager (object): Manager for model lifecycle
        stage (str): Update stage
    """
    sft_params = config.get_default_dict_module("train_sft")
    dpo_params = config.get_default_dict_module("train_dpo")

    updates = {}

    if stage == "SFT":
        params = sft_params
    elif stage == "DPO":
        params = dpo_params
    else:
        return updates

    train_componet_elem_list = manager.get_dependencies("basic.best_config.train")
    basic_componet_elem_list = manager.get_dependencies("basic.best_config.basic")

    for full_id in train_componet_elem_list["dependent_ids"]:
        if full_id.startswith("train."):
            elem_id = full_id[len("train") + 1 :]
            if elem_id in params:
                param_value = params[elem_id]

                try:
                    component = manager.get_elem_by_id("train", elem_id)
                    fixed_value = fix_component_value(component, param_value)

                    if param_value is None:
                        fixed_value = get_default_value_for_component(component)

                    updates[full_id] = gr.update(value=fixed_value)

                except Exception as e:
                    print(f"Error processing component {full_id}: {e}")
                    updates[full_id] = gr.update(value=param_value)

    for full_id in basic_componet_elem_list["dependent_ids"]:
        if full_id.startswith("basic."):
            elem_id = full_id[len("basic") + 1 :]
            if elem_id in params:
                param_value = params[elem_id]

                try:
                    component = manager.get_elem_by_id("basic", elem_id)

                    fixed_value = fix_component_value(component, param_value)

                    if param_value is None:
                        fixed_value = get_default_value_for_component(component)

                    updates[full_id] = gr.update(value=fixed_value)

                except Exception as e:
                    print(f"Error processing component {full_id}: {e}")
                    updates[full_id] = gr.update(value=param_value)

    return updates


def get_default_value_for_component(component):
    """
    Return appropriate default value based on component type.

    Args:
        component (str): Component identifier or type

    Returns:
        Any: Default value corresponding to the component type
    """
    component_type = type(component).__name__

    if hasattr(component, 'value') and component.value is not None:
        return component.value

    if component_type in ['Textbox', 'TextArea']:
        return ""
    elif component_type in ['Number', 'Slider']:
        return 0
    elif component_type == 'Checkbox':
        return False
    elif component_type == 'Dropdown':
        if hasattr(component, 'multiselect') and component.multiselect:
            return []
        else:
            return None
    elif component_type == 'Radio':
        return None
    elif component_type == 'CheckboxGroup':
        return []
    else:
        return ""


def fix_component_value(component, value):
    """
    Validate and adjust component value to acceptable range.

    Args:
        component (str): Component identifier or type
        value (Any): Input value to be validated
    """

    component_type = type(component).__name__

    if component_type == "Number":
        if isinstance(value, str):
            try:
                float_val = float(value)
                if float_val.is_integer():
                    return int(float_val)
                return float_val
            except (ValueError, TypeError):
                print(f"Warning: Cannot convert '{value}' to number, using 0")
                return 0
        elif isinstance(value, (int, float)):
            return value
        else:
            print(f"Warning: Unexpected type {type(value)} for Number component, using 0")
            return 0

    elif component_type == "Dropdown":
        if hasattr(component, "choices") and component.choices:
            choices = component.choices
            choices_str = [str(choice) for choice in choices]

            if set(choices_str) == {"True", "False"}:
                if isinstance(value, bool):
                    return str(value)
                elif isinstance(value, str):
                    if value.lower() in ["true", "1", "yes"]:
                        return "True"
                    elif value.lower() in ["false", "0", "no"]:
                        return "False"
                    return value
                elif isinstance(value, (int, float)):
                    return str(bool(value))

            elif all(isinstance(c, (int, float)) for c in choices):
                try:
                    return type(choices[0])(value)
                except (ValueError, TypeError):
                    return choices[0] if choices else value
            elif isinstance(value, list):
                return value
            else:
                return str(value)
        return value

    elif component_type == "Slider":
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return 0.0

    elif component_type == "Textbox":
        return str(value)

    elif component_type == "Checkbox":
        if isinstance(value, str):
            return value.lower() in ["true", "1", "yes", "on"]
        elif isinstance(value, (int, float)):
            return bool(value)
        elif isinstance(value, bool):
            return value
        else:
            return False

    return value


def setup_update_stage(manager):
    """
    Configure the update process lifecycle stages.

    Args:
        manager (object): Manager for update process coordination
    """
    best_config_elem = manager.get_elem_by_id("basic", "best_config")

    train_componet_elem_list = manager.get_dependencies("basic.best_config.train")
    basic_componet_elem_list = manager.get_dependencies("basic.best_config.basic")
    train_components = []
    basic_components = []

    for full_id in train_componet_elem_list["dependent_ids"]:
        if full_id.startswith("train."):
            elem_id = full_id[len("train") + 1 :]
            component = manager.get_elem_by_id("train", elem_id)
            train_components.append((full_id, component))

    for full_id in basic_componet_elem_list["dependent_ids"]:
        if full_id.startswith("basic."):
            elem_id = full_id[len("basic") + 1 :]
            component = manager.get_elem_by_id("basic", elem_id)
            basic_components.append((full_id, component))

    all_components = train_components + basic_components

    def on_component_value_change(value):
        manager._update_component_value("basic", "best_config", value)

        updates = model_update_callback(manager, value)

        output_list = []
        for full_id, component in all_components:
            if full_id in updates:
                output_list.append(updates[full_id])
            else:
                output_list.append(gr.update())

        return output_list

    best_config_elem.change(
        fn=on_component_value_change,
        inputs=[best_config_elem],
        outputs=[component for _, component in all_components],
    )


def setup_chatbot_response(manager):
    """
    Configure chatbot response logic, including event bindings for submit and stop buttons.

    Args:
        manager (object): Manager for chatbot UI and state

    """
    chat_input = manager.get_elem_by_id("chat", "chat_input")
    chatbot = manager.get_elem_by_id("chat", "chatbot")
    role_setting = manager.get_elem_by_id("chat", "role_setting")
    system_prompt = manager.get_elem_by_id("chat", "system_prompt")
    max_new_tokens = manager.get_elem_by_id("chat", "max_new_tokens")
    top_p = manager.get_elem_by_id("chat", "top_p")
    temperature = manager.get_elem_by_id("chat", "temperature")
    submit_btn = manager.get_elem_by_id("chat", "submit_btn")
    port = manager.get_elem_by_id("chat", "save_port")
    model_name = manager.get_elem_by_id("basic", "model_name")
    stop_btn = manager.get_elem_by_id("chat", "stop_btn")
    clear_btn = manager.get_elem_by_id("chat", "clear_btn")

    async def on_submit(message, history, role, system_prompt, max_new_tokens, top_p, temperature, port, model_name):
        update_config_yaml(manager, "chat_yaml_path", "chat")

        use_thought_model = chat_generator.check_thought_model(model_name=model_name)

        if use_thought_model:
            async for result in chat_generator.thought_response(
                message,
                history,
                role,
                system_prompt,
                max_new_tokens,
                top_p,
                temperature,
                port,
            ):
                yield result
        else:
            async for result in chat_generator.mm_response(
                message,
                history,
                role,
                system_prompt,
                max_new_tokens,
                top_p,
                temperature,
                port,
            ):
                yield result

    def on_stop():
        chat_generator.stop()

    def on_clear():
        return [], ""

    submit_btn.click(
        fn=on_submit,
        inputs=[
            chat_input,
            chatbot,
            role_setting,
            system_prompt,
            max_new_tokens,
            top_p,
            temperature,
            port,
            model_name,
        ],
        outputs=[
            chatbot,
            chat_input,
        ],
        api_name=None,
    )

    chat_input.submit(
        fn=on_submit,
        inputs=[
            chat_input,
            chatbot,
            role_setting,
            system_prompt,
            max_new_tokens,
            top_p,
            temperature,
            port,
            model_name,
        ],
        outputs=[
            chatbot,
            chat_input,
        ],
        api_name=None,
    )

    stop_btn.click(fn=on_stop, inputs=[], outputs=[])

    clear_btn.click(
        fn=on_clear,
        inputs=[],
        outputs=[
            chatbot,
            chat_input,
        ],
    )


def chat_status_button_handler(manager, runner):
    """
    Handle chat status button interactions (e.g., start/stop/pause).

    Args:
        manager (object): Manager for chat UI components
        runner (object): Execution handler for chat operations
    """

    button = manager.get_elem_by_id("chat", "status_button")
    save_port = manager.get_elem_by_id("chat", "save_port")

    async def update_status(port_value):
        url = f"http://0.0.0.0:{port_value}/health"

        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        success = alert.get(
                            "chat_check_load_model",
                            "tips",
                        )
                        status_message = f"{success}: " + f"http://0.0.0.0:{port_value}"
                        gr.Success(status_message)
                    else:
                        status_message = alert.get(
                            "chat_check_load_model",
                            "error",
                        ).format(port_value)
                        gr.Warning(f"{status_message} - Status: {response.status}")

        except asyncio.TimeoutError:
            error_message = alert.get("chat_check_load_model", "error").format(port_value)
            gr.Warning(f"{error_message} - Timeout")

        except Exception as e:
            error_message = alert.get("chat_check_load_model", "error").format(port_value)
            gr.Warning(f"{error_message} - Exception: {e!s}")

    button.click(fn=update_status, inputs=[save_port], outputs=[])


def chat_role_setting_system_prompt_handler(manager):
    """
    Handle system prompt configuration for chat role settings.

    Args:
        manager (object): Manager for chat role configuration
    """
    role_setting = manager.get_elem_by_id("chat", "role_setting")
    system_prompt = manager.get_elem_by_id("chat", "system_prompt")
    model_name = manager.get_elem_by_id("basic", "model_name")

    def update_role_setting(input):
        if config.is_thought_model(input):
            gr.Warning(alert.get("chat_role_setting", "warning"))

    def update_system_prompt(input):
        if config.is_thought_model(input):
            gr.Warning(alert.get("chat_system_prompt", "warning"))

    role_setting.change(
        fn=update_role_setting,
        inputs=[model_name],
        outputs=[],
    )

    system_prompt.change(
        fn=update_system_prompt,
        inputs=[model_name],
        outputs=[],
    )


def chat_update_max_new_len_max(manager):
    """
    Update the maximum value for the 'max_new_tokens' parameter.

    Args:
        manager (object): Manager for chat configuration settings
    """
    max_model_len = manager.get_elem_by_id("chat", "max_model_len")
    max_new_tokens = manager.get_elem_by_id("chat", "max_new_tokens")

    def update_max_new_tokens(input):
        return gr.update(maximum=input - 1)

    max_model_len.change(
        fn=update_max_new_tokens,
        inputs=[max_model_len],
        outputs=[max_new_tokens],
    )


def update_config_yaml(manager, execute_path, module, is_preview=False):
    """
    Update configuration in YAML file for specified module.

    Args:
        manager (object): Configuration manager
        execute_path (str): Path to execution directory
        module (str): Target module identifier
    """

    config_dict = manager.get_all_component_values()
    common.merge_dict_to_yaml(
        manager,
        config_dict,
        config.get_execute_yaml_path(execute_path),
        ["basic", module],
        [
            "preview_command_btn",
            "start_btn",
            "language",
            "stop_btn",
            "best_config",
            "command_preview",
            "output_text",
            "output_container",
            "clean_btn",
            "train_dataset_preview_btn",
            "start_merge_btn",
            "start_split_btn",
            "chat_input",
            "chatbot",
            "load_model_btn",
            "response_display",
            "status_button",
            "submit_btn",
            "thinking_display",
            "unload_model_btn",
            "role_setting",
            "system_prompt",
            "eval_dataset_preview_btn",
            "model_output_tab",
            "train_tab",
            "chat_tab",
            "clear_btn",
            "eval_tab",
            "start_split_btn",
            "start_merge_btn",
            "save_port",
            "export_tab",
            "gpu_num",
            "dataloader_parameters_tab",
            "distributed_parameters_tab",
            "optimizer_parameters_tab",
            "other_parameters_tab",
            "stop_btn",
            "output_dir_view",
            "prev_btn",
            "train_dataset_group",
            "eval_dataset_group",
            "eval_dataset",
            "model_source",
            "chat_info",
            "eval_dataset",
            "train_dataset",
            "model_name",
            "eval_dataset_save_btn",
            "train_dataset_save_btn",
            "eval_dataset_btn",
            "train_dataset_btn",
            "eval_dataset_preview_btn",
            "train_dataset_preview_btn",
            "progress_display"
        ],
        is_preview,
    )


def create_dynamic_form_component(
        manager,
        demo: gr.Blocks,
        default_dataset,
        max_rows: int = 30
):

    """
    Build Dataset Button

    Args:
        manager (object): Configuration manager
        demo: Demo built via webui
        default_dataset: Default dataset
        max_rows: Number of rows to initialize with
    """

    columns = config.get_init_dataset_elem()
    language = manager.get_elem_by_id("basic", "language")

    def get_default_value(col_config):
        if col_config["type"] == "dropdown":
            return col_config["options"][0] if col_config["options"] else None
        elif col_config["type"] == "textbox":
            return ""
        return ""

    def get_next_available_row(form_data):
        if not form_data:
            return 0
        int_keys = [int(k) for k in form_data.keys()]
        return max(int_keys) + 1

    def get_display_to_actual_mapping(form_data):
        if not form_data:
            return {}
        sorted_rows = sorted([int(k) for k in form_data.keys()])
        return {i: sorted_rows[i] for i in range(len(sorted_rows))}

    def format_form_data_for_display(form_data):
        if not form_data:
            return ""

        import json
        try:
            return json.dumps(form_data, indent=2, ensure_ascii=False)
        except:
            return str(form_data)

    def initialize_default_datasets():
        form_data = {}
        dataset_info = config.get_dataset_info()

        if default_dataset and isinstance(default_dataset, list):
            for i, dataset_name in enumerate(default_dataset):
                if dataset_name in dataset_info:
                    dataset_config = dataset_info[dataset_name]
                    new_row = {
                        "col0": dataset_name,
                        "col1": dataset_config["type"],
                        "col2": dataset_config["path"],
                        "col3": dataset_config["prob"]
                    }
                    form_data[str(i)] = new_row
                else:
                    print(f"Warning: Dataset '{dataset_name}' not found in config, skipping...")
                    continue

        return form_data

    def initialize_form_rows():
        initial_form_data = initialize_default_datasets()

        visibility_states = []
        num_rows = len(initial_form_data) if initial_form_data else 0

        for i in range(max_rows):
            visibility_states.append(i < num_rows)

        all_component_updates = get_all_component_updates(initial_form_data)

        return [initial_form_data,
                format_form_data_for_display(initial_form_data)] + visibility_states + all_component_updates

    def update_data(form_data, actual_row_idx, col_name, value):
        if form_data is None:
            form_data = {}

        row_key = str(actual_row_idx)

        if row_key not in form_data:
            new_row = {}
            for i, col_config in enumerate(columns):
                new_row[f"col{i}"] = get_default_value(col_config)
            form_data[row_key] = new_row

        form_data[row_key][col_name] = value

        if col_name == "col0" and value != "Customization":
            dataset_info = config.get_dataset_info().get(value)
            if dataset_info is not None:
                form_data[row_key]["col1"] = dataset_info["type"]
                form_data[row_key]["col2"] = dataset_info["path"]
                form_data[row_key]["col3"] = dataset_info["prob"]
        elif col_name == "col0" and value == "Customization":
            form_data[row_key]["col2"] = ""


        return form_data

    def get_ui_updates_for_row(col_name, value):
        if col_name == "col0" and value != "Customization":
            dataset_info = config.get_dataset_info().get(value)
            if dataset_info is not None:
                ui_updates = [
                    gr.update(value=dataset_info["type"]),
                    gr.update(value=dataset_info["path"]),
                    gr.update(value=dataset_info["prob"])
                ]
            else:
                ui_updates = [gr.update(), gr.update(), gr.update()]
        elif col_name == "col0" and value == "Customization":
            ui_updates = [
                gr.update(),
                gr.update(value=""),
                gr.update(),
            ]

        else:
            ui_updates = [gr.update(), gr.update(), gr.update()]

        return ui_updates

    def get_all_component_updates(form_data):
        all_updates = []
        sorted_rows = sorted([int(k) for k in form_data.keys()]) if form_data else []

        for i in range(max_rows):
            row_updates = []
            for j, col_config in enumerate(columns):
                if i < len(sorted_rows):
                    actual_row_idx = sorted_rows[i]
                    row_key = str(actual_row_idx)
                    value = form_data[row_key][f"col{j}"]
                    row_updates.append(gr.update(value=value))
                else:
                    default_value = get_default_value(col_config)
                    row_updates.append(gr.update(value=default_value))
            all_updates.extend(row_updates)

        return all_updates

    def add_row(form_data, *visibility_checks):
        if form_data is None:
            form_data = {}

        if len(form_data) < max_rows:
            next_row_idx = get_next_available_row(form_data)
            row_key = str(next_row_idx)

            default_dataset_info = config.get_dataset_info()[config.get_default_dataset_name()]
            new_row = {}
            if default_dataset_info:
                new_row["col0"] = config.get_default_dataset_name()
                new_row["col1"] = default_dataset_info["type"]
                new_row["col2"] = default_dataset_info["path"]
                new_row["col3"] = default_dataset_info["prob"]
            else:
                for i, col_config in enumerate(columns):
                    new_row[f"col{i}"] = get_default_value(col_config)

            form_data[row_key] = new_row

            new_vis = list(visibility_checks)
            for i in range(len(new_vis)):
                if not new_vis[i]:
                    new_vis[i] = True
                    break

            all_component_updates = get_all_component_updates(form_data)

            return [form_data] + new_vis + all_component_updates

        all_component_updates = get_all_component_updates(form_data)
        return [form_data] + list(visibility_checks) + all_component_updates

    def delete_row_and_reset(form_data, actual_row_idx, *visibility_checks):
        if form_data is None:
            form_data = {}

        row_key = str(actual_row_idx)
        if row_key in form_data:
            del form_data[row_key]

        new_vis = list(visibility_checks)
        sorted_rows = sorted([int(k) for k in form_data.keys()]) if form_data else []

        for i in range(len(new_vis)):
            new_vis[i] = i < len(sorted_rows)

        all_row_updates = get_all_component_updates(form_data)

        return [form_data] + new_vis + all_row_updates

    def show_popup():
        return gr.Column(visible=True), gr.Column(visible=True)

    def save_and_hide(form_data):
        if form_data is not None and form_data:
            for row_key, row_data in form_data.items():
                dataset_name = row_data.get("col0", "")
                if dataset_name == "":
                    continue
                missing_fields = []

                if not row_data.get("col1"):
                    missing_fields.append("type")
                if not row_data.get("col2"):
                    missing_fields.append("path")
                if not row_data.get("col3"):
                    missing_fields.append("prob")

                if missing_fields:
                    missing_str = "、".join(missing_fields)
                    warnings_str = alert.get("dataset_none_data", "warning").format(dataset_name, missing_str)
                    gr.Warning(warnings_str)

        return gr.Column(visible=False), gr.Column(visible=False)

    def add_row_with_output(form_data, *visibility_checks):
        result = add_row(form_data, *visibility_checks)
        return [result[0], format_form_data_for_display(result[0])] + result[1:]

    def get_outputs():
        row_outputs = []
        for row in row_components:
            row_outputs.extend(row)

        return [
            add_dataset_btn,
            save_dataset_btn,
            dataset_tip,
            *delete_buttons,
            *row_outputs
        ]

    def update_language(language):
        add_dataset_btn_la = la.get(key="add_dataset_btn", lang=language, prop="value")
        save_dataset_btn_la = la.get(key="save_dataset_btn", lang=language, prop="value")
        delete_btn_la = la.get(key="delete_dataset_btn", lang=language, prop="value")
        dataset_tip_la = la.get(key="dataset_tip", lang=language, prop="value")

        delete_btn_updates = [gr.update(value=delete_btn_la) for _ in delete_buttons]
        row_updates = []

        for _ in row_components:
            row_update = [
                gr.update(label=la.get(key="dataset", lang=language, prop="label")),
                gr.update(label=la.get(key="dataset_type", lang=language, prop="label")),
                gr.update(label=la.get(key="dataset_path", lang=language, prop="label")),
                gr.update(label=la.get(key="dataset_prob", lang=language, prop="label")),
            ]
            row_updates.extend(row_update)

        return (
            gr.update(value=add_dataset_btn_la),
            gr.update(value=save_dataset_btn_la),
            gr.update(value=dataset_tip_la),
            *delete_btn_updates,
            *row_updates
        )

    form_data = gr.State(value={})
    dataset_btn = gr.Button(variant="primary", size="lg")
    output_textbox = gr.Textbox(visible=False)
    overlay = gr.Column(visible=False, elem_classes="modal-overlay-1")

    with gr.Column(visible=False, elem_classes="modal-box-1") as popup:
        dataset_tip = gr.Markdown()

        with gr.Row():
            add_dataset_btn = gr.Button(variant="primary")
            save_dataset_btn = gr.Button(variant="secondary")

        visibility_checks = []
        delete_buttons = []
        row_components = []

        for i in range(max_rows):
            vis_check = gr.Checkbox(value=False, visible=False)
            visibility_checks.append(vis_check)

            with gr.Row(visible=False, elem_classes="form-row-1") as row:
                row_cols = []
                for j, col_config in enumerate(columns):
                    if col_config["type"] == "dropdown":
                        col = gr.Dropdown(
                            choices=col_config["options"],
                            scale=col_config.get("scale", 1),
                            value=col_config["options"][0] if col_config["options"] else None
                        )
                    elif col_config["type"] == "textbox":
                        col = gr.Textbox(
                            scale=col_config.get("scale", 1),
                            value=""
                        )
                    row_cols.append(col)

                delete_btn = gr.Button(scale=1, size="sm", variant="stop")
                delete_buttons.append(delete_btn)
                row_components.append(row_cols)

            vis_check.change(
                fn=lambda val: gr.update(visible=val),
                inputs=[vis_check],
                outputs=[row]
            )

    for i, row_cols in enumerate(row_components):
        for j, col in enumerate(row_cols):
            def create_update_handler(display_idx, col_idx):
                def update_handler(data, val):
                    if not data:
                        return [data, format_form_data_for_display(data)] + [gr.update() for _ in range(3)]

                    mapping = get_display_to_actual_mapping(data)

                    if display_idx not in mapping:
                        return [data, format_form_data_for_display(data)] + [gr.update() for _ in range(3)]

                    actual_row_idx = mapping[display_idx]

                    updated_data = update_data(data, actual_row_idx, f"col{col_idx}", val)

                    if col_idx == 0:
                        ui_updates = get_ui_updates_for_row(f"col{col_idx}", val)
                        return [updated_data, format_form_data_for_display(updated_data)] + ui_updates
                    else:
                        return [updated_data, format_form_data_for_display(updated_data)] + [gr.update() for _ in range(3)]

                return update_handler

            col.blur(
                fn=create_update_handler(i, j),
                inputs=[form_data, col],
                outputs=[form_data, output_textbox, row_cols[1], row_cols[2], row_cols[3]]
            )

    for i, delete_btn in enumerate(delete_buttons):
        all_row_components = []
        for row_cols in row_components:
            all_row_components.extend(row_cols)

        def create_delete_handler(display_idx):
            def delete_handler(data, *vis):
                if not data:
                    return [data, format_form_data_for_display(data)] + list(vis) + [gr.update() for _ in
                                                                                     range(max_rows * len(columns))]

                mapping = get_display_to_actual_mapping(data)

                if display_idx not in mapping:
                    return [data, format_form_data_for_display(data)] + list(vis) + [gr.update() for _ in
                                                                                     range(max_rows * len(columns))]

                actual_row_idx = mapping[display_idx]
                result = delete_row_and_reset(data, actual_row_idx, *vis)
                return [result[0], format_form_data_for_display(result[0])] + result[1:]

            return delete_handler

        delete_btn.click(
            fn=create_delete_handler(i),
            inputs=[form_data] + visibility_checks,
            outputs=[form_data, output_textbox] + visibility_checks + all_row_components
        )

    all_row_components = []
    for row_cols in row_components:
        all_row_components.extend(row_cols)

    add_dataset_btn.click(
        fn=add_row_with_output,
        inputs=[form_data] + visibility_checks,
        outputs=[form_data, output_textbox] + visibility_checks + all_row_components
    )

    dataset_btn.click(
        show_popup,
        outputs=[overlay, popup]
    )

    save_dataset_btn.click(
        save_and_hide,
        inputs=[form_data],
        outputs=[overlay, popup]
    )

    demo.load(
        fn=initialize_form_rows,
        inputs=[],
        outputs=[form_data, output_textbox] + visibility_checks + all_row_components
    )

    language.change(
        fn=update_language,
        inputs=[language],
        outputs=get_outputs()
    )

    demo.load(
        fn=update_language,
        inputs=[language],
        outputs=get_outputs()
    )

    return {
        "output": output_textbox,
        "save_dataset_btn": save_dataset_btn,
        "dataset_btn": dataset_btn
    }
