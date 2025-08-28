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
Process execution management, initiation, and supervision
"""

import asyncio
import glob
import subprocess
import os
import re
import time
import threading
import pandas as pd
from typing import AsyncGenerator, Tuple, Optional

from visualdl import LogReader

import erniekit.webui.common as common
from erniekit.webui.alert import alert
from erniekit.webui.common import config


class CommandRunner:
    def __init__(self):
        self.current_process = None
        self.process_lock = asyncio.Lock()
        self.was_terminated_by_user = False
        self.lines_history = []
        self.track_progress = True
        self.current = 0
        self.total = 0
        self.percentage = 0
        self.progress_line_buffer = {}
        self.loss_tracker = LossTracker()
        self._loss_monitoring_active = False

        # 实时更新优化
        self.last_update_time = 0
        self.min_update_interval = 0.05
        self.force_update_threshold = 0.2
        self.line_count_since_update = 0
        self.max_lines_before_update = 3

    async def execute(self, command: str) -> AsyncGenerator[Tuple[str, float], None]:
        """
        Asynchronously execute a shell command and stream its output.

        Args:
            self: Instance reference
            command (str):  command to execute

        Returns:
            AsyncGenerator[str, None]: Asynchronous generator yielding output lines
        """
        self.lines_history = []
        self.progress_line_buffer = {}
        self.last_update_time = time.time()
        self.line_count_since_update = 0

        separator = "\n" + "-" * 50 + "\n"
        start_text = f"{separator}Executing: {command}\n"
        self.lines_history.append(start_text)
        self._loss_monitoring_active = True
        self.loss_tracker.start_monitoring()

        yield "\n".join(self.lines_history), 0
        print(start_text, flush=True)

        process = None
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["FORCE_COLOR"] = "1"

            process = await asyncio.create_subprocess_shell(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
            )
            self.current_process = process
            self.percentage = 0

            buffer = b""
            while True:
                chunk = await process.stdout.read(1024)
                if not chunk:
                    break

                buffer += chunk
                current_time = time.time()

                while b"\n" in buffer or b"\r" in buffer:
                    line, buffer = self._extract_next_line(buffer)
                    if not line:
                        break

                    line_str = line.decode("utf-8", errors="replace")
                    print(line_str, end="", flush=True)

                    line_clean = re.sub(r"\x1b\[[0-9;]*[mGKH]", "", line_str)
                    line_clean = line_clean.rstrip("\n\r").strip()

                    if line_clean:
                        should_update = self._process_line_realtime(line_clean)
                        self._parse_progress(line_clean)
                        self.line_count_since_update += 1

                        time_since_update = current_time - self.last_update_time

                        if (
                            should_update
                            or self.line_count_since_update
                            >= self.max_lines_before_update
                            or time_since_update > self.min_update_interval
                            or time_since_update > self.force_update_threshold
                        ):

                            yield (
                                "\n".join(self.lines_history),
                                self.compute_percentage(self.current, self.total),
                            )
                            self.last_update_time = current_time
                            self.line_count_since_update = 0

        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            self.lines_history.append(error_msg)
            print(error_msg, flush=True)
            yield (
                "\n".join(self.lines_history),
                self.compute_percentage(self.current, self.total),
            )

        finally:
            self._flush_progress_buffer()

            if process:
                return_code = await process.wait()
                if return_code == 0:
                    success_msg = "Command completed successfully!"
                    self.lines_history.append(f"\n{success_msg}")
                    print(f"\n{success_msg}", flush=True)
                    yield (
                        "\n".join(self.lines_history),
                        self.compute_percentage(self.current, self.total),
                    )

            self.current_process = None
            self.loss_tracker.stop_monitoring()

    def _process_line_realtime(self, line_clean: str) -> bool:
        """
        Extract a complete line from the buffer, handling both LF (\n) and CR+LF (\r\n) endings.

        Args:
            self: Instance reference
            buffer (bytes): Byte buffer containing partial or complete lines
        """
        progress_key = self._get_progress_key(line_clean)

        if progress_key:
            self.progress_line_buffer[progress_key] = line_clean
            self._update_progress_in_history(progress_key, line_clean)
            return True
        else:
            self.lines_history.append(line_clean)

            if self._is_important_line(line_clean):
                return True

            return False

    def _is_important_line(self, line: str) -> bool:
        """
        Determine if a line is important or not.
        Args:
            self: Instance reference
            line (str): Line to check
        Returns:
            bool: True if the line is important, otherwise False
        """
        important_keywords = [
            "error",
            "Error",
            "ERROR",
            "warning",
            "Warning",
            "WARNING",
            "failed",
            "Failed",
            "FAILED",
            "success",
            "Success",
            "SUCCESS",
            "complete",
            "Complete",
            "COMPLETE",
            "finished",
            "Finished",
            "FINISHED",
            "starting",
            "Starting",
            "STARTING",
            "epoch",
            "Epoch",
            "EPOCH",
            "step:",
            "Step:",
            "STEP:",
            "loss:",
            "Loss:",
            "LOSS:",
            "%",
        ]

        return any(keyword in line for keyword in important_keywords)

    def _get_progress_key(self, line: str) -> Optional[str]:
        """
        Get a progress key from a line and return it if it exists.
        Args:
            self: Instance reference
            line (str): Line to check
        Returns:
            str: Progress key if found, otherwise None
        """
        patterns = [
            r"(Loading\s+\w+):\s*\d+%",
            r"(\w+\s+\w+):\s*\d+%",
            r"(Epoch\s+\d+)",
            r"(Step\s+\d+)",
            r"(\d+%)",
            r"(\w+):\s*\d+/\d+",
            r"(Progress)",
            r"(Training)",
            r"(Validation)",
        ]

        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1)

        progress_keywords = ["progress", "epoch", "step", "batch", "%", "/", "loss"]
        for keyword in progress_keywords:
            if keyword in line.lower():
                words = line.split()
                if len(words) > 0:
                    return words[0]

        return None

    def _update_progress_in_history(self, progress_key: str, line: str):
        """
        Update the progress history with a new line.
        Args:
            self: Instance reference
            progress_key (str): Progress key to update
            line (str): Line to update
        Returns:
            None
        """
        for i, history_line in enumerate(self.lines_history):
            if progress_key in history_line and self._get_progress_key(history_line):
                self.lines_history[i] = line
                return

        self.lines_history.append(line)

    def _flush_progress_buffer(self):
        """
        Flush buffer and flush progress buffer.
        Args:
            self: Instance reference
        """
        for progress_key, line in self.progress_line_buffer.items():
            self._update_progress_in_history(progress_key, line)

    def _parse_progress(self, line: str):
        """
        Parse a line and return a progress key.
        Args:
            self: Instance reference
            line (str): Line to parse
        Returns:
            str: Progress key
        """
        try:
            patterns = [
                (
                    r"global_step:\s*(\d+)",
                    lambda m: setattr(self, "current", int(m.group(1))),
                ),
                (r"step:\s*(\d+)", lambda m: setattr(self, "current", int(m.group(1)))),
                (
                    r"(\d+)/(\d+)",
                    lambda m: (
                        setattr(self, "current", int(m.group(1))),
                        setattr(self, "total", int(m.group(2))),
                    ),
                ),
                (r"(\d+)%", lambda m: setattr(self, "percentage", int(m.group(1)))),
                (
                    r"epoch:\s*(\d+)",
                    lambda m: setattr(self, "current", int(m.group(1))),
                ),
            ]

            for pattern, action in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    action(match)
                    break

        except Exception:
            pass

    def _extract_next_line(self, buffer):
        """
        Extract the next line from the buffer.
        Args:
            self: Instance reference
            buffer (bytes): Buffer containing partial or complete lines
        """
        nl_pos = buffer.find(b"\n")
        cr_pos = buffer.find(b"\r")

        if nl_pos >= 0 and cr_pos >= 0:
            end_pos = min(nl_pos, cr_pos) + 1
        elif nl_pos >= 0:
            end_pos = nl_pos + 1
        elif cr_pos >= 0:
            end_pos = cr_pos + 1
        else:
            return buffer, b""

        return buffer[:end_pos], buffer[end_pos:]

    def compute_percentage(self, current: int, total: int) -> float:
        """
        Compute the percentage between current and total.
        Args:
            self: Instance reference
            current (int): Current progress
            total (int): Total progress
        Returns:
            float: Percentage
        """
        if total > 0:
            progress_ratio = current / total
            self.percentage = progress_ratio * 100
        return self.percentage

    async def stop(self):
        """
        Terminate the currently running process asynchronously.

        Args:
            self: Instance reference
        """
        async with self.process_lock:
            process = self.current_process

            if process is None:
                no_terminated_msg = "\n" + alert.get("progress", "no_progress") + "\n"
                self.lines_history.append(no_terminated_msg)
                return "\n".join(self.lines_history)

            try:
                if process.returncode is not None:
                    progress_end_msg = (
                        "\n" + alert.get("progress", "progress_end") + "\n"
                    )
                    self.lines_history.append(progress_end_msg)
                    return "\n".join(self.lines_history)

                try:
                    common.abort_process(process.pid)
                except Exception:
                    process.terminate()

                await asyncio.sleep(0.5)

                if process.returncode is None:
                    process.kill()
                    force_terminated_msg = (
                        "\n" + alert.get("progress", "force_terminated") + "\n"
                    )
                    print(force_terminated_msg)
                    self.lines_history.append(force_terminated_msg)
                    await process.wait()

                self.was_terminated_by_user = True
                user_terminated_msg = (
                    "\n" + alert.get("progress", "user_terminated") + "\n"
                )
                self.lines_history.append(user_terminated_msg)
                print(user_terminated_msg)
            except Exception as e:
                error_msg = alert.get("progress", "terminate_error").format(str(e))
                self.lines_history.append(error_msg)
                print(error_msg.strip())
            finally:
                self.current_process = None

            return "\n".join(self.lines_history)

    def clear_output(self):
        """
        Clear the current output buffer.

        Args:
            self: Instance reference

        Returns:
            str: Empty string indicating successful clearing
        """
        self.output_reset = True
        self.current_output = ""
        self.progress_line_buffer = {}
        return ""

    def is_running(self) -> bool:
        """
        Check if there is an active process being executed.

        Args:
            self: Instance reference

        Returns:
            bool: True if there is an active process, False otherwise
        """
        process = self.current_process
        return process is not None and process.returncode is None

    def get_plot(self):
        """
        Retrieve the plot data.

        Args:
        self: Instance reference

        Returns:
        The plot data obtained from the loss tracker.
        """

        self.loss_tracker.update_loss_config()
        return self.loss_tracker.get_plot_data()

    def is_loss_monitoring_active(self):
        """
        Check if loss monitoring is active.

        Args:
        self: Instance reference

        Returns:
        bool: True if loss monitoring is active and there is an active process, False otherwise.
        """
        return self._loss_monitoring_active and self.is_running()


class LossTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.log_path = None
        self.log_module = None
        self.log_tag = None
        self.loss_history = []
        self.step_history = []
        self.monitoring_task = None
        self.is_monitoring = False
        self.last_logging_path = None
        self.latest_plot_data = pd.DataFrame({"Step": [0], "Loss": [0]})

    def start_monitoring(self):
        """
        Start the loss monitoring coroutine.

        Args:
        self: Instance reference

        Returns:
        None
        """
        if self.monitoring_task is None or self.monitoring_task.done():
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    def stop_monitoring(self):
        """
        Stop the loss monitoring coroutine and clear history data.

        Args:
        self: Instance reference

        Returns:
        None
        """
        self.is_monitoring = False
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        self.clear_history_data()

    async def _monitoring_loop(self):
        """
        Background monitoring loop that periodically reads plot data.

        Continuously checks for new data while monitoring is active,
        with a 3-second interval between checks.

        Args:
        self: Instance reference

        Returns:
        None
        """

        try:
            while self.is_monitoring:
                try:
                    plot_data = self._read_plot_data()
                    if plot_data is not None:
                        self.latest_plot_data = plot_data
                except Exception as e:
                    print("Loss Tracker Error: ", e)
                await asyncio.sleep(3)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print("Loss Tracker Error: ", e)
        finally:
            self.is_monitoring = False

    def update_loss_config(self):
        """
        Update log configuration parameters including path, module and tag.

        Retrieves user configurations and sets default values if none are provided.
        Automatically finds the latest log file if no path is specified.

        Args:
        self: Instance reference

        Returns:
        None
        """

        user_log_path = config.get_default_user_dict("basic", "log_path")
        user_log_module = config.get_default_user_dict("basic", "log_module")
        user_log_tag = config.get_default_user_dict("basic", "log_tag")
        logging_dir = config.get_default_user_dict("train", "logging_dir")

        try:
            if self.log_path is None and logging_dir != self.last_logging_path:
                if user_log_path is None:
                    search_pattern = os.path.join(logging_dir, "*.log")
                    log_files = glob.glob(search_pattern)

                    if log_files:
                        latest_file = max(log_files, key=os.path.getmtime)
                        self.log_path = os.path.abspath(latest_file)
                        self.last_logging_path = logging_dir
                    else:
                        self.log_path = None
                else:
                    self.log_path = os.path.join(logging_dir, user_log_path)
                    self.last_logging_path = logging_dir

            if user_log_module is None:
                self.log_module = "scalar"
            else:
                self.log_module = user_log_module

            if user_log_tag is None:
                self.log_tag = "train/loss"
            else:
                self.log_tag = user_log_tag
        except Exception:
            self.log_path = None
            self.log_module = "scalar"
            self.log_tag = "train/loss"

    def _read_plot_data(self):
        """
        Read and process plot data from the log file (internal method)

        Reads log data using the configured parameters and returns it in
        a DataFrame format with steps and corresponding loss values.

        Args:
        self: Instance reference

        Returns:
        pandas.DataFrame: DataFrame containing "Step" and "Loss" columns.
        Returns default empty data if log path is None or error occurs.
        """

        try:
            if self.log_path is None:
                return pd.DataFrame({"Step": [0], "Loss": [0]})

            reader = LogReader(file_path=self.log_path)
            data = reader.get_data(self.log_module, self.log_tag)

            with self.lock:
                self.loss_history = []
                self.step_history = []

                for item in data:
                    value = item.value
                    step = item.id
                    self.loss_history.append(value)
                    self.step_history.append(step)

                if not self.loss_history:
                    return pd.DataFrame({"Step": [0], "Loss": [0]})

                return pd.DataFrame(
                    {"Step": self.step_history, "Loss": self.loss_history}
                )

        except Exception as e:
            print("Loss Tracker Error: ", e)
            return pd.DataFrame({"Step": [0], "Loss": [0]})

    def get_plot_data(self):
        """
        Retrieve the latest plot data.

        Args:
        self: Instance reference

        Returns:
        pandas.DataFrame: The most recently updated plot data containing
        "Step" and "Loss" columns.
        """

        return self.latest_plot_data

    def clear_history_data(self):
        """
        Clear all historical data and reset log path and plot data.

        Uses a lock to ensure thread-safe data clearing.

        Args:
        self: Instance reference

        Returns:
        None
        """

        with self.lock:
            self.step_history = []
            self.loss_history = []
            self.latest_plot_data = self.get_plot_data()

    def reset_latest_plot_data(self):
        self.latest_plot_data = pd.DataFrame({"Step": [0], "Loss": [0]})
        self.log_path = None
