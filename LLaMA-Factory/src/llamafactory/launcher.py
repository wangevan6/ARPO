# Copyright 2025 the LlamaFactory team.
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

import os
import sys

# Fix for Windows PyTorch distributed training libuv error
# Must be set before any torch imports and at the very beginning
os.environ["USE_LIBUV"] = "0"

# # Also set it as a command line argument to ensure it persists
# if "USE_LIBUV=0" not in str(sys.argv):
#     sys.argv.insert(1, "USE_LIBUV=0")

from llamafactory.train.tuner import run_exp  # use absolute import


def launch():
    run_exp()


if __name__ == "__main__":
    launch()
