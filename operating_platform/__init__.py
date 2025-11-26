"""
This file contains lists of available environments, dataset and policies to reflect the current state of LeRobot library.
We do not want to import all the dependencies, but instead we keep it lightweight to ensure fast access to these variables.

Example:
    ```python
        import operating_platform
        print(operating_platform.available_policies)
        print(operating_platform.available_robots)
    ```
"""

import itertools

from operating_platform.__version__ import __version__

available_policies = [
    "act",
    "diffusion",
    "tdmpc",
    "vqbet",
]

available_robots = [
    "aloha_v1",
    "adora_v1",
]
