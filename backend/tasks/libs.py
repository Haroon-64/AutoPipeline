import os
import subprocess
import sys
from typing import List

def install_libraries(task):
    libraries = {"image": ""}

    for lib in libraries.task:
        subprocess.call(["pip", "install", lib])


libs_ = []
install_libraries(libs_)
