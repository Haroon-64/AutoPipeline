import subprocess
import sys
from typing import List

class ManageEnv:
    def __init__(self,tools,name,env_path,requirements: List[str]):
        self.tools = tools
        self.name = name
        self.env_path = env_path
        self.requirements = requirements

    def create_env(self):
        if "conda" in self.tools:
            self._create_conda_env(self.name, self.env_path, self.requirements)
        elif "pip" in self.tools:
            self._create_pip_env(self.name, self.env_path, self.requirements)
        elif "poetry" in self.tools:
            self._create_poetry_env(self.name, self.env_path, self.requirements)
        else:
            raise Exception("No valid tools provided")
        
    def _create_conda_env(self, name, env_path, requirements):
        subprocess.run(["conda", "create", "--name", name, "--prefix", env_path, "python=3.8", "-y"])
        self._install_requirements("conda", env_path, requirements)

    def _create_pip_env(self, name, env_path, requirements):
        subprocess.run(["python", "-m", "venv", env_path])
        self._install_requirements("pip", env_path, requirements)

    def _create_poetry_env(self, name, env_path, requirements):
        subprocess.run(["poetry", "new", env_path])
        self._install_requirements("poetry", env_path, requirements)

    def _install_requirements(self, tool, env_path, requirements):
        if tool == "conda":
            subprocess.run(["conda", "install", "--name", self.name, "--prefix", self.env_path, "-y"] + requirements)
        elif tool == "pip":
            subprocess.run([f"{env_path}/bin/pip", "install"] + requirements)
        elif tool == "poetry":
            subprocess.run(["poetry", "add"] + requirements)

if __name__ == "__main__":
    tools = sys.argv[1]
    name = sys.argv[2]
    env_path = sys.argv[3]
    requirements = sys.argv[4:]
    ManageEnv(tools, name, env_path, requirements).create_env()
        