import os
import venv
def create_or_load_env(env_name="myenv"):
    if not os.path.exists(env_name):
        print(f"Creating a new virtual environment: {env_name}")
        venv.create(env_name, with_pip=True)
    else:
        print(f"Virtual environment '{env_name}' already exists.")

    
    activate_script = os.path.join(env_name, 'Scripts', 'activate') if os.name == 'nt' else os.path.join(env_name, 'bin', 'activate')

    if os.name == 'nt':
        activate_command = f"{env_name}\\Scripts\\activate"
    else:
        activate_command = f"source {env_name}/bin/activate"

    print(f"To activate the virtual environment, run:\n{activate_command}")

def install_libs(env):
    pass

if __name__ == "__main__":
    create_or_load_env("myenv")

