# Backend For Pipelines

- generate code for ML tasks using templates
  - tasks✔️
  - data processing✔️
  - model definition✔️
  - training configuration✔️
- communicate with UI using FastAPI
  - receive config✔️
  - send back generated code✔️
  - send training command, inference results
- run training or inference scripts and return result to UI

## Running the server

- install uv with `brew install uv` or `python -m pip install uv`
- sync dependencies `uv sync` from root folder of repo
- activate venv `source .venv/bin/activate` (/activate.<shell_ext> if present in .venv/bin)
- run server `python modules/main.py`
- check api at `http://127.0.0.1:8000/docs`

- or
- run the release `./main`

## TODO: Build into a cross platform binary

- build one using pyinstaller, need to build on other OS ✔️
- Need to bundle witht The UI using tauri  side-car ✔️* (needs system libs/venv for running)

## TODO: Pending functionalities

1. run the generated code using the UI instead of needing to manually save the file and run
2. Test the server, currently the App only builds something that is basically a boilerplate builder,

- - need to test with different configs, datasets, and actually building the dataloaders.

3. convert the fastAPI app into rust based. there isnt much benefit of using python here.
4. the dataloaders specifically have too much variation, trying to only
   use subset like other tasks isnt viable as you'd be either recreating the whole pytorch dataset or hardcode everything which defeats the point of a generic expandable system


## adding capabilities 
To add extra functions, add them as a jinja file in templates/{function_dir},

and make another python version in modules/configs/registries/

rarely you might need to modify configs in modules/configs.


for UI, just add the function similarly as typescript as in [client repo](https://github.com/suhaib-us/PytorchForge/tree/main/client%2Fapp%2Froutes%2Fnew-pipeline.dl._index%2Fsections)