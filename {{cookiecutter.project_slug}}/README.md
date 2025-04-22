# {{cookiecutter.project_slug}}

We recommend: 
- install `pixi`
- run `pixi install` in {{cookiecutter.project_slug}} 
- modify the `config` to match your requirements: 
    - specifically, add paths to the data
- check your configuration by running `pixi run notebooks/test_config.py` -- make sure that this runs without errors
- use `pixi run scripts/01_split.py` ... etc to run