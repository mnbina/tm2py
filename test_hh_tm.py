from tm2py.controller import RunController
base_configs = ["examples/model_config.toml","examples/scenario_config.toml"]
my_run = RunController(base_configs, run_dir='examples/UnionCity', run_components='household')
my_run.run_next()