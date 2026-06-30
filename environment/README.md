## For container running with `pytorch 2.7` and `dig`
1. Use the def file `oneprot-panda-pytorch-25-03-py3.def`, in the `requirements_pyg_25_03.txt` comment out `dive-into-graphs` and `rdkit-pypi`
2. Create an env (I used `sc_venv_template` without modules) using `requirements_env_dig.txt` file (within the container)
3. activate the env within the container and run `pip install dive-into-graphs --no-dependencies`
