tests:
	pipenv run pytest

pythonpath:
	export PYTHONPATH=${PYTHONPATH}:${PWD}