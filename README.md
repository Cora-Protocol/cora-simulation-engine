# Cora Protocol Simulator

This is a Work-In-Progress.

This repository contains the code necessary for simulating the behaviour of the Cora DeFi Protocol using a Python replica and an agent-based environment.

## Organization

 * [`simulator`](simulator) contains the base code for the simulation engine
 * [`protocols/cora`](protocols/cora) contains the specific business logic for the Cora Protocol
 * [`libs`](libs) contains additional dependencies not externally available. In this case, code for pricing options through the Kelly Criterion
 * [`data`](data) contains cached time series data for cryptocurrency spot values used for the simulations
 * [`apis`](apis) contains interactors for external APIs used to get data
 * [`simlogs`](simlogs) is the default folder for simulation logs
 * [`studies`](studies) is the location for storing the studies performed by using this simulator

## How to
### How to run a Monte Carlo simulation study

 1. Install the dependencies using Pipenv: `pipenv install`, or `pipenv install --dev` if you want to develop.
 2. Make sure you have the project path in `$PYTHONPATH`: `export PYTHONPATH=${PYTHONPATH}:${PWD}`.
 3. Define the conditions for your simulation and run it. An example is available [here](example/run_monte_carlo_study.ipynb).

### How to find the maximum LTV for a Cora lending pool

 1. Install the dependencies using Pipenv: `pipenv install`.
 2. Make sure you have the project path in `$PYTHONPATH`: `export PYTHONPATH=${PYTHONPATH}:${PWD}`.
 3. Define your risk pofile, the pool fee model, and any other parameters, in `example/find_ltv.ipynb`.
 4. Run the notebook. If equivalent simulations have already been run and cached, the process is quick. Otherwise, it will take a while.

### How to add a new price model

Fee models are subclasses of [`protocols.cora.v1.business_logic.fee_models.base_fee_model.BaseFeeModel`](protocols/cora/v1/business_logic/fee_models/base_fee_model.py). The `BaseFeeModel` class defines the interface for fee models, and the [`protocols.cora.v1.business_logic.fee_model.black_scholes_model.BlackScholesFeeModel`](protocols/cora/v1/business_logic/fee_models/black_scholes_model.py) class is a simple example of a fee model.

In order to add a new fee model:

 1. Create a new class that inherits from `BaseFeeModel` and implements its abstract methods. These methods define the fee calculation as well as, optionally, the algorithm used by the pool manager that updates the fee model parameters as the market conditions evolve.
 2. Add a reference to the new fee model class in the [`protocols.cora.v1.strategies.FEE_MODELS`](protocols/cora/v1/strategies.p) dictionary, giving it a name that can be used to refer to it in the simulation configuration.

In order to use this new fee model, you pass its reference to the `fee_model` parameter in the examples. You will also need to pass a dictionary with the parameters for the update algorithm, if any, in the `fee_model_update_params` parameter. If there are no update parameters, pass an empty dictionary.
### Tests

You can simply run the current suite of tests with `pipenv run pytest`


