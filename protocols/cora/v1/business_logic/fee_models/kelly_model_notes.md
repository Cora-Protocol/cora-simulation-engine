# Kelly model notes

Notes for [Cora-Protocol/pricing-simulator](https://github.com/Cora-Protocol/pricing-simulator)

## Understanding [2_kelly_price_simulator](https://github.com/Cora-Protocol/pricing-simulator/blob/9ec54bd64d8062c9184545ef5648f606b34d01c5/2_kelly_price_simulator.ipynb)

 * Config (`libs.curve_config.curve_config.Config`) for operation has:
   * coin_id (coingecko)
   * number of days of history to download and use
   * strike percentage = 0.8
   * expiration = 30 / 28
   * training label = "bear" / "PastYear" / "bull" / "Full"
 * Start with downloading historical data
   * `hd = libs.historic_downloader.historic_downloader.HistoricDownloader()`
   * `hd.download_history_file(coin_id="weth", number_of_days=365)`
   * This uses the Coingecko API through `pycoingecko` 
   * Outputs a CSV files with columns "Master calendar" and "{coin_id}"
 * Using the history and config, creates an input file ([`libs.utils.create_input_file`](https://github.com/Cora-Protocol/pricing-simulator/blob/9ec54bd64d8062c9184545ef5648f606b34d01c5/libs/utils/create_input_file.py#L4))
   * Reads the history file
   * Creates a single row CSV file with the following data:
     * Asset: coin_id
     * TrainingLabel: "bear" / "PastYear" / "bull" / "Full"
     * TrainingStart: 0
     * TrainingEnd: length of history file
     * StrikePct: strike percentage from before, like 0.8. It is ratio of ATM price 0.8 is considered 20% OTM (so no strick moneyness definition here for the Put)
     * Expiration: like above 30 / 28. Still don't know what it means exactly, but I guess it is the expiration time of the option
     * CurrentPrice: Last value in the time series
  * Instantiate a Generator ([`libs.curve_gen.gen.Generator`](https://github.com/Cora-Protocol/pricing-simulator/blob/9ec54bd64d8062c9184545ef5648f606b34d01c5/libs/curve_gen/gen.py#L289)), no input args
  * Build generator config [`libs/curve_gen/utils.build_generator_config`](https://github.com/Cora-Protocol/pricing-simulator/blob/9ec54bd64d8062c9184545ef5648f606b34d01c5/libs/curve_gen/utils.py#L153) using the paths of the previous two files as input (SMH)
    * There is a GeneratorConfigBuilder within this and it's not even independent fuck me
    * There are three parts in the configuration, Training, Constraints, and Fit
    * There is a builder for each of them, why not
      * Training:
        * This fucking loads the CSV files as fucking DataFrames what the fucking fuck
        * Literally does nothing else
        * Fuck these people
      * Constraints:
        * This stores two list of functions, `upper_bounds` and `lower_bounds`. Each function is suposed to be univariate. 
        * By default, single element with constant functions which return 1.0 and 1e-10 respectively
      * FitConfig stores just a `fit_type` parameter, which is always `"COSH"`, unless you change it, which they don't. 
        * Looking more deep it can be one of: fit_function_table = {'EXP': _fit_exp,'POLY': _fit_poly,'COSH': _fit_cosh}. 
        * Those values are a function like this: `_fit_cosh(bet_fractions: np.ndarray, premiums: np.ndarray, maxfev=100000, lower_bounds=(0.0, 0.0, 0.0, 0.0), upper_bounds=(100.0, 100.0, 100.0, 100.0)):`
        * This just looks for a hiperbolic cosine based function for which to fit the Xs and Ys points from the Kelly Curve (X are bet fractions, Y are premiums), using `scipy.optimize.curve_fit`
        * Not called at this point, just chooses which kind of curve to try to fit later on
  * Call `configure_curve_gen` method in the Generator with the generator config as input
    * This just stores the config from earlier and updates some weird global objects with the actual business logic. This repo is great at following worst practices
  * Everything until this point has been downloading data and cofiguring literally 5 parameters, jesus fucking christ
  * Call `generate_curves` method, which outputs results as dataframes with the a, b, c, d parameters
    * First, it does `conv_dfs, conv_configs, payoff_configs = _perform_training(*initial_guess, dist=dist, payoff_dict=payoff_dict)`
      * The actual output is the first list of dfs, the rest are derived from it
      * This is creates something referred to as a convolution group
      * This apparently only depends on the historical data used (asset, label, start, end)
      * 

