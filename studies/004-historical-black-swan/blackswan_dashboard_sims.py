import argparse
from generate_study_blackswan import run_and_save_study_array
from functools import partial

# What do you want to run?
# start_type = "uniform"  # "uniform" or "parabolic"
# duration_type = "parabolic"  # "uniform" or "parabolic"
# fee_model = "bsm"  # bsm kelly aave trad bsmaavecombo bsmaavesum bsmtradcombo bsmtradsum kellytradcombo kellytradsum


parser = argparse.ArgumentParser(description="Run a blackswan simulation")
parser.add_argument(
    "--start_type", type=str, required=True, choices=["uniform", "parabolic"]
)
parser.add_argument(
    "--duration_type", type=str, required=True, choices=["uniform", "parabolic"]
)
parser.add_argument(
    "--fee_model",
    type=str,
    required=True,
    choices=[
        "bsm",
        "kelly",
        "trad",
        "bsmaavecombo",
        "bsmaavesum",
        "bsmtradcombo",
        "bsmtradsum",
        "kellytradcombo",
        "kellytradsum",
    ],
)

args = parser.parse_args()

num_randoms = 60
max_ltv_options = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
utilization_param_options = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]

folder = f"studies/004-blackswan/start_{args.start_type}_duration_{args.duration_type}_x{num_randoms}/"

run = partial(
    run_and_save_study_array,
    save_location=folder,
    max_ltvs=max_ltv_options,
    utilization_params=utilization_param_options,
    loan_start_type=args.start_type,
    loan_duration_type=args.duration_type,
    num_randoms=num_randoms,
    random_seed=52324,
    cancel_if_exists=True,
    continue_if_failed=True,
    compress_to_zip=True,
)

if __name__ == "__main__":
    run(fee_model=args.fee_model)
