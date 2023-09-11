from simulation_setup import run_study

if __name__ == "__main__":
    run_study(
        max_ltv_options=[0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        fee_model_options=[
            "bsm",
            "kelly",
            "trad",
            "aave",
            "bsmaavecombo",
            "bsmaavesum",
            "bsmtradcombo",
            "bsmtradsum",
            "kellytradcombo",
            "kellytradsum",
        ],
        borrower_demand_options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        loan_start_options=["parabolic"],
        loan_duration_options=["parabolic"],
        volatility_factor_options=[1.0],
        num_seeds=20,
        base_seed=295739,
    )
