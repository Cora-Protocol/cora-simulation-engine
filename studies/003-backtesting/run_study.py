from generate_study_backtesting import generate_study


if __name__ == "__main__":
    study_results = generate_study(
        ltv=0.5,
        fee_model="black_scholes",
        fee_model_update_params={"lookback_days": 365, "volatility_factor": 1.0},
        utilization_param_array=[0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
        loan_start_type_options=["uniform"],
        loan_duration_type_options=["uniform", "triangular", "parabolic"],
        num_start_times=5,
        random_seed=0,
    )

    study_results.to_csv("results.csv")
