touch log.txt
nohup pipenv run python studies/005-monte-carlo/montecarlo_dashboard_sims.py --start_type $1 --duration_type $2 --fee_model bsm --volatility_factor $3 >> log.txt 2>&1 &
nohup pipenv run python studies/005-monte-carlo/montecarlo_dashboard_sims.py --start_type $1 --duration_type $2 --fee_model kelly --volatility_factor $3 >> log.txt 2>&1 &
nohup pipenv run python studies/005-monte-carlo/montecarlo_dashboard_sims.py --start_type $1 --duration_type $2 --fee_model aave --volatility_factor $3 >> log.txt 2>&1 &
nohup pipenv run python studies/005-monte-carlo/montecarlo_dashboard_sims.py --start_type $1 --duration_type $2 --fee_model trad --volatility_factor $3 >> log.txt 2>&1 &
nohup pipenv run python studies/005-monte-carlo/montecarlo_dashboard_sims.py --start_type $1 --duration_type $2 --fee_model bsmaavecombo --volatility_factor $3 >> log.txt 2>&1 &
sleep 5
nohup pipenv run python studies/005-monte-carlo/montecarlo_dashboard_sims.py --start_type $1 --duration_type $2 --fee_model bsmaavesum --volatility_factor $3 >> log.txt 2>&1 &
nohup pipenv run python studies/005-monte-carlo/montecarlo_dashboard_sims.py --start_type $1 --duration_type $2 --fee_model bsmtradcombo --volatility_factor $3 >> log.txt 2>&1 &
nohup pipenv run python studies/005-monte-carlo/montecarlo_dashboard_sims.py --start_type $1 --duration_type $2 --fee_model bsmtradsum --volatility_factor $3 >> log.txt 2>&1 &
nohup pipenv run python studies/005-monte-carlo/montecarlo_dashboard_sims.py --start_type $1 --duration_type $2 --fee_model kellytradcombo --volatility_factor $3 >> log.txt 2>&1 &
nohup pipenv run python studies/005-monte-carlo/montecarlo_dashboard_sims.py --start_type $1 --duration_type $2 --fee_model kellytradsum --volatility_factor $3 >> log.txt 2>&1 &