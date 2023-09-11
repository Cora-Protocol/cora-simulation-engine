class ExistingNameError(Exception):
    def __init__(self):
        self.message = "Name already exists"


class InsufficientLiquidityError(Exception):
    def __init__(self):
        self.message = "Insufficient Liquidity"


class InsufficientBalanceError(Exception):
    def __init__(self):
        self.message = "Insufficient Balance"


class InsuficientCollateralError(Exception):
    def __init__(self, collateral_balance: float, collateral_needed: float):
        self.message = f"Insufficient Collateral to reach the max allowed LTV: {collateral_balance} < {collateral_needed}"
        super().__init__(self.message)


class NonExistingBorrowerAddress(Exception):
    def __init__(self):
        self.message = "Wallet has no outstanding loans"


class InvalidLoanId(Exception):
    def __init__(self):
        self.message = "Loan ID is invalid for borrower address"


class LoanAmountTooLow(Exception):
    def __init__(self):
        self.message = "Loan amount is too low"


class InvalidLoanPeriodLong(Exception):
    def __init__(self):
        self.message = "Requested loan period is too long"


class InvalidLoanPeriodShort(Exception):
    def __init__(self):
        self.message = "Requested loan period is too short"


class LoanExpiredError(Exception):
    def __init__(self):
        self.message = "Loan has expired"


class LendingPoolNotRunning(Exception):
    def __init__(self):
        self.message = "Lending pool is not running, cannot call this function"


class LendingPoolRunning(Exception):
    def __init__(self):
        self.message = "Lending pool is running"
