# instruments.py

from abc import ABC, abstractmethod
from datetime import date
from dataclasses import dataclass
from dateutil.relativedelta import relativedelta

class Instrument(ABC):
    """Abstract Base Class for all financial instruments."""
    @abstractmethod
    def __repr__(self):
        pass

@dataclass
class Bond(Instrument):
    """Represents a simple fixed-rate bond."""
    issuer: str
    notional: float
    coupon_rate: float
    maturity_date: date
    issue_date: date
    coupon_freq_months: int = 6 # Semi-annual coupons

    def get_cashflows(self, valuation_date: date) -> list[tuple[date, float]]:
        cashflows = []
        
        # Generate coupon payment dates
        current_date = self.issue_date
        while current_date < self.maturity_date:
            current_date += relativedelta(months=self.coupon_freq_months)
            if current_date > valuation_date:
                coupon_payment = self.notional * self.coupon_rate / (12 / self.coupon_freq_months)
                cashflows.append((current_date, coupon_payment))

        # Add the final principal + coupon payment
        if self.maturity_date > valuation_date:
            final_coupon = self.notional * self.coupon_rate / (12 / self.coupon_freq_months)
            cashflows.append((self.maturity_date, self.notional + final_coupon))

        return cashflows

    def __repr__(self):
        return f"Bond({self.issuer}, Mat: {self.maturity_date}, Cpn: {self.coupon_rate:.2%})"

@dataclass
class EuropeanOption(Instrument):
    """Represents a European-style vanilla option."""
    underlying_ticker: str
    option_type: str # 'Call' or 'Put'
    strike_price: float
    expiry_date: date

    def __post_init__(self):
        if self.option_type not in ['Call', 'Put']:
            raise ValueError("option_type must be 'Call' or 'Put'")

    def __repr__(self):
        return f"EuropeanOption({self.underlying_ticker}, K={self.strike_price}, T={self.expiry_date}, {self.option_type})"
