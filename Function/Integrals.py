import sympy as sp
from .base_calculator import BaseCalculator

class Integrals(BaseCalculator):
    def calculate(self, var=None, lower=None, upper=None):
        """Computes indefinite integrals or definite integrals when bounds are supplied."""
        target_var = var if var is not None else self.x
        if lower is not None and upper is not None:
            return self.safe_execute(sp.integrate, self.expr, (target_var, lower, upper))
        return self.safe_execute(sp.integrate, self.expr, target_var)