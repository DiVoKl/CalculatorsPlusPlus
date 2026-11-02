import sympy as sp
from typing import Dict, Any
from .base_calculator import BaseCalculator

class Differentiation(BaseCalculator):
    def calculate(self, var=None, order: int = 1):
        """Calculates single variable derivatives or default derivatives wrt x."""
        target_var = var if var is not None else self.x
        return self.safe_execute(sp.diff, self.expr, target_var, order)

    def partial_derivative(self, var_name: str, order: int = 1):
        """Calculates partial derivative with respect to a target variable name (e.g., 'y')."""
        target_var = sp.Symbol(var_name)
        return self.safe_execute(sp.diff, self.expr, target_var, order)