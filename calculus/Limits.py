import sympy as sp
from common.base_calculator import BaseCalculator

class Limits(BaseCalculator):
    def calculate(self, point, var=None, direction="+"):
        """Evaluates targeted variable limits from the specified direction ('+' or '-')."""
        target_var = var if var is not None else self.x
        return self.safe_execute(sp.limit, self.expr, target_var, point, dir=direction)