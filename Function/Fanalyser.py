import sympy as sp
from sympy.calculus.util import continuous_domain
from typing import Dict, Any
from .base_calculator import BaseCalculator
from .Differentiation import Differentiation
from .Limits import Limits


class Fanalyser(BaseCalculator):
    def __init__(self, expr_str: str, symbols_str: str = "x y z t"):
        super().__init__(expr_str, symbols_str)
        self.diff_engine = Differentiation(expr_str, symbols_str)
        self.limit_engine = Limits(expr_str, symbols_str)

    def calculate(self) -> Dict[str, Any]:
        try:
            domain = continuous_domain(self.expr, self.x, sp.S.Reals)
            roots = sp.solve(self.expr, self.x)

            d1 = self.diff_engine.calculate(var=self.x, order=1)
            critical_points = sp.solve(d1["sympy_obj"], self.x) if d1["success"] else []

            return {
                "domain": str(domain),
                "roots": [str(r) for r in roots],
                "critical_points": [str(cp) for cp in critical_points],
                "first_derivative": d1.get("text", ""),
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}