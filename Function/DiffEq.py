import sympy as sp
from typing import Dict, Any
from .base_calculator import BaseCalculator, safe_step


class DiffEq(BaseCalculator):
    def __init__(self, expr_str: str, symbols_str: str = "x y z t"):
        super().__init__(expr_str, symbols_str)
        self.f = sp.Function('f')(self.x)

    @safe_step
    def calculate(self, **kwargs) -> Dict[str, Any]:
        return self.solve_separable(**kwargs)

    def solve_separable(self, initial_condition: Dict[Any, Any] = None) -> Dict[str, Any]:
        try:
            ode_expr = self.expr.subs(self.y, self.f)
            eq = sp.Eq(self.f.diff(self.x), ode_expr)

            if initial_condition:
                solution = sp.dsolve(eq, self.f, hint='separable', ics=initial_condition)
            else:
                solution = sp.dsolve(eq, self.f, hint='separable')

            return {"equation": str(eq), "solution": str(solution), "latex": sp.latex(solution), "success": True}
        except NotImplementedError:
            return {
                "error": "SymPy could not solve this as a separable differential equation. It may not be separable.",
                "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}