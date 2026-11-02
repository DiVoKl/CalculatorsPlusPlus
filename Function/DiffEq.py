import sympy as sp
from typing import Dict, Any
from .base_calculator import BaseCalculator, safe_step


class DiffEq(BaseCalculator):
    def __init__(self, expr_str: str, symbols_str: str = "x y z t"):
        super().__init__(expr_str, symbols_str)
        self.f = sp.Function('f')(self.x)

    @safe_step
    def calculate(self, **kwargs) -> Dict[str, Any]:
        return self.solve_ode(**kwargs)

    def solve_ode(self, initial_condition: Dict[Any, Any] = None) -> Dict[str, Any]:
        try:
            ode_expr = self.expr.subs(self.y, self.f)
            eq = sp.Eq(self.f.diff(self.x), ode_expr)

            # Step 1: Attempt strict separable solve
            try:
                if initial_condition:
                    solution = sp.dsolve(eq, self.f, hint='separable', ics=initial_condition)
                else:
                    solution = sp.dsolve(eq, self.f, hint='separable')
                method = "Separable"
            # Step 2: Fallback to general solver for Linear and other 1st order ODEs
            except (ValueError, NotImplementedError):
                if initial_condition:
                    solution = sp.dsolve(eq, self.f, ics=initial_condition)
                else:
                    solution = sp.dsolve(eq, self.f)
                method = "General First-Order (e.g., Linear)"

            return {
                "equation": str(eq),
                "solution": str(solution),
                "latex": sp.latex(solution),
                "method": method,
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}