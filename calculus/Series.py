import sympy as sp
from typing import Dict, Any
from common.base_calculator import BaseCalculator, safe_step


class Series(BaseCalculator):
    @safe_step
    def calculate(self, **kwargs) -> Dict[str, Any]:
        return self.taylor_expansion(**kwargs)

    @safe_step
    def taylor_expansion(self, center=0, order=6) -> Dict[str, Any]:
        expansion = sp.series(self.expr, self.x, center, order).removeO()
        return {"text": str(expansion), "latex": sp.latex(expansion), "sympy_obj": expansion, "success": True}

    @safe_step
    def test_convergence(self, n_var: str = 'n') -> Dict[str, Any]:
        n = sp.Symbol(n_var, integer=True, positive=True)

        nth_term_limit = sp.limit(self.expr, n, sp.oo)
        if nth_term_limit != 0:
            return {"result": "Diverges", "reason": f"Limit of n-th term is {nth_term_limit} (not 0)", "success": True}

        a_n = self.expr
        a_n_plus_1 = a_n.subs(n, n + 1)
        ratio = sp.simplify(sp.Abs(a_n_plus_1 / a_n))
        L = sp.limit(ratio, n, sp.oo)

        if L < 1:
            status = "Converges Absolutely"
        elif L > 1:
            status = "Diverges"
        else:
            status = "Inconclusive (Ratio test limit is 1, further testing needed)"

        return {"result": status, "ratio_limit": str(L), "success": True}

    @safe_step
    def radius_and_interval(self, n_var: str = 'n') -> Dict[str, Any]:
        n = sp.Symbol(n_var, integer=True, positive=True)
        a_n = self.expr
        a_n_plus_1 = a_n.subs(n, n + 1)

        ratio = sp.simplify(sp.Abs(a_n_plus_1 / a_n))
        ratio_limit = sp.limit(ratio, n, sp.oo)

        if ratio_limit == 0:
            radius, interval = "oo", "(-oo, oo)"
        elif ratio_limit == sp.oo:
            radius, interval = "0", "Converges only at the center"
        else:
            try:
                interval_cond = sp.solve(ratio_limit < 1, self.x)
                interval, radius = str(interval_cond), "Calculate from interval bounds"
            except:
                interval, radius = f"Solve {ratio_limit} < 1 manually", "Unknown"

        return {"ratio_limit": str(ratio_limit), "interval_condition": interval, "radius": str(radius), "success": True}

    @safe_step
    def term_by_term_calculus(self) -> Dict[str, Any]:
        diff_term = sp.diff(self.expr, self.x)
        int_term = sp.integrate(self.expr, self.x)
        return {"derivative": str(diff_term), "integral": str(int_term), "success": True}