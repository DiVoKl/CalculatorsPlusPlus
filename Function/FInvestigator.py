import sympy as sp
from sympy.calculus.util import continuous_domain
from typing import Dict, Any
from .base_calculator import BaseCalculator
from .Differentiation import Differentiation
from .Limits import Limits

class FInvestigator(BaseCalculator):
    def __init__(self, expr_str: str, symbols_str: str = "x y z t"):
        super().__init__(expr_str, symbols_str)
        self.diff_engine = Differentiation(expr_str, symbols_str)
        self.limit_engine = Limits(expr_str, symbols_str)

    def calculate(self, **kwargs) -> Dict[str, Any]:
        try:
            # 1. Domain & Intercepts
            domain = continuous_domain(self.expr, self.x, sp.S.Reals)
            y_intercept = self.expr.subs(self.x, 0) if 0 in domain else "Undefined"
            x_intercepts = sp.solve(self.expr, self.x)

            # 2. Symmetry Check
            f_neg_x = self.expr.subs(self.x, -self.x)
            if sp.simplify(f_neg_x - self.expr) == 0:
                symmetry = "Even (Symmetric about y-axis)"
            elif sp.simplify(f_neg_x + self.expr) == 0:
                symmetry = "Odd (Symmetric about origin)"
            else:
                symmetry = "Neither Even nor Odd"

            # 3. Limits / Horizontal Asymptotes
            lim_pos = self.limit_engine.calculate(sp.oo, var=self.x)
            lim_neg = self.limit_engine.calculate(-sp.oo, var=self.x)

            # 4. First Derivative & Critical Points
            d1_res = self.diff_engine.calculate(var=self.x, order=1)
            d1_expr = d1_res.get("sympy_obj") if d1_res.get("success") else None
            critical_points = sp.solve(d1_expr, self.x) if d1_expr is not None else []

            # 5. Second Derivative & Inflection Points
            d2_res = self.diff_engine.calculate(var=self.x, order=2)
            d2_expr = d2_res.get("sympy_obj") if d2_res.get("success") else None
            inflection_points = sp.solve(d2_expr, self.x) if d2_expr is not None else []

            return {
                "domain": str(domain),
                "symmetry": symmetry,
                "y_intercept": str(y_intercept),
                "x_intercepts": [str(p) for p in x_intercepts],
                "asymptote_pos_inf": lim_pos.get("text", "N/A"),
                "asymptote_neg_inf": lim_neg.get("text", "N/A"),
                "first_derivative": d1_res.get("text", ""),
                "critical_points": [str(cp) for cp in critical_points],
                "second_derivative": d2_res.get("text", ""),
                "inflection_points": [str(ip) for ip in inflection_points],
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def investigate_2d(self) -> Dict[str, Any]:
        try:
            fx = sp.diff(self.expr, self.x)
            fy = sp.diff(self.expr, self.y)
            critical_points = sp.solve([fx, fy], (self.x, self.y), dict=True)

            f_xx = sp.diff(fx, self.x)
            f_yy = sp.diff(fy, self.y)
            f_xy = sp.diff(fx, self.y)
            hessian_det = f_xx * f_yy - (f_xy ** 2)

            classified_points = []
            for pt in critical_points:
                det_val = hessian_det.subs(pt)
                f_xx_val = f_xx.subs(pt)
                if det_val > 0:
                    kind = "Local Minimum" if f_xx_val > 0 else "Local Maximum"
                elif det_val < 0:
                    kind = "Saddle Point"
                else:
                    kind = "Inconclusive (Hessian determinant = 0)"
                classified_points.append({"point": str(pt), "type": kind})

            return {
                "gradient": [str(fx), str(fy)],
                "critical_points": classified_points,
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}