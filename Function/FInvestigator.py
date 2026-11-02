import sympy as sp
from typing import Dict, Any
from sympy.calculus.util import continuous_domain
from .base_calculator import BaseCalculator, safe_step
from .Differentiation import Differentiation
from .Limits import Limits


class FInvestigator(BaseCalculator):
    def __init__(self, expr_str: str, symbols_str: str = "x y z t"):
        super().__init__(expr_str, symbols_str)
        self.diff_engine = Differentiation(expr_str, symbols_str)
        self.limit_engine = Limits(expr_str, symbols_str)

    @safe_step
    def local_extrema_2d(self) -> Dict[str, Any]:
        fx, fy = self.get_gradient_2d()
        critical_points = sp.solve([fx, fy], (self.x, self.y), dict=True)

        f_xx, f_yy, f_xy = sp.diff(fx, self.x), sp.diff(fy, self.y), sp.diff(fx, self.y)
        hessian_det = f_xx * f_yy - (f_xy ** 2)

        classified_points = []
        if isinstance(critical_points, dict): critical_points = [critical_points]

        for pt in (critical_points or []):
            if self.x in pt and self.y in pt:
                det_val, f_xx_val = hessian_det.subs(pt), f_xx.subs(pt)

                if det_val > 0:
                    kind = "Local Minimum" if f_xx_val > 0 else "Local Maximum"
                elif det_val < 0:
                    kind = "Saddle Point"
                else:
                    kind = "Inconclusive"

                classified_points.append({
                    "point": f"({pt[self.x]}, {pt[self.y]})",
                    "value": str(self.expr.subs(pt)),
                    "type": kind
                })

        return {
            "gradient": [str(fx), str(fy)],
            "hessian_determinant": str(hessian_det),
            "critical_points": classified_points,
            "success": True
        }

    @safe_step
    def lagrange_multipliers(self, constraint_str: str) -> Dict[str, Any]:
        g = self.parse_secondary(constraint_str)
        lam = sp.Symbol('lam')

        fx, fy = self.get_gradient_2d()
        gx, gy = sp.diff(g, self.x), sp.diff(g, self.y)

        eq1, eq2, eq3 = sp.Eq(fx, lam * gx), sp.Eq(fy, lam * gy), sp.Eq(g, 0)
        solutions = sp.solve([eq1, eq2, eq3], (self.x, self.y, lam), dict=True)
        if isinstance(solutions, dict): solutions = [solutions]

        points_evaluated = [
            {
                "point": f"({sol[self.x]}, {sol[self.y]})",
                "lambda": str(sol.get(lam, "N/A")),
                "value": str(sp.simplify(self.expr.subs({self.x: sol[self.x], self.y: sol[self.y]})))
            }
            for sol in (solutions or []) if self.x in sol and self.y in sol
        ]

        return {"system": [str(eq1), str(eq2), str(eq3)], "solutions": points_evaluated, "success": True}

    @safe_step
    def absolute_extrema(self, boundary_str: str) -> Dict[str, Any]:
        g = self.parse_secondary(boundary_str)
        fx, fy = self.get_gradient_2d()
        valid_points = []

        # 1. Interior points
        interior_sols = sp.solve([fx, fy], (self.x, self.y), dict=True)
        if isinstance(interior_sols, dict): interior_sols = [interior_sols]
        for sol in (interior_sols or []):
            if self.x in sol and self.y in sol and sp.simplify(g.subs(sol)) <= 0:
                valid_points.append((sol[self.x], sol[self.y], "Interior"))

        # 2. Boundary points
        lam = sp.Symbol('lam')
        gx, gy = sp.diff(g, self.x), sp.diff(g, self.y)
        bound_sols = sp.solve([sp.Eq(fx, lam * gx), sp.Eq(fy, lam * gy), sp.Eq(g, 0)], (self.x, self.y, lam), dict=True)
        if isinstance(bound_sols, dict): bound_sols = [bound_sols]
        for sol in (bound_sols or []):
            if self.x in sol and self.y in sol:
                valid_points.append((sol[self.x], sol[self.y], "Boundary"))

        # 3. Evaluate valid points
        evaluated = []
        for px, py, loc in valid_points:
            val = self.expr.subs({self.x: px, self.y: py})
            if val.is_real or val.is_real is None:
                try:
                    evaluated.append((float(sp.N(val)), val, px, py, loc))
                except TypeError:
                    pass

        if not evaluated: return {"result": "No real critical points found in domain", "success": True}

        abs_max, abs_min = max(evaluated, key=lambda i: i[0]), min(evaluated, key=lambda i: i[0])
        return {
            "absolute_maximum": {"value": str(abs_max[1]), "point": f"({abs_max[2]}, {abs_max[3]})",
                                 "location": abs_max[4]},
            "absolute_minimum": {"value": str(abs_min[1]), "point": f"({abs_min[2]}, {abs_min[3]})",
                                 "location": abs_min[4]},
            "success": True
        }