import sympy as sp
from typing import Dict, Any
from common.base_calculator import BaseCalculator, safe_step


class Integrals(BaseCalculator):
    @safe_step
    def calculate(self, var=None, lower=None, upper=None) -> Dict[str, Any]:
        target_var = var if var is not None else self.x
        if lower is not None and upper is not None:
            l, u = sp.sympify(lower), sp.sympify(upper)
            return self.safe_execute(sp.integrate, self.expr, (target_var, l, u))
        return self.safe_execute(sp.integrate, self.expr, target_var)

    @safe_step
    def double_integral(self, inner_var: str, inner_lower: str, inner_upper: str,
                        outer_var: str, outer_lower: str, outer_upper: str) -> Dict[str, Any]:
        i_var, o_var = sp.Symbol(inner_var), sp.Symbol(outer_var)
        i_l, i_u = sp.sympify(inner_lower), sp.sympify(inner_upper)
        o_l, o_u = sp.sympify(outer_lower), sp.sympify(outer_upper)

        result = sp.integrate(self.expr, (i_var, i_l, i_u), (o_var, o_l, o_u))
        return {"result": str(result), "latex": sp.latex(result), "success": True}

    @safe_step
    def polar_double_integral(self, r_lower: str, r_upper: str, theta_lower: str, theta_upper: str) -> Dict[str, Any]:
        r, theta = sp.Symbol('r', positive=True), sp.Symbol('theta')
        polar_expr = self.expr.subs({self.x: r * sp.cos(theta), self.y: r * sp.sin(theta)})
        integrand = sp.simplify(polar_expr * r)

        r_l, r_u = sp.sympify(r_lower), sp.sympify(r_upper)
        th_l, th_u = sp.sympify(theta_lower), sp.sympify(theta_upper)

        result = sp.integrate(integrand, (r, r_l, r_u), (theta, th_l, th_u))
        return {"polar_integrand": str(integrand), "result": str(result), "latex": sp.latex(result), "success": True}

    @safe_step
    def area_2d(self, inner_var: str, inner_lower: str, inner_upper: str,
                outer_var: str, outer_lower: str, outer_upper: str) -> Dict[str, Any]:
        original_expr = self.expr
        self.expr = sp.S.One
        res = self.double_integral(inner_var, inner_lower, inner_upper, outer_var, outer_lower, outer_upper)
        self.expr = original_expr
        if res.get("success"):
            res["area"] = res.pop("result")
        return res

    @safe_step
    def center_of_mass(self, density_str: str, inner_var: str, inner_lower: str, inner_upper: str,
                       outer_var: str, outer_lower: str, outer_upper: str) -> Dict[str, Any]:
        density = self.parse_secondary(density_str)
        i_var, o_var = sp.Symbol(inner_var), sp.Symbol(outer_var)
        i_l, i_u = sp.sympify(inner_lower), sp.sympify(inner_upper)
        o_l, o_u = sp.sympify(outer_lower), sp.sympify(outer_upper)

        mass = sp.integrate(density, (i_var, i_l, i_u), (o_var, o_l, o_u))
        Mx = sp.integrate(self.y * density, (i_var, i_l, i_u), (o_var, o_l, o_u))
        My = sp.integrate(self.x * density, (i_var, i_l, i_u), (o_var, o_l, o_u))

        x_bar, y_bar = sp.simplify(My / mass), sp.simplify(Mx / mass)
        return {"mass": str(mass), "moment_x": str(Mx), "moment_y": str(My), "center_of_mass": f"({x_bar}, {y_bar})",
                "success": True}