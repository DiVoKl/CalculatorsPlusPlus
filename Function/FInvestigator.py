import sympy as sp
import math


class DifferentiationAndApplications:
    def __init__(self, expr_str: str = "0", var_name: str = 'x'):
        self.x = sp.Symbol(var_name, real=True)
        if expr_str:
            self.f = sp.sympify(expr_str.replace('^', '**'))
        else:
            self.f = sp.Integer(0)

    # ==========================================
    # 1. HIGHER ORDER DERIVATIVES
    # ==========================================
    def higher_order_derivative_step_by_step(self, n: int = 1) -> list[str]:
        """Calculates the n-th derivative of the function."""
        steps = [f"### {n}-th Order Derivative Calculation"]
        steps.append(f"1. **Function:** $f(x) = {sp.latex(self.f)}$")

        current_f = self.f
        for i in range(1, n + 1):
            current_f = sp.diff(current_f, self.x)
            current_f = sp.simplify(current_f)
            prime_notation = "'" * i if i <= 3 else f"({i})"
            steps.append(f"   * $f^{prime_notation}(x) = {sp.latex(current_f)}$")

        steps.append(f"2. **Final Result:** $f^{{({n})}}(x) = {sp.latex(current_f)}$")
        return steps

    # ==========================================
    # 2. IMPLICIT & PARAMETRIC DIFFERENTIATION
    # ==========================================
    @staticmethod
    def implicit_differentiation_step_by_step(eq_str: str) -> list[str]:
        steps = ["### Implicit Differentiation ($y'$)"]
        x, y = sp.symbols('x y', real=True)

        if '=' in eq_str:
            lhs_str, rhs_str = eq_str.split('=', 1)
            eq = sp.sympify(lhs_str.replace('^', '**')) - sp.sympify(rhs_str.replace('^', '**'))
        else:
            eq = sp.sympify(eq_str.replace('^', '**'))

        steps.append(f"1. **Equation:** ${sp.latex(eq)} = 0$")

        try:
            y_prime = sp.idiff(eq, y, x)
            steps.append(f"2. **First Derivative ($y'$):** $y' = {sp.latex(sp.simplify(y_prime))}$")
        except Exception as e:
            steps.append(f"2. **Error:** Could not isolate $y'$. ({str(e)})")

        return steps

    @staticmethod
    def parametric_differentiation_step_by_step(x_t_str: str, y_t_str: str) -> list[str]:
        steps = ["### Parametric Differentiation ($y'$ and $y''$)"]
        t = sp.Symbol('t', real=True)
        x_t = sp.sympify(x_t_str.replace('^', '**'))
        y_t = sp.sympify(y_t_str.replace('^', '**'))

        steps.append(f"1. **Parametric Equations:** $x(t) = {sp.latex(x_t)}, \\quad y(t) = {sp.latex(y_t)}$")

        dx_dt = sp.diff(x_t, t)
        dy_dt = sp.diff(y_t, t)

        steps.append(f"2. **Derivatives with respect to $t$:**")
        steps.append(f"   * $\\frac{{dx}}{{dt}} = {sp.latex(dx_dt)}$")
        steps.append(f"   * $\\frac{{dy}}{{dt}} = {sp.latex(dy_dt)}$")

        if dx_dt == 0:
            steps.append("3. **Error:** $\\frac{dx}{dt} = 0$, $y'$ is undefined.")
            return steps

        y_prime = sp.simplify(dy_dt / dx_dt)
        steps.append(f"3. **First Derivative ($y'$):** $\\frac{{dy}}{{dx}} = {sp.latex(y_prime)}$")

        dy_prime_dt = sp.diff(y_prime, t)
        y_double_prime = sp.simplify(dy_prime_dt / dx_dt)
        steps.append(f"4. **Second Derivative ($y''$):** $\\frac{{d^2y}}{{dx^2}} = {sp.latex(y_double_prime)}$")

        return steps

    # ==========================================
    # 3. TANGENTS, NORMALS, AND ANGLES
    # ==========================================
    def tangent_and_normal_step_by_step(self, x0_str: str) -> list[str]:
        steps = ["### Tangent and Normal Lines"]
        x0 = sp.sympify(x0_str)
        y0 = self.f.subs(self.x, x0)

        steps.append(f"1. **Point of Interest:** $x_0 = {sp.latex(x0)} \\implies y_0 = {sp.latex(y0)}$")

        df = sp.diff(self.f, self.x)
        m = df.subs(self.x, x0)

        steps.append(f"2. **Derivative & Slope:** $f'(x) = {sp.latex(df)} \\implies m = {sp.latex(m)}$")

        tangent_eq = sp.simplify(m * (self.x - x0) + y0)
        steps.append(f"3. **Tangent Line:** $y = {sp.latex(tangent_eq)}$")

        if m == 0:
            steps.append(f"4. **Normal Line:** Vertical: $x = {sp.latex(x0)}$")
        elif abs(m) == sp.oo:
            steps.append(f"4. **Normal Line:** Horizontal: $y = {sp.latex(y0)}$")
        else:
            m_norm = -1 / m
            normal_eq = sp.simplify(m_norm * (self.x - x0) + y0)
            steps.append(f"4. **Normal Line:** $m_n = {sp.latex(m_norm)} \\implies y = {sp.latex(normal_eq)}$")

        return steps

    def angle_between_curves_step_by_step(self, other_expr_str: str) -> list[str]:
        steps = ["### Angle Between Curves"]
        g = sp.sympify(other_expr_str.replace('^', '**'))

        eq = sp.expand(self.f - g)
        raw_intersections = sp.solve(eq, self.x)
        intersections = [pt for pt in raw_intersections if pt.is_real or pt.is_real is None]

        if not intersections:
            steps.append("1. **Conclusion:** Curves do not intersect on $\\mathbb{R}$.")
            return steps

        steps.append(
            f"1. **Intersection Points:** $x \\in \\{{{', '.join([sp.latex(pt) for pt in intersections])}\\}}$")

        df = sp.diff(self.f, self.x)
        dg = sp.diff(g, self.x)

        for pt in intersections:
            m1, m2 = df.subs(self.x, pt), dg.subs(self.x, pt)
            steps.append(f"\n* **At $x = {sp.latex(pt)}$:** Slopes: $m_1 = {sp.latex(m1)}$, $m_2 = {sp.latex(m2)}$")

            if m1 * m2 == -1:
                steps.append("  * **Angle:** Curves are orthogonal $\\implies 90^\\circ$")
            else:
                tan_theta = sp.Abs((m1 - m2) / (1 + m1 * m2))
                theta = sp.atan(tan_theta)
                theta_deg = float(sp.N(theta * 180 / sp.pi))
                steps.append(f"  * **Angle:** $\\theta = {sp.latex(theta)} \\approx {theta_deg:.2f}^\\circ$")

        return steps

    # ==========================================
    # 4. L'HOPITAL'S RULE SOLVER
    # ==========================================
    def apply_lhopital_step_by_step(self, point_str: str = "0") -> list[str]:
        target = sp.sympify(point_str)
        num, den = sp.fraction(self.f)
        steps = ["### L'Hôpital's Rule Evaluation"]

        if den == 1:
            steps.append("1. **Error:** Expression is not a fraction.")
            return steps

        num_val = sp.limit(num, self.x, target)
        den_val = sp.limit(den, self.x, target)

        is_zero_over_zero = (num_val == 0 and den_val == 0)
        is_inf_over_inf = (abs(num_val) == sp.oo and abs(den_val) == sp.oo)

        if not (is_zero_over_zero or is_inf_over_inf):
            steps.append("1. **Notice:** L'Hôpital's Rule not applicable.")
            return steps

        num_diff = sp.diff(num, self.x)
        den_diff = sp.diff(den, self.x)
        final_lim = sp.limit(num_diff / den_diff, self.x, target)

        steps.append(f"1. **Derivatives:** $f'(x) = {sp.latex(num_diff)}$, $g'(x) = {sp.latex(den_diff)}$")
        steps.append(
            f"2. **New Limit:** $\\lim_{{x \\to {sp.latex(target)}}} \\frac{{{sp.latex(num_diff)}}}{{{sp.latex(den_diff)}}} = {sp.latex(final_lim)}$")
        return steps

    # ==========================================
    # 5. TAYLOR & MACLAURIN POLYNOMIALS
    # ==========================================
    def taylor_polynomial_step_by_step(self, x0_str: str = "0", n: int = 3) -> list[str]:
        """Generates the n-th degree Taylor/Maclaurin polynomial around x0."""
        x0 = sp.sympify(x0_str)
        series_name = "Maclaurin" if x0 == 0 else "Taylor"

        steps = [f"### {series_name} Polynomial of degree $n={n}$"]
        steps.append(f"1. **Center ($x_0$):** $x_0 = {sp.latex(x0)}$")

        poly = sp.Integer(0)
        current_deriv = self.f
        steps.append("2. **Derivatives and Coefficients:**")

        for k in range(n + 1):
            if k > 0:
                current_deriv = sp.simplify(sp.diff(current_deriv, self.x))

            val_at_x0 = current_deriv.subs(self.x, x0)
            term = (val_at_x0 / sp.Integer(math.factorial(k))) * ((self.x - x0) ** k)
            poly += term

            notation = f"f({sp.latex(x0)})" if k == 0 else (
                f"f'({sp.latex(x0)})" if k == 1 else f"f^{{({k})}}({sp.latex(x0)})")
            steps.append(f"   * $k={k}$: ${notation} = {sp.latex(val_at_x0)}$ $\\implies$ Term: ${sp.latex(term)}$")

        steps.append(f"3. **Final Polynomial ($P_{{{n}}}(x)$):** $P_{{{n}}}(x) = {sp.latex(poly)}$")
        return steps

    def lagrange_remainder_step_by_step(self, x0_str: str = "0", n: int = 3, target_x_str: str = None) -> list[str]:
        """Calculates the Lagrange remainder and estimates error bound."""
        x0 = sp.sympify(x0_str)
        c = sp.Symbol('c', real=True)
        steps = [f"### Lagrange Remainder $R_{{{n}}}(x)$"]

        deriv_np1 = self.f
        for _ in range(n + 1):
            deriv_np1 = sp.diff(deriv_np1, self.x)
        deriv_np1 = sp.simplify(deriv_np1)

        deriv_c = deriv_np1.subs(self.x, c)
        remainder = (deriv_c / sp.Integer(math.factorial(n + 1))) * ((self.x - x0) ** (n + 1))

        steps.append(f"1. **$(n+1)$-th Derivative:** $f^{{({n + 1})}}(x) = {sp.latex(deriv_np1)}$")
        steps.append(
            f"2. **Lagrange Formula:** $R_{{{n}}}(x) = {sp.latex(remainder)}$ *(c is between {sp.latex(x0)} and x)*")

        if target_x_str:
            target_x = sp.sympify(target_x_str)
            val_x0 = sp.Abs(deriv_np1.subs(self.x, x0))
            val_tx = sp.Abs(deriv_np1.subs(self.x, target_x))

            try:
                max_error = (sp.Max(val_x0, val_tx) / sp.Integer(math.factorial(n + 1))) * (
                            sp.Abs(target_x - x0) ** (n + 1))
                steps.append(
                    f"3. **Estimated Error Bound at $x={sp.latex(target_x)}$:** $|R_{{{n}}}| \\le {sp.latex(max_error)} \\approx {float(sp.N(max_error)):.6e}$")
            except Exception:
                pass

        return steps


# ==========================================
# EXAMPLES AND TESTS
# ==========================================
if __name__ == "__main__":
    calc = DifferentiationAndApplications("exp(x)")

    print("\n".join(calc.taylor_polynomial_step_by_step("0", n=3)))
    print("\n" + "=" * 40 + "\n")
    print("\n".join(calc.lagrange_remainder_step_by_step("0", n=3, target_x_str="0.5")))