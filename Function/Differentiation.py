import sympy as sp


class IntegrationAndApplications:
    def __init__(self, expr_str: str, var_name: str = 'x', verbose: bool = True):
        """
        :param expr_str: The mathematical expression as a string.
        :param var_name: The variable of integration (default is 'x').
        :param verbose: If True, prints each step to the console automatically.
        """
        self.x = sp.Symbol(var_name, real=True)
        self.f = sp.sympify(expr_str.replace('^', '**'))
        self.verbose = verbose

    def _log(self, steps_list: list, message: str):
        """Helper method to append a step and print it immediately if verbose is True."""
        steps_list.append(message)
        if self.verbose:
            print(message)

    # ==========================================
    # 1. INDEFINITE INTEGRAL (ANTIDERIVATIVE)
    # ==========================================
    def indefinite_integral_step_by_step(self) -> list[str]:
        """Calculates the indefinite integral (antiderivative) + C."""
        steps = []
        self._log(steps, "### Indefinite Integral (Antiderivative)")
        self._log(steps, f"1. **Integral Setup:** $\\int \\left({sp.latex(self.f)}\\right) dx$")

        try:
            F = sp.integrate(self.f, self.x)
            self._log(steps, f"2. **Antiderivative Computed:** $F(x) = {sp.latex(F)}$")
            self._log(steps, f"3. **Final Result:** $\\int {sp.latex(self.f)} dx = {sp.latex(F)} + C$")
        except Exception as e:
            self._log(steps, f"2. **Error:** SymPy could not compute the integral. ({str(e)})")

        return steps

    # ==========================================
    # 2. DEFINITE INTEGRAL (NEWTON-LEIBNIZ)
    # ==========================================
    def definite_integral_step_by_step(self, a_str: str, b_str: str) -> list[str]:
        """Calculates definite integral using the Newton-Leibniz formula."""
        a = sp.sympify(a_str)
        b = sp.sympify(b_str)
        steps = []

        self._log(steps, f"### Definite Integral on $[{sp.latex(a)}, {sp.latex(b)}]$")
        self._log(steps,
                  f"1. **Integral Setup:** $\\int_{{{sp.latex(a)}}}^{{{sp.latex(b)}}} \\left({sp.latex(self.f)}\\right) dx$")

        F = sp.integrate(self.f, self.x)
        self._log(steps, f"2. **Antiderivative (Newton-Leibniz):** $F(x) = {sp.latex(F)}$")

        F_b = F.subs(self.x, b)
        F_a = F.subs(self.x, a)

        self._log(steps, f"3. **Evaluate Boundaries:**")
        self._log(steps, f"   * $F({sp.latex(b)}) = {sp.latex(F_b)}$")
        self._log(steps, f"   * $F({sp.latex(a)}) = {sp.latex(F_a)}$")

        result = sp.simplify(F_b - F_a)
        self._log(steps,
                  f"4. **Final Area/Result:** $\\int_{{{sp.latex(a)}}}^{{{sp.latex(b)}}} f(x) dx = F(b) - F(a) = {sp.latex(result)}$")

        try:
            val_float = float(sp.N(result))
            self._log(steps, f"   * **Approximate Value:** $\\approx {val_float:.4f}$")
        except Exception:
            pass

        return steps

    # ==========================================
    # 3. AREA BETWEEN TWO CURVES
    # ==========================================
    def area_between_curves_step_by_step(self, g_str: str, a_str: str, b_str: str) -> list[str]:
        """Calculates the absolute area between f(x) and g(x) from a to b."""
        g = sp.sympify(g_str.replace('^', '**'))
        a = sp.sympify(a_str)
        b = sp.sympify(b_str)
        steps = []

        self._log(steps, f"### Area Between Curves on $[{sp.latex(a)}, {sp.latex(b)}]$")
        self._log(steps, f"1. **Functions:** $f(x) = {sp.latex(self.f)}$, $g(x) = {sp.latex(g)}$")

        diff_eq = sp.simplify(self.f - g)
        raw_intersections = sp.solve(diff_eq, self.x)
        real_intersections = [pt for pt in raw_intersections if pt.is_real or pt.is_real is None]

        valid_pts = []
        for pt in real_intersections:
            try:
                if float(sp.N(a)) < float(sp.N(pt)) < float(sp.N(b)):
                    valid_pts.append(pt)
            except Exception:
                pass

        valid_pts = sorted(list(set(valid_pts)), key=lambda p: float(sp.N(p)))
        intervals = [a] + valid_pts + [b]

        if valid_pts:
            self._log(steps,
                      f"2. **Intersection Points in interval:** $x \\in \\{{{', '.join([sp.latex(p) for p in valid_pts])}\\}}$")
            self._log(steps, f"3. **Integral Splitting:** Area must be split at intersections to maintain positivity.")
        else:
            self._log(steps, "2. **No intersections in interval:** Evaluating as a single region.")

        total_area = sp.Integer(0)

        for i in range(len(intervals) - 1):
            left, right = intervals[i], intervals[i + 1]
            integral_val = sp.integrate(diff_eq, (self.x, left, right))
            area_part = sp.simplify(sp.Abs(integral_val))
            total_area += area_part

            self._log(steps,
                      f"   * $\\int_{{{sp.latex(left)}}}^{{{sp.latex(right)}}} |f(x) - g(x)| dx = {sp.latex(area_part)}$")

        total_area = sp.simplify(total_area)
        self._log(steps, f"4. **Total Area:** $S = {sp.latex(total_area)}$")

        try:
            self._log(steps, f"   * **Approximate:** $S \\approx {float(sp.N(total_area)):.4f}$")
        except Exception:
            pass

        return steps

    # ==========================================
    # 4. VOLUMES OF REVOLUTION
    # ==========================================
    def volume_of_revolution_step_by_step(self, a_str: str, b_str: str, axis: str = 'x') -> list[str]:
        """Calculates volume of revolution around x-axis (Disk) or y-axis (Shell)."""
        a = sp.sympify(a_str)
        b = sp.sympify(b_str)
        steps = []

        self._log(steps, f"### Volume of Solid of Revolution (Around {axis}-axis)")
        self._log(steps, f"1. **Function:** $f(x) = {sp.latex(self.f)}$ on interval $[{sp.latex(a)}, {sp.latex(b)}]$")

        if axis == 'x':
            self._log(steps, "2. **Method:** Disk Method $\\implies V_x = \\pi \\int_a^b (f(x))^2 dx$")
            integrand = self.f ** 2
            self._log(steps, f"   * Integrand: $(f(x))^2 = {sp.latex(sp.expand(integrand))}$")
            volume = sp.pi * sp.integrate(integrand, (self.x, a, b))
        elif axis == 'y':
            self._log(steps, "2. **Method:** Cylindrical Shells Method $\\implies V_y = 2\\pi \\int_a^b x f(x) dx$")
            integrand = self.x * self.f
            self._log(steps, f"   * Integrand: $x f(x) = {sp.latex(sp.expand(integrand))}$")
            volume = 2 * sp.pi * sp.integrate(integrand, (self.x, a, b))
        else:
            self._log(steps, "2. **Error:** Axis must be 'x' or 'y'.")
            return steps

        volume = sp.simplify(volume)
        self._log(steps, f"3. **Final Volume:** $V_{axis} = {sp.latex(volume)}$")

        try:
            self._log(steps, f"   * **Approximate:** $V_{axis} \\approx {float(sp.N(volume)):.4f}$")
        except Exception:
            pass

        return steps

    # ==========================================
    # 5. IMPROPER INTEGRALS & CONVERGENCE
    # ==========================================
    def improper_integral_step_by_step(self, a_str: str, b_str: str) -> list[str]:
        """Evaluates improper integrals and checks for convergence/divergence."""
        a = sp.sympify(a_str)
        b = sp.sympify(b_str)
        steps = []

        self._log(steps, "### Improper Integral & Convergence Test")
        self._log(steps,
                  f"1. **Integral Setup:** $\\int_{{{sp.latex(a)}}}^{{{sp.latex(b)}}} \\left({sp.latex(self.f)}\\right) dx$")

        try:
            result = sp.integrate(self.f, (self.x, a, b))
            self._log(steps, f"2. **Evaluation:** Result = ${sp.latex(result)}$")

            if result.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
                self._log(steps,
                          "3. **Conclusion:** The integral **Diverges** (does not evaluate to a finite real number).")
            else:
                self._log(steps, "3. **Conclusion:** The integral **Converges** to a finite value.")

        except Exception as e:
            self._log(steps, f"2. **Evaluation Error:** SymPy could not resolve the improper integral. ({str(e)})")

        return steps


# ==========================================
# EXAMPLES AND TESTS
# ==========================================
if __name__ == "__main__":
    calc_ind = IntegrationAndApplications("x * exp(x)")
    calc_ind.indefinite_integral_step_by_step()
    print("\n" + "=" * 40 + "\n")

    calc_def = IntegrationAndApplications("sin(x)")
    calc_def.definite_integral_step_by_step("0", "pi")
    print("\n" + "=" * 40 + "\n")

    calc_area = IntegrationAndApplications("x**2")
    calc_area.area_between_curves_step_by_step("x", "0", "2")
    print("\n" + "=" * 40 + "\n")

    calc_vol = IntegrationAndApplications("sqrt(x)")
    calc_vol.volume_of_revolution_step_by_step("0", "4", axis='x')