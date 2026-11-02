import sympy as sp


class LimitsAndContinuity:
    def __init__(self, expr_str: str, var_name: str = 'x'):
        self.x = sp.Symbol(var_name, real=True)
        self.f = sp.sympify(expr_str.replace('^', '**'))

    # ==========================================
    # 1. LIMIT SOLVER & SPECIAL CASES
    # ==========================================
    def compute_limit_step_by_step(self, point_str: str = "0", dir_str: str = "+-") -> list[str]:
        """Computes limits as x approaches a target point or infinity."""
        target = sp.sympify(point_str)
        steps = ["### Limit Evaluation"]
        steps.append(
            f"1. **Target Expression:** $\\lim_{{x \\to {sp.latex(target)}}} \\left({sp.latex(self.f)}\\right)$")

        special_step = self._check_special_limits(target)
        if special_step:
            steps.extend(special_step)
            return steps

        if dir_str == "+-":
            if target in (sp.oo, -sp.oo):
                limit_val = sp.limit(self.f, self.x, target)
                steps.append(
                    f"2. **Evaluation at Infinity:** $\\lim_{{x \\to {sp.latex(target)}}} {sp.latex(self.f)} = {sp.latex(limit_val)}$")
            else:
                lim_right = sp.limit(self.f, self.x, target, dir="+")
                lim_left = sp.limit(self.f, self.x, target, dir="-")

                steps.append(f"2. **Right-Hand Limit ($x \\to {sp.latex(target)}^+$):** ${sp.latex(lim_right)}$")
                steps.append(f"3. **Left-Hand Limit ($x \\to {sp.latex(target)}^-$):** ${sp.latex(lim_left)}$")

                if lim_right == lim_left:
                    steps.append(
                        f"4. **Conclusion:** Both one-sided limits are equal $\\implies \\lim_{{x \\to {sp.latex(target)}}} {sp.latex(self.f)} = {sp.latex(lim_right)}$")
                else:
                    steps.append(
                        "4. **Conclusion:** Left-hand and right-hand limits differ $\\implies$ **Limit does not exist.**")
        else:
            limit_val = sp.limit(self.f, self.x, target, dir=dir_str)
            direction_label = "+" if dir_str == "+" else "-"
            steps.append(
                f"2. **One-Sided Limit ($x \\to {sp.latex(target)}^{{{direction_label}}}$):** Result $= {sp.latex(limit_val)}$")

        return steps

    def _check_special_limits(self, target) -> list[str] | None:
        """Helper to match fundamental course limits."""
        if target == 0 and self.f == sp.sin(self.x) / self.x:
            return [
                "2. **Fundamental Limit Identified:** $\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1$",
                "3. **Result:** $= 1$"
            ]

        if target == sp.oo and self.f == (1 + 1 / self.x) ** self.x:
            return [
                "2. **Fundamental Limit Identified:** $\\lim_{x \\to \\infty} \\left(1 + \\frac{1}{x}\\right)^x = e$",
                "3. **Result:** $= e$"
            ]

        if target == 0 and self.f == (1 + self.x) ** (1 / self.x):
            return [
                "2. **Fundamental Limit Identified:** $\\lim_{x \\to 0} (1 + x)^{\\frac{1}{x}} = e$",
                "3. **Result:** $= e$"
            ]

        return None

    def apply_lhopital_step_by_step(self, point_str: str = "0") -> list[str]:
        """Evaluates limit using L'Hôpital's Rule for indeterminate forms."""
        target = sp.sympify(point_str)
        num, den = sp.fraction(self.f)

        steps = ["### L'Hôpital's Rule Evaluation"]
        steps.append(f"1. **Fraction Form:** $\\frac{{f(x)}}{{g(x)}} = \\frac{{{sp.latex(num)}}}{{{sp.latex(den)}}}$")

        num_val = sp.limit(num, self.x, target)
        den_val = sp.limit(den, self.x, target)

        steps.append(f"2. **Direct Limits:** $\\lim f(x) = {sp.latex(num_val)}$, $\\lim g(x) = {sp.latex(den_val)}$")

        is_zero_over_zero = (num_val == 0 and den_val == 0)
        is_inf_over_inf = (abs(num_val) == sp.oo and abs(den_val) == sp.oo)

        if not (is_zero_over_zero or is_inf_over_inf):
            steps.append(
                "3. **Notice:** L'Hôpital's Rule not applicable (not $\\left[\\frac{0}{0}\\right]$ or $\\left[\\frac{\\infty}{\\infty}\\right]$).")
            return steps

        form_str = "\\left[\\frac{0}{0}\\right]" if is_zero_over_zero else "\\left[\\frac{\\infty}{\\infty}\\right]"
        steps.append(
            f"3. **Indeterminate Form Detected:** {form_str} $\\implies$ Differentiate numerator and denominator.")

        num_diff = sp.diff(num, self.x)
        den_diff = sp.diff(den, self.x)

        steps.append(f"4. **Derivatives:** $f'(x) = {sp.latex(num_diff)}$, $g'(x) = {sp.latex(den_diff)}$")

        final_lim = sp.limit(num_diff / den_diff, self.x, target)
        steps.append(
            f"5. **New Limit:** $\\lim_{{x \\to {sp.latex(target)}}} \\frac{{{sp.latex(num_diff)}}}{{{sp.latex(den_diff)}}} = {sp.latex(final_lim)}$")
        return steps

    # ==========================================
    # 2. DISCONTINUITY CLASSIFIER
    # ==========================================
    def classify_discontinuities_step_by_step(self) -> list[str]:
        """Finds candidate discontinuity points and classifies them."""
        steps = ["### Discontinuity Classification"]
        steps.append(f"1. **Function:** $f(x) = {sp.latex(self.f)}$")

        den = sp.denom(self.f)
        if den == 1:
            sings = []
        else:
            # Solve denominator roots reliably
            raw_sings = sp.solve(den, self.x)
            sings = [s for s in raw_sings if s.is_real or s.is_real is None]

        if not sings:
            steps.append("2. **Conclusion:** Function is continuous everywhere on $\\mathbb{R}$.")
            return steps

        steps.append(
            f"2. **Candidate Discontinuity Points:** $x \\in \\{{{', '.join([sp.latex(s) for s in sings])}\\}}$")

        for pt in sings:
            lim_left = sp.limit(self.f, self.x, pt, dir="-")
            lim_right = sp.limit(self.f, self.x, pt, dir="+")

            steps.append(f"\n---\n* **Analyzing point $x = {sp.latex(pt)}$:**")
            steps.append(f"  * Left limit ($x \\to {sp.latex(pt)}^-$): ${sp.latex(lim_left)}$")
            steps.append(f"  * Right limit ($x \\to {sp.latex(pt)}^+$): ${sp.latex(lim_right)}$")

            if abs(lim_left) == sp.oo or abs(lim_right) == sp.oo:
                dis_type = "**Essential / Infinite Discontinuity (Second Kind)**"
            elif lim_left == lim_right:
                dis_type = "**Removable Discontinuity**"
            else:
                dis_type = "**Jump Discontinuity (First Kind)**"

            steps.append(f"  * **Classification:** {dis_type}")

        return steps

    # ==========================================
    # 3. NUMERICAL SOLVER (BISECTION METHOD)
    # ==========================================
    def bisection_method_step_by_step(self, a: float, b: float, tol: float = 1e-3, max_iter: int = 10) -> list[str]:
        """Finds root of f(x) = 0 in interval [a, b] using Intermediate Value Theorem / Bisection."""
        steps = ["### Bisection Method (Intermediate Value Theorem)"]
        steps.append(f"1. **Interval:** $[{a}, {b}]$, **Tolerance:** {tol}")

        # Use lambdify to create a pure Python function for safe numerical evaluation
        f_num = sp.lambdify(self.x, self.f, 'math')

        f_a = float(f_num(float(a)))
        f_b = float(f_num(float(b)))

        steps.append(f"2. **Check Endpoints:** $f({a}) = {f_a:.4f}$, $f({b}) = {f_b:.4f}$")

        if f_a * f_b > 0:
            steps.append("3. **Error:** $f(a)$ and $f(b)$ must have opposite signs by IVT.")
            return steps

        steps.append("3. **Iterations:**")

        c = a
        for i in range(1, max_iter + 1):
            c = (a + b) / 2.0
            f_c = float(f_num(float(c)))

            steps.append(
                f"   * Iteration {i}: $c = \\frac{{{a:.4f} + {b:.4f}}}{{2}} = {c:.4f} \\implies f(c) = {f_c:.4f}$")

            if abs(f_c) < tol or (b - a) / 2.0 < tol:
                steps.append(f"4. **Root Found:** $x \\approx {c:.4f}$ (within tolerance)")
                return steps

            if f_a * f_c < 0:
                b = c
                f_b = f_c
            else:
                a = c
                f_a = f_c

        steps.append(f"4. **Approximate Root after {max_iter} iterations:** $x \\approx {c:.4f}$")
        return steps


# Example Execution
if __name__ == "__main__":
    # Test 1: Discontinuity Classifier for 1 / (x * (x - 2))
    calc_dis = LimitsAndContinuity("1 / (x * (x - 2))")
    print("\n".join(calc_dis.classify_discontinuities_step_by_step()))
    print("\n" + "=" * 40 + "\n")

    # Test 2: Bisection Method for x^3 - x - 2 in [1, 2]
    calc_bisect = LimitsAndContinuity("x**3 - x - 2")
    print("\n".join(calc_bisect.bisection_method_step_by_step(1, 2, tol=1e-2)))