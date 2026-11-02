import sympy as sp
from sympy.calculus.util import continuous_domain


class FunctionAnalyzer:
    def __init__(self, expr_str: str, var_name: str = 'x'):
        self.x = sp.Symbol(var_name, real=True)
        self.f = sp.sympify(expr_str.replace('^', '**'))

    def check_parity_step_by_step(self) -> list[str]:
        """Check whether f(x) is even, odd, or neither."""
        steps = ["### Parity Check"]
        steps.append(f"1. **Function:** $f(x) = {sp.latex(self.f)}$")

        f_minus_x = self.f.subs(self.x, -self.x)
        steps.append(f"2. **Substitute $-x$:** $f(-x) = {sp.latex(f_minus_x)}$")

        # Symbolic equality check using simplify difference
        is_even = sp.simplify(f_minus_x - self.f) == 0
        is_odd = sp.simplify(f_minus_x + self.f) == 0

        if is_even:
            steps.append("3. **Conclusion:** $f(-x) = f(x) \\implies$ **Even**")
        elif is_odd:
            steps.append("3. **Conclusion:** $f(-x) = -f(x) \\implies$ **Odd**")
        else:
            steps.append("3. **Conclusion:** Neither even nor odd")

        return steps

    def find_domain_step_by_step(self) -> list[str]:
        """Find domain of definition for f(x)."""
        steps = ["### Domain of Definition"]
        steps.append(f"1. **Function:** $f(x) = {sp.latex(self.f)}$")

        # Correct import usage
        domain = continuous_domain(self.f, self.x, sp.S.Reals)
        steps.append(f"2. **Domain:** $D_f = {sp.latex(domain)}$")
        return steps

    def compose_with_step_by_step(self, g_str: str) -> list[str]:
        """Compute composition (f o g)(x)."""
        g = sp.sympify(g_str.replace('^', '**'))
        steps = ["### Function Composition $(f \\circ g)(x)$"]
        steps.append(f"1. **Functions:** $f(x) = {sp.latex(self.f)}$, $g(x) = {sp.latex(g)}$")

        composed = sp.simplify(self.f.subs(self.x, g))
        steps.append(f"2. **Result:** $(f \\circ g)(x) = f(g(x)) = {sp.latex(composed)}$")
        return steps

    def find_inverse_step_by_step(self) -> list[str]:
        """Calculate inverse function f^{-1}(x)."""
        steps = ["### Inverse Function $f^{-1}(x)$"]
        steps.append(f"1. **Equation:** $y = {sp.latex(self.f)}$")

        y = sp.Symbol('y', real=True)
        solutions = sp.solve(sp.Eq(y, self.f), self.x)

        if not solutions:
            steps.append("2. **Result:** Explicit inverse could not be found.")
            return steps

        inverse_expr = solutions[0].subs(y, self.x)
        steps.append(f"2. **Solve for $x$ and swap variables:** $f^{{-1}}(x) = {sp.latex(inverse_expr)}$")
        return steps

    def check_periodicity_step_by_step(self) -> list[str]:
        """Check whether f(x) is periodic and find its fundamental period."""
        steps = ["### Periodicity Check"]
        steps.append(f"1. **Function:** $f(x) = {sp.latex(self.f)}$")

        period = sp.periodicity(self.f, self.x)
        if period is not None:
            steps.append(
                f"2. **Conclusion:** The function is **periodic** with fundamental period $T = {sp.latex(period)}$.")
        else:
            steps.append("2. **Conclusion:** The function is **not periodic**.")

        return steps
if __name__ == "__main__":
    analyzer = FunctionAnalyzer("(x**3 + 1) / 2")

    print("\n".join(analyzer.check_parity_step_by_step()))
    print("\n" + "-" * 30 + "\n")
    print("\n".join(analyzer.find_domain_step_by_step()))
    print("\n" + "-" * 30 + "\n")
    print("\n".join(analyzer.compose_with_step_by_step("2*x - 1")))
    print("\n" + "-" * 30 + "\n")
    print("\n".join(analyzer.find_inverse_step_by_step()))