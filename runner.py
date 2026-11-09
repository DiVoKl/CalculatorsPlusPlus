from calculus.Series import Series
from functionanalysis.Vectors3D import Vectors3D
from functionanalysis.FInvestigator import FInvestigator
from calculus.Integrals import Integrals
from common.DiffEq import DiffEq


def run_edge_case_checks():
    print("==================================================")
    print("         CALCULUS 2 EDGE CASES TEST RUNNER        ")
    print("==================================================\n")

    # --- 1. SERIES EDGE CASES ---
    print("--- 1. Series: Taylor Expansion Around a Singularity ---")
    # 1/x cannot be expanded around x=0 (Pole/Singularity)
    s1 = Series("1 / x")
    print(s1.taylor_expansion(center=0))

    print("\n--- 2. Series: Ratio Test Inconclusive (L = 1) ---")
    # Harmonic series 1/n -> Limit of ratio is 1
    s2 = Series("1 / n")
    print(s2.test_convergence())

    print("\n--- 3. Series: Zero Radius of Convergence (R = 0) ---")
    # n! * x^n converges only at x = 0
    s3 = Series("factorial(n) * x**n")
    print(s3.radius_and_interval())

    # --- 2. VECTORS 3D EDGE CASES ---
    print("\n\n--- 4. Vectors3D: Parallel Vectors (Cross Product = 0) ---")
    v1 = Vectors3D("0")
    print(v1.vector_algebra(v1=[1, 2, 3], v2=[2, 4, 6]))

    print("\n--- 5. Vectors3D: Degenerate Plane (Zero Normal Vector) ---")
    # Normal vector [0, 0, 0] produces 0 = 0
    print(v1.plane_equation(point=[1, 1, 1], normal=[0, 0, 0]))

    # --- 3. FINEVESTIGATOR EDGE CASES ---
    print("\n\n--- 6. FInvestigator: Saddle Point (Hessian Det < 0) ---")
    # f(x,y) = x^2 - y^2 has a saddle point at (0,0)
    f1 = FInvestigator("x**2 - y**2")
    print(f1.local_extrema_2d())

    print("\n--- 7. FInvestigator: Inconclusive Hessian (Hessian Det = 0) ---")
    # f(x,y) = x^4 + y^4 has f_xx*f_yy - f_xy^2 = 0 at (0,0)
    f2 = FInvestigator("x**4 + y**4")
    print(f2.local_extrema_2d())

    # --- 4. INTEGRALS EDGE CASES ---
    print("\n\n--- 8. Integrals: Singular / Improper Double Integral ---")
    # 1/(x*y) integrated from 0 to 1 causes division by zero / log(0)
    integ = Integrals("1 / (x * y)")
    print(
        integ.double_integral(
            inner_var="x",
            inner_lower="0",
            inner_upper="1",
            outer_var="y",
            outer_lower="0",
            outer_upper="1",
        )
    )

    # --- 5. DIFFERENTIAL EQUATIONS EDGE CASES ---
    print("\n\n--- 9. DiffEq: Non-Separable Differential Equation ---")
    # dy/dx = x + y cannot be solved by separation of variables
    ode = DiffEq("x + y")
    print(ode.solve_ode())  # <--- Change this from solve_separable()

if __name__ == "__main__":
    run_edge_case_checks()