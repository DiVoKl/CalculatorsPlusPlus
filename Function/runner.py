from Function.Series import Series
from Function.Vectors3D import Vectors3D
from Function.FInvestigator import FInvestigator
from Function.Integrals import Integrals
from Function.DiffEq import DiffEq
import sympy as sp

def run_calculus2_checks():
    print("==================================================")
    print("          CALCULUS 2 MODULES - TEST RUNNER        ")
    print("==================================================\n")

    # --- STEP 1: Infinite Series & Power Series ---
    print("--- 1. Testing Series (Taylor Expansion) ---")
    series_engine = Series("exp(x)")
    print(series_engine.calculate(order=5))

    print("\n--- Testing Series (Convergence) ---")
    series_convergence = Series("1 / (2**n)")
    print(series_convergence.test_convergence())

    print("\n--- Testing Series (Radius & Interval) ---")
    series_power = Series("(x**n) / factorial(n)")
    print(series_power.radius_and_interval())


    # --- STEP 2: 3D Vectors & Analytic Geometry ---
    print("\n\n--- 2. Testing Vectors3D (Vector Algebra) ---")
    geo_engine = Vectors3D("0") # Main expression unused for basic vectors
    print(geo_engine.vector_algebra(v1=[1, 2, 3], v2=[4, 5, 6], v3=[7, 8, 9]))

    print("\n--- Testing Vectors3D (Line and Plane) ---")
    print(geo_engine.plane_equation(point=[1, 1, 1], normal=[2, -1, 3]))
    print(geo_engine.line_equation(point=[0, 0, 0], direction=[1, 2, 3]))

    print("\n--- Testing Vectors3D (Level Surfaces) ---")
    surface_engine = Vectors3D("x**2 + y**2")
    print(surface_engine.level_surface(C=4))


    # --- STEP 4: Extrema & Function Analysis (Multivariable) ---
    print("\n\n--- 4. Testing FInvestigator (Local Extrema via Hessian) ---")
    extrema_engine = FInvestigator("x**2 + y**2")
    print(extrema_engine.local_extrema_2d())

    print("\n--- Testing FInvestigator (Lagrange Multipliers) ---")
    # Maximize f(x,y) = x*y subject to constraint x + y - 10 = 0
    lagrange_engine = FInvestigator("x * y")
    print(lagrange_engine.lagrange_multipliers("x + y - 10"))

    print("\n--- Testing FInvestigator (Absolute Extrema on Closed Domain) ---")
    # Absolute max/min of f(x,y) = x^2 + y^2 - x - y on closed disk x^2 + y^2 <= 1
    domain_engine = FInvestigator("x**2 + y**2 - x - y")
    print(domain_engine.absolute_extrema("x**2 + y**2 - 1"))


    # --- STEP 5: Double Integrals & Applications ---
    print("\n\n--- 5. Testing Integrals (Double Integral - Volume) ---")
    integ_engine = Integrals("x * y")
    print(integ_engine.double_integral(inner_var="x", inner_lower="0", inner_upper="2",
                                       outer_var="y", outer_lower="0", outer_upper="1"))

    print("\n--- Testing Integrals (Area of General Domain) ---")
    print(integ_engine.area_2d(inner_var="y", inner_lower="x**2", inner_upper="x",
                               outer_var="x", outer_lower="0", outer_upper="1"))

    print("\n--- Testing Integrals (Polar Double Integral) ---")
    polar_engine = Integrals("exp(-(x**2 + y**2))")
    print(polar_engine.polar_double_integral(r_lower="0", r_upper="R", theta_lower="0", theta_upper="2*pi"))

    print("\n--- Testing Integrals (Center of Mass) ---")
    com_engine = Integrals("0")
    print(com_engine.center_of_mass(density_str="x + y",
                                    inner_var="x", inner_lower="0", inner_upper="1",
                                    outer_var="y", outer_lower="0", outer_upper="1"))


    # --- STEP 6: Differential Equations ---
    print("\n\n--- 6. Testing DiffEq (Separable ODE without Initial Condition) ---")
    ode_engine = DiffEq("x * y")
    print(ode_engine.solve_separable())

    print("\n--- Testing DiffEq (Separable ODE with Initial Condition) ---")
    ode_engine_ic = DiffEq("x * y")
    x_sym = sp.Symbol('x')
    f_sym = sp.Function('f')(x_sym)
    ic = {f_sym.subs(x_sym, 0): 2}
    print(ode_engine_ic.solve_separable(initial_condition=ic))
    print("\n==================================================")


if __name__ == "__main__":
    run_calculus2_checks()