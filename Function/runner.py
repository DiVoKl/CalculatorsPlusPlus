from Function.Differentiation import Differentiation
from Function.Integrals import Integrals
from Function.Limits import Limits
from Function.Fanalyser import Fanalyser
from Function.FInvestigator import FInvestigator


def run_checks():
    print("--- Testing Differentiation ---")
    diff = Differentiation("x**3 + 2*x")
    print(diff.calculate())

    print("\n--- Testing Integrals ---")
    integ = Integrals("2*x")
    print(integ.calculate())

    print("\n--- Testing Limits ---")
    lim = Limits("sin(x)/x")
    print(lim.calculate(0))

    print("\n--- Testing Fanalyser ---")
    analyser = Fanalyser("x**2 - 4")
    print(analyser.calculate())

    print("\n--- Testing FInvestigator (1D Analysis) ---")
    investigator = FInvestigator("x**2 - 1")
    print(investigator.calculate())

    print("\n--- Testing FInvestigator (2D Calculus 2 Analysis) ---")
    investigator_2d = FInvestigator("x**2 + y**2")
    print(investigator_2d.investigate_2d())


if __name__ == "__main__":
    run_checks()