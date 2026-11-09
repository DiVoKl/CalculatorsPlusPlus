import sympy as sp
from typing import Dict, Any, Tuple

def parse_expression(expr_str: str, symbols_str: str = "x y z t") -> Tuple[sp.Expr, Tuple[sp.Symbol, ...]]:
    """
    Normalizes mathematical string syntax, instantiates multi-variable symbols,
    and returns a symbolic SymPy expression object.
    """
    try:
        symbols = sp.symbols(symbols_str)
        cleaned_str = expr_str.replace('^', '**')  # Converts user exponent syntax
        parsed_expr = sp.sympify(cleaned_str)
        return parsed_expr, symbols
    except Exception as e:
        raise ValueError(f"Expression parsing error for '{expr_str}': {str(e)}")

def format_result(result: sp.Expr) -> Dict[str, Any]:
    """
    Wraps raw SymPy evaluation outputs into a standardized dictionary containing
    plain text, LaTeX representations, and raw symbolic objects.
    """
    return {
        "text": str(result),
        "latex": sp.latex(result),
        "sympy_obj": result,
        "success": True
    }


# ==========================================
# Phase 6: Integration and Input Parsing
# ==========================================

# File 1: Common/utils.py (Add or update these parser methods)
import re
from typing import List, Union


def parse_vector_input(input_string: str) -> List[float]:
    """
    Parses a string representation of a vector into a list of floats.
    Example input: "(1.5, 2, -3.4)" or "[1, 2, 3]"
    """
    cleaned = input_string.strip("()[]{} ")
    if not cleaned:
        raise ValueError("Empty vector input.")

    parts = cleaned.split(",")
    try:
        return [float(p.strip()) for p in parts]
    except ValueError:
        raise ValueError(f"Invalid numeric data in vector input: {input_string}")


def parse_matrix_input(input_string: str) -> List[List[float]]:
    """
    Parses a string representation of a matrix into a 2D list of floats.
    Expects rows to be separated by semicolons (;) or newlines, and elements by commas.
    Example input: "[1, 2; 3, 4]" or "1, 2, 3 \n 4, 5, 6"
    """
    cleaned = input_string.strip("[]{} ")
    if not cleaned:
        raise ValueError("Empty matrix input.")

    # Split by semicolon or newline to separate rows
    rows_str = re.split(r'[;\n]', cleaned)

    matrix_data = []
    for row_str in rows_str:
        if not row_str.strip():
            continue
        # Split by comma to separate elements in a row
        parts = row_str.split(",")
        try:
            row = [float(p.strip()) for p in parts]
            matrix_data.append(row)
        except ValueError:
            raise ValueError(f"Invalid numeric data in matrix row: {row_str}")

    # Validate dimensions
    if not matrix_data:
        raise ValueError("Matrix input parsed to empty structure.")
    cols = len(matrix_data[0])
    if any(len(row) != cols for row in matrix_data):
        raise ValueError("All rows in the matrix must have the same number of columns.")

    return matrix_data


# File 2: runner.py (Update your main CLI menu structure)

import sys
# Existing calculus imports...
from common.utils import parse_vector_input, parse_matrix_input
from linearalgebra.Linearalgebra import LinearAlgebraCalculator, Matrix
from linearalgebra.VectorsND import VectorND


def run_linear_algebra_menu():
    """Handles the sub-menu for Linear Algebra operations."""
    calc = LinearAlgebraCalculator()

    while True:
        print("\n=== Linear Algebra Menu ===")
        print("1. Matrix Properties (Rank, Trace, Determinant)")
        print("2. Solve System of Equations (Ax = b)")
        print("3. Vector Space Analysis (Null Space, Row Space, Col Space)")
        print("4. Eigenvalues & Diagonalizability (2x2)")
        print("5. Return to Main Menu")

        choice = input("Select operation (1-5): ").strip()

        if choice == '1':
            mat_str = input("Enter matrix (e.g., '1,2; 3,4'): ")
            try:
                data = parse_matrix_input(mat_str)
                mat = Matrix(data)
                print(f"\nMatrix:\n{mat.data}")
                if mat.is_square():
                    print(f"Trace: {mat.trace()}")
                    print(f"Determinant: {mat.determinant()}")
                print(f"Rank: {mat.rank()}")
                print(f"Is Square: {mat.is_square()}")
                print(f"Is Diagonal: {mat.is_diagonal()}")
                print(f"Is Triangular: {mat.is_triangular()}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '2':
            a_str = input("Enter coefficient matrix A (e.g., '1,2; 3,4'): ")
            b_str = input("Enter solution vector/matrix b (e.g., '5; 11'): ")
            try:
                a_data = parse_matrix_input(a_str)
                b_data = parse_matrix_input(b_str)

                print("\nSolving using general solver...")
                result = calc.solve_system(a_data, b_data)
                print(f"Solution X:\n{result.data}")

                # Try Cramer's rule if square and single column b
                mat_a = Matrix(a_data)
                mat_b = Matrix(b_data)
                if mat_a.is_square() and mat_b.cols == 1 and mat_a.determinant() != 0:
                    print("\nSolving using Cramer's Rule...")
                    cramer_res = calc.cramers_rule(mat_a, mat_b)
                    print(f"Solution X:\n{cramer_res.data}")

            except Exception as e:
                print(f"Error solving system: {e}")

        elif choice == '3':
            mat_str = input("Enter matrix (e.g., '1,2; 3,4'): ")
            try:
                data = parse_matrix_input(mat_str)
                mat = Matrix(data)
                print(f"\nRow Space Basis: {mat.row_space_basis()}")
                print(f"Column Space Basis: {mat.col_space_basis()}")
                print(f"Null Space Basis: {mat.null_space_basis()}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '4':
            mat_str = input("Enter 2x2 matrix (e.g., '1,2; 3,4'): ")
            try:
                data = parse_matrix_input(mat_str)
                mat = Matrix(data)
                if mat.rows != 2 or mat.cols != 2:
                    print("Error: Must be a 2x2 matrix for explicit eigenvalues.")
                    continue

                evals = mat.eigenvalues_2x2()
                print(f"\nEigenvalues: {evals}")
                for val in evals:
                    print(f"Eigenspace Basis for lambda={val}: {mat.eigenspace_basis(val)}")

                print(f"Is Diagonalizable: {mat.is_diagonalizable(evals)}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '5':
            break
        else:
            print("Invalid choice.")


def main():
    """Main application loop."""
    print("Welcome to CalculatorsPlusPlus!")
    while True:
        print("\n=== Main Menu ===")
        print("1. Calculus (Differentiation, Limits, etc.)")
        print("2. Linear Algebra (Matrices, Vectors, Transforms)")
        print("3. Exit")

        main_choice = input("Select module (1-3): ").strip()

        if main_choice == '1':
            # Run your existing calculus menu here
            print("Loading Calculus module...")
            # run_calculus_menu() # Assuming you have something like this
        elif main_choice == '2':
            run_linear_algebra_menu()
        elif main_choice == '3':
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid module selection.")


if __name__ == "__main__":
    main()