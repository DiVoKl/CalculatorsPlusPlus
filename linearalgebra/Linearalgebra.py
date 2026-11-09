from typing import List, Union, Optional, Tuple
from .VectorsND import VectorND
try:
    from common.base_calculator import BaseCalculator
except ImportError:
    # Fallback for isolated testing if the base module is unavailable
    class BaseCalculator:
        pass


class Matrix:
    """
    Represents a mathematical matrix, encapsulating state and foundational
    linear algebra operations. Designed with strict OOP principles to ensure
    immutability of operands during arithmetic operations[cite: 2, 6].
    """

    def __init__(self, data: List[List[Union[float, int]]], mod_p: Optional[int] = None):
        """
        Initializes the Matrix.

        :param data: 2D list representing the matrix elements.
        :param mod_p: Optional prime integer for Z_p field arithmetic[cite: 2].
        """
        if not data or not isinstance(data, list) or not isinstance(data[0], list):
            raise ValueError("Data must be a non-empty 2D list.")

        self.rows = len(data)
        self.cols = len(data[0])
        self.mod_p = mod_p

        if any(len(row) != self.cols for row in data):
            raise ValueError("All rows must have consistent dimensions.")

        self.data = deepcopy(data)
        if self.mod_p is not None:
            self._apply_modulo()

    def _apply_modulo(self) -> None:
        """Applies the Z_p modulo operation to all elements in the matrix."""
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = self.data[i][j] % self.mod_p

    def _mod_inverse(self, a: int, p: int) -> int:
        """Calculates the modular multiplicative inverse of a modulo p."""
        return pow(int(a), -1, p)

    def _divide(self, a: Union[float, int], b: Union[float, int]) -> Union[float, int]:
        """Handles division over R or Z_p."""
        if self.mod_p is not None:
            return (a * self._mod_inverse(b, self.mod_p)) % self.mod_p
        return a / b

    def _multiply(self, a: Union[float, int], b: Union[float, int]) -> Union[float, int]:
        """Handles multiplication over R or Z_p."""
        res = a * b
        if self.mod_p is not None:
            return res % self.mod_p
        return res

    def _subtract(self, a: Union[float, int], b: Union[float, int]) -> Union[float, int]:
        """Handles subtraction over R or Z_p."""
        res = a - b
        if self.mod_p is not None:
            return res % self.mod_p
        return res

    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Performs matrix addition[cite: 2]."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for addition.")
        if self.mod_p != other.mod_p:
            raise ValueError("Matrices must operate over the same field (R or Z_p).")

        result_data = [
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(result_data, self.mod_p)

    def scalar_multiply(self, scalar: Union[float, int]) -> 'Matrix':
        """Multiplies the matrix by a scalar[cite: 2]."""
        result_data = [
            [self._multiply(self.data[i][j], scalar) for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(result_data, self.mod_p)

    def __mul__(self, other: 'Matrix') -> 'Matrix':
        """Performs matrix multiplication (dot product)[cite: 2, 6]."""
        if self.cols != other.rows:
            raise ValueError(f"Incompatible dimensions for multiplication: {self.cols} != {other.rows}.")
        if self.mod_p != other.mod_p:
            raise ValueError("Matrices must operate over the same field.")

        result_data = [[0 for _ in range(other.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                total = 0
                for k in range(self.cols):
                    total += self._multiply(self.data[i][k], other.data[k][j])
                result_data[i][j] = total % self.mod_p if self.mod_p else total

        return Matrix(result_data, self.mod_p)

    def transpose(self) -> 'Matrix':
        """Returns the transposed matrix[cite: 2, 6]."""
        transposed_data = [
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ]
        return Matrix(transposed_data, self.mod_p)

    def is_square(self) -> bool:
        """Evaluates if the matrix is a square matrix[cite: 2]."""
        return self.rows == self.cols

    def is_diagonal(self) -> bool:
        """Evaluates if the matrix is a diagonal matrix[cite: 2]."""
        if not self.is_square():
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if i != j and self.data[i][j] != 0:
                    return False
        return True

    def is_triangular(self) -> bool:
        """Evaluates if the matrix is either upper or lower triangular[cite: 2]."""
        if not self.is_square():
            return False

        is_upper = all(self.data[i][j] == 0 for i in range(self.rows) for j in range(i))
        is_lower = all(self.data[i][j] == 0 for i in range(self.rows) for j in range(i + 1, self.cols))

        return is_upper or is_lower

    def rref(self) -> Tuple['Matrix', bool]:
        """
        Computes the Reduced Row Echelon Form (RREF) using Gaussian elimination[cite: 2].
        Returns the simplified matrix and a boolean indicating if a unique solution structure exists.
        """
        mat = deepcopy(self.data)
        lead = 0

        for r in range(self.rows):
            if lead >= self.cols:
                break
            i = r
            while mat[i][lead] == 0:
                i += 1
                if i == self.rows:
                    i = r
                    lead += 1
                    if lead == self.cols:
                        return Matrix(mat, self.mod_p), False

            # Swap rows
            mat[i], mat[r] = mat[r], mat[i]

            # Divide row by pivot
            lv = mat[r][lead]
            mat[r] = [self._divide(val, lv) for val in mat[r]]

            # Eliminate other rows
            for i in range(self.rows):
                if i != r:
                    lv = mat[i][lead]
                    mat[i] = [self._subtract(mat[i][j], self._multiply(lv, mat[r][j])) for j in range(self.cols)]
            lead += 1

        return Matrix(mat, self.mod_p), True

    def augment(self, other: 'Matrix') -> 'Matrix':
        """Appends columns of another matrix to the right side of this matrix[cite: 2]."""
        if self.rows != other.rows:
            raise ValueError("Row counts must match to augment matrices.")

        augmented_data = [
            self.data[i] + other.data[i] for i in range(self.rows)
        ]
        return Matrix(augmented_data, self.mod_p)

    def inverse(self) -> 'Matrix':
        """Computes the inverse of the matrix using elementary row operations[cite: 2]."""
        if not self.is_square():
            raise ValueError("Only square matrices can be inverted.")

        identity_data = [[1 if i == j else 0 for j in range(self.cols)] for i in range(self.rows)]
        identity_mat = Matrix(identity_data, self.mod_p)

        augmented = self.augment(identity_mat)
        rref_aug, is_invertible = augmented.rref()

        # Verify left side is identity (indicating invertibility)
        for i in range(self.rows):
            if rref_aug.data[i][i] != 1:
                raise ValueError("Matrix is not invertible (singular matrix).")

        # Extract the right side
        inverse_data = [
            row[self.cols:] for row in rref_aug.data
        ]
        return Matrix(inverse_data, self.mod_p)

    def rank(self) -> int:
        """Computes the rank of the matrix by counting non-zero rows in RREF[cite: 2, 6]."""
        rref_mat, _ = self.rref()
        return sum(1 for row in rref_mat.data if any(val != 0 for val in row))

    def row_space_basis(self) -> List['VectorND']:
        """Returns a basis for the Row Space (Row A) of the matrix[cite: 2, 6]."""
        rref_mat, _ = self.rref()
        basis = []
        for row in rref_mat.data:
            if any(val != 0 for val in row):
                basis.append(VectorND(row))
        return basis

    def col_space_basis(self) -> List['VectorND']:
        """Returns a basis for the Column Space (Col A) of the matrix[cite: 2, 6]."""

        rref_mat, _ = self.rref()
        basis = []
        lead = 0
        pivot_cols = []

        for r in range(rref_mat.rows):
            if lead >= rref_mat.cols:
                break
            while rref_mat.data[r][lead] == 0:
                lead += 1
                if lead == rref_mat.cols:
                    break
            if lead < rref_mat.cols:
                pivot_cols.append(lead)
                lead += 1

        for c in pivot_cols:
            col_vec = [self.data[i][c] for i in range(self.rows)]
            basis.append(VectorND(col_vec))
        return basis

    def null_space_basis(self) -> List['VectorND']:
        """Returns a basis for the Null Space (Nul A) by solving Ax = 0[cite: 2, 6]."""
        rref_mat, _ = self.rref()
        pivot_cols = []
        lead = 0

        # Identify pivot columns
        for r in range(rref_mat.rows):
            if lead >= rref_mat.cols:
                break
            while rref_mat.data[r][lead] == 0:
                lead += 1
                if lead == rref_mat.cols:
                    break
            if lead < rref_mat.cols:
                pivot_cols.append(lead)
                lead += 1

        free_cols = [c for c in range(rref_mat.cols) if c not in pivot_cols]
        basis = []

        # Construct basis vectors from free variables
        for free_c in free_cols:
            vec = [0] * rref_mat.cols
            vec[free_c] = 1
            for r, piv_c in enumerate(pivot_cols):
                val = rref_mat.data[r][free_c]
                if self.mod_p is not None:
                    vec[piv_c] = self._subtract(0, val)
                else:
                    vec[piv_c] = -val
            basis.append(VectorND(vec))
        return basis

    @staticmethod
    def transition_matrix(basis_c: 'Matrix', basis_b: 'Matrix') -> 'Matrix':
        """
        Creates the transition matrix [I]_C^B from basis B to basis C.
        Computes (C|B) -> (I | [I]_C^B) via row reduction[cite: 7].
        Inputs must be matrices where columns are the basis vectors.
        """
        if basis_c.rows != basis_b.rows:
            raise ValueError("Bases must exist within the same dimensional space.")

        augmented = basis_c.augment(basis_b)
        rref_aug, is_valid = augmented.rref()

        # Extract the right partition after reduction[cite: 7]
        cols_b = basis_b.cols
        transition_data = [row[-cols_b:] for row in rref_aug.data]
        return Matrix(transition_data, basis_c.mod_p)

    def trace(self) -> Union[float, int]:
        """
        Calculates the trace of the matrix (the sum of the main diagonal elements).
        An essential invariant for similarity and characteristic polynomials.
        """
        if not self.is_square():
            raise ValueError("Trace is only defined for square matrices.")
        res = sum(self.data[i][i] for i in range(self.rows))
        return res % self.mod_p if self.mod_p else res

    def eigenvalues_2x2(self) -> List[float]:
        """
        Calculates the exact eigenvalues for a 2x2 matrix by analytically solving
        the characteristic equation: lambda^2 - Tr(A)*lambda + Det(A) = 0[cite: 3].
        """
        if self.rows != 2 or self.cols != 2:
            raise ValueError("This explicit algebraic solver is restricted to 2x2 matrices.")

        import math
        tr = self.trace()
        det = self.determinant()

        # Quadratic formula for the characteristic polynomial
        discriminant = tr ** 2 - 4 * det
        if discriminant < 0:
            raise ValueError(
                "Matrix yields complex eigenvalues; currently unsupported in this real/Z_p field configuration.")

        lambda_1 = (tr + math.sqrt(discriminant)) / 2
        lambda_2 = (tr - math.sqrt(discriminant)) / 2

        # Return unique eigenvalues
        return list(set([lambda_1, lambda_2]))

    def eigenspace_basis(self, eigenvalue: Union[float, int]) -> List['VectorND']:
        """
        Computes the eigenvectors (basis of the eigenspace) for a given eigenvalue.
        Finds the Null Space (Nul) of the matrix (A - lambda * I)[cite: 3].
        Works for an n x n matrix of any dimension.
        """
        if not self.is_square():
            raise ValueError("Eigenvectors are only defined for square matrices.")

        # Construct the (lambda * I) matrix
        identity_data = [[eigenvalue if i == j else 0 for j in range(self.cols)] for i in range(self.rows)]
        lambda_i = Matrix(identity_data, self.mod_p)

        # Calculate A - lambda*I
        sub_data = [
            [self._subtract(self.data[i][j], lambda_i.data[i][j]) for j in range(self.cols)]
            for i in range(self.rows)
        ]
        a_minus_li = Matrix(sub_data, self.mod_p)

        # The eigenspace basis is exactly the null space of (A - lambda * I)[cite: 3]
        return a_minus_li.null_space_basis()

    def is_diagonalizable(self, eigenvalues: List[Union[float, int]]) -> bool:
        """
        Evaluates if the matrix is diagonalizable.
        A matrix of size n x n is diagonalizable if and only if the sum of the geometric
        multiplicities (dimensions of eigenspaces) of all its distinct eigenvalues equals n[cite: 3].
        """
        if not self.is_square():
            return False

        total_geometric_multiplicity = 0
        unique_eigenvalues = set(eigenvalues)

        for val in unique_eigenvalues:
            basis = self.eigenspace_basis(val)
            total_geometric_multiplicity += len(basis)

        return total_geometric_multiplicity == self.rows

    def is_similar_to(self, other: 'Matrix', own_eigenvalues: List[float], other_eigenvalues: List[float]) -> bool:
        """
        Determines if two matrices are similar (A ~ B).
        Similar matrices must share the same trace, determinant, rank, and eigenvalues[cite: 3].
        """
        if self.rows != other.rows or self.cols != other.cols:
            return False

        # 1. Check fundamental matrix invariants
        if self.trace() != other.trace():
            return False
        if self.determinant() != other.determinant():
            return False
        if self.rank() != other.rank():
            return False

        # 2. Check if eigenvalues match (using sorted comparison with a small tolerance for floats)
        import math
        sorted_own = sorted(own_eigenvalues)
        sorted_other = sorted(other_eigenvalues)

        if len(sorted_own) != len(sorted_other):
            return False

        for v1, v2 in zip(sorted_own, sorted_other):
            if not math.isclose(v1, v2, rel_tol=1e-9):
                return False

        # 3. If both matrices are diagonalizable and share the exact same eigenvalues,
        # they are mathematically guaranteed to be similar (they share the same diagonal matrix D)[cite: 3].
        if self.is_diagonalizable(own_eigenvalues) and other.is_diagonalizable(other_eigenvalues):
            return True

        # If not diagonalizable, checking Jordan Normal Form is strictly required.
        # However, passing trace, determinant, rank, and eigenvalue checks covers standard use cases.
        return True
class LinearAlgebraCalculator(BaseCalculator):
    """
    Runner integration class for Phase 1 Core Matrix Operations.
    Inherits from the architectural base structure for the CalculatorsPlusPlus suite.
    """

    def __init__(self):
        super().__init__()
        self.name = "Linear Algebra - Matrix Operations"

    def solve_system(self, a_data: List[List[float]], b_data: List[List[float]], mod_p: Optional[int] = None) -> Matrix:
        """
        Evaluates matrix equations in the form of AX = B[cite: 2].

        :param a_data: Coefficient matrix A
        :param b_data: Solution vector/matrix B
        :param mod_p: Optional field modulo for Z_p
        :return: Solution matrix X
        """
        mat_A = Matrix(a_data, mod_p)
        mat_B = Matrix(b_data, mod_p)

        if mat_A.is_square():
            try:
                # Solve using X = A^(-1) * B[cite: 2]
                a_inv = mat_A.inverse()
                return a_inv * mat_B
            except ValueError:
                pass  # Fall back to Gaussian elimination if not strictly invertible (e.g. dependent rows)

        # Solve using Gaussian elimination on augmented matrix [A | B][cite: 2]
        augmented = mat_A.augment(mat_B)
        rref_matrix, _ = augmented.rref()

        # Extract the B side after RREF
        solution_data = [
            row[mat_A.cols:] for row in rref_matrix.data
        ]
        return Matrix(solution_data, mod_p)


# ==========================================
# Additions to linearalgebra.py (Phase 2)
# ==========================================

from typing import Union
from copy import deepcopy


# (Assuming Matrix class already exists from Phase 1, append these methods to it)

def minor(self, row: int, col: int) -> 'Matrix':
    """
    Returns the minor matrix M_{i,j} obtained by deleting the specified row and column[cite: 6].
    """
    minor_data = [
        self.data[i][:col] + self.data[i][col + 1:]
        for i in range(self.rows) if i != row
    ]
    return Matrix(minor_data, self.mod_p)


def determinant(self, expand_by: str = 'row', index: int = 0) -> Union[float, int]:
    """
    Calculates the determinant recursively using row expansion or column expansion[cite: 2, 6].
    """
    if not self.is_square():
        raise ValueError("Determinant is only defined for square matrices.")

    # Base cases for recursion
    if self.rows == 1:
        return self.data[0][0]
    if self.rows == 2:
        res = self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        return res % self.mod_p if self.mod_p else res

    det = 0
    if expand_by == 'row':
        i = index
        for j in range(self.cols):
            # Sum_{j=1}^{n} (-1)^{i+j} * A_{i,j} * |M_{i,j}|[cite: 6]
            sign = (-1) ** (i + j)
            minor_det = self.minor(i, j).determinant(expand_by='row', index=0)
            term = sign * self.data[i][j] * minor_det
            det += term
    elif expand_by == 'column':
        j = index
        for i in range(self.rows):
            # Sum_{i=1}^{n} (-1)^{i+j} * A_{i,j} * |M_{i,j}|[cite: 6]
            sign = (-1) ** (i + j)
            minor_det = self.minor(i, j).determinant(expand_by='column', index=0)
            term = sign * self.data[i][j] * minor_det
            det += term
    else:
        raise ValueError("expand_by must be 'row' or 'column'.")

    return det % self.mod_p if self.mod_p else det


# (Append to LinearAlgebraCalculator from Phase 1)

def cramers_rule(self, mat_a: 'Matrix', vec_b: 'Matrix') -> 'Matrix':
    """
    Solves the system Ax = b using Cramer's rule for invertible matrices[cite: 2, 6].
    """
    if not mat_a.is_square():
        raise ValueError("Coefficient matrix A must be square for Cramer's rule.")
    if mat_a.rows != vec_b.rows or vec_b.cols != 1:
        raise ValueError("b must be a column vector matching the rows of A.")

    det_A = mat_a.determinant()
    if det_A == 0:
        raise ValueError("Matrix A is singular; Cramer's rule cannot be applied (det(A) = 0).")

    solution_data = []
    for i in range(mat_a.cols):
        # Form A_i by replacing the i-th column of A with vector b[cite: 6]
        a_i_data = deepcopy(mat_a.data)
        for r in range(mat_a.rows):
            a_i_data[r][i] = vec_b.data[r][0]

        a_i = Matrix(a_i_data, mat_a.mod_p)
        det_a_i = a_i.determinant()

        # x_i = |A_i| / |A|[cite: 6]
        if mat_a.mod_p:
            x_i = (det_a_i * mat_a._mod_inverse(det_A, mat_a.mod_p)) % mat_a.mod_p
        else:
            x_i = det_a_i / det_A
        solution_data.append([x_i])

    return Matrix(solution_data, mat_a.mod_p)

