import math
from typing import List, Union
from Linearalgebra import Matrix

class VectorND:
    """
    Represents an abstract vector in n-dimensional space (R^n).
    Supports operations between vectors in R^n[cite: 2].
    """

    def __init__(self, coordinates: List[Union[float, int]]):
        if not coordinates:
            raise ValueError("Vector must have at least one coordinate.")
        self.coordinates = list(coordinates)
        self.dim = len(self.coordinates)

    def __add__(self, other: 'VectorND') -> 'VectorND':
        """Performs vector addition in R^n[cite: 2]."""
        if self.dim != other.dim:
            raise ValueError(f"Cannot add vectors of different dimensions: {self.dim} and {other.dim}.")
        return VectorND([self.coordinates[i] + other.coordinates[i] for i in range(self.dim)])

    def __sub__(self, other: 'VectorND') -> 'VectorND':
        """Performs vector subtraction in R^n[cite: 2]."""
        if self.dim != other.dim:
            raise ValueError(f"Cannot subtract vectors of different dimensions: {self.dim} and {other.dim}.")
        return VectorND([self.coordinates[i] - other.coordinates[i] for i in range(self.dim)])

    def scalar_multiply(self, scalar: Union[float, int]) -> 'VectorND':
        """Multiplies the vector by a scalar value[cite: 2]."""
        return VectorND([c * scalar for c in self.coordinates])

    def __mul__(self, scalar: Union[float, int]) -> 'VectorND':
        """Overloads the * operator for scalar multiplication."""
        return self.scalar_multiply(scalar)

    def __rmul__(self, scalar: Union[float, int]) -> 'VectorND':
        """Overloads the * operator for scalar multiplication (scalar * vector)."""
        return self.scalar_multiply(scalar)

    def dot_product(self, other: 'VectorND') -> Union[float, int]:
        """Calculates the dot product of two n-dimensional vectors."""
        if self.dim != other.dim:
            raise ValueError("Vectors must be of the same dimension for dot product.")
        return sum(self.coordinates[i] * other.coordinates[i] for i in range(self.dim))

    def norm(self) -> float:
        """Calculates the Euclidean norm (length/magnitude) of the vector."""
        return math.sqrt(self.dot_product(self))

    def cross_product(self, other: 'VectorND') -> 'VectorND':
        """
        Calculates the cross product. Strictly limited to 3-dimensional vectors.
        Preserved to ensure backward compatibility with earlier 3D vector features.
        """
        if self.dim != 3 or other.dim != 3:
            raise ValueError("Cross product is only defined for 3-dimensional vectors.")

        a1, a2, a3 = self.coordinates
        b1, b2, b3 = other.coordinates

        return VectorND([
            a2 * b3 - a3 * b2,
            a3 * b1 - a1 * b3,
            a1 * b2 - a2 * b1
        ])

    def to_column_matrix(self):
        """
        Converts the vector into an n x 1 column Matrix format.
        Used for matrix-vector multiplication T(u) = Au[cite: 7].
        """

        return Matrix([[c] for c in self.coordinates])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorND):
            return False
        if self.dim != other.dim:
            return False
        # Using a small tolerance for floating point comparisons
        return all(math.isclose(self.coordinates[i], other.coordinates[i], rel_tol=1e-9) for i in range(self.dim))

    def __str__(self) -> str:
        coords_str = ", ".join(f"{c:g}" for c in self.coordinates)
        coords_str = ", ".join(f"{c:g}" for c in self.coordinates)
        return f"({coords_str})"

    def __repr__(self) -> str:
        return f"VectorND({self.coordinates})"

    @staticmethod
    def are_linearly_independent(vectors: List['VectorND']) -> bool:
        """
        Verifies if a set of vectors is linearly independent by evaluating the rank
        of their combined column matrix[cite: 2, 6].
        """
        if not vectors:
            return True

        # Assemble matrix where each vector is a column
        mat_data = [[vec.coordinates[i] for vec in vectors] for i in range(vectors[0].dim)]
        mat = Matrix(mat_data)

        # Independent if rank equals the number of vectors[cite: 2, 6]
        return mat.rank() == len(vectors)

    def coordinate_vector(self, basis: List['VectorND']) -> 'VectorND':
        """
        Calculates the unique coordinate vector [v]_B relative to a specified basis B[cite: 6, 7].
        """
        if not basis:
            raise ValueError("Basis cannot be empty.")
        if self.dim != basis[0].dim:
            raise ValueError("Vector and basis must be in the same dimensional space.")

        # Build matrix B from basis columns
        b_data = [[b_vec.coordinates[i] for b_vec in basis] for i in range(self.dim)]
        mat_b = Matrix(b_data)

        # Convert self to a column matrix for augmentation
        v_data = [[c] for c in self.coordinates]
        mat_v = Matrix(v_data)

        # Solve Bx = v using augmented matrix[cite: 2, 7]
        augmented = mat_b.augment(mat_v)
        rref_aug, _ = augmented.rref()

        # Extract the final column (the coordinates)
        coord_data = [rref_aug.data[i][-1] for i in range(len(basis))]
        return VectorND(coord_data)