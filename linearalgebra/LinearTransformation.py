from typing import Callable, List, Optional
from Linearalgebra import Matrix
from VectorsND import VectorND


class LinearTransformation:
    """
    Represents a linear transformation T: U -> V.
    Implements standard matrix generation, kernel/image analysis, and
    matrix representation relative to arbitrary bases[cite: 3, 7].
    """

    def __init__(self, transform_func: Callable[['VectorND'], 'VectorND'], domain_dim: int, codomain_dim: int,
                 mod_p: Optional[int] = None):
        """
        :param transform_func: A python function representing the mapping T(u).
        :param domain_dim: Dimension of the input space U.
        :param codomain_dim: Dimension of the output space V.
        """
        self.func = transform_func
        self.domain_dim = domain_dim
        self.codomain_dim = codomain_dim
        self.mod_p = mod_p

    def standard_matrix(self) -> 'Matrix':
        """
        Generates the standard representing matrix A such that T(u) = Au.
        Computes T(e_i) for all standard basis vectors e_i in the domain[cite: 7].
        """
        t_outputs = []
        for i in range(self.domain_dim):
            # Create standard basis vector e_i
            coords = [1 if j == i else 0 for j in range(self.domain_dim)]
            e_i = VectorND(coords)

            # Apply transformation T(e_i)
            t_e_i = self.func(e_i)
            t_outputs.append(t_e_i.coordinates)

        # Transform the list of column vectors into a 2D array of rows for the Matrix class
        mat_data = [
            [t_outputs[col_idx][row_idx] for col_idx in range(self.domain_dim)]
            for row_idx in range(self.codomain_dim)
        ]
        return Matrix(mat_data, self.mod_p)

    def kernel_basis(self) -> List['VectorND']:
        """
        Computes the basis for the kernel (Ker T) by evaluating the
        Null Space (Nul A) of the standard representing matrix[cite: 3, 7].
        """
        std_mat = self.standard_matrix()
        return std_mat.null_space_basis()

    def image_basis(self) -> List['VectorND']:
        """
        Computes the basis for the image (Im T) by evaluating the
        Column Space (Col A) of the standard representing matrix[cite: 3, 7].
        """
        std_mat = self.standard_matrix()
        return std_mat.col_space_basis()

    def is_one_to_one(self) -> bool:
        """
        A linear transformation is one-to-one (injective) if and only if dim(Ker T) = 0[cite: 3].
        """
        ker = self.kernel_basis()
        return len(ker) == 0

    def is_onto(self) -> bool:
        """
        A linear transformation is onto (surjective) if dim(Im T) = dim(V)[cite: 3].
        """
        im = self.image_basis()
        return len(im) == self.codomain_dim

    def matrix_relative_to_bases(self, basis_b: List['VectorND'], basis_c: List['VectorND']) -> 'Matrix':
        """
        Calculates the representing matrix [T]_C^B from basis B (domain) to basis C (codomain).
        Uses the row reduction method: (C | T(b_1), ..., T(b_n)) -> (I | [T]_C^B).
        """
        if len(basis_b) != self.domain_dim or len(basis_c) != self.codomain_dim:
            raise ValueError("Bases do not match domain and codomain dimensions.")

        # 1. Construct matrix of T(b_i) as columns
        t_b_cols = [self.func(b).coordinates for b in basis_b]
        t_b_data = [
            [t_b_cols[col_idx][row_idx] for col_idx in range(self.domain_dim)]
            for row_idx in range(self.codomain_dim)
        ]
        mat_t_b = Matrix(t_b_data, self.mod_p)

        # 2. Construct matrix C from basis_c as columns
        c_cols = [c.coordinates for c in basis_c]
        c_data = [
            [c_cols[col_idx][row_idx] for col_idx in range(self.codomain_dim)]
            for row_idx in range(self.codomain_dim)
        ]
        mat_c = Matrix(c_data, self.mod_p)

        # 3. Augment and row reduce: (C | T(B)) -> (I | [T]_C^B)[cite: 8]
        augmented = mat_c.augment(mat_t_b)
        rref_aug, is_valid = augmented.rref()

        if not is_valid:
            raise ValueError("Basis C does not span the space or is not linearly independent.")

        # 4. Extract the right partition, which is [T]_C^B[cite: 8]
        cols_t_b = mat_t_b.cols
        t_c_b_data = [row[-cols_t_b:] for row in rref_aug.data]
        return Matrix(t_c_b_data, self.mod_p)