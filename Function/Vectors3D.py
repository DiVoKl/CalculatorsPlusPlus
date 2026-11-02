import sympy as sp
from typing import Dict, Any, List
from .base_calculator import BaseCalculator, safe_step


class Vectors3D(BaseCalculator):
    @safe_step
    def calculate(self, **kwargs) -> Dict[str, Any]:
        return self.level_surface(kwargs.get('C', 0))

    def _to_vector(self, vec_list: List[Any]) -> sp.Matrix:
        return sp.Matrix(vec_list)

    @safe_step
    def vector_algebra(self, v1: List[Any], v2: List[Any], v3: List[Any] = None) -> Dict[str, Any]:
        vec1, vec2 = self._to_vector(v1), self._to_vector(v2)
        dot_prod = vec1.dot(vec2)
        cross_prod = vec1.cross(vec2)

        res = {"dot_product": str(dot_prod), "cross_product": str(list(cross_prod)), "success": True}

        if v3 is not None:
            vec3 = self._to_vector(v3)
            mixed_prod = vec1.dot(vec2.cross(vec3))
            res["mixed_product"] = str(mixed_prod)
            res["volume"] = str(sp.Abs(mixed_prod))

        return res

    @safe_step
    def plane_equation(self, point: List[Any], normal: List[Any]) -> Dict[str, Any]:
        p, n = self._to_vector(point), self._to_vector(normal)
        r = sp.Matrix([self.x, self.y, self.z])
        plane_eq = sp.Eq(n.dot(r - p), 0)
        return {"plane_equation": str(plane_eq), "latex": sp.latex(plane_eq), "success": True}

    @safe_step
    def line_equation(self, point: List[Any], direction: List[Any]) -> Dict[str, Any]:
        p, d = self._to_vector(point), self._to_vector(direction)
        line_x = sp.Eq(self.x, p[0] + self.t * d[0])
        line_y = sp.Eq(self.y, p[1] + self.t * d[1])
        line_z = sp.Eq(self.z, p[2] + self.t * d[2])
        return {"parametric_x": str(line_x), "parametric_y": str(line_y), "parametric_z": str(line_z), "success": True}

    @safe_step
    def level_surface(self, C: float = 0) -> Dict[str, Any]:
        level_eq = sp.Eq(self.expr, C)
        return {"level_equation": str(level_eq), "latex": sp.latex(level_eq), "constant": C, "success": True}