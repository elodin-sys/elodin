"""Typed EQL expression builders (Phase 2).

Expressions are validated against a schema when ``strict=True`` and emit
canonical EQL strings for the existing schematic/KDL pipeline.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


class ExprError(ValueError):
    """Raised when an expression fails schema validation."""


class Expr:
    """Typed expression that stringifies to EQL."""

    __slots__ = ("_eql",)

    def __init__(self, eql: str):
        self._eql = eql

    def __str__(self) -> str:
        return self._eql

    def __repr__(self) -> str:
        return f"Expr({self._eql!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Expr):
            return self._eql == other._eql
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._eql)

    def _binop(self, op: str, other: Any) -> Expr:
        return Expr(f"({self} {op} {_as_eql(other)})")

    def __add__(self, other: Any) -> Expr:
        return self._binop("+", other)

    def __radd__(self, other: Any) -> Expr:
        return Expr(f"({_as_eql(other)} + {self})")

    def __sub__(self, other: Any) -> Expr:
        return self._binop("-", other)

    def __rsub__(self, other: Any) -> Expr:
        return Expr(f"({_as_eql(other)} - {self})")

    def __mul__(self, other: Any) -> Expr:
        return self._binop("*", other)

    def __rmul__(self, other: Any) -> Expr:
        return Expr(f"({_as_eql(other)} * {self})")

    def __truediv__(self, other: Any) -> Expr:
        return self._binop("/", other)

    def __rtruediv__(self, other: Any) -> Expr:
        return Expr(f"({_as_eql(other)} / {self})")

    def __neg__(self) -> Expr:
        return Expr(f"(-{self})")

    def __getitem__(self, index: int) -> Expr:
        if not isinstance(index, int):
            raise TypeError("expression index must be an int")
        if index < 0:
            raise ExprError(f"negative index {index} is not allowed")
        return Expr(f"{self}[{index}]")

    def sqrt(self) -> Expr:
        return Expr(f"{self}.sqrt()")

    def abs(self) -> Expr:
        return Expr(f"{self}.abs()")

    def norm(self) -> Expr:
        return Expr(f"{self}.norm()")

    def degrees(self) -> Expr:
        return Expr(f"{self}.degrees()")

    def translate(self, x: Any, y: Any, z: Any) -> Expr:
        return Expr(f"{self}.translate({_as_eql(x)}, {_as_eql(y)}, {_as_eql(z)})")

    def translate_x(self, distance: Any) -> Expr:
        return Expr(f"{self}.translate_x({_as_eql(distance)})")

    def translate_y(self, distance: Any) -> Expr:
        return Expr(f"{self}.translate_y({_as_eql(distance)})")

    def translate_z(self, distance: Any) -> Expr:
        return Expr(f"{self}.translate_z({_as_eql(distance)})")

    def direction(self, x: Any, y: Any, z: Any) -> Expr:
        return Expr(f"{self}.direction({_as_eql(x)}, {_as_eql(y)}, {_as_eql(z)})")


class ComponentHandle(Expr):
    """Schema-backed component reference with element / index access."""

    __slots__ = ("_name", "_element_names", "_shape", "_strict")

    def __init__(
        self,
        name: str,
        *,
        element_names: Sequence[str] | None = None,
        shape: Sequence[int] | None = None,
        strict: bool = True,
    ):
        super().__init__(name)
        self._name = name
        self._element_names = list(element_names or [])
        self._shape = list(shape) if shape is not None else None
        self._strict = strict

    @property
    def element_names(self) -> list[str]:
        return list(self._element_names)

    @property
    def shape(self) -> list[int] | None:
        return list(self._shape) if self._shape is not None else None

    def __getattr__(self, item: str) -> Expr:
        if item.startswith("_"):
            raise AttributeError(item)
        if self._element_names:
            if item not in self._element_names:
                if self._strict:
                    raise ExprError(
                        f"{self._name}.{item} is not an element "
                        f"(known: {', '.join(self._element_names)})"
                    )
            else:
                return Expr(f"{self._name}.{item}")
        return Expr(f"{self._name}.{item}")

    def __getitem__(self, index: int) -> Expr:
        if not isinstance(index, int):
            raise TypeError("component index must be an int")
        if index < 0:
            raise ExprError(f"{self._name}[{index}] negative index is not allowed")
        if self._strict and self._element_names and index >= len(self._element_names):
            raise ExprError(
                f"{self._name}[{index}] out of range for shape ({len(self._element_names)},)"
            )
        if self._strict and self._shape and len(self._shape) == 1 and index >= self._shape[0]:
            raise ExprError(f"{self._name}[{index}] out of range for shape {tuple(self._shape)}")
        return Expr(f"{self._name}[{index}]")


def _as_eql(value: Any) -> str:
    if isinstance(value, Expr):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(float(value)) if isinstance(value, float) else str(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"cannot convert {type(value).__name__} to EQL")


def tuple_expr(*values: Any) -> Expr:
    """Build an EQL tuple expression."""
    return Expr("(" + ", ".join(_as_eql(v) for v in values) + ")")


def pose(
    *,
    quat: Any | None = None,
    pos: Any | None = None,
    order: str = "xyzw",
) -> Expr:
    """Build a 7-vector pose ``(qx,qy,qz,qw, x,y,z)`` or position-only tuple.

    ``order`` is the quaternion component order of ``quat`` when it is a
    4-element sequence of expressions/scalars. Component handles are expanded
    via ``.x/.y/.z/.w`` or ``[0]..[3]`` when element names are unknown.
    """
    if quat is None and pos is None:
        raise ExprError("pose() requires quat and/or pos")

    quat_parts: list[Any] = []
    if quat is not None:
        if order not in ("xyzw", "wxyz"):
            raise ExprError(f"unsupported quaternion order {order!r}")
        parts = _expand_vec(quat, 4, ("x", "y", "z", "w"))
        if order == "wxyz":
            # Input is w,x,y,z → emit x,y,z,w
            w, x, y, z = parts
            quat_parts = [x, y, z, w]
        else:
            quat_parts = list(parts)

    pos_parts: list[Any] = []
    if pos is not None:
        pos_parts = list(_expand_vec(pos, 3, ("x", "y", "z")))

    if quat_parts and pos_parts:
        return tuple_expr(*quat_parts, *pos_parts)
    if quat_parts:
        return tuple_expr(*quat_parts)
    return tuple_expr(*pos_parts)


def sym_mat3(values: Any, *, packing: str = "lower_row") -> Expr:
    """Pack a 6-element covariance vector into a display helper expression.

    Today this emits a 6-tuple in the requested packing order so authors stop
    hand-reordering indices. Full matrix ops land in Tier B.
    """
    elems = list(_expand_vec(values, 6, None))
    if packing == "lower_row":
        # Already lower-triangular row-wise: [00, 10, 11, 20, 21, 22]
        ordered = elems
    elif packing == "upper_row":
        # [00, 01, 02, 11, 12, 22] → lower_row
        a00, a01, a02, a11, a12, a22 = elems
        ordered = [a00, a01, a11, a02, a12, a22]
    else:
        raise ExprError(f"unknown sym_mat3 packing {packing!r}")
    return tuple_expr(*ordered)


def _expand_vec(value: Any, length: int, element_names: Sequence[str] | None) -> list[Any]:
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ExprError(f"expected {length} elements, got {len(value)}")
        return list(value)
    if isinstance(value, ComponentHandle):
        if value.element_names and len(value.element_names) >= length:
            names = value.element_names[:length]
            if element_names and all(n in value.element_names for n in element_names):
                names = list(element_names)
            return [getattr(value, n) for n in names]
        return [value[i] for i in range(length)]
    if isinstance(value, Expr):
        return [value[i] for i in range(length)]
    raise TypeError(f"expected component/expr/sequence, got {type(value).__name__}")


def as_eql_strings(values: Iterable[Any]) -> list[str]:
    return [_as_eql(v) for v in values]
