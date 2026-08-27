"""Class-D fallback data — placeholders, NOT measured BDX truth.

The baseline elodin_package deliberately publishes `aero.derivatives = null`
and no inertia tensor: no lateral-directional/rate/control derivative data or
measured inertia exists for this aircraft yet. Interactive 6-DOF flight still
needs those terms, so this module carries the whitepaper's estimated set
(BDX_Simulation_Whitepaper.md §3.3/§5.2/§7.1) under the handoff's class-D
rules (improvement guide §9.2):

- scenarios must opt in explicitly (`bdx_model.MODE_CLASS_D_6DOF`);
- every selected fallback is logged at startup;
- fallback values are never merged into package data structures; and
- entries are deleted as measured tiers arrive from open-air
  (`flight_dynamics` stage; see the package integration_guide.md §6).

Sign conventions are standard aerospace (source-coefficient frame); the one
frame adapter in aero.py converts to Elodin body axes. `C_mde` is -1.5 here:
the pre-campaign config.py carried +0.5, sign-inverted relative to the
convention it was used in (guide §3.3).
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ClassDAero:
    """Whitepaper §3.3 estimates for terms absent from the package."""

    # Longitudinal rate/control increments (linearization covers CL0/CLa/Cm0/Cma).
    C_Lq: float = 8.0
    C_Lde: float = 0.4
    C_mq: float = -20.0
    C_mde: float = -1.5
    C_Dde: float = 0.02

    # Lateral-directional (all whitepaper class-D estimates).
    C_Ybeta: float = -0.5
    C_Yp: float = 0.0
    C_Yr: float = 0.3
    C_Yda: float = 0.0
    C_Ydr: float = 0.15
    C_lbeta: float = -0.08
    C_lp: float = -0.5
    C_lr: float = 0.1
    C_lda: float = 0.15
    C_ldr: float = 0.01
    C_nbeta: float = 0.1
    C_np: float = -0.03
    C_nr: float = -0.15
    C_nda: float = -0.01
    C_ndr: float = -0.1


@dataclass(frozen=True)
class ClassDInertia:
    """Whitepaper §7.1 component-buildup estimate; no measured tensor exists.

    Ixz is intentionally absent: el.SpatialInertia consumes diagonals only.
    Held constant through fuel burn (no mass-distribution model to scale by).
    """

    ixx_kg_m2: float = 0.8
    iyy_kg_m2: float = 2.5
    izz_kg_m2: float = 3.0


@dataclass(frozen=True)
class ClassDActuators:
    """Generic servo/linkage estimates; EA publishes only linear throws (mm),
    unconverted to hinge angles (handoff §13)."""

    servo_tau_s: float = 0.05
    max_deflection_deg: float = 25.0
    max_rudder_deflection_deg: float = 30.0
    max_rate_deg_s: float = 400.0
    max_rudder_rate_deg_s: float = 350.0


@dataclass(frozen=True)
class ClassDPropulsionDynamics:
    """Spool response estimate (whitepaper §4.1 gives 0.4-0.8 s typical);
    the package propulsion map is steady-state only."""

    spool_tau_s: float = 0.4


@dataclass(frozen=True)
class ClassDFallbacks:
    aero: ClassDAero = ClassDAero()
    inertia: ClassDInertia = ClassDInertia()
    actuators: ClassDActuators = ClassDActuators()
    propulsion: ClassDPropulsionDynamics = ClassDPropulsionDynamics()

    def describe(self) -> list[str]:
        lines = []
        for group_field in fields(self):
            group = getattr(self, group_field.name)
            names = ", ".join(f.name for f in fields(group))
            lines.append(f"class-D {group_field.name}: {names}")
        return lines

    def log_selection(self, scenario_name: str) -> None:
        print(f"[class-D] scenario {scenario_name!r} opted into fallback data:")
        for line in self.describe():
            print(f"[class-D]   {line}")
        print("[class-D] these are whitepaper estimates, not measured BDX values")


FALLBACKS = ClassDFallbacks()
