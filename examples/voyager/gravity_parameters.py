"""DE440 gravitational parameters used by the Voyager example.

Values are from NAIF's ``gm_de440.tpc`` and converted from km^3/s^2 to
m^3/s^2. Barycenter entries intentionally use the system GM rather than the
central planet's GM.
"""

DE440_GM_M3_S2 = {
    "SUN": 1.3271244004127942e20,
    "MERCURY BARYCENTER": 2.2031868551400003e13,
    "VENUS BARYCENTER": 3.2485859200000000e14,
    "EARTH": 3.9860043550702266e14,
    "MARS BARYCENTER": 4.2828375815756102e13,
    "JUPITER BARYCENTER": 1.2671276409999998e17,
    "SATURN BARYCENTER": 3.7940584841799997e16,
    "URANUS BARYCENTER": 5.7945563999999985e15,
    "NEPTUNE BARYCENTER": 6.8365271005803989e15,
}
