You *can* totally build a nice BDX example – there just isn’t a plug‑and‑play BDX FDM out there yet. What I *did* find is a pretty rich ecosystem of:

* RC‑scale fixed‑wing simulations you can crib structure and parameters from
* Small‑UAV aero modeling papers that show exactly how to go from geometry → 6‑DOF model
* Micro‑turbine / turbojet models you can simplify for a JetCat‑class engine
* Tools (XFLR5, etc.) that play the same role OpenRocket/RasAero do for rockets

Below is a curated list plus a concrete “stitch them together” plan for an Elodin BDX example.

---

## 1. Grounding the model: BDX specs & geometry sources

**Elite Aerosports / UAS BDX pages**

These give you the basic physical scales and mission envelope:

* **EA BDX (RC sport jet)** – wingspan 2.65 m, length 2.8 m, weight 18.1–19 kg (40–42 lb), recommended turbine power 180–210 N. ([AIR-RC][1])
* **UAS BDX variant** – same core airframe but spec’d as a UAS: 2.65 m span, 2.8 m length, empty weight 42 lb (max 125 lb), turbine 210–320 N, 21 L fuel, >200 kt max speed, 18 g maneuverability, ~50 min cruise endurance. ([Elite Aerosports][2])

**Air‑RC BDX page**

* Confirms the same dimensions and gives an explicit weight range (18.1–19 kg) and power recommendation (180–210 N). ([AIR-RC][1])

**BD‑5 lineage for airfoils**

BDX is explicitly “what a modern BD‑5 would look like.” ([AIR-RC][1])
The BD‑5J used NACA 64‑212 at the root and 64‑218 at the tip. ([BD5][3])

👉 For a **first‑pass aero model**, it’s totally reasonable to:

* Use a NACA 64‑212 (or 64‑212 → 64‑218 taper) as the wing airfoil in XFLR5.
* Use the EA/AIr‑RC geometry for span, area estimate, and tail sizing.
* Assume mass ~19 kg with a thrust range of 180–220 N (T/W ≈ 1 at full burn for the RC version; higher for the UAS version).

---

## 2. “Heavy hitters”: General flight‑dynamics engines to mine

These are the JSBSim/OpenRocket‑level things worth reading, even if you don’t depend on them.

### 2.1 JSBSim & JSBSim‑UAVs

**JSBSim** – open‑source flight dynamics model used by FlightGear, ArduPilot SITL, etc. ([GitHub][4])

* Has a battle‑tested **6‑DOF rigid‑body formulation**, atmosphere, gravity, and an aero coefficient architecture (coefficient tables, polynomials, etc.).
* Python wheels exist, so you can experiment quickly from Python and mirror the structure in Elodin.

**JSBSim‑UAVs** – collection of JSBSim aircraft XMLs for small UAVs, including RC‑scale models such as EPP FPV and Giant Big Stick. ([GitHub][5])

What to crib:

* State layout, sign conventions, and how they factor forces/moments into components.
* Example aero coefficient tables for **RC‑scale aircraft** (span, weight, CLα magnitudes are actually close to BDX scale).
* Example propulsion blocks (even if they’re propellers, you can replace with your turbine model).

### 2.2 FlightGear Rascal 110 RC plane model

**ThunderFly FlightGear‑Rascal** repo: JSBSim + YASim model of the Rascal 110 RC plane. ([GitHub][6])

* Full RC‑plane FDM with geometry, inertia approximations, and tuned aero derivatives.
* Shows how people handle **RC‑scale stability & control derivatives** and how they scale them.

What to crib:

* Inertia estimation workflow (mass distribution, scaling).
* Aero vs control‑input layout: CLδe, Clδa, Cnδr, etc.
* Approach to modeling landing gear and idle thrust in a small aircraft.

### 2.3 NASA SimuPy Flight Vehicle Toolkit

**simupy‑flight** (NASA): a Python 6‑DOF flight vehicle package with a `Vehicle` and `Planet` abstraction and an F‑16 example. ([GitHub][7])

* Very clean, modular implementation of equations of motion in Python.
* Built around explicit force/moment callbacks, which is conceptually close to what you do in `nox`.

What to crib:

* How they structure a **vehicle** object: inertias, aero coeffs, control inputs.
* Their F‑16 example: good pattern for high‑performance jet aero and trimming.

### 2.4 Misc 6‑DOF aircraft repos worth skimming

All of these are small, readable 6‑DOF implementations whose patterns you can echo:

* **SixDOF.jl** – clean 6DOF rigid‑body implementation in Julia. ([GitHub][8])
* **Aircraft‑6DOF‑Simulation (C++17)** – light 6DOF simulator with flat‑earth EoM and a small aero module. ([GitHub][9])
* **Flight6DOF‑Simulator (Python+Streamlit)** – 6DOF light‑aircraft model with real‑time visualization. ([GitHub][10])
* **Aircraft‑Dynamics‑Simulation (MATLAB)** – trim + 6DOF + linearization for a Navion‑class aircraft. ([GitHub][11])

You probably don’t want to *depend* on these, but they’re handy for cross‑checking your math and sign conventions against simple examples.

---

## 3. RC‑scale fixed‑wing examples (closest analogs to your drone example)

These are closer in spirit to what you want: full plant models for small aircraft.

### 3.1 Simulink Drone Reference Application (Multiplex Mentor)

MathWorks’ **Simulink Drone Reference Application** implements a full **6‑DOF dynamics model** of a Multiplex Mentor RC plane (65" foam trainer). ([GitHub][12])

Model includes:

* 6‑DOF rigid‑body equations.
* Aerodynamic force/moment model for a small RC aircraft.
* **Motor, actuator, sensor, and wind models**.

What to crib:

* Top‑level decomposition: airframe block, motor block, actuator block, sensor block. This maps very cleanly to Elodin’s ECS style.
* Their approach to mapping control surface deflections → coef increments (e.g., CLδe, Clδa).
* How they choose trim points and initial conditions for an RC‑scale aircraft.

### 3.2 Nonlinear longitudinal small‑UAV model (Aerosonde)

`Nonlinear_Longitudinal_Dynamic_Simulation_of_UAV` (MATLAB) implements longitudinal dynamics for a small fixed‑wing UAV using aerodynamic parameters from **Aerosonde**. ([GitHub][13])

What to crib:

* A compact, **single‑axis (longitudinal)** non‑linear set of equations.
* Thrust + elevator → pitch/altitude behavior; nice testbed for verifying your pitch dynamics before enabling full 6DOF.

### 3.3 fixed‑wing‑sim: glider with VLM‑based aero

`fixed-wing-sim` implements nonlinear dynamics of a fixed‑wing unmanned glider, with aerodynamic coefficients computed via a vortex lattice method. ([GitHub][14])

Why it’s interesting:

* Shows an end‑to‑end path: **geometry → VLM → aero coeffs → 6DOF sim**.
* Great structural inspiration if you route XFLR5 output into Elodin.

---

## 4. Small‑UAV / RC aerodynamics papers

These are the “RasAero‑like” whitepapers for airplanes: they show how to boil geometry and data into simulation‑ready CL/CD/CM models.

### 4.1 Selig – “Modeling Full‑Envelope Aerodynamics of Small UAVs in Real Time”

Michael S. Selig’s AIAA paper is basically a full framework for RC / small‑UAV aerodynamics over ridiculous envelopes (high‑α, harriers, spins, etc.). ([Applied Aerodynamics Group][15])

Key ideas:

* Component‑based model: wing, tail, fuselage, prop/jet, ground effect, etc. all computed separately and summed.
* Transition away from classical stability derivatives toward more general look‑up/analytic models when you leave the linear regime.
* Supports both prop and **jet configurations** in the same framework.

For Elodin:

* You can **cherry‑pick just the low‑angle, moderate‑Mach subset** for a simple BDX model, and treat it as an advanced reference if you ever want to extend into crazy aerobatics.

### 4.2 “Modeling and identification of a small fixed‑wing UAV…” (Skywalker X8)

This CEAS Aeronautical Journal paper develops a **simulation‑ready 6DOF aerodynamic model** for a Skywalker X8 flying wing and explains the structure of the polynomials and the identification process. ([SpringerLink][16])

Why it’s useful:

* Very clear layout of forces/moments as polynomial functions of (V_a, \alpha, \beta, p, q, r, \delta_e, \delta_a, \delta_r).
* Shows one concrete, reasonably low‑order structure that still fits flight‑test data.

You can largely copy their **model structure** (which terms to include) and then:

* Fill coefficients using a mix of XFLR5 / DATCOM analysis and hand‑tuning.
* Skip the actual system ID step unless you want to test your estimator.

### 4.3 “Modeling and Control of a Fixed‑Wing High‑Speed Mini‑UAV”

This mini‑UAV paper designs a high‑speed 1.2 m aircraft and builds a 6DOF model. Aerodynamics come from **XFLR5 + USAF DATCOM**; they then build a Simulink 6DOF sim. ([IJAST][17])

Key bits:

* Full equations of motion for a fast RC‑scale airframe.
* Explicit description of using **XFLR5 for low‑Re RC aerodynamics** and DATCOM to fill gaps.
* Thrust, servomotor dynamics, and gyro effects included.

For BDX:

* Almost exactly your use case (high‑speed RC aircraft) – it’s a perfect template for your “level of fidelity”.

### 4.4 Preliminary sizing & aerodynamics of an electric RC aircraft (Mazio/Ciliberti)

This bachelor thesis walks through **preliminary sizing and aerodynamics of an electric RC aircraft**, again combining XFLR5 and basic aerodynamics. ([wpage.unina.it][18])

Why it’s helpful:

* Step‑by‑step from mission requirements → wing área → tail volumes → stability margins → XFLR5 analysis.
* Great for sanity‑checking your BDX mass, wing loading, and tail sizing.

### 4.5 Off‑board aerodynamic measurements of small UAVs (The Aeronautical Journal)

This paper uses motion capture to estimate aerodynamic coefficients of small UAVs in glide. ([Cambridge University Press & Assessment][19])

Use it as:

* A benchmark table of typical **CLα, CD0, Cmα, etc.** for small wings at low Reynolds numbers.
* A sanity check on the slopes you get from XFLR5 / DATCOM.

---

## 5. Turbojet / micro gas‑turbine models for your BDX engine

You probably don’t want a full thermodynamic model in an example, but these show you how to collapse a real turbojet into a low‑order dynamic block.

### 5.1 JetCat P100‑RX Identification (DGLR Turbojet paper)

“Identification of a Turbojet Engine using Multi‑Sine Inputs in Ground Testing” models a JetCat P100 RX using a **0D thermodynamic model** and then fits it to ground test data. ([dglr.de][20])

Takeaways:

* Shows a **state‑space turbojet model** with states like spool speed, gas temperature, etc.
* You can simplify this into a **first‑order lag** on rotor speed and a static map (T(n)), then map throttle → spool speed.

For an Elodin example you can:

* Treat engine state as: ( \dot{n} = (n_\text{cmd} - n)/\tau )
* Map thrust as (T = T_\text{max} \cdot f(n)), with f(n) linear or quadratic.
* Use JetCat P220‑class thrust numbers (~49–52 lbf / 220–230 N static). ([Chief Aircraft][21])

### 5.2 Micro gas‑turbine modeling papers

* **“Static and Dynamic Mathematical Modeling of a Micro Gas Turbine”** – presents a linear dynamic model of a micro gas turbine suitable for real‑time simulation. ([OUP Academic][22])
* **“Modeling performance characteristics of a turbojet engine”** – lumped‑parameter 0D turbojet model with first‑order ODEs over conservation laws. ([EA Journals][23])

You can pillage:

* Ideas for how many states you can get away with while remaining “physically flavored”.
* Typical **time constants** for spool up/down and fuel dynamics (hundreds of ms to a few seconds).

For an example, I’d strongly lean toward:

* **1–2 engine states** (spool speed ± maybe gas temp) and a static thrust map.
* Use spec data for T_max and simple throttle → n profile.

---

## 6. Tools: the OpenRocket / RasAero analogs for RC aircraft

### 6.1 XFLR5

**XFLR5** is *the* low‑Re modeling tool for RC aircraft and acts almost exactly like “OpenRocket for airplanes”: ([XFLR5][24])

* Uses XFOIL for 2D airfoil analysis.
* Includes lifting‑line, vortex‑lattice, and 3D panel methods to compute wing and full‑aircraft aerodynamics.
* Has stability analysis modes that will give you **static margins, CLα, Cmα, etc.** for the whole plane.

There are nice intros/tutorials you can follow to drive it: ([University of Notre Dame][25])

Suggested workflow for the BDX example:

1. Sketch a simplified BDX geometry (span, taper, sweep, tail volumes) in XFLR5 using:

   * NACA 64‑212 / 64‑218 or a more “friendly” airfoil if you prefer.
2. Run polars over a range of α and flap/aileron deflections.
3. Export CL, CD, Cm vs α and δ control.
4. Fit low‑order polynomials or store as lookup tables for Elodin.

### 6.2 Tornado VLM and related tools

If you prefer scripting:

* **Tornado** is a MATLAB VLM code used in conceptual aircraft design and education. ([MDO Lab][26])

You could:

* Build a minimal Tornado model of a BDX‑like wing/body, compute aerodynamic coefficients, and embed them as static tables, similar to how you might use RasAero outputs for rockets.

---

## 7. How I’d stitch this into an Elodin BDX example

Given the way your **rocket** and **drone** examples are set up, here’s a concrete path that stays “simple but accurate”:

### 7.1 States and dynamics (structure)

Use the same rigid‑body 6‑DOF structure as:

* JSBSim or SimuPy Flight (for equations of motion). ([GitHub][4])
* Multiplex Mentor / Simulink drone model as a pattern for wiring actuators/engine. ([GitHub][12])

State vector might look like:

* Position in NED: (x, y, z)
* Velocity in body: (u, v, w)
* Attitude: quaternion (or Euler, though Elodin probably likes quats)
* Body rates: (p, q, r)
* Engine spool state: (n) (normalized 0–1)

### 7.2 Airframe parameters

From EA / Air‑RC specs: ([Elite Aerosports][2])

* Mass: m ≈ 19 kg (RC BDX)
* Wingspan: 2.65 m
* Length: 2.8 m
* Wing area S: you can estimate ≈ 0.7–0.8 m² from photos, or better, from your XFLR5 geometry.
* Tail volumes: proportioned similar to BD‑5 or to the high‑speed mini‑UAV paper. ([IJAST][17])

Moments of inertia:

* Use Rascal 110 FDM and Simulink Mentor model as guides for **non‑dimensional inertia ratios** at RC scales, then scale via (I \propto mL^2). ([GitHub][6])

### 7.3 Aerodynamic model

Pick a mid‑complexity structure inspired by:

* Skywalker X8 model structure (force/moment polynomials). ([SpringerLink][27])
* High‑speed mini‑UAV modeling (XFLR5+DATCOM). ([IJAST][17])

For example:

[
\begin{aligned}
C_L &= C_{L_0} + C_{L_\alpha} \alpha + C_{L_q} \frac{q c}{2V} + C_{L_{\delta_e}} \delta_e \
C_D &= C_{D_0} + k C_L^2 \
C_m &= C_{m_0} + C_{m_\alpha}\alpha + C_{m_q} \frac{q c}{2V} + C_{m_{\delta_e}} \delta_e \
C_Y &= C_{Y_\beta} \beta + C_{Y_p} \frac{p b}{2V} + C_{Y_r} \frac{r b}{2V} + C_{Y_{\delta_a}} \delta_a + C_{Y_{\delta_r}} \delta_r \
C_l &= C_{l_\beta}\beta + C_{l_p} \frac{p b}{2V} + C_{l_r} \frac{r b}{2V} + C_{l_{\delta_a}} \delta_a + C_{l_{\delta_r}} \delta_r \
C_n &= C_{n_\beta}\beta + C_{n_p} \frac{p b}{2V} + C_{n_r} \frac{r b}{2V} + C_{n_{\delta_a}} \delta_a + C_{n_{\delta_r}} \delta_r
\end{aligned}
]

Where:

* CL/CD/CM vs α and control deflections are **fit from XFLR5** for your BDX‑like geometry.
* The lateral derivatives can be pulled/adjusted from Rascal 110 / other RC FDMs if you don’t want to compute them.

You can keep stall simple:

* Piecewise CL(α): linear up to α_stall, then roll off.
* Optionally add a crude post‑stall CL plateau if you care about deep stall behavior.

### 7.4 Turbine model

Use a minimal engine block inspired by the JetCat/DGLR and micro‑turbine papers. ([dglr.de][20])

* State: normalized spool speed **n**
  [
  \dot{n} = \frac{n_\text{cmd} - n}{\tau}
  ]
  where (n_\text{cmd}) is throttle command (0–1) and τ ~ 0.3–0.8 s.
* Static thrust map:
  [
  T(n) = T_\text{max}(a_1 n + a_2 n^2)
  ]
  with (T_\text{max} \approx 220\text{ N}) for a P220‑class turbine.
* Optionally modulate thrust with Mach number and altitude via the classic turbojet thrust lapse formula from the turbojet modeling paper.

That is **simple enough for an example**, but:

* Still captures spool lag and non‑linear throttle response.
* Looks familiar to anyone reading turbojet literature.

### 7.5 Control surfaces and actuators

Borrow patterns from:

* Mentor RC plane Simulink model (actuator rate limits, saturations, delays). ([GitHub][12])
* JSBSim / Rascal 110 (control deflection ranges and hinge moments). ([GitHub][6])

Actuator model:

* First‑order or rate‑limited:
  [
  \dot{\delta} = \text{sat}\left(\frac{\delta_\text{cmd} - \delta}{\tau_\text{servo}}, \dot{\delta}_\text{max}\right)
  ]
* Travel limits: e.g., ±25° aileron/elevator, ±30° rudder.

---

## 8. Suggested “reading / cribbing order”

If you want to be efficient about it, I’d go in this order:

1. **SimuPy Flight F‑16 example** – copy their 6DOF structuring into Elodin terms. ([GitHub][7])
2. **Simulink Mentor RC model** – see how a small RC fixed‑wing plant is broken into blocks; mimic that decomposition. ([GitHub][12])
3. **JSBSim‑UAVs / Rascal 110** – grab inertia ratios, control derivative magnitudes, and sign conventions at RC scale. ([GitHub][5])
4. **High‑speed mini‑UAV paper** – use its model structure and XFLR5 workflow as the main blueprint for BDX aero. ([IJAST][17])
5. **JetCat/DGLR turbojet model** – collapse it into a 1‑state turbine block with realistic thrust range. ([dglr.de][20])
6. **Selig full‑envelope paper** – optional deep dive if you later want to extend the BDX example to crazy aerobatics. ([Applied Aerodynamics Group][15])

---

If you’d like, next step I can help you sketch an actual `nox-py` system graph for “BDX:Body”, “BDX:Aero”, “BDX:Turbine”, and “BDX:Actuators”, using these references as the underlying assumptions so the example is both honest about its approximations and easy to tweak.

[1]: https://www.air-rc.com/aircraft/elite-aerosport_ea-bdx_bdx-wbr "AIR-RC"
[2]: https://www.eliteaerosports.com/products/uas-bdx "
      UAS BDX
 – Elite Aerosports"
[3]: https://bd5.com/reprofile.htm?utm_source=chatgpt.com "BD-5 Wing Reprofile Specs"
[4]: https://github.com/JSBSim-Team/jsbsim?utm_source=chatgpt.com "GitHub - JSBSim-Team/jsbsim: An open source flight dynamics & control ..."
[5]: https://github.com/ocereceda/JSBSim-UAVs "GitHub - ocereceda/JSBSim-UAVs: See README"
[6]: https://github.com/ThunderFly-aerospace/FlightGear-Rascal "GitHub - ThunderFly-aerospace/FlightGear-Rascal: JSBsim and YASim small UAV plane model"
[7]: https://github.com/nasa/simupy-flight "GitHub - nasa/simupy-flight"
[8]: https://github.com/byuflowlab/SixDOF.jl?utm_source=chatgpt.com "GitHub - byuflowlab/SixDOF.jl: 6-DOF nonlinear dynamic model (primarily ..."
[9]: https://github.com/chrisfrancisque/Aircraft-6DOF-Simulation?utm_source=chatgpt.com "chrisfrancisque/Aircraft-6DOF-Simulation - GitHub"
[10]: https://github.com/engrkhert/Flight6DOF-Simulator?utm_source=chatgpt.com "GitHub - engrkhert/Flight6DOF-Simulator: A 6-Degree-of-Freedom (6-DOF ..."
[11]: https://github.com/JDOR-Hub/Aircraft-Dynamics-Simulation?utm_source=chatgpt.com "GitHub - JDOR-Hub/Aircraft-Dynamics-Simulation: MATLAB implementation ..."
[12]: https://github.com/mathworks/simulinkDroneReferenceApp?utm_source=chatgpt.com "GitHub - mathworks/simulinkDroneReferenceApp: This Simulink Project ..."
[13]: https://github.com/nrlhozkan/Nonlinear_Longitudinal_Dynamic_Simulation_of_UAV?utm_source=chatgpt.com "nrlhozkan/Nonlinear_Longitudinal_Dynamic_Simulation_of_UAV"
[14]: https://github.com/jrgenerative/fixed-wing-sim?utm_source=chatgpt.com "Simulation of a Fixed-Wing Unmanned Aerial Glider - GitHub"
[15]: https://m-selig.ae.illinois.edu/pubs/Selig-2010-AIAA-2010-7635-FullEnvelopeAeroSim.pdf?utm_source=chatgpt.com "Modeling Full-Envelope Aerodynamics of Small UAVs in Realtime"
[16]: https://link.springer.com/article/10.1007/s13272-025-00816-3?utm_source=chatgpt.com "Modeling and identification of a small fixed-wing UAV using ... - Springer"
[17]: https://ijast.org/volume-3-issue-1-article-4/ "Modeling and Control of a Fixed-Wing High-Speed Mini-UAV - IJAST"
[18]: https://wpage.unina.it/danilo.ciliberti/doc/Mazio.pdf?utm_source=chatgpt.com "PRELIMINARY SIZING AND AERODYNAMICS OF AN ELECTRIC POWERED RC AIRCRAFT"
[19]: https://www.cambridge.org/core/journals/aeronautical-journal/article/offboard-aerodynamic-measurements-of-smalluavs-in-glide-flight-using-motion-tracking/4F2CB54FF6A9C3DF9870305D0EA92044?utm_source=chatgpt.com "Off-board aerodynamic measurements of small-UAVs in glide flight using ..."
[20]: https://www.dglr.de/publikationen/2022/550281.pdf "Identification of a Turbojet Engine using Multi-Sine Inputs in Ground Testing"
[21]: https://www.chiefaircraft.com/radio-control/turbine-jet-aircraft/turbines/jc-71152-0000.html?utm_source=chatgpt.com "P220-RXi Turbine Engine, 52 lbs Thrust, from JetCat, jc-71152-0000"
[22]: https://academic.oup.com/jom/article/29/2/327/5948243?utm_source=chatgpt.com "Static and Dynamic Mathematical Modeling of a Micro Gas Turbine"
[23]: https://eajournals.org/wp-content/uploads/MODELING-PERFORMANCE-CHARACTERISTICS-OF-A-TURBOJET-ENGINE.pdf?utm_source=chatgpt.com "MODELING PERFORMANCE CHARACTERISTICS OF A TURBOJET ENGINE"
[24]: https://www.xflr5.tech/xflr5.htm?utm_source=chatgpt.com "xflr5"
[25]: https://www3.nd.edu/~prumbach/XFLR5_Tutorial_AME40451_FA2024.pdf?utm_source=chatgpt.com "Introduction to XFLR5"
[26]: https://mdolab.engin.umich.edu/wiki/aircraft-design-software?utm_source=chatgpt.com "Aircraft Design Software · MDO Lab"
[27]: https://link.springer.com/article/10.1007/s13272-025-00816-3 "Modeling and identification of a small fixed-wing UAV using estimated aerodynamic angles | CEAS Aeronautical Journal"
