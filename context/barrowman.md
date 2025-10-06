+++
title = "Barrowman"
description = "The Theoretical Prediction of the Center of Pressure"
draft = false
weight = 140
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 40
+++

# The Theoretical Prediction of the Center of Pressure

by
James S. Barrowman (#6883)
and
Judith A. Barrowman (#7489)

Presented as a
RESEARCH AND DEVELOPMENT
Project at
NARAM-8
on
August 18, 1966

## TABLE OF CONTENTS

1. Background............................................1
2. Objectives............................................3
3. Method of Procedure...................................3
4. Assumptions...........................................3
5. Portioning of Rocket..................................4
6. Symbols...............................................6
7. Body Normal Force Coefficient Slope...................8
8. Body Center of Pressure...............................13
9. Fin Normal Force Coefficient Slope....................22
10. Fin Center of Pressure...............................28
11. Interference Effects.................................36
12. Combination Calculations.............................38
13. Experimental Verifications...........................39
14. Conclusions..........................................39
15. References...........................................51
16. Compilation of Derived Equations.....................52

# CENTER OF PRESSURE CALCULATIONS

## Background

The most important characteristic of a model rocket is its stability.

A rocket's static stability is affected by the relative positions of its center of gravity (C.G.) and its center of pressure (C.P.). As is well known, the static margin of a rocket is the distance between the C.G. and C.P. A rocket is statically stable if the C.P. is behind the C.G.; also, it is more stable for a larger static margin.

The center of gravity of a rocket is easily determined by a simple balance test. The center of pressure determination is much more difficult. Many methods for determining the C.P. have been proposed. The majority of them boil down to the determination of the center of lateral area which is the C.P. of the rocket if it were flying sideways. The center of lateral area is a conservative estimate; that is, it is forward of the actual C.P.; and, as such, is not a bad method for the beginner. However, as model rocketry becomes more sophisticated, and rocketeers become more concerned with reducing the static margin to the bare minimum; to reduce weathercocking a more accurate method is called for.

The center of pressure is the furthest aft at zero angle of attack. By calculating the C.P. at α=0°, therefore, one has the least conservative value. It is this value to which any safety margin should be added.

The advantage of this method is that it reduces the static margin to a safe and predetermined minimum.

The existence of an easily applicable set of equations for the calculation of the C.P. allows the rocketeer to truly design his
birds before any construction takes place. Since, by necessity, the derivation of any equations requires a predetermined configuration, a method of iteration must be used to determine the final design.

## Objective

To derive the subsonic theoretical center of pressure equations of a general rocket configuration. And to simplify the resulting equations without a great loss of accuracy so that the average leader can use them.

## Method of Approach

1. Divide rocket into separate portions.
2. Analyse each portion separately.
3. Analyse the interference effects between the portions.
4. Simplify the calculations where necessary.
5. Recombine the results of the separate analyses to obtain the final answer.
6. Verify Analysis by experiment.

## Assumptions

1. Flow over rocket is potential flow, i.e., no vortices or friction.
2. Point of the nose is sharp.
3. Fins are thin flat plates with no cant.
4. The angle of attack is very near zero.
5. The flow is steady state and subsonic.
6. The rocket is a rigid body.
7. The rocket is axially symmetric.

# Portioning of Rocket

A rocket is, in general, composed of the following portions:

1. **Nose**
   - ![Cone Diagram Placeholder](#)
   - ![Ogive Diagram Placeholder](#)

2. **Cylindrical Body**
   - Different diameters before and after any conical shoulder.

3. **Conical Shoulder**
   - ![Conical Shoulder Diagram Placeholder](#)

4. **Conical Boattail**
   - ![Conical Boattail Diagram Placeholder](#)

5. **Fins**

   ![Fins Diagram Placeholder](#)

## Symbols

- **A** = Reference area = π/4 * d²
- **Af** = Area of one fin
- **R** = Aspect ratio
- **C** = General fin chord length
- **Cm** = Nondimensional aerodynamic pitching moment coefficient = M/½ρV²d
- **Cmx** = Slope of moment coefficient curve at α = 0, ∂Cm/∂α | α=0
- **Cma** = Mean aerodynamic chord
- **CN** = Nondimensional aerodynamic normal force
- **CNα** = Slope of the force coefficient at α = 0, ∂Cy/∂α | α=0
- **Cr** = Root chord length
- **Ct** = Tip chord length
- **d** = Reference length = diameter at the base of the nose
- **F** = Diederich's correlation parameter
- **f** = Thickness ratio
- **K** = Interference factor
- **L** = Body portion length
- **l** = Length of fin mid-chord line
- **M(x)** = Local aerodynamic pitching moment about the nose of the body portion
- **n(α)** = Local aerodynamic normal force
- **q** = Dynamic pressure = ½ρV²
- **r** = Radius of body between fins
- **S** = Fin semi-span
- **S(x)** = Local cross-sectional area
- **V∞** = Free stream velocity
- **V'** = Body portion volume
- **W(x)** = Local downwash velocity
- **x** = General distance along body
- **X̅** = Center of pressure location
- **Xf** = Distance between the nose tip and the leading edge of the fin root chord
- **Xr** = Distance between the root chord leading edge and the tip chord leading edge in a direction parallel to the body
- **y** = General fin spanwise coordinate
- **Y̅** = Spanwise location of mean aerodynamic chord
- **α** = Angle of attack
- **λ** = Swept of fin leading edge
- **σ** = Sweep of fin mid-chord line
- **λ** = Fin taper ratio = Ct/Cr
- **ρ** = Free stream density

## Subscripts

- **B** = Body
- **F** = Tail or fins
- **N** = Nose
- **CS** = Conical shoulder
- **CB** = Conical boattail
- **T(B)** = Tail in the presence of the body

## BODY AERODYNAMICS DERIVATIONS

### Normal Force Coefficient Slope

**General:**

For an axially symmetric body of revolution, from reference 4, the subsonic steady state aerodynamic running normal load is given by:

$ n(x) = \rho V∞ \frac{d}{dx} [S(x)w(x)] $ ①

where:

- \( n(x) \) = The running normal load per unit length
- \( \rho \) = Free stream density
- \( V∞ \) = Free stream air speed
- \( S(x) \) = Local cross-sectional area
- \( w(x) \) = Local downwash at a given point on the body

A rigid body has the downwash given by:

$ w(x) = V∞ \alpha $ ②

Thus:

$ n(x) = \rho V∞² \alpha \frac{dS(x)}{dx} $ ③

By the definition of normal force coefficient:

$ C_N(x) = \frac{n(x)}{qA} = \frac{n(x)}{\frac{1}{2}\rho V∞²A} $ ④

Substituting equation ③ into ④:

$ C_N(x) = 2 \frac{\alpha}{A} \frac{dS(x)}{dX} $ ⑤

![Figure 1: Body Station Parameters](#)

This figure illustrates the body station parameters, showing section A-A and the relationships between \( S(x) \), \( n(x) \), \( x \), and the other relevant geometric parameters.

but;

$ A = \frac{\pi}{4} d^2 $

therefore;

$ C_N(x) = \frac{8 \alpha}{\pi d^2} \frac{dS(x)}{dx} $ ⑥

By the definition of the normal force coefficient curve slope;

$ C_{N\alpha}(x) = \left. \frac{dC_N}{d\alpha} \right|_{\alpha=0} = \frac{8}{\pi d^2} \frac{dS(x)}{dx} $ ⑦

In order to obtain the total \( C_{N\alpha} \), Equation ⑦ is integrated over the length of the body;

$ C_{N\alpha} = \int_0^L C_{N\alpha}(x) d x = \int_0^L \frac{8}{\pi d^2} \frac{dS(x)}{dx} dx $ ⑧

Since \(\frac{8}{\pi d^2}\) is not a function of \( x \);

$ C_{N\alpha} = \frac{8}{\pi d^2} \int_0^L \frac{dS(x)}{dx} dx $ ⑨

Performing the integration in ⑨; and noting that the antiderivative of \(\frac{dS(x)}{dx}\) is:

$ S(x) $

Then;

$ C_{N\alpha} = \frac{8}{\pi d^2} \left[S(L) - S(0)\right] $ ⑩

Immediately it is noticed that \( C_{N\alpha} \) is independent of the shape of the body as long as the body is such that the integration is valid. Equation 10 is now applied to the different portions of the body.

**Nose**

For the nose; \( S(0) = 0 \)

$ S(0) \rightarrow \text{Nozzle Diagram Placeholder} \rightarrow S(L) $

Thus:

$ C_{N\alpha} = \frac{8}{\pi d^2} [S(L) - 0] $ ⑪

But;

$ S(L) = \frac{\pi d^2}{4} $

Thus:

$ (C_{N\alpha})_N = 2 \quad (\text{per radian}) $ ⑫

This result holds for ogives, cones, or parabolic shapes, as well as any other shape that varies smoothly.

**Cylindrical Body**

For any cylindrical body; \( S(L) = S(0) \)

Thus:

$ C_{N\alpha} = 0 $ ⑬

This says that there is no lift on the cylindrical body portions at low angles of attack.

**Conical Shoulder**

Equation 10 is directly applicable to both conical shoulders and boattails.

![Conical Shoulder Diagram Placeholder](#)

$ (C_{N\alpha})_{CS} = \frac{8}{\pi d^2} (S_2 - S_1) $ ⑭

**Conical Boattail**

![Conical Boattail Diagram Placeholder](#)

$ (C_{N\alpha})_{CB} = \frac{8}{\pi d^2} (S_2 - S_1) $ ⑮

Since \( S_2 \) is less than \( S_1 \), for a conical boattail, the value of \( (C_{N\alpha})_{CB} \) is negative for angles of attack near zero.

### Center of Pressure

**General:**

By definition, the pitching moment of the local normal aerodynamic force about the front of the body (\(x=0\)) is:

$ M(x) = X \cdot n(x) $ ⑯

Substituting equation 3 into equation 16:

$ M(x) = \rho V_∞^2 \alpha X \frac{dS(x)}{dx} $ ⑰

By definition of the aerodynamic pitching moment coefficient,

$ C_m(x) = \frac{M(x)}{8Ad} = \frac{M(x)}{\frac{1}{2} \rho V_∞^2A} $ ⑱

Substituting equation 17 into equation 18:

$ C_m(x) = \frac{2 \alpha X}{Ad} \frac{dS(x)}{dx} $ ⑲

but;

$ A = \frac{\pi}{4} d^2 $

Therefore,

$ C_m(x) = \frac{8 \alpha X}{\pi d^3} \frac{dS(x)}{dx} $ ⑳

By the definition of moment coefficient curve slope:

$ C_{M\alpha}(x) = \left. \frac{dC_m(x)}{d\alpha} \right|_{\alpha=0} = \frac{8X}{\pi d^3} \frac{dS(x)}{dx} $ ⑭

In order to obtain the total \( C_{M\alpha} \), equation 21 is integrated over the length of the body:

$ C_{M\alpha} = \int_0^L \frac{8X}{\pi d^3} \frac{dS(x)}{dx} dx $ ⑮

Since \(\frac{8X}{\pi d^3}\) is not a function of \( x \);

$ C_{M\alpha} = \frac{8}{\pi d^3} \int_0^L X \frac{dS(x)}{dx} dx $ ⑯

Performing the integration in 23 by parts:

$$
\begin{align*}
u &= X \\
dv &= \frac{dS(x)}{dx} dx \\
du &= dX \\
v &= S(x)
\end{align*}
$$

$ C_{M\alpha} = \frac{8}{\pi d^3} \left\{ \left[ XS(x) \right]_0^L - \int_0^L S(x) dX \right\} $

$ = \frac{8}{\pi d^3} \left\{ L S(L) - 0S(0) - \int_0^L S(x) dx \right\} $

$ C_{M\alpha} = \frac{8}{\pi d^3} \left[ L S(L) - \int_0^L S(x) dx \right] $ ⑰

By definition, the second term in 24 is the volume of the body:

$ V = \int_0^L S(x) dx $ ⑮

Thus:

$ C_{M\alpha} = \frac{8}{\pi d^3} \left[ L S(L) - V \right] $ ⑯

The center of pressure of the body is defined as:

$ \bar{X} = d \left( \frac{C_{M\alpha}}{C_{N\alpha}} \right) $ ⑰

Substituting equations 10 and 26 into equation 27:

$ \bar{X} = \frac{L S(L) - V}{S(L) - S(0)} $ ⑱

Dividing numerator and denominator by \( S(L) \):

$ \bar{X} = \frac{L - \frac{V}{S(L)}}{1 - \frac{S(0)}{S(L)}} $ ⑲

The center of pressure, then, is a definite function of the body shape which determines the volume.

Equation 29 is now applied to the different portions of the body.

**Nose**

The nose shapes most often used are that of either a cone or an ogive. Thus, \(\bar{X}\) is determined for those particular shapes.

## Cone

![Cone Diagram Placeholder](#)

$ V = \frac{\pi}{3} r^2 L = \frac{1}{3} L S(L) $

Thus;

$ \frac{V}{S(L)} = \frac{L}{3} $ ⑳

also;
$ S(0) = 0 $
thus;
$ \frac{S(0)}{S(L)} = 0 $ ㉑

Therefore;

$ \bar{X} = \frac{L - \frac{L}{3}}{1 - 0} $

or,

$ \bar{X}_N = \frac{2}{3} L \quad \text{(CONE)} $ ㉒

## Ogive

From reference 2; for a tangent ogive,

$$
\frac{V}{\frac{\pi d^2}{4}} = f(f^2 + \frac{l}{4})^2 - \frac{f^3}{3} - (f^2 - \frac{l}{4})(f^2 + \frac{l}{4})^2 \ln \left(\frac{f}{f^2 + \frac{l}{4}} \right)
$$ ㉓

![Ogive Diagram Placeholder](#)

where;

$ f = \frac{L}{d} $ ㉔

Again, the denominator is 1, since \( S(0) = 0 \). Thus;

$ \bar{X} = L - \frac{V}{S(L)} $ ㉕

Dividing equation 35 by \( d \);

$ \frac{\bar{X}}{d} = f - \frac{V}{d S(L)} $ ㉖

or, substituting equation 33 in equation 36:

$ \frac{\bar{X}}{d} = f + f(f^2+\frac{l}{4})^2 + \frac{f^3}{3} + (f^2-\frac{l}{4})(f^2+\frac{l}{4})^2 \ln \left(\frac{f}{f^2+\frac{l}{4}}\right) $ ㉗

Equation 37 is solved numerically and plotted in figure 2. A computer program, as listed on the next page, was used to do the calculation with extreme accuracy. As can be seen in figure 2, the resultant curve is very nearly a straight line. Thus, equation 37 may be approximated very well by the equation of the straight line as long as \( f \) is greater than one (1).

$ \frac{\bar{X}}{d} = .466 f = .466 \frac{L}{d} $ ㉘

dividing both sides of equation 38 by \( d \);

$ \bar{X}_N = .466 L \quad \text{(OGIVE)} $ ㉙

![Figure 2: Tangent Ogive Center of Pressure](#)

This figure illustrates the center of pressure for a tangent ogive, showing the relationship between \(\frac{\bar{X}}{d}\) and \(f = \frac{L}{d}\). It demonstrates the linear approximation \(\frac{\bar{X}}{d} = .466f\).



```
C     CENTER OF PRESSURE OF AN OGIVE
      DOUBLE PRECISION A,B,C,D,E,F,G,H,XCP
      WRITE(6,2)
2     FORMAT(13H1    F        X/D)
      DO 10 I=1,10
      F=I
      A=F*F
      B=A*.25
      C=A-.25
      D=B*D
      E=A*F
      G=F/E
      H=DATAN(DABS(G/DSQRT(1.-G*G)))
      XCP = F + 4.*DAT(C*H - F) + 4.*E/3.
10    WRITE(6,1)F,XCP
1     FORMAT(1TH ,F5.0,F9.3)
      STOP
      END

OUTPUT

 F      X/D
1.   0.430
2.   0.914
3.   1.307
4.   1.657
5.   2.326
6.   2.794
7.   3.261
8.   3.729
9.   4.196
10.  4.663
```

## Cylindrical Body

Since \( C_{N\alpha} = 0 \) for a cylindrical body, calculation of \( \bar{X} \) is not necessary.

## Conical Shoulder

![Conical Shoulder Diagram Placeholder](#)

The volume of a conical frustum is;

$ V = \frac{\pi L}{12} \left( d_1^2 + d_1 d_2 + d_2^2 \right) $ ㉚

or

$ V = \frac{L}{3} \left( S_1 + S_2 \frac{d_1}{d_2} + S_2 \right) $

or

$ V = \frac{L S_2}{3} \left(\frac{S_1}{S_2} + \frac{d_1}{d_2} + 1 \right) $

But, since

$ S_2 = S(L) $

then,

$ \frac{V}{S(L)} = \frac{L}{3} \left(\frac{S_1}{S_2} + \frac{d_1}{d_2} + 1 \right) $ ㉛





Also,

$ \frac{S_1}{S_2} = \left(\frac{d_1}{d_2}\right)^2 $

thus,

$ \frac{V}{S(L)} = \frac{L}{3} \left[ 1 + \frac{d_1}{d_2} + \left(\frac{d_1}{d_2}\right)^2 \right] $ ㉜

Substituting equation 42 in equation 29;

$ \bar{X} = \frac{L - \frac{L}{3}\left[1 + \frac{d_1}{d_2} + \left(\frac{d_1}{d_2}\right)^2\right]}{1 - \frac{S(0)}{S(L)}} $ ㉝

Again, noting that

$ \frac{S(0)}{S(L)} = \frac{S_1}{S_2} = \left(\frac{d_1}{d_2}\right)^2 $

and expanding;

$ \bar{X} = \frac{L}{3} \left[ \frac{3 - 1 - \frac{d_1}{d_2} - \left(\frac{d_1}{d_2}\right)^2}{1 - \left(\frac{d_1}{d_2}\right)^2}\right] $

$ = \frac{L}{3} \left[ \frac{2 - \frac{d_1}{d_2} - \left(\frac{d_1}{d_2}\right)^2}{1 - \left(\frac{d_1}{d_2}\right)^2}\right] $

$ = \frac{L}{3} \left[ \frac{1 - \left(\frac{d_1}{d_2}\right)^2}{1 - \left(\frac{d_1}{d_2}\right)^2} + \frac{1 - \frac{d_1}{d_2}}{1 - \left(\frac{d_1}{d_2}\right)^2}\right] $

$ \bar{X}_{CS} = \frac{L}{3} \left[ 1 + \frac{1 - \frac{d_1}{d_2}}{1 - \left(\frac{d_1}{d_2}\right)^2}\right]\quad \text{from front of shoulder or boattail} $ ㉞

$ \bar{X}_{N} = \frac{L}{3} \left(1 + \frac{1}{1 + \frac{d_1}{d_2}}\right) \quad \text{as per the provision} $





## Conical Boattail

Since no distinction as to direction of the conical frustum was made in deriving equation 44, it holds true also for a frustum with the dimensions shown:

![Conical Boattail Diagram Placeholder](#)

Here's the representation with dimensions \(d_1\), \(d_2\), and length \(L\).



## FIN AERODYNAMICS DERIVATIONS

### Normal Force Coefficient Slope

From Reference 1, by a theory of Diederich, \( C_{N\alpha} \) of a finite flat plate is given by:

$ C_{N\alpha} = \frac{C_{N\alpha_0} F \left(\frac{A_f}{A}\right) \cos \sigma}{2 + F \sqrt{1 + \frac{4F^2}{R^2}}} $ ㉟

where:

- \( C_{N\alpha_0} \) = Normal force coefficient slope of a two-dimensional airfoil
- \( F \) = Diederich's correlation parameter
- \( A_f \) = Area of one fin

According to Diederich:

$ F = \frac{R}{2\pi C_{N\alpha_0} \cos \sigma} $ ㊱

By the thin airfoil theory of potential flow:

$ C_{N\alpha_0} = 2\pi $ ㊲

Thus:

$ F = \frac{R}{\cos \sigma} $ ㊳

Substituting equations 47 and 48 into 45:

$ C_{N\alpha} = \frac{2\pi R \left(\frac{A_f}{A}\right)}{2 + \frac{R}{\cos \sigma} \sqrt{1 + \frac{4\cos^2 \sigma}{R^2}}} $ ㊴





Simplifying:

$ C_{N\alpha} = \frac{2\pi R \left(\frac{A_f}{A}\right)}{2 + \sqrt{4 + \left(\frac{R}{C_{N\alpha_0} \cos \sigma}\right)^2}} $  ㊵

This is \( C_{N\alpha} \) for a single fin.

A typical fin has the geometry shown in Figure 3. All fins can be idealized into a fin or a set of fins having straight line edges as shown in Figure 3.

By definition:

$ R = \frac{2s^2}{A_f} $ ㊶

also;

$ A = \frac{\pi d^2}{4} $ ㊷

Substituting 51 and 52 into the numerator of equation 50:

$ 2\pi R \left(\frac{A_f}{A}\right) = 2\pi \left(\frac{2s^2}{A_f}\right)\left(\frac{A_f}{\frac{\pi d^2}{4}}\right) $

$ 2\pi R \left(\frac{A_f}{A}\right) = 16 \left(\frac{s}{d}\right)^2 $ ㊸

By trigonometric definition:

$ \cos \sigma = \frac{s}{l} $ ㊹





![Figure 3: Fin Geometry](#)

This figure illustrates the geometry of a fin, showing dimensions such as root chord (\(C_r\)), tip chord (\(C_t\)), span (\(S\)), mean aerodynamic chord, quarter chord line, and mid-chord line, along with other geometric parameters.



Then;

$ \frac{R}{\cos \sigma} = \frac{2s^2 l}{A_f b} = \frac{2ls}{A_f} $

But, from geometry:

$ A_f = \left(\frac{C_r + C_t}{2}\right) s $

Therefore;

$ \frac{R}{\cos \sigma} = \frac{2ls}{\left(\frac{C_r + C_t}{2}\right) s} $

$ \frac{R}{\cos \sigma} = \frac{4l}{C_r + C_t} $ ㊺

Substituting 55 into the denominator of 50:

$ 2 + \sqrt{4 + \left(\frac{R}{\cos \sigma}\right)^2} = 2 + \sqrt{4 + \left(\frac{4l}{C_r + C_t}\right)^2} $

$ = 2 + \sqrt{4 + \left(\frac{2l}{C_r + C_t}\right)^2} $

$ 2 + \sqrt{4 + \left(\frac{R}{\cos \sigma}\right)^2} = 2 + 2\sqrt{1 + \left(\frac{2l}{C_r + C_t}\right)^2} $ ㊻

Substituting equation 53 and 55 into 50:

$ C_{N\alpha} = \frac{16 \left(\frac{s}{d}\right)^2}{2 + 2\sqrt{1 + \left(\frac{2l}{C_r + C_t}\right)^2}} $





Simplifying:

$ C_{N\alpha} = \frac{8 \left(\frac{s}{d}\right)^2}{1 + \sqrt{1 + \left(\frac{2l}{C_r + C_t}\right)^2}} $ ㊼

Equation 57 gives \( C_{N\alpha} \) for a single fin. A four fin rocket, having two fins in the plane normal to the plane of the angle of attack (see Figure 4a) has the \( C_{N\alpha} \) of:

$ (C_{N\alpha})_F = \frac{16 \left(\frac{s}{d}\right)^2}{1 + \sqrt{1 + \left(\frac{2l}{C_r + C_t}\right)^2}} $ ㊽

A three-finned rocket has its fins spaced 120° apart. Assuming that the 3 finned rocket flies with one fin in the plane of the angle of attack, with \( (C_{N\alpha})_B = C_{N\alpha} \) of one fin (see Figure 4b):

$ C_{N\alpha} = 2(C_{N\alpha})_1 \cos 30° $
$ = 2(C_{N\alpha})_1 \frac{\sqrt{3}}{2} $
$ C_{N\alpha} = \sqrt{3}(C_{N\alpha})_1 $

Thus:

$ (C_{N\alpha})_F = \frac{8\sqrt{3} \left(\frac{s}{d}\right)^2}{1 + \sqrt{1 + \left(\frac{2l}{C_r + C_t}\right)^2}} $

or (see addendum 2)

$ (C_{N\alpha})_F = \frac{12 \left(\frac{s}{d}\right)^2}{1 + \sqrt{1 + \left(\frac{2l}{C_r + C_t}\right)^2}} $ ㊾





![Figure 4a: Four Fins](#)

This diagram illustrates a configuration with four fins, each contributing to the normal force coefficient \( (C_{N\alpha})_1 \).

![Figure 4b: Three Fins](#)

This diagram depicts a configuration with three fins spaced 120° apart, showing the contributions to the normal force coefficient, including the effects at angles of 30° and 60°.



## FIN AERODYNAMICS DERIVATIONS

### Center of Pressure

From the potential theory of subsonic flow, the center of pressure of a two-dimensional airfoil is located at 1/4 the length of its chord from its leading edge. Thus, on a three-dimensional fin, the center of pressure should be located along the quarter chord line. By definition, the span-wise center of pressure is located along the mean aerodynamic chord. Therefore, by the above argument, the fin center of pressure is located at the intersection of the quarter chord line and the mean aerodynamic chord. (See Figure 5)

It remains to determine the length position of the mean aerodynamic chord. By definition, the mean aerodynamic chord is:

$ C_{MA} = \frac{1}{A_f} \int_0^s c^2 dy $ ㊿

where: (See Figure 5)

- \( A_f \) = Area of one fin
- \( s \) = Semispan of one fin
- \( c \) = Generalized chord
- \( y \) = Spanwise coordinate

The generalized chord is a function of the span. To find this function, a proportionality relation is set up. (See Figure 6)

$ \frac{C_r}{L^*} = \frac{c}{L^* - y} = \frac{C_t}{L^* - s} $ 5①

From the first two terms:

$ c = \frac{C_r (L^* - y)}{L^*} $

$ c = C_r - \frac{y}{L^*} C_r $ 5②

![Figure 5: Diagram Placeholder](#)
![Figure 6: Diagram Placeholder](#)



![Figure 5: Coordinate System for the Determination of the Mean Aerodynamic Chord](#)

This figure shows the coordinate system used to determine the mean aerodynamic chord, illustrating dimensions such as the chord length (\(c\)), semispan (\(s\)), and coordinates (\(x\), \(y\), and \(x_f\)).



![Figure 6: Triangle of Proportionality for the Determination of the General Chord Length](#)

This figure depicts the triangle of proportionality used to determine the general chord length, showing the relationship between dimensions including root chord (\(C_r\)), tip chord (\(C_t\)), semispan (\(s\)), and coordinates along the span such as \(L^*\), \(y\), and \(L^*-y\).



From the first and last terms:

$ C_t L^* = C_r L^* - C_r s $

or

$ L^*(C_r - C_t) = C_r s $

or

$ L^*(C_r - C_t) = L^* C_r (1 - \lambda) = C_r s $

thus

$ L^* = \frac{s}{1 - \lambda} $ 5③

Substituting 63 into 62:

$ c = C_r \left[ 1 + \left(\lambda^{-1}\right)\frac{y}{s} \right] $ 5④

Substituting 64 into 60:

$ C_{MA} = \frac{1}{A_f} \int_0^s \left( C_r^2 \left[ 1 + \left(\lambda^{-1}\right)\frac{y}{s} \right]^2 \right) dy $

Expanding:

$ C_{MA} = \frac{C_r^2}{A_f} \int_0^s \left[ 1 + 2\left(\lambda^{-1}\right)\frac{y}{s} + \left(\lambda^{-1}\right)^2\left(\frac{y}{s}\right)^2 \right] dy $





Let; \(\mathcal{R} = \frac{\lambda - 1}{s}\)

$ C_{MA} = \frac{C_r^2}{A_f} \int_0^s \left[ 1 + 2 \mathcal{R} y + \mathcal{R}^2 y^2 \right] dy $ 5⑤

Performing the integration:

$ C_{MA} = \frac{C_r^2}{A_f} \left\{ \int_0^s dy + 2\mathcal{R} \int_0^s y \, dy + \mathcal{R}^2 \int_0^s y^2 \, dy \right\} $

$ = \frac{C_r^2}{A_f} \left\{ \left[ y \right]_0^s + 2\mathcal{R} \left[ \frac{y^2}{2} \right]_0^s + \mathcal{R}^2 \left[ \frac{y^3}{3} \right]_0^s \right\} $

$ C_{MA} = \frac{C_r^2}{A_f} \left[ s + \mathcal{R} s^2 + \frac{1}{3} \mathcal{R}^2 s^3 \right] $ 5⑥

Substitute 66 in 65 and simplifying:

$ C_{MA} = \frac{C_r^2 s}{A_f} \left[ 1 + \lambda s + \frac{1}{3} \lambda^2 s^2 \right] $

$ = \frac{C_r^2 s}{A_f} \left[ 1 + (\lambda - 1) + \frac{1}{3} (\lambda - 1)^2 \right] $

$ = \frac{C_r^2 s}{A_f} \left[ \lambda + \frac{1}{3} (\lambda^2 - 2\lambda + 1) \right] $

$ C_{MA} = \frac{1}{3} \frac{C_r^2 s}{A_f} \left[ \lambda^2 + \lambda + 1 \right] $ 5⑦

But, by geometry:

$ A_f = \frac{1}{2} (C_r + C_t) s $ 5⑧





Thus, substituting 68 in 57:

$ C_{MA} = \frac{2}{3} \frac{C_r^2}{C_r + C_t} \left[ \lambda^2 + \lambda + 1 \right] $

$ = \frac{2}{3} \frac{1}{C_r + C_t} \left[ C_t^2 + C_r C_t + C_r^2 \right] $

$ = \frac{2}{3} \frac{1}{C_r + C_t} \left[ (C_r + C_t)^2 - C_r C_t \right] $

$ C_{MA} = \frac{2}{3} \left[ C_r + C_t - \frac{C_r C_t}{C_r + C_t} \right] $ 5⑨

It is now necessary to find the x surface position of \( C_{MA} \). This is done by equating equation 69 and 64 and solving for \(\bar{Y}\).

$$
\frac{2}{3}\left[C_r + C_t - \frac{C_r C_t}{C_r + C_t} \right] = C_r \left[1 + \left(\lambda^{-1}\right)\frac{\bar{Y}}{s}\right]
$$

$$
= C_r + \left(\frac{C_r - C_t}{s}\right)\bar{Y}
$$

Thus:

$$
\bar{Y} = \left[ \frac{2}{3} C_r + \frac{2}{3} C_t - \frac{2 C_r C_t}{3(C_r + C_t)} - C_r \right] \frac{s}{C_r - C_t}
$$





$$
\bar{Y} = \frac{s}{3(C_r - C_t)} \left[ 2C_r - C_t - \frac{2C_r C_t}{C_r + C_t} \right]
$$

$$
= \frac{s}{3(C_r - C_t)(C_r + C_t)} \left[ 2C_r(C_r + C_t) + 2C_t^2 - C_t^2 - C_r C_t - 2C_r C_t \right]
$$

$$
= \frac{s}{3(C_r - C_t)(C_r + C_t)} \left[ 2C_r^2 - C_r C_t - C_t^2 \right]
$$

$$
\bar{Y} = \frac{s}{3} \left[ \frac{(C_r + 2C_t)(C_r - C_t)}{(C_r - C_t)(C_r + C_t)} \right]
$$ 7⓪

By trigonometry: (see Figure 5)

$ d_{MA}^* = \bar{Y} \tan \Gamma_L $ 7①

And,

$ \tan \Gamma_L = \frac{x_f}{s} $ 7②

Thus,

$ d_{MA}^* = \frac{\bar{Y}}{s} x_f $ 7③





Substituting 73 into 70:

$ d_{MA}^* = \frac{x_f}{3} \frac{(C_r + 2C_t)}{(C_r - C_t)} $ 7④

From the argument at the beginning of this section:

$ \bar{X} = d + \frac{1}{4} C_{MA} $ 7⑤

Substituting equations 74 and 69 into 75:

$ \bar{X}_F = \frac{x_f}{3} \frac{(C_r + 2C_t)}{(C_r + C_t)} + \frac{1}{6} \left[C_r + C_t - \frac{C_r C_t}{C_r + C_t} \right] $ 7⑥a

The \(\bar{X}\) is from the leading edge of the root chord. To get the center of pressure of the fin from the nose tip, \(X_F\) must be added to \(\bar{X}_F\).

- \( X_F \) = Distance from nose tip to leading edge of fin root chord.

$ (\bar{X})_{T(B)} = X_F + \bar{X}_F $ 7⑥b





## INTERFERENCE EFFECTS

The major interference effects encountered on any rocket are the change of lift of the fin alone when it is brought into the presence of the body and the change of lift on the body between the fins. Reference 3 discusses these effects in detail. They are handled by the use of correction factors which are applied to the fins alone. The values of these factors are shown in Figure 7. The plots of interest are underlined in red. In this figure, "5 IF" is "5 if" normally in my nomenclature.

- \( K_{T(B)} \) = Correction factor for the fins in the presence of the body
- \( K_{B(F)} \) = Correction factor for the body in the presence of the fins

As can be seen in Figure 7, the value of \( K_{T(B)} \) is occasionally greater than that for \( K_{B(F)} \) in the range of \( F/(S+d) \), in which most model rockets fall (\(<3+\)). Thus, it is a conservative and reasonable approximation to do two things to simplify the interference calculations:

1. Approximate the \( K_{T(B)} \) curve by a straight line (red line on Figure 7).
2. Neglect the influence of \( K_{B(F)} \).

In this way;

$ K_{T(B)} = 1 + \frac{r_e}{s + r_e} $ 7⑦

Thus:

$ (C_{N\alpha})_{T(B)} = K_{T(B)} (C_{N\alpha})_{\text{fins, alone}} $ 7⑧

where; \( (C_{N\alpha})_{\text{fins, alone}} \) is obtained from equation 57 or equation 59, and \( K_{T(B)} \) comes from equation 77.





![Figure 7: Correction Factors](#)

This chart illustrates the correction factors \(K_{T(B)}\) and \(K_{B(F)}\) used for calculating the lift changes due to interference effects. The axis represents the radius-to-semispan ratio \((r/s)_w\) or \((r/s)_f\), with curves showing the values of lift ratios based on slender-body theory.



## COMBINATION CALCULATIONS

The total vehicle \( C_{N\alpha} \) is the sum of the \( C_{N\alpha} \)'s of the individual portions:

$ C_{N\alpha} = (C_{N\alpha})_N + (C_{N\alpha})_{T(B)} + (C_{N\alpha})_{CS} + (C_{N\alpha})_{CB} $ 7⑨

The center of pressure is determined by a moment balance about the nose of the rocket:

$ \bar{X} = \frac{(C_{N\alpha})_N \bar{X}_N + (C_{N\alpha})_{T(B)} \bar{X}_{T(B)} + (C_{N\alpha})_{CS} \bar{X}_{CS} + (C_{N\alpha})_{CB} \bar{X}_{CB}}{C_{N\alpha}} $ 8⓪

Of course, if there are more than one conical shoulder, conical boattail, and/or fins, these are also included in equations 79 and 80.





## REFERENCES

1. Shapiro, A. H.; *The Dynamics and Thermodynamics of Compressible Fluid Flow, Vol. 1*; Ronald; New York; 1955.

2. Kaye, M. H.; *Cone Cylinder and Ogive Cylinder Geometric and Flow Characteristics*; Memo to code 721 at NASM-3859; RC Dept. 1955.

3. Pitts, W. C., Nielsen, J. N., and Kazaroff, C. Z.; *Lift and Center of Pressure of Air-Boattail Combinations at Subsonic, Transonic, and Supersonic Speeds*; NACA TN-1307; C.A.C., Washington, D.C.; 1955.

4. Miles, J. W.; *Unsteady Supersonic Flow*; M.A.C.; Baltimore; 1955; Research R-44.

5. McCueney, J. R.; *Aerobee 350 Windtunnel Test Analysis*; Aerac General Corp.; La Habra, Calif.; January 1965.
