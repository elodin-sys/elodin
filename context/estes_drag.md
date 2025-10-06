+++
title = "Estes Drag"
description = "Aerodynamic Drag of Model Rockets"
draft = false
weight = 140
sort_by = "weight"

[extra]
toc = true
top = false
icon = ""
order = 40
+++

# TR-11 Model Rocket Technical Report

## Aerodynamic Drag of Model Rockets

By Dr. Gerald M. Gregorek

Published as a service to its customers by
ESTES INDUSTRIES, INC.
Box 227
Penrose, Colorado 81240

Copyright 1970
ESTES INDUSTRIES, INC.

![Estes Logo](logo-placeholder)
*A SUBSIDIARY OF DAMON*

# TR-11 Model Rocket Technical Report

R&D-359-71

# Table of Contents

I. Aerodynamic Drag of Model Rockets ................................................. 1
II. Basic Concepts ............................................................................. 1
III. Introduction To Rocket Drag Analysis ............................................ 8
IV. Drag of Rocket Bodies .............................................................. 10
V. Drag of Rocket Fins .................................................................... 16
VI. Total Rocket Drag .................................................................... 23
VII. Drag Reduction Techniques ....................................................... 32
VIII. Putting It All Together ............................................................ 39

Appendix I .......................................................................................... 47
Appendix II, Suggested Reading List .................................................. 50
Appendix III, List of Symbols ........................................................... 51


# I. Aerodynamic Drag of Model Rockets

Up to the launcher you go, slip the rocket on the rod, connect the micro-clips. Step back, hold your breath during the countdown. Your first model -- and all those experienced rocketeers watching!

Three...two...one...Fire!

The model arcs up, twisting a little. There goes the ejection charge.... The chute? There it is. Good shoot! You feel pleased as the model floats slowly down.

Another bird about to be launched. Hmm, same kit as yours, wonder who built it? Launch! The model goes straight up -- no twisting -- look at that altitude! Why’d that model do so much better than yours? Better check.

At the Ready Area you get a chance to look at that high performance bird. Colorful paint job, smooth finish. Fins are even sanded, sort of round in front and sharp at the rear. Looks a lot different than your model which was stuck together in a couple of hours. “Is it really worth all that work?” you ask the modeler.

“What do you think?” comes back, “you saw the height I got. You cut the drag and you really can get up there!”

“Oh”, you nod your head and wander away. “Wonder what drag is?”

## Drag

Drag is the theme of this report. In technical terms, drag is the resistance caused by the motion of bodies through fluids like air or water. When we push our way through a swimming pool, the water resists our passage. This is hydrodynamic drag which means, literally, the drag due to the motion of the water about a body. When we stick our hand out the window of a moving automobile we feel a force due to the motion of the air past our hand. When you hold your hand at different angles you can feel the air push on it, sometimes up, sometimes down, sometimes just back. These motions are caused by aerodynamic forces; again the term refers to forces due to the motion of air about a body (in this case your hand).

Aerodynamic forces can be put to use -- airplanes work because we know how to shape the wings to get a favorable aerodynamic force. This favorable force is called “Lift”. Unfortunately, we must pay a price for lift -- even in nature we can never get something for nothing -- and an unfavorable force is also generated. The unfavorable force is called “Drag”. Model rockets experience drag, just as airplanes and all other bodies do. Together with gravity, drag opposes the rocket’s thrust and is a very important factor in determining your rocket’s performance.

### Why Bother With Drag

Drag is important because it retards our models, holding them back and preventing them from reaching their potential altitudes. When rocket engines are ignited, the models lift off the pad because the upward thrust is greater than the downward pull of gravity on the rocket. As the model gains speed, the aerodynamic resistance builds up rapidly; the drag and the weight determine the top speed of the model at rocket burnout. After rocket burnout, during the coast upward, aerodynamic drag and gravity slow the model until the upward velocity reaches zero at the peak altitude and the rocket falls back to earth. Thus the greater the drag of a rocket the lower the burnout speed and the more rapid deceleration during coast -- both conditions reduce the height achieved with a given model and engine. Many rocket competitions depend upon peak altitude (payload, egg-lofting, even parachute duration, since the higher a bird goes the longer it takes to come down) so the desirability of cutting drag to a minimum is obvious.

Another reason drag is important is that it is the one factor we can do something about. Although we can strive for light weight models, we can not change the pull of gravity -- so maybe we can find the rocket nose cone that gives the least drag or fin the shape that is best. These are the problems we’ll look at in this report. We will give practical examples of ways to improve model rocket performance.

As we proceed with our discussions, we’ll point out some basic concepts of drag. We’ll find that a few equations will be helpful in our study. By treating these equations as a sort of shorthand, we’ll be able to tell precisely how certain factors affect the aerodynamic resistance of a model rocket. Further, just as practicing engineers and scientists, we’ll use graphs and illustrations to help us visualize the various features of aerodynamic drag.

# II. Basic Concepts

## Factors Affecting Drag

If we think back to the example of holding our hand out the window of a moving car we can get a little insight, or physical feeling, about factors that influence the drag of a body. Referring to Fig. 1, we might sense, intuitively, that the size of a hand should make a difference; the larger the hand, the greater the force anticipated. The area presented to the stream must, therefore, be a factor. The speed of the air that hits your hand certainly has an effect. If you hold your hand out when the auto is doing twenty miles an hour, you feel a lot less force or drag than when your hand is exposed to a 40 mile an hour wind, so speed certainly is important.

![FIG. 1: Feeling Aerodynamic Drag with Your Hand](image-placeholder)

Maybe less obvious, but again important when we think about it, is the density of the air. Density refers to the amount of material present in a given space or volume. Air has a certain amount of material in a cubic foot (about 0.077 lb); water has more material in the same space (about 62.4 lbs). The higher density of water explains why it’s more difficult to move your hand through water than through air. Density of the fluid, therefore, is a contributing factor to the resistance of a body.

Other, more subtle factors, influence the drag of your hand. For example, if you hold your hand parallel to the road or perpendicular to it, you'll feel a different amount of aerodynamic force. When you cup your hand or make it into a fist, the air resistance or aerodynamic drag is different, so we can say both the angle of the wind and shape influence the total resistance encountered by a moving body.

## Bringing the Factors Together

Looking back to see what factors we thought about that could influence the drag of a body we note that the size, speed, shape, density of the air and angle will all contribute to the drag. It’s important to be able to identify the things which affect drag, but it’s even better to be able to calculate exactly what the effects are, so we are led to our first equation. From experimental information accumulated during the last 70 years, we know that it is possible to represent the drag of a body by the shorthand statement below:

Drag = Drag coefficient $\times \frac{1}{2} \times$ density $\times$ velocity$^2 \times$ area

To save time and space, we will represent this word equation by letters and drop the $\times$ meaning multiplication. Thus

$ D = \frac{1}{2} CD \rho V^2 A $

(1)

Examining Eq. 1 we see direct evidence of the factors that we said influence aerodynamic drag. The drag of the body is represented by V$^2$ (note that V$^2$ means V by V, so the influence of velocity on drag D is very strong). The “$\rho$” stands for the density of the fluid.

One factor that needs a little explanation is the drag coefficient, CD. This term has no dimensions; it is simply a number used to describe how the shape of the body and its angle to the wind influence drag. All shapes that move through the air possess drag coefficients: your hand, autos, airplanes, and, of course, your model rockets. If we can find the value of CD for a rocket, we’ll be able to compute its actual aerodynamic drag in pounds or grams.

The full advantage of the shorthand notation of Eq. 1 and the use of drag coefficients can be demonstrated if we work out an example. It might be interesting to find the force of wind on your hand as you hold it out the window of a moving auto. Looking at the equation, we see we must determine the area of your hand, the air density, and the speed of the car. We’ll assume that the drag coefficient of your hand has been determined by an earlier experiment. (CD is approximately 1.2 for a hand held perpendicular to the air flow.)

We estimate the area of your hand and arm exposed to the wind to be about 1/4 of a square foot and the air density to be 0.00238 slugs/ft$^3$. Note that we converted the units of the air density from weight units of lbs/ft$^3$ to mass units of slugs/ft$^3$ to have the solution come out correctly in lbs.

Substituting these values into the drag equation,

$ D = CD \frac{1}{2} \rho V^2 A $

$ = 1.2 \times \frac{1}{2} \times 0.00238 \times \frac{1}{4} V^2 $

$ D = 0.000357 V^2 \text{ in lbs} $

(2)

In the above form, we can find the drag for any velocity. If V = 10 ft/sec for example

$ D = 0.000357 \times 10000 = 3.57 \text{ lbs} $

We can get a better feeling for how drag changes with speed by constructing a graph by substituting different velocities into Eq. (2). This is shown in Fig. 2. We can also change the area in Eq. 1 and repeat the process. For a smaller hand of just 1/8 of a square foot we see the drag is just 1/2 the value of the large hand. You can see how useful a curve of this type would be for the study of the drag of your model rockets. Once we determine looking for: we divide the weight of an object by the acceleration due to gravity to obtain a value for the mass of an object that is not affected by the gravitational field. In equation form

$ \text{Mass} = \frac{\text{Weight}}{\text{Acceleration due to Gravity}} $

Thus, on Earth an astronaut has mass

$ \frac{M = \frac{180}{32.2} = 5.6 \text{ slugs}} $

while on the moon, his reduced weight combines with the slower acceleration to produce an identical value for the mass:

$ \frac{M = \frac{30}{5.6} = 5.6 \text{ slugs}} $

For use in aerodynamic studies, the air density is converted to mass units.

$ \frac{0.077}{32.2} = 0.00238 \text{ slugs/ft}^3 $



## Pressure Drag

Looking at drag due to pressure first, we might consider a baseball. When it is sitting on the ground, as illustrated in Fig. 3, the pressure around it, represented by the arrows pushing perpendicularly to the surface, is the same. At sea level, this atmospheric pressure, which is due to all the air piled above us, is 14.7 pounds on every square inch of surface. Since the pressure on all parts of the baseball has the same value, there is no unbalance of pressure forces, and hence, no drag. But, when you throw the ball, what happens? The air starts to move around the ball, the pressures about the sphere change, a pressure unbalance occurs, and aerodynamic drag is created. In the illustration, the arrows represent the pressure distribution on the ball; the longer the arrow, the higher the pressure. The unbalance in pressures and resulting drag is exhibited by the way the ball slows after it is thrown.

More than 95% of the drag on a sphere comes from pressure drag. We'll see later that more streamlined shapes will have less pressure drag but more friction drag. Because this type of drag depends on the shape of the body, it is sometimes called "profile" drag by aeronautical engineers.

## Friction Drag

To get a feeling for friction drag, we might consider a very sharp, thin plate moving through the air as in Fig. 4. When moving at zero angle to the air stream you can see that there will be no unbalance of pressure forces. Does this mean that the drag is zero? No, the air is rubbing on the surface. The influence of this friction is confined to a thin region close to the body surface. The second sketch in Fig. 4 indicates how the friction affects this layer after the air velocity near the surface. On the surface, the velocity is zero, increasing to the air velocity in the free stream outside the layer. This behavior is due to another property of fluids called "viscosity". In the thin region close to the body surface, termed the "boundary layer", viscosity is important.

Just like density, viscosity is a property of air. Instead of measuring mass, however, viscosity measures the resistance of a fluid to flowing over a surface. The viscosity of molasses, for example, is very high and we know that molasses is hard to pour; water has a much lower viscosity and flows quite freely. The viscosity of air is extremely low and air flows easily over surfaces.

![FIG. 2: Drag of Hand](image-placeholder)

### Where Drag Comes From

As we go deeper into the study of drag, we may ask where does this drag come from? We've managed to describe the factors that affect drag and we've been successful in determining the number of pounds of drag for a particular case. (Incidentally, determining how much of a quantity exists is always a lot harder than just telling how a quantity is obtained.) There are only two methods by which a force can be communicated between the air and a model rocket. The first way is through an unbalance in the air pressure on the rocket and the second is through the friction of the air sliding over its surfaces.

![FIG. 3: Illustration of Pressure Drag](image-placeholder)

### Table of Values

| V (ft/sec) | V²   | D (lbs) | gms |
|------------|------|---------|-----|
| 0          | 0    | 0       | 0   |
| 10         | 100  | 0.036   | 16  |
| 20         | 400  | 0.143   | 65  |
| 40         | 1600 | 0.57    | 259 |
| 60         | 3600 | 1.28    | 581 |
| 80         | 6400 | 2.28    | 1036|
| 100        | 10000| 3.57    | 1620|

Where:
- \( D = C_D \frac{1}{2} \rho V^2 A \)
- \( C_D = 1.2 \)
- \( A = 0.25 \, \text{ft}^2 \)



Many times, in fact, the viscosity of air can be neglected and flow patterns past bodies correctly represented. However, in the boundary layer, viscous effects are the dominant ones that give rise to the friction drag.

![FIG. 4: Illustration of Friction Drag](image-placeholder)

## Combining Friction and Pressure Drag

To show how both pressure and friction effect the drag on a family of shapes, examine Fig. 5. In this graph, the drag coefficient for ellipsoids (which are just elongated round bodies) is presented. A special ellipsoid is the sphere which has a drag coefficient of 0.47. An interesting ellipsoid is the football, which has a length to diameter ratio of about 2 so that its drag coefficient according to Fig. 5 is 0.28. We could use this information to find the drag force on these two shapes. For a softball with a diameter of 0.3 feet moving at 100 feet per second we could use Eq. 1 to obtain the drag in pounds:

$ D = C_D \frac{1}{2} \rho V^2 A = 0.47 \times \frac{1}{2} \times 0.00238 \times (100)^2 \times 0.0706 $

where $ A = \pi R^2 = 3.14 \times (0.15)^2 = 0.0706 $

The drag is \( D = 0.395 \, \text{lbs} \). A football with a diameter of about 0.5 feet and a drag coefficient of 0.28 would have a larger drag force of 0.65 pounds at 100 ft/sec. What speed must the football have for it to feel the same aerodynamic resistance as the softball? Just 78 ft/sec. Check for yourself by calculating the football’s drag at this speed.

![FIG. 5: Drag of Ellipsoids](image-placeholder)

Figure 5 illustrates another important feature of ellipsoidal shapes. As bodies get more elongated, that is, as their length to diameter ratio increases, the total drag decreases rapidly to a minimum near a length to diameter ratio of 5, then drag increases slowly. Pressure drag is observed to be the major cause of aerodynamic resistance for the blunt shapes, but friction is the major contributor to drag of the high length to diameter bodies. Observations like these are important, since they allow us to concentrate on the correct factors to reduce the drag of our model rockets.

## Another Look at Viscosity

We’ve just introduced a property of air called viscosity. Because viscosity has such a strong influence on the aerodynamic flow, let’s look at this property more thoroughly. After all, the viscosity of air at "standard conditions" is very, very small (0.000,000,39 lb sec/ft$^2$); any property that small can produce sizable drag forces must be important.

Viscosity plays a large part in the production of both types of drag, pressure drag and friction drag. For friction drag, viscosity acts directly to produce shearing stresses in the boundary layer. For pressure drag, viscosity acts indirectly to trigger a flow “separation” from the body. Separation refers to the behavior of the flow when the air does not follow the body contour, but breaks away into a turbulent wake. This separation of the air flow is the real reason that the pressure unbalance occurs on aerodynamic shapes.

Let’s re-examine pressure drag, this time drawing streamlines past the body instead of representing the pressure forces. Streamlines are simply lines drawn to represent the path of air as it moves past an object. In a wind tunnel, thin lines of smoke would trace out a streamline pattern much like that shown in Fig. 6 about a circular cylinder held perpendicular to the flow. As shown in this figure, the lines move smoothly around the front of the cylinder but break away (or separate) on the back side. The wake is the large turbulent region behind the cylinder. The drag coefficient, CD, is about 0.4 for this shape, mainly due to this separation and large wake. Because of the wake the pressure on the back of the cylinder is low relative to the pressure on the front and the unbalance in pressure causes the drag. Therefore, if we can prevent the flow from breaking away, we should be able to decrease the drag.

In Fig. 7, the flow pattern about a different shape is shown. This shape is designed to reduce the amount of flow separation by filling in the base region of the cylinder to more gently contour (or streamline) the flow.



![FIG. 6: Flow About a Circular Cylinder](image-placeholder)
_Large Wake CD = 0.4_

The effect of the flow pattern is clean, the streamlines follow the body, flow separation is minimized, and the size of the wake is significantly reduced. The drag coefficient is decreased by a factor greater than — — to CD = 0.03 for this streamline shape. That is surely a worthwhile reduction.

![FIG. 7: Flow About a Streamlined Shape](image-placeholder)
_Small Wake CD = 0.03_

How was this drag reduction accomplished? By cutting down on flow separation. How did this do it? By allowing the pressure to increase on the back side of the body. Remember the low pressure in the wake of the cylinder which caused the pressure unbalance and hence high drag? Keeping the flow attached to the body allows the pressure to build back up to levels near the pressure on the nose and thereby reduce this pressure unbalance which, of course, cuts drag.

The basic rule to follow for preventing flows from separating is to always use aerodynamic shapes that are rounded gently and never have any sharp changes in direction. The viscosity of the air makes the flow resist these changes in direction and forces the flow to break away. As an illustration of the practical importance of this idea, consider the two egg-lofting birds of Fig. 8. The bird on the left, with the sharp transition between the payload and body tube, will have a higher drag than the rocket on the right because of separation at the transition. To prevent flow separation on any model rocket that uses transition pieces (adapters), always keep the angle less than 5° and you’ll have a low drag design with attached flow.

## Turbulent vs Laminar Flow

Just as we’ve learned more about drag by studying separation phenomena, we can increase our knowledge still further by re-examining the role of viscosity in the boundary layer. About a century ago, a scientist named Osborne Reynolds conducted experiments with water flows to determine how viscosity effected the flow patterns. He discovered two basic patterns of viscous flows: one he called laminar, the other turbulent. Aerodynamicists later found these same patterns existed in the air boundary layers moving over aerodynamic shapes. These two patterns are shown in Fig. 9. The laminar boundary layer, so-named because the different layers (or "lamina") of air slide smoothly over each other, has an almost straight variation of velocity from the outer edge of the layer to the zero surface velocity. The other velocity pattern, termed turbulent because of the large fluctuations of velocity and the mixing of different layers of air, has a much fuller pattern, with the greatest variation of velocity occurring nearest the surface.

![FIG. 9: Velocity Distribution in a Boundary Layer Above a Surface](image-placeholder)

These velocity variations within the boundary layer become more significant when we find the shearing stress — hence, friction drag — depends upon the rapidly with which the velocity changes. We’ll use another equation to give a precise definition:

$ \text{Shearing Stress} = \text{Coefficient of Viscosity} \times \text{velocity change at surface} $

Using our short hand lettering notation:

$ \text{S.S.} = \mu \frac{\Delta V}{\Delta y} $

(3)

In Eq. 3, \( \mu \) (the Greek letter 'mu'), stands for the coefficient of viscosity, \( \Delta V \) means the change in velocity over a small distance \( \Delta y \) from the surface. If we look at the velocity profiles and then the equation we can observe immediately that the turbulent profile must have much more drag than the laminar profile. That’s because it has the greatest velocity change nearest the surface.

We can illustrate this point by taking an example. Consider the two velocity profiles shown in Fig. 10. The laminar profile reaches 100 ft/sec at 0.001 feet from the




surface while the turbulent profile attains 100 ft/sec at 0.0001 feet. Employing Eq. 3 and the coefficient of viscosity given earlier we find:

Laminar Shear Stress = \(\mu \frac{\Delta V}{\Delta y}\)

$$
= 0.00000037 \times \frac{100}{.001} = 0.037 \, \text{lb/ft}^2
$$

and

Turbulent Shear Stress = \(\mu \frac{\Delta V}{\Delta y}\)

$$
= 0.00000037 \times \frac{100}{.0001} = 0.37 \, \text{lb/ft}^2
$$

![FIG. 10: Illustration of Increased Surface Shearing Stress in Turbulent Boundary Layer](image-placeholder)

Now, since skin friction drag is simply the area times the shear stress on a flat plate, we can see that the turbulent drag on a one foot square plate would be 0.37 lbs. If a laminar boundary layer existed on this plate the drag would be just one-tenth of this, only 0.037 lbs. So we have another important piece of practical information: Our models will have less skin friction drag if we can keep the flows laminar.

But, you can correctly ask, what factors determine whether a boundary layer is laminar or turbulent? How can we build models with laminar boundary layers? These are certainly pertinent questions. To answer them in part we must dig a little deeper into this study of viscous flows.

## Introducing the Reynolds Number

In his experiments, Reynolds found that by properly grouping the physical quantities of importance (velocity, viscosity, density and a factor describing the size of the experiment), he obtained a number which allowed him to predict whether the viscous flow would be laminar or turbulent in a particular experiment. The particular groupings of items leads to a number which we now call Reynolds’ number in his honor. In equation form it is given by Eq. (4). First in words:

Reynolds Number = density x Velocity x length / Viscosity

Then
$ RN = \frac{PVl}{\mu} $

$ = \frac{0.00238 \times 100 \times 1}{0.00000039} $

$ = 610,000 $

This value of Reynolds number is right in the transition region! That means we can’t tell what kind of boundary layer we’ll have on the rocket. But this is not bad, for we now have the opportunity to keep the boundary layer laminar and thereby keep the drag down. We'll go into these details later, but just think: Real rockets with lengths like 30 feet would have RN = 18,300,000. This is certainly in the turbulent boundary layer range and there is little chance to cut friction drag by keeping the layer laminar.

The Reynolds number is a term that is used continually in aerodynamics. We can see its importance in our work because it helps to predict whether the flow will be laminar or turbulent. In the next section we’ll use the Reynolds number again to find the drag on a golf ball. Any problem that includes viscosity will usually involve the Reynolds number, so get used to using the term.

## The Golf Ball

So far, we’ve introduced many ideas which are useful to our study of model rocket drag. Drag coefficients, the difference between pressure and skin friction drag, influence of viscosity, the concepts of boundary layers and separation all have been defined. Before we continue our discussion it might be wise to pull these concepts together. One of the most intriguing illustrations is the aerodynamics of a golf ball. Did you ever wonder why golf balls have dimples? The answer, as we shall see, is aerodynamic.

![FIG. 11: Drag of Smooth Spheres](image-placeholder)




Imagine a test of a smooth sphere, six inches in diameter, in a wind tunnel. When the tunnel is started, we'll measure the drag and, as the speed of the air builds up to 400 ft/sec, we'll continue to record the drag force on the sphere. If we plot the measurements, as shown in Fig. 11, we'll see the drag build up rapidly until 100 ft/sec, then hold constant and at about 150 ft/sec suddenly drop off. At 200 ft/sec the drag again increases. This is certainly erratic behavior. Now we’ll repeat the test sequence with a smaller sphere of 3 inches in diameter. The same abnormal behavior is exhibited, but shifted towards the higher velocities, as shown in the dashed lines in Fig. 11.

![FIG. 12: Drag Coefficient of Smooth Spheres](image-placeholder)

Possibly we can reduce these fluctuating drag values by examining the corresponding drag coefficients instead of the force. We obtain this information by solving Eq. 1 for the drag coefficient, \( C_D \):

$ C_D = \frac{D}{\frac{1}{2} \rho V^2 A} $

(5)

![FIG. 13: Drag Coefficient vs Reynolds Number for Smooth and Dimpled Spheres](image-placeholder)

This procedure produces the curves shown in Fig. 12, when appropriate values of density, area and velocity are used to divide into the measured drag. Although the curves are much smoother, the drag coefficients show a sudden drop at about 95 ft/sec for the 6 inch sphere, and a similar drop at 185 ft/sec for the 3 inch sphere. Another step is even more illuminating: instead of plotting the drag coefficient against velocity, plot it against Reynolds number. Now see what happens! Figure 13 shows that all the data has collapsed into a single solid line. This is a fine example of the advantages that can be obtained by a proper choice of factors. Look back at Fig. 11 and see how complicated the curve of drag vs. velocity looks; further, a new curve must be generated for each diameter sphere. But by using drag coefficient and Reynolds number, we have determined a universal curve that can be used for a wide variety of velocities and sphere diameters.

## Dimpling the Sphere

If we were to run a test of a dimpled sphere we would find that the drag coefficient would follow the dashed line of Fig. 13. For a given Reynolds number in the range of 105 (100,000) the drag coefficient is lower for the dimpled sphere than for the smooth sphere. Now if we apply this data to a 1.7 inch diameter sphere, the size of a golf ball, we find the drag as a function of velocity shown in Fig. 14. It is obvious that the drag is a lot less for the dimpled sphere whenever the sphere velocity is greater than 150 ft/sec. This is, of course, the range of velocities encountered by the golf ball. By keeping the drag low, our golf ball will travel a lot farther, since the distance a ball can be driven with a given swing depends upon gravity (which pulls the ball down) and drag (which slows the ball).

![FIG. 14: Drag of Smooth and Dimpled Spheres](image-placeholder)

We’ve shown the effect of dimpling the golf ball, and in the process, reviewed drag, drag coefficients, and Reynolds number, but we have not found why the drag is lowered. To determine this, we conduct another series of tests, this time in a tunnel where we can use smoke to observe the flow streamlines. Looking at the low Reynolds number (or low velocity) case we see the first pattern shown in Fig. 15. The flow is observed to separate from the sphere and a large wake is formed. From our previous discussions, we know that the large wake is associated with a high drag because we have a large pressure unbalance. The high Reynolds number case shown in Fig. 15 produces less flow separation and therefore exhibits a much smaller wake; we recall that this should lead to the lower drag observed in the tests at high speed. In our examinations of drag, we have uncovered one feature of aerodynamic flows which depend strongly upon Reynolds numbers — that was the character of the boundary layer.




For high Reynolds numbers a turbulent boundary layer exists, for low Reynolds numbers, the laminar one. It appears, then, that the turbulent boundary layer will tend to resist separation to a greater extent than the laminar layer. This appears reasonable since the velocity profiles of the turbulent layer were much fuller and had higher velocities near the surface. This higher velocity allows the turbulent layer to cling to the surface of the sphere more than the laminar layer.

The last step in our discussion of the golf ball is the observation that the size and speed of a smooth ball places it in the range of Reynolds numbers which have laminar flows, therefore early separation, and large wakes with corresponding high drag. In order to promote turbulent flow, the dimples are added to the ball. Because of this roughness the boundary layer becomes turbulent, and as shown in Fig. 16, the wake is reduced in size, lowering the drag of the ball considerably. Now don’t be misled. Turbulent boundary layers still have higher skin friction drag than laminar boundary layers.

![FIG. 15: Flow Patterns Past Smooth Spheres at High and Low Reynolds Numbers](image-placeholder)

CD = 0.47 at RN = 100,000 (Low Speed, Large Wake)
CD = 0.10 at RN = 1,000,000 (High Speed, Small Wake)

We were able to decrease the total drag of the golf ball by producing a turbulent boundary layer only because the majority of this drag was due to pressure drag caused by flow separation.

The golf ball has served as a review of many aspects of aerodynamics. The concept of finding what actually causes the drag on a particular body and then taking steps to reduce this drag is the lesson we must apply to our model rocket designs.

![FIG. 16: Comparison of Flows Over Smooth and Dimpled Spheres](image-placeholder)

CD = .47 at RN = 300,000 (Smooth Sphere)
CD = 0.10 at RN = 300,000 (Dimpled Sphere)

## III. Introduction to Rocket Drag Analysis

The speed, altitude and range of a full size rocket, just like our models, depends upon its aerodynamic drag. Therefore, one of the first tasks of a rocket designer is to estimate the drag of any new configuration. The drag analysis of rockets, which have fairly complicated aerodynamic shapes (compared to our golf ball, for example) is usually simplified by considering the rocket to be made up of several simple basic components. In this process the drag of each separate part is determined and any portion of the rocket which develops excessive drag is identified. Steps to improve this high drag component can then be taken. This is exactly the procedure we will follow. We’ll determine where the drag of our model comes from, reduce it where possible, and thus improve the overall performance of our birds.

### Sources of Drag

How should we break our model into basic components? Our models have two major parts — a rocket body and a set of fins. Drag of these two components will have to be examined in detail. Note how these two parts fit into our previous discussion of basic aerodynamics. The fins are similar to the flat plate shown in Fig. 4; fins will generate mostly friction drag. The nose cone

![FIG. 17: Rocket Components for Drag Analysis](image-placeholder)

= Nose Cone + Body Tube + Base + Fins = Total Rocket Components




and body tube are much like the ellipsoids shown in Fig. 5, and will produce both pressure drag and skin friction drag. Like the ellipsoids, the amount of each type of drag will depend upon the nose shape and the length to diameter ratio of a particular rocket. Unlike the ellipsoids, our rockets have squared rear portions (engineers say the rocket have “blunt bases”) which create more pressure drag than the pressure drag of the ellipsoids.

In the next section, we’ll examine rocket body drag. In order that we may use our drag analysis directly for the improvement of our models, we’ll break the drag of the body into three parts; the drag of the nose cone, the drag of the cylindrical body and the drag of the base. In this manner, we’ll be able to evaluate the drag of the individual parts which is, of course, the way we build our models. We will then be able to take practical steps to reduce the drag of these components to the lowest value.

The fin drag will be treated in the section following the rocket body drag analysis. The total drag is the sum of the drag of the two basic portions; Fig. 17 illustrates how all the parts combine to give an indication of the total rocket drag.

We can write a word equation to represent this procedure:

Nose Cone Drag + Body Tube Drag + Base Drag + Fin Drag = Total Component Drag

As we learned in Section II, the drag depends upon rocket size, velocity and air density, which means that if any of these quantities changes, the drag will also change. We can be more flexible if we deal with drag coefficients, rather than with the drag force in pounds or grams. Once we get the coefficients for a particular rocket we can multiply by the proper factors (as indicated in Eq. 1) to obtain the aerodynamic drag in pounds or grams. Therefore, we will use the coefficient form of the word equation:

$ C_{DN} + C_{DBT} + C_{DB} + C_{DF} = C_{DC} $

(6)

where
\( C_{DN} \) is the drag coefficient of the nose shape
\( C_{DBT} \) is the drag coefficient of the body tube
\( C_{DB} \) is the drag coefficient of the base
\( C_{DF} \) is the drag coefficient of the fins
\( C_{DC} \) is the drag coefficient of the sum of the components

### Interference Drag

To these basic components of Eq. (6), two other drag increments must be added to obtain the total rocket drag (or drag coefficient). The first additional amount of drag is caused by the joining of the fins to the rocket body. When joined together, the air flows about the rocket body and fin tend to “interfere” with each other. This altered flow pattern causes the drag to increase above the value of the simple sum of the two components. The increased drag is termed “interference drag”; we’ll give it the symbol \( C_{Dint} \). Interference drag can be as much as 10% above the sum of the fin and body tube drag. In Section VI we’ll use a simple method to estimate the value of \( C_{Dint} \).

Additional drag is caused by any other rocket components — launch lugs, for example. We can use the symbol \( C_{DLL} \) to represent the drag coefficient of the launch lug. So, the final equation becomes:

$ C_{DO} = C_{DN} + C_{DBT} + C_{DB} + C_{DF} + C_{Dint} + C_{DLL} $

(7)

Our job then, is to find the drag coefficients of all the components, using theory or experiment, and then add them up to find the total rocket drag coefficient, \( C_{DO} \).

### Induced Drag

The subscript, O, in \( C_{DO} \) is used for a special reason. This drag coefficient represents the drag of the rocket when it is moving directly into the wind. The angle between the rocket’s centerline and the oncoming air stream is zero in this case, as shown in Fig. 18. This figure also defines the “relative wind”: it’s the oncoming air stream, directly opposed to the flight path of the rocket. By finding the drag coefficient at zero angle to the relative wind, \( C_{DO} \) which is given by Eq. (7), we find the lowest possible drag of the rocket. Any angle to the wind will produce higher drag coefficients than the value given by \( C_{DO} \).

![FIG. 18: Rocket Moving at Zero Angle to Relative Wind](image-placeholder)

This is an important point; let’s look at Fig. 19 to see where this extra drag comes from. In this figure, the rocket is shown at an angle to the wind. The angle between the relative wind and the rocket centerline is termed the “angle of attack”. Any time the rocket is at an angle of attack, the air flow is altered and the rocket develops an aerodynamic force at right angles to the wind. On airplanes, which travel horizontally, this force is called lift; we’ll use the same name. Now, remember one of our first statements about aerodynamic forces? Whenever we get lift, we’ll get some drag. Aerodynamicists say the lift causes or “induces” this drag, so we call this type of drag due to lift “induced drag”. This kind of drag comes about solely from the angle of attack; that’s why we wanted to examine the rocket flight at an angle to the relative wind.



![FIG. 19: Rocket at Angle of Attack to Relative Wind](image-placeholder)

Obviously, we could get rid of the fin induced drag by eliminating the fins. However, our birds do not have the automatic guidance systems of “big birds” so, in order to assure straight up flights, we have to build in stability. We make rockets stable by putting fins on them, and the price we pay for this stability is the added drag of the fins — friction drag, pressure drag and induced drag.

It's not possible to discuss rocket stability in detail at this time, but we must recognize that stability does affect the drag of our models. For example, as shown in Fig. 20, when a model climbs vertically at 200 ft/sec and encounters a sudden horizontal wind of 20 ft/sec, an angle of attack near 5° is created. How the model reacts to this angle depends upon its stability. Stable rockets, which have sufficient fin lift to overcome the body lift, will return to zero angle of attack by rotating about the center of gravity. How long the rocket requires to return to zero angle, and the magnitude of the angles attained during this return to zero depends upon the particular design. In general, very stable rockets return rapidly to zero angle, while less stable rockets require more time. The more time spent at an angle to the wind, the higher the total air resistance encountered by a model rocket. So we can see how stability can affect our model drag.

It is time to return to the detailed drag analysis of our rockets and find how we can use our knowledge of aerodynamics to predict the drag. I’ll be fun to employ our new terms, angle of attack, relative wind, and induced drag too. We’ll observe, for example, that the induced drag from the fins will be influenced by the planform, so maybe we can give some ideas how to shape fins to reduce this type of drag.

## IV. Drag of Rocket Bodies

### Nose Cones

The drag of the nose cone consists of both pressure drag and skin friction drag. When we examine the body tube drag later, we'll find the tube drag is all skin friction, so we’ll include the skin friction part of the nose cone drag in the discussion of the body tube drag. For now, let’s look only at the pressure drag of the nose cone.

Probably the worst nose cone we could use on a rocket is the flat nose shown in Fig. 21. It is not too difficult to imagine the high drag we’d get from this nose shape; just look at the high pressure region in the front (shown by the large arrows) that pushes on this flat surface. The drag coefficient for this shape is \( C_{DN} = 0.80 \).

![FIG. 20: Effect of Wind Gust on Model](image-placeholder)

![FIG. 21: Pressure Distribution on Flat Faced Cylinder \( C_{DN} = 0.8 \)](image-placeholder)

- **Small Body Lift** tries to rotate nose to right
- **Large Fin Lift** rotates nose to left towards zero angle

- **Low Pressure Region**: Free stream pressure
- **High Pressure Region**



![FIG. 22: Pressure Distribution on Ogive Nose Cone \( C_{DN} = 0.004 \)](image-placeholder)

- **Low Pressure Region**
- **High Pressure Region**

![FIG. 23: % of Drag of Flat Face Nose for Various Nose Shapes](image-placeholder)

Now, by simply rounding the nose we can really cut this nose drag. A rounded contour is shown in Fig. 22; shown also is the pressure distribution that would occur on this nose shape. If you examine this figure carefully, you’ll see that the pressure on the surface falls below the atmospheric level; this means there is a suction on certain parts of the nose cone. If enough of the nose cone has a suction on it we can attain a negative pressure drag coefficient. This means the nose cone will actually contribute in a small thrust to the rocket!

![FIG. 24: Blunted Three to One Ogive Nose Cone for Low Drag](image-placeholder)

- **Note**: Blunt Nose \(\frac{1}{16}"\) to \(\frac{1}{8}"\) Radius

![FIG. 25: Construction of Three-to-One and Two-to-One Ogive Contours](image-placeholder)

- **4.25 d for 2 to 1 Ogive**
- **9.25 d for 3 to 1 Ogive**

**Note**: d is body tube diameter



To be sure, we've neglected the skin friction drag so far. We've also neglected the low pressure at the rear of the rocket. We'll pick up this base drag as a separate contribution later, just as we will include the skin friction with the body tube analysis. The fact does remain, though, that the rounded nose cone shown in Fig. 22 has far less pressure drag than the flat faced shape of Fig. 21.

Many nose cone shapes can be used on your rockets, of course. Some of these are shown in Fig. 23. To show how the drag of each shape compares with the drag of the flat face, the drag coefficient of each shape has been divided by the flat-face drag coefficient. You might be surprised to find that the pointed cones show much higher drag than the rounded nose shapes. Many model rocketeers, recalling pictures of real rockets with pointed cones, use pointed shapes, believing these will give low drag. These modelers forget that the real rockets fly many times faster than the speed of sound. In supersonic flight pointed cones, which can cut through the shock waves generated by such high speed, are indeed better than the round shapes. Our model rockets, however, do not fly faster than sound, and therefore do not build up any shock waves. Without shock waves the rounded shapes which guide the flow gently around the nose contour are definitely superior.

A good, low drag nose cone contour is shown in Fig. 24. It is a blunted, three-to-one ogive. An ogive is a simple shape, generated by an arc of a circle; specific ogives are named by giving the length to diameter ratio of the ogive. Thus, a three-to-one ogive has a length three times the diameter. The small radius at the tip is used to prevent flow separation when the nose oscillates slightly.

For example, an ogive nose cone to fit a BT-50 body tube is obtained by drawing an arc of 9.25 x .976 = 9.03 inches. After drawing this arc on a cardboard template the ogive contour can be cut out of the cardboard and used to shape your own low drag nose cone.

### Body Tube Drag

Now, let's look at the drag of the body tube. We'll limit this examination to the drag at zero lift; this will give us the lowest drag for the body tube since it is the body drag at zero angle to the relative wind. It's always a good idea to get the minimum drag because this gives us a goal to try to attain. In this section we'll include the drag of the nose cone as well as the drag of the body tube, since we neglected the friction drag of the component earlier.

To begin, what factors contribute to the drag of the nose cone and body tube? Any pressure unbalance and air friction, of course; that's why we discussed these terms. Recall that, in Fig. 5, we found that the drag of a series of ellipsoids depended upon the length to diameter ratio (L/d) of the ellipsoid. We can expect, therefore, that the nose cone and body tube drag will also depend upon length to diameter ratio. Further, Fig. 5 shows that as the length to diameter ratio increases, friction drag gets more and more important; so we can anticipate that our model rockets, which usually have high L/d, will encounter a large amount of skin friction. That means the boundary layer will be important in our drag analysis.

Let’s imagine a wind tunnel test of a specific rocket body, say a one inch diameter tube with a 3 to 1 ogive nose cone. We’ll make the model one foot long, so its L/d will be 12. These dimensions are representative of many sport model rockets; therefore our drag analysis can be applied directly to these types of models.

![FIG. 26: Drag of Nose Cone and Body Tube for L/d = 12 Model with Laminar and Turbulent Boundary Layer as Speed Varies](image-placeholder)

Just as we did for our golf ball illustration, we’ll illustrate how the drag force builds up on the rocket body. We’ll also show how two types of boundary layer will affect the total drag. We assume that the boundary layer can be either laminar or turbulent. At very low speeds, the laminar layer will build up and so will the drag force. At high speed, when L/d = 12, the body layer will also build up. The level of the drag force, as shown in Fig. 26, will depend upon the kind of boundary layer that covers the model. We’ll get the lowest drag if we can keep the boundary layer laminar; we’ll get the highest drag if the boundary layer is turbulent everywhere. These two limits are shown in Fig. 26. An intermediate case, with the nose cone laminar and body tube turbulent (which is likely) is also shown in the figure; its drag falls between the two limits as we would expect.

![FIG. 27: Nose Cone and Body Tube Drag Coefficient as Speed Varies for Several L/d](image-placeholder)

- Fully Laminar Boundary Layer
- Fully Turbulent Boundary Layer



We can see from this figure that doubling the speed from 200 ft/sec to 400 ft/sec increases the drag by a factor of three—from 0.014 to 0.042 lbs for the laminar boundary layer. For the turbulent case, the drag is increased by a factor of almost four—from 0.095 lbs to 0.19 lbs. We see also that, at any given speed, the turbulent boundary layer gives much more drag (four to five times greater) than the laminar boundary layer. The laminar layer is certainly one we’d like to use on our models if we can. (We’ll discuss this in more detail later, but a smooth surface is one of the most important ways to keep the boundary layer laminar, so put a smooth finish on your model rockets!)

We found that a good way to make any aerodynamic test more usable was to put the results into coefficient form. This allows us to apply the results to different size models, different air densities and different speeds. Equation 5 was used before to convert a drag force to a coefficient; remember we have to divide the drag by the density of the air, the “square” of the velocity (V x V), and the cross-sectional area of the body tube. When we do this arithmetic, we come up with Fig. 27. You'll note that the drag coefficient is labeled \( C_{DN} + C_{DBT} \).

This is because the drag force in our imaginary experiment was due to the nose cone and body tube. These two drag coefficients were part of Eq. 6, the total zero lift drag coefficient of our model rockets. Our practical drag analysis is proceeding. Figure 27 also includes the drag coefficients for models of two different L/d, 8 and 16, so we now have the nose cone and body tube drag coefficients for three different size rockets.

Before we extend our analysis further, we’d better make a comment about the drag coefficients of Fig. 27, which decrease as the speed of the rocket bodies increases. This does not mean that the drag goes down; that’s a mistake many people make. Remember, the drag force is obtained by multiplying by velocity squared, so the drag is still rising as shown in Fig. 26.

![FIG. 28: Nose Cone and Body Tube Drag Coefficient for Different Length to Diameter Ratio](image-placeholder)

Figure 28 has been prepared to illustrate the nose cone and body tube drag coefficients for more length to diameter ratios. The range from L/d = 4 to L/d = 20 is covered in this type of design chart, with the drag coefficients plotted for three different rocket velocities. Curves like these help us in model rocket design by showing the trends of the drag coefficient as we change length to diameter ratio, speed, or type of boundary layer. For example, it is clear from the descent that the drag coefficient for the turbulent case drops more strongly upon the length to diameter ratio than do the laminar coefficients. On the other hand, the chart shows that the laminar drag coefficients are more sensitive to velocity changes; this can be noted from the 54% decrease in drag coefficient for the laminar case contrasted to the 25% decrease of drag coefficient for the turbulent boundary layer. These values are for the L/d = 12 model as it undergoes a change in speed from 100 to 500 ft/sec.

These curves can be used for all 1” models with 3:1 ogive nose cones, but what about other shapes and model rocket diameters? For other rocket configurations we would have to run more wind tunnel tests or find a mathematical way to determine the drag coefficients. Luckily, aerodynamicists have been working on this problem for many years, so short hand equations are available for our use. The one equation which we can use for the nose cone and body tube is

$ C_{DN} + C_{DBT} = 1.02 \, C_f \left[ 1 + \frac{1.5}{(L/d)^{3/2}} \right] \frac{S_w}{S_{BPT}} $

(8)

Not surprisingly, the value for \( C_{DN} + C_{DBT} \) depends upon length to diameter ratio, L/d, and upon skin friction, \( C_f \). The \( S_w \) in Eq. (8) stands for the "wetted surface area" of the rocket; this is the area of the rocket "scrubbed" (or wet) by the boundary layer. You can think of it as the total area of the nose cone and rocket that would get wet if the model were dunked in water. The \( S_{BPT} \) in the equation is the cross-sectional area of the body tube. The skin friction is represented by the skin friction coefficient, \( C_f \), which depends upon the rocket speed, air density and air viscosity, as well as rocket size (in other words, the Reynolds number-- oh, oh, that term again).

Instead of going into the details of these calculations at this time, let's reserve this work for the appendix. In Appendix A, we have charts to find the Reynolds number for our rockets and the \( C_f \) to be used in Eq. (8) once we have found the correct Reynolds number. At this time, let’s continue our drag analysis by examining the base drag of our models.

### Base Drag

The first thing to do when we consider base drag is to find where it comes from. Base drag is due entirely to low pressure at the rear of the rocket caused by flow separation. That’s the aerodynamicist's description, but let’s look at this definition in an existing flight in Fig. 29 to clear up this definition. As the air flowing past the rocket (represented by the streamlines) reaches the rear, it tries to make the sharp turn to follow the base contour. However, the viscosity of the air prevents any sudden change of flow direction, so the flow separates from the surface, creating a partial vacuum at the base of the rocket. This low pressure region at the rear produces the pressure unbalance that gives rise to base drag.

How do we find the level of this base drag? This is a very difficult problem even for full-scale rocket designers. We might suspect that the character of the boundary layer may have something to do with the actual value of



base drag, since viscosity has caused the flow separation. The golf ball, remember, had a pressure drag that varied greatly as the boundary layer changed from laminar to turbulent conditions. A similar behavior for base drag of rocket shapes has been observed by experimenters. Our base drag coefficient, \( C_{DB} \), must therefore depend upon the flow Reynolds number, because the Reynolds number determines if the boundary layer will be turbulent or laminar.

![FIG. 29: Base Pressure Coefficient Dependence](image-placeholder)

After many experiments, aerodynamicists have been able to come up with an approximate equation for the base drag:

$ C_{DB} = \frac{0.029}{\sqrt{C_{DN} + C_{DBT}}} $

(9)

Note how convenient this expression is; the base drag coefficient is determined from the sum of the nose cone and body tube drag coefficients, \( C_{DN} + C_{DBT} \) which has just been found from Eq. (8). The reason we can write the base drag in terms of \( C_{DN} + C_{DBT} \) can be seen by considering the pressure on the flat base of the rocket. It will vary according to how the flow turns at the rear. A sharp turn produces a strong pressure gradient and greater base drag. Changing from a laminar to a turbulent state allows the flow to make the turn more towards the flat base, raising the base pressure to higher levels than obtainable with laminar layers (although still below the local atmospheric pressure). The drag coefficients are therefore lower than at values of \( C_{DN} + C_{DBT} \) associated with laminar boundary layers.

We now have three terms of Eq. (6), \( C_{DN} \), \( C_{DBT} \), and \( C_{DB} \). The sum of these three drag coefficients gives the zero lift drag coefficient of rocket bodies; it is instructive to examine the drag of these bodies before going on to discuss fin drag. Even before this, though, we should emphasize that we are considering coasting flight. During boost, the rocket exhausts into the base region, completely altering the flow field at the rear of the rocket. The level of base drag depends on whether the exit pressure of the rocket gases and the velocity of the jet from the rocket. These factors differ with each rocket design and are difficult to account for in our drag analysis. About the only statement that can be made is that if the pressure in the rocket exhaust is the local atmospheric pressure, the base pressure would be atmospheric and there would be no pressure unbalance. For this case, base drag would be zero. An estimate for the base drag during boost flight, since model rocket engines operate with near atmospheric exhausts, is that the base drag coefficient \( C_{DB} \) is approximately zero.

This base drag problem during rocket firing is a part of our drag analysis that could use some good aerodynamic research by a model rocketeer.

### Zero Lift Rocket Body Drag

To find the drag of the rocket body at zero angle to the relative wind, all the drag coefficients are merely summed up. This is the zero lift drag coefficient. We’ll represent the coefficient by \( C_{DOB} \) and write another word equation:

$ C_{DOB} = C_{DN} + C_{DBT} + C_{DB} $

(10)

where \( C_{DOB} \) is the zero lift drag of the rocket body.

We show the result of this addition in Fig. 31 for the different length to diameter models moving at 100 ft/sec.

Some very important observations can be made from this figure. First, for the laminar case, the base drag coefficient is observed to be more than 50% of the rocket body drag. Further, \( C_{DOB} \) for the laminar case is practically constant with length to diameter, varying from 0.180 to 0.193 as L/d changes from 4 to 20. In contrast, the turbulent case shows a smaller fraction of \( C_{DOB} \) comes from the base drag contribution and the total drag coefficient increases in an almost straight line from the \( C_{DOB} = 0.185 \) at L/d = 4 to \( C_{DOB} = 0.40 \) at L/d = 20.

These observations are quite helpful to us in designing our model rockets. For example, if our model can maintain a laminar boundary layer its entire length, then the place to look for drag reduction is in the base. We’ll find later that boat-tailing (tapering the rocket at the aft end) will be an effective method to cut the base drag. Also, the length of the rocket does not appear to matter for the laminar boundary layer, as the drag coefficient doesn’t change with length. This is not the case for the turbulent boundary layer because the drag coefficient is shown to increase with length. When the layer is completely turbulent, therefore, the shorter models will have less drag. We have to be careful when we



apply this rule, however, because too much shortening of a rocket can lead to an unstable model. These comments are illustrations of some of the practical uses of the drag analysis, as we apply aerodynamic concepts to improve our model rocket designs.

![FIG. 31: Rocket Body Drag Coefficient at 100 ft./sec. for Different L/d Models](image-placeholder)

The variation of zero lift body drag coefficient for three different speeds is shown in Fig. 32. The curves indicate that the drag coefficients for the turbulent case decrease with speed (that doesn't mean the drag force drops, remember), while the laminar layer exhibits the opposite behavior at the lower L/d values because of high base drag. Once again, drag coefficients for the fully laminar and fully turbulent boundary layers are presented. This procedure is necessary because of the difficulty in predicting how the boundary layer will behave on our models. We know that the boundary layer will start out as laminar but, at some point along the rocket surface, it will become turbulent. The exact point is difficult to determine (a good guess, however, is the junction of the nose cone and body tube) which makes the exact drag coefficient difficult to obtain also.

![FIG. 32: Rocket Body Drag Coefficient at Zero Angle of Attack](image-placeholder)

This does not mean, of course, that all our work has been in vain. Just as aeronautical engineers do when faced with similar problems (which they often are) we have placed some pretty important boundaries on the drag coefficient. We know, for example, that the lowest possible drag coefficient (for L/d greater than 8) occurs with the laminar layer; we can strive for this value, but realize we cannot reach it except at very low speeds where turbulent flow cannot exist. Similarly, we can be assured that the body zero lift drag coefficient will be less than that predicted by the fully turbulent case, since some portion of the model must have a laminar layer. Probably the simplest approach, which best represents the true flow situation, can be obtained by choosing a drag coefficient about 75% of the distance between the laminar and turbulent curves. For example, an L/d = 12 model moving at 300 ft/sec has a \( C_{DOB} = 0.255 \) for turbulent flow and \( C_{DOB} = 0.182 \) for laminar flow with a difference of 0.073. Taking 75% of this difference gives 0.055, which, when added to the laminar \( C_{DOB} \) yields \( C_{DOB} = 0.237 \) as the most realistic value of the zero lift drag coefficient.

### Rocket Body Drag at Angle of Attack

Up to this point, we have not touched upon rocket body drag at an angle of attack. This is with good reason, since a theoretical analysis of the flow about rocket bodies at an angle to the relative wind is very difficult, and there is no point in estimating the amount of drag whenever our models get forced from the minimum drag, or zero angle condition.

Without getting involved in the details of a theory applicable to our model rockets at small angles of attack, let's look directly at the results. The increment of additional drag caused by angle of attack for three rocket bodies with 3:1 ogive nose cones is presented in Fig. 33. The chart allows a rapid calculation of the drag coefficient at any angle of attack up to 10° for the three length to diameter ratios of the bodies. All that is required is the addition of the zero lift drag coefficient, \( C_{DOB} \), to the incremental drag coefficient, \( \Delta C_{DA} \). In equation form this is written as:

$ C_{DA_B} = C_{DOB} + \Delta C_{DA} $

(11)

where \( C_{DA_B} \) is called the total rocket body drag coef-

![FIG. 33: Body Drag Coefficient Increment with Angle of Attack for 3:1 Ogive Nose Body](image-placeholder)



ficient at angle of attack and \( C_{DOB} \) is obtained from Eq. (10).

A typical solution for both the laminar and turbulent boundary layers is presented in Fig. 34 for the three different length to diameter ratio bodies. This chart is based on earlier solutions for \( C_{DOB} \) for the 1” diameter models moving at a speed of 300 ft/sec. The symmetric form of the total rocket drag is clearly illustrated, as is the occurrence of the minimum drag coefficient at zero angle of attack. Another point to be seen from the curves is that at low angles of attack the drag coefficient changes very little; however, as the rocket angle increases towards 10° angle of attack, the drag due to angle of attack forms a significant part of the total rocket body drag. Taking the L/d = 12 body for example, the drag coefficient at 10° is more than 50% above the zero angle value. Hopefully, our stable model rockets will not reach such high angles of attack, so that the drag penalty will not be this severe.

In any case, Figs. 33 and 34 show how angle of attack can increase the drag of a model rocket. Figure 33, incidentally, can be used for any diameter, 3:1 ogive nosed model rocket at angle of attack; all that need be calculated is the zero lift drag coefficient, \( C_{DOB} \), either from the charts or from the Appendix.

We are now ready to move on to fin drag.

![FIG. 34: Body Drag Coefficient Variation with Angle of Attack at 300 ft./sec.](image-placeholder)

## V. Drag of Rocket Fins

### Fin Shapes and Terms

Before we start our discussion of fin drag, we’d better develop a method to describe the many different fin designs. The amount of each type of fin drag — pressure, skin friction and induced — will depend upon the shape. We would like to be able to identify the various types of fins. When we cut a fin from a sheet of balsa, we trace out a particular “planform”. Typical planforms used for model rockets (and full size birds) are shown in Fig. 35. These shapes can be described as rectangular, straight-tapered, swept-tapered, and elliptical.

![FIG. 35: Fin Designs](image-placeholder)

- **Rectangular**
- **Straight-Tapered**
- **Swept-Tapered**
- **Elliptical**

To aid in a more complete description of the various planforms we have to introduce additional terminology. For example, tapered planforms can have many different shapes so we should think about a term to describe the various possible taper ratios. We'll use the symbol \(\lambda\) to represent taper ratio, which we’ll define as the ratio of the tip chord to root chord of a fin.

$$
\lambda = \frac{C_T}{C_R}
$$

Figure 36 illustrates these new terms. It is also a good idea to define a sweep angle, because tapered planforms can also be swept. If we describe the sweep of the mid chord point by \(\Delta C/2\), we will be able to classify the swept and tapered planforms by specifying \(\lambda\) and \(\Delta C/2\). This definition has the advantage that the sweep of the mid chord is zero for the straight taper, although the leading edge does have a sweep angle as shown in Figure 36.

Even with taper ratio and sweep angle determined, we still need another factor to completely specify the configuration of the fin. The term required is called aspect ratio; it is used to indicate the relationship of the length to width of the fin. Aspect ratio, A.R., was originally used to describe the wings of aircraft so, as shown in Fig. 37, the total span of the fins and the total surface area, including the portion covered up by the rocket body, is used to calculate the value for aspect ratio. For rectangular wings, the aspect ratio is simply the span, b, divided by the chord, c. Therefore, high aspect ratio fins are long and narrow while low aspect ratio fins are short and stubby. When fins have shapes different from the rectangular, the aspect ratio is calculated from Eq. 12,

$$
A.R. = \frac{b \times b}{S}
$$

(12)

where \( S \) is the entire surface area.



![FIG. 36: Definitions of Taper Ratio and Sweep for Model Rocket Fins](image-placeholder)

- Straight-Taper: \(\lambda = \frac{C_T}{C_R}, \Delta C/2 = 0\)
- \(\lambda = \frac{C_T}{C_R}, \Delta C/2 = 45^\circ\)

![FIG. 37: Rectangular Fins](image-placeholder)

- AR = \(\frac{b}{c}\)
- \(S = \text{Shaded Area}\)
- A.R. = \(\frac{b \times b}{S}\)

Planform shape is not the only factor that determines fin drag; the cross-sectional shape also contributes to drag. Three typical model rocket fin cross-sections or airfoil shapes are shown in Fig. 38. The first shape is the rectangular section you get if you simply cut the fin out of the balsa and don’t sand or finish it. The second shape has rounded leading and trailing edges; this is the airfoil we have when we sand the edges lightly. The third cross-section is a streamlined airfoil, like that shown earlier in Fig. 6, but much thinner. It has a rounded nose and sharp trailing edge. As shown in the figure, the ratio of the thickness to the chord is also used to aid in our description of the fin. A \( \frac{1}{16} \)" sheet fin with a 1 inch chord has a thickness ratio, \(\frac{t}{c}\), of 0.0625. If the same sheet were used to cut a fin with a 2" chord, \(\frac{t}{c} = 0.03125\). These numbers give us an idea of the thickness ratios our model rocket fins usually have. Our fins are quite thin; rarely will we have a fin greater than \(\frac{t}{c} = 0.1\).

![FIG. 38](image-placeholder)

It appears then, that we must examine two basic features of rocket fins if we wish to find the fin drag. Both planform and cross-sectional shape (airfoil) will contribute to this drag, so let’s examine both of these factors to see if we can gain some insight into the types of fins which will give the least drag and which will therefore give us the best performance.

## Drag of Airfoil Sections

To study airfoil section drag characteristics we will limit ourselves to fins with rectangular planforms operated at zero angle to the relative wind. In this fashion, we will be able to study the effects of the variation of cross-section on fin drag alone and not encounter drag variations due to planform shape and angle of attack. We’ll look at these effects in a later section.

As we have done before, let’s imagine a wind tunnel test on three rectangular fins, each with a different cross-sectional shape. We will make these fins out of \(\frac{1}{8}"\) balsa sheet and cut them with a 2" chord and a 6" span. The three cross-sections will be rectangular, rounded, and streamlined. When we measure the drag force on these airfoils we’ll come up with the curves shown in Fig. 39 as we increase the wind tunnel speed. The drag for all three sections increases rapidly with speed, but the rectangular cross-section certainly has a lot more drag than the streamlined shape. At 200 ft/sec, for example, the rectangular airfoil has a drag force of 0.22 lbs, while the streamlined airfoil has a drag force of about 0.036 lbs. Note that, just by rounding the leading and trailing edges, we can reduce the drag of the section sizeably; at 200 ft/sec the drag is re-



duced from the rectangular section by almost 50% down to 0.124 lbs. The advantage of streamlining your fins is clearly shown in this figure.

![FIG. 39: Drag of Three Fin Cross-Sections](image-placeholder)

## Fin Drag Coefficients

Of course, if we change the size of the fins the drag will vary, so it would be more convenient if we would use fin drag coefficients, rather than drag force. This is also in keeping with our drag analysis, for we want to find \( C_{DF} \) for Eq. (6) so that we may find the total rocket drag coefficient. We have to be a little careful about this fin drag coefficient; we must be certain we represent the size factor in the drag equation correctly. Equation (5) for the drag coefficient still applies, of course, but we must be sure of what reference area should be used. For rocket bodies, we used the cross-sectional area, \( S_B \), to obtain the coefficient. Since we’re talking about fins which are usually quite thin, cross-section area is not the most convenient area to measure or to use. Planform area is a much better reference area; therefore, we will use the planform area, termed \( S_F \), to find the zero lift drag coefficient of our fins as shown in Eq. 13.

$ C_{DOF^*} = \frac{DF}{\frac{1}{2}dV^2S_F} $

(13)

When we apply this equation to the results of Fig. 38, we come up with drag coefficients \( C_{DOF^*} = 0.0563, 0.0313, \) and \( 0.009 \) for the rectangular, rounded and streamlined cross-sections respectively. These values apply to the \(\frac{1}{8}"\) thick fin with a 2" chord; this gives a thickness to chord ratio \(\frac{t}{c}\) of 0.0625. What about other thicknesses? We can answer this by referring to Fig. 40, which indicates how fin drag coefficients vary with thickness ratio.

The rectangular sections increase very rapidly with thickness. This is because the flat front and back surfaces give a large pressure drag (similar to the flat nose cone we considered in Chapter IV). Rounding the nose and back edges decreases the pressure drag, but base drag still forms a large portion of the drag. Streamlining by sharpening the trailing edge reduces this base drag, just as we discussed for the shape shown in Fig. 7. We also note that as the thickness of each section decreases, the drag coefficients are reduced; the pressure drag is of less importance as the fin gets thinner. Ultimately, if the thickness were zero, we would have no pressure drag. Does that mean no fin drag? No, friction drag still remains, of course. See how our basic aerodynamic concepts keep coming back to help us understand model rocket drag?

For streamlined cross-sections, in fact, the drag is due almost entirely to skin friction. This means that the drag coefficient of a fin depends on the type of boundary layer and, again, the Reynolds number. Although the values of drag coefficient shown in Fig. 40 are good average values for fin drag coefficients, more precise results can be obtained, for streamlined fins at least, by using Eq. (14), below, and the skin friction chart given in the Appendix.

$ C_{DOF^*} = 2C_f \left(1 + 2\frac{t}{c}\right) $

(14)

![FIG. 40: Zero Lift Drag Coefficients for Three Cross-Sections of Various Thickness Ratio](image-placeholder)

In the chart, \( C_f \) is found according to the Reynolds number. The Reynolds number, in turn, is determined from the average chord length.

Figure 41 is the result of applying Eq. (14) to streamlined fins, \(\frac{1}{16}"\) thick, for three flight speeds. We can observe the difference in coefficients for the two types of boundary layers. Once again, it is clear that the laminar boundary layer will produce the least drag at a given speed. Remembering that surface finish will help us keep the boundary layer laminar, the value of smoothing the fins and giving them a good finish is obvious.



![FIG. 41: Zero Lift Fin Drag Coefficient for Streamlined Cross-Sections of Several Thickness Ratios](image-placeholder)

## Drag of Fin Planforms

At zero angle of attack, the fin cross-sections that we’ve just examined have zero lift. For this case we’ve found that the particular planform does not contribute to the drag except through the amount of surface area for the fin. However, anytime our model rocket gets pushed to an angle of attack, the fin’s “lift” to bring the model back to zero angle to the relative wind. As discussed earlier, this lifting action induces additional drag. Since both the planform and angle of attack of the fins determine the lift, these factors will also determine the level of the “induced” drag. Let’s see if we can understand where this drag comes from and then find the fin designs which will give us the least drag.

Figure 42 presents a series of sketches showing the flow field about the fins of a model rocket. The first sketch shows how the air flows about a section of a fin at angle of attack. The high pressure and low pressure regions are indicated by the arrows; this pressure unbalance produces the lift. The greater the angle of attack, the larger the lift force generated. At about a 15° angle of attack, the air flow separates from the upper surface of the fin and the lift force is destroyed (aerodynamicists say the fin, or wing, has “stalled”). The second sketch is a view of the rocket fins from the front, showing how the high and low pressure regions are distributed along the span of the fin. Air tends to flow from high pressure zones to low pressure zones, we know, so we might expect some portion of the air to try to move around the tip of the fin from the high pressure undersurface to the low pressure upper surface.

![FIG. 42: How Tip Vortex is Produced](image-placeholder)

1. **Flow over Cross-section of Fin at Angle of Attack**
   - \( \alpha \), Angle of Attack
   - Low Pressure Zone
   - High Pressure Zone

2. **Flow About Fins, Front View of Rocket at Angle of Attack**
   - Air tries to move around tip to equalize the pressure

3. **Flow About Fins, Top View Showing Vortex Pattern from Tip**

![FIG. 43: The Origin of Induced Drag](image-placeholder)

1. **Front View of Rocket Showing Downwash on Fin Created by Tip Vortex**
   - \(\alpha\), Angle of Attack
   - Relative Wind with No Downwash

2. **Downwash Tilts Lift Force Back Creating Induced Drag, \( D_i \)**

Such motion does actually occur; it's complicated, though, by the forward motion of the rocket. As the air tries to curl around the fin tips, the fin moves out of the way and the swirling airflow pattern shown in sketch C of Fig. 42 is set up.

### Vortex Flow

Swirling flow patterns are called vortex flow patterns (pull the plug in your bathtub and you have an example of a bathtub vortex which is very similar to



the vortex from the fin tip). The strength of the tip vortex depends upon the spanwise pressure distribution, but, as this distribution determines the lift, we can also state that the vortex depends upon the lift. The greater the lift, the stronger the vortex.

What do we mean by a strong vortex and how does the vortex cause this extra drag? Two good questions; we’ll use Fig. 43 to explain. In sketch (a) of the figure, the front view of the rocket, the vortex is shown coming from the tip. The dotted lines indicate how a vortex creates an additional flow velocity along the span. This additional velocity, shown to be “downward” in the sketch, changes the relative wind over the fin. As shown in the second sketch, a cross-sectional view along the span of the fin, the additional air velocity, called "downwash", combines with the original wind direction to form a new relative wind. The effect of this combination is to cause the lift of the fin to tilt backward, thereby generating a component to the rear, which is a drag force. The drag caused by this rearward tilt of the lift is called the “induced” drag; it is a direct result of the vortex pattern which created the downwash, which itself was produced by the pressure distribution about the fin.

That's all there is to explanation of induced drag. We see that if the fin were at zero angle of attack there would be no lift and the pressure on the upper and lower surfaces of the fin would be the same. There would then be no flow towards the tip, so no tip vortex would form, the relative wind would not be changed, and therefore, we would have no induced drag. Because this pressure distribution along the span is so important to the generation of the tip vortex, and because this span-wise pressure distribution is determined from the shape of the planform, we can begin to understand how fin planforms can influence the induced drag. Let’s continue our discussion by examining typical fin shapes.

## Evaluating Fin Shapes

Aerodynamicists have been studying wings for many years; let’s take advantage of their work by using the experimental results and theories these investigators have accumulated on wings. Rocket fins are, after all, just half wings, so the wing results will apply to our model fins. At speeds below that of sound, an elliptical wing planform has been determined to be the most efficient; that is, the elliptical shape gives the least induced drag of all comparable size wings. The elliptical wing, therefore, is often used as the standard to compare the performance of other wings.

$ C_{D_{i}} = \frac{C_{L} \times C_{L}}{\pi \times A.R.} e_{w} $

(15)

\( C_{L} \) is the wing lift coefficient; it has values from 0 to 1.0 depending upon the angle of attack. We can find lift of the wing if we use the equation below:

$ L = C_{L} \frac{1}{2} dV^2 \text{wing} $

For elliptical wings, \( e_{w} = 1 \); for all other wings, \( e_{w} \) is less than one. This means that \( C_{D_{i}} \) will be larger for all planforms other than the elliptical ones. Equation 15 also illustrates the importance of aspect ratio in determining the induced drag. The higher the aspect ratio, the lower the value of \( C_{D_{i}} \), all other factors being equal. Long narrow fins give lower induced drag than short stubby planforms.

![FIG. 44: Wing Efficiency Factor](image-placeholder)

Typical values of \( e_w \) for rectangular and tapered planforms with zero sweep angle, \(\Delta C/2\) are presented in Fig. 44 for different values of aspect ratio. Using this information, we can find the induced drag from Eq. (15) for different planforms for constant values of lift coefficient as we vary the aspect ratio. Figure 45 presents these results for three lift coefficients for rectangular and elliptical wings. The tapered wings would have an induced drag that falls between these two limits. From this figure we can see that the higher aspect ratio wings have much less induced drag than the low aspect ratios. For example the elliptical wing at \( C_{L} = 0.2 \), has \( C_{D_i} = 0.007 \) for an aspect ratio of 2 but only 0.002 for an aspect ratio of 6. The rectangular wing has \( C_{D_i} = 0.0086 \) and 0.0025 for the same conditions -- an increase of more than 20% of induced drag.

![FIG. 45: Induced Drag Coefficient for Elliptical and Rectangular Wings](image-placeholder)

Swept wings have not been considered as yet. One of the reasons is that these wings are difficult to analyze. We know, in general, that wing sweeping in subsonic flows will lead to greater values of induced drag than for straight wings. Again you may wonder why so many full-scale rockets and missiles have swept fins; the reason is the same as the one given for pointed nose cones. Full scale rockets fly faster than sound and therefore generate very strong, drag producing, shock waves. Wing sweep allows the wing or fin to cut through the shock wave more easily and does reduce the supersonic wave drag. Model rockets, however fly slower than sound; they generate no shock systems. Therefore, sweep is not necessary for efficient design.

Another interesting point can also be made from Fig. 45. How do we get a lift coefficient of 0.2? The lift is caused by an angle of attack of the wing (or fin) due to a gust or other de-stabilizing influence. This gust, in turn, puts the rocket body at an angle of attack. This



means that the amount of lift we can generate for each degree of angle of attack is also important in our choice of planform. Which wing gives the most lift for the least angle? Figure 46 provides the answer, this time including swept planforms. Once again, the elliptical planform is the best, followed by the straight rectangular and then by the swept planforms. Suppose a \( C_{L} = 0.2 \) is required to correct the flight path of the aircraft. According to Fig. 46, an elliptic wing of \( A_{R} = 2 \) will produce a lift coefficient of 0.054 per degree angle of attack. That means that the angle that will produce \( C_{L} = 0.2 \) is given by

$$
\frac{C_I}{0.2} = \frac{0.2}{0.054} = 3.7°
$$

If the \( A_{R} \) were 6, then \( C_I \) per degree is 0.082

$$
\alpha = \frac{.2}{.082} = 2.4°
$$

We could look back to Fig. 34 to see what increment increase we would have for this body angle for an L/d = 2 rocket body. You’ll note that the increase in \( C_D \) is about 0.013 and 0.007; we can observe that the fin planform couples with the body to determine the overall drag.

With rectangular planforms the wings are less efficient, so the wing and body must go to a higher angle of attack to obtain the necessary lift coefficient to recover from the gust. Let’s look at a swept wing with \(\Lambda = 60^\circ\); we find \( C_{L} \) per degree is 0.034 and 0.046 for \( AR = 2 \) and 6, respectively. This means the angle necessary to produce \( C_{L} = 0.2 \) is 5.8° and 4.3°, for the two aspect ratios. The corresponding increase in body drag coefficient is 0.027 and 0.020, factors of two and three above the drag coefficient increase with elliptic wings. For the rectangular case, with no sweep, you would get drag increases of 0.015 and 0.008. You might check these values. Again, for all cases, the elliptic planform not only produces the least induced drag but causes the least incremental increase in body drag.

To sum up this discussion of planform drag, wings (and therefore fins) that have elliptical planforms and high aspect ratio will produce the least drag due to lift. Tapered and unswept planforms would be the next best choice; in fact, a straight taper with \(\Lambda = 0.4\) is just about as good as the elliptical planform. Swept planforms look pretty, but give more induced drag than unswept wings.

![FIG. 46: Lift Coefficient for Angle of Attack for Several Fin Planforms](image-placeholder)

![FIG. 47: Typical Planforms](image-placeholder)

- **Elliptical:**
  - \( S = 0.785 \, C_T \, b \)
  - \( A.R. = \frac{1.27 \, b}{C_R} \)

- **Rectangular:**
  - \( S = C_T \times b \)
  - \( A.R. = \frac{b}{C_T} \)

- **Swept:**
  - \( S = C_T \times b \)
  - \( A.R. = \frac{b}{C_T} \)

- **Tapered:**
  - \( S = \frac{(C_T + C_R) \, b}{2} \)
  - \( A.R. = \frac{2 \, b}{C_T + C_R} \)



A word of caution about aspect ratio: don’t use too high a value or you’ll have structural problems - aerodynamic forces can pull the fins off. As with most aerodynamic problems, fin design must be a compromise of aerodynamic efficiency and structural strength. Figure 47 presents typical fin planforms and gives equations to find areas and aspect ratios of these shapes.

## Total Fin Drag

At the beginning of this section of fins, we noted that the drag depended upon the cross-section and planform. We have been able to find the drag coefficient for both types of fin drag; the total drag coefficient is the sum of these two components.

Thus

$ C_{DF^*} = C_{DOF^*} + C_{DiF^*} $

(16)

where \( C_{DF^*} \) is the total drag coefficient based on fin area, \( C_{DOF^*} \) is the zero lift drag coefficient, found from Fig. 39 or for streamlined sections from Eq. (14), and \( C_{DiF^*} \) is the induced drag coefficient found from Eq. (15).

We conclude this section on fins by finding the total drag of four fin designs. As usual, we’ll pick the worst case and the best case to try to bound the fin drag coefficient. We will consider unswept planforms; rectangular with rectangular cross-sections and elliptical with streamline cross-sections. Further, we will consider two aspect ratios, \( A.R. = 2 \) and \( A.R. = 6 \).

To ensure that we may apply these results to our model rockets, we’ll construct the fins of \( \frac{1}{16}" \) sheet balsa with an area of 6 square inches. The geometries selected are shown in Fig. 48. The fin drag coefficients are shown as a function of lift coefficient in Fig. 49. The increase in \( C_{DF^*} \) with \( C_L \) is much more rapid for the \( A.R. = 2 \) planforms compared to the \( A.R. = 6 \) fin shapes.

Because we’ve kept the thickness of the fin at \( \frac{1}{16}" \), the thickness ratio for the \( A.R. = 6 \) fins is greater than \(\frac{t}{c}\) for \( A.R. = 2 \). This causes the zero lift drag coefficient for the low aspect ratio fins to be below the high aspect ratio fins. If we had kept the same thickness ratio for both aspect ratio fins the zero lift drag coefficient would be the same. The curves do point out the trades that can be made in fin design.

We can observe from this figure that the low aspect ratio fins have a more rapid increase in \( C_{DF^*} \) with \( C_L \) than the high aspect ratio fins. We might also note that the minimum \( C_{DF^*} \) occurs at zero angle of attack as we would anticipate, but what about the fact that \( C_{DF^*} \) at zero angle for the low \( A.R. \) shapes is below the \( C_{DF^*} \) for high \( A.R. \)? This is a consequence of the way we made our fins: we held the thickness at \( \frac{1}{16}" \) and kept area at 6 square inches. That way, when we lowered the \( A.R. \), the fin chord increased, then when we divided the constant thickness by the larger cross-section width the thickness ratio decreased. Recall that the zero lift drag coefficient, \( C_{DOF^*} \), of the fins decreased as the fins got thinner. In fact, for the streamlined, elliptical fin planforms, the curves show that the low \( A.R. \) fin gives lower total drag until \( C_L \) gets greater than 0.15. For the rectangular fin, the advantage of the low aspect ratio is even greater--up to \( C_L = 0.35 \). If we have stable rockets it may be advantageous to use low aspect ratio fins because they have smaller thickness ratios.

![FIG. 48: S(2 fins) = 3](image-placeholder)

- **Elliptical:**
  - \( C_R = 0.9, b = 4.2 \)
  - \( A.R = 6 \)

- **Rectangular:**
  - \( C_T = 0.71, b = 4.23 \)
  - \( A.R = 6 \)

- **Elliptical:**
  - \( C_R = 1.56, b = 2.43 \)
  - \( A.R = 2 \)

- **Rectangular:**
  - \( C_T = 1.22, b = 2.44 \)
  - \( A.R = 2 \)

![FIG. 49: Drag Coefficient of Elliptical and Rectangular Fins; Variation with Fin Lift Coefficient](image-placeholder)



We have to watch out though; we must find what angle the fin must go to in order to generate a \( C_L \) of 0.15 or 0.35. We’d find that for elliptic fins with \( AR = 6 \) the angle is 2° but with \( AR = 2 \), the angle is 3°. Thus the body drag for the low AR wing will increase more. For rectangular wings the angles are 4.8 and 8.2 respectively. Maybe we don’t want low A.R. after all.

The above illustration makes clear some of the problems (and fun) of model rocketry (and full scale aeronautical engineering). It’s very difficult to make a flat statement that this particular fin design is the best. It all depends. It depends on induced drag, lift coefficient with angle, body drag coefficient change with angle, surface finish...Wow! What we try to do is understand the basic concepts, get some general guide lines (like streamline cross-sections shapes are best, keep the surface smooth for laminar boundary layers, and for given thickness ratio, high AR is best) build the design, fly it, and find out which fin planform suits a particular task best.

![FIG. 50: Drag of Elliptical and Rectangular Fins: Variation with Angle of Attack](image-placeholder)

## VI. Total Rocket Drag

### Drag at Zero Lift

As we’ve learned earlier, the drag of our model rockets will depend upon their velocity, the air density, and the size of the rocket. In Chapter 1 we wrote Eq. (1) to show an exact relationship for drag. We rewrite that relation now to obtain the zero lift drag in pounds, \( D_0 \), for a model rocket with a body cross-sectional area, \( S_{BT} \), in ft.²:

$ D_0 = C_{DO} \frac{1}{2} dV^2 S_{BT} $

(17)

Correct units are assured by measuring air density \( d \) in slugs and model rocket velocity in ft./sec. The zero lift drag coefficient, \( C_{DO} \), has no dimensions. We need only select the flight conditions, velocity and altitude since altitude determines air density, decide on a model size to establish \( S_{BT} \) and compute the model rocket drag from the \( C_{DO} \).

We can illustrate this procedure before getting into detailed calculations by rearranging Eq. (17) to a more simple form:

$ \frac{D_0}{S_{BT}} = 0.0827 \, C_{DO} $

(18)

This relationship assumes a sea level average value for \( d \) ( = 0.00238 slugs/ft.³) and a velocity of 100 ft./sec. to determine the drag of a model rocket in pounds for every square inch of body tube cross-sectional area. Thus a drag coefficient of one yields a drag of 0.0827 lbs. for every square inch of body tube cross-section. A formula such as Eq. (18) can be used to construct a chart like Fig. 51, showing the drag in pounds of a rocket per square inch of cross-section area for several flight speeds.

To use the chart, simply enter with the drag coefficient -- we have yet to find \( C_{DO} \) exactly, but let’s say

$ C_{DO} = 0.5 $

for a model rocket moving at 300 ft./sec.—then a value of

$ \frac{D_0}{S_B} = 0.371 $

is read from the chart at \( V = 300 \) ft./sec. If the rocket has a BT-50 body tube, then

$ D_0 = \frac{D_0}{S_{BT}} \times S_{BT} = 0.371 \times .785(0.976)^2 = 0.279 \text{ lbs} $

![FIG. 51: Zero Lift Drag of Model Rocket vs \( C_{DO} \) for Different Flight Speeds at Sea Level and at 5000 Feet Altitude](image-placeholder)

To obtain drag of model rocket in pounds for different body tubes, multiply chart value

- By 0.425 for BT-20
- By 0.748 for BT-50
- By 1.380 for BT-55
- By 2.105 for BT-60



in a similar fashion a BT-60 body tube, with a 1.637” diameter, has an area of 2.105 square inches. Therefore

$ D_0 = 0.371 \times 2.105 = 0.768 \text{ lbs.} $

The effect of altitude is also shown on the chart. The dashed line indicated the drag in pounds per square inch when the density is 0.002048 \(\frac{\text{slugs}}{\text{ft}^3}\), the value of at 5000 ft on what is termed a "Standard Day". At this altitude, the drag of the BT-50 model rocket drops to 0.242 lbs, a reduction to 86.1% of the sea level value. This reduction is exactly that of the density ratio, since

$ \frac{\rho_{5000}}{\rho_{\text{sea level}}} = \frac{0.002048}{0.00238} = 0.861 $

All we need to do to correct the drag for any altitude, then, is to multiply the sea level value of drag by the density ratio. A chart of "Standard Day" density at altitudes up to 20,000 feet is given in the Appendix for this purpose.

Incidentally, before passing on to a calculation of \( C_{DO} \) we should take note that Fig. 51 is a universal chart. It applies to full size rockets, as well as model rockets. Consider a full-scale rocket with a body diameter of 1.5 feet moving at sea level at 500 ft/sec with \( C_{DO} = 0.4 \). The drag of this bird at 500 ft/sec is a little less than 140 pounds. Even more impressive is the drag acting on a 33 foot diameter Saturn V climbing through 5,000 feet altitude at 500 ft/sec with \( C_{DO} = 0.4 \); the aerodynamic resistance of this giant is 88,200 pounds! You might take the time to check these numbers; then just for fun try to find the aerodynamic resistance of the family automobile, which has a drag coefficient near 0.6.

Let’s return to our model rocket, now, and attack the main problem - finding \( C_{DO} \). We have made real progress in our study so far; recall that in Chapter III we used Eq. (7) to write the \( C_{DO} \) in terms of the component drag coefficients:

$ C_{DO} = C_{DN} + C_{DBT} + C_{DB} + C_{DOF^*} + C_{Dint} + C_{DLL} $

(7)

We've been working in the past two chapters to estimate the first four of these drag coefficients. All we need to do now is to find the interference drag coefficient, \( C_{Dint} \) and the launch lug drag coefficient, \( C_{DLL} \), to be able to compute the zero lift drag.

In Chapter V we found the fin drag coefficient, \( C_{DF^*} \) which was based on fin area. We’ll have to correct that drag coefficient to be based on the same area as the other coefficients; that is, the body tube cross-sectional area. As we examine this correction, we will also find a method to estimate the interference drag. First, the correction to the proper area is made by Eq. (19):

$ C_{DOF} = C_{DOF^*} \frac{S_F}{S_{BT}} $

(19)

The value for \( C_{DOF^*} \) may be obtained from Eq. 14 or from Fig. 41 for the particular fin design being used. Then all that is required is to multiply by the ratio of fin areas using the reference formulas from Fig. 47.

The equations of Fig. 47 are based on wings; what about the fact that the body tube covers a portion of the fins? What do we do if we have three fins, not four? As shown in Fig. 52, we can subtract the area covered by the body tube to obtain equations similar to Eq. (20) for rectangular fins.

$ S_F = \frac{C_R}{2} [b - d] \times \text{Number of fins} $

(20)

![FIG. 52: Calculation of Exposed Fin Area](image-placeholder)

$ S_F = \frac{C_R}{2} [b - d] \times \text{No of Fins} $

$ \text{Since } \frac{1}{2} dV^2 \text{ is the same for both equations:} $

$ C_{DOF} S_F = C_{DOF^*} S_{BT} $
or
$ C_{DOF} = C_{DOF^*} \frac{S_F}{S_{BT}} $

*Remember we are finding the zero lift drag of the fins, therefore no induced drag is encountered. Further the drag is*

$ D_{OF} = C_{DOF^*} \frac{1}{2}dV^2S_F = C_{DOF^*} \frac{1}{2}dV^2S_{B} $



However, by a fortunate coincidence, one method used to estimate fin interference is to neglect the fact that the air does not flow past the portion of the fin area covered by the body tube. The increased surface area will result in a higher drag coefficient, but the increase has been shown to be about equal to the interference drag coefficient. In other words, we can find \( C_{Dint} \) from the equation:

$ C_{Dint} = C_{DOF^*} \frac{C_R}{S_{BT}} \frac{d}{2} \times \text{Number of fins} $

(21)

We can combine this expression with Eqs. (19) and (20) to come up with a simple relation for both \( C_{DOF} \) and \( C_{Dint} \):

$ C_{DOF} + C_{Dint} = C_{DOF^*} \frac{\text{Area} \times \text{Number of fins}}{S_{BT}} $

(22)

The area is found for any fin design by the relations shown in Fig. 47.

It is true that this is an estimate of interference drag. It certainly is a better method than simply multiplying the \( C_{DO} \) by some correction factor like 1.05 or 1.10 to guess at the effect of fin interference. Interference problems are difficult to analyze and improvements to ways to estimate interference effects are an important area for model rocketeers to examine.

### Launch Lug Drag

The last component necessary to find \( C_{DO} \) from Eq. (7) is the launch lug drag coefficient. \( C_{DLL} \) we’ll have to do this all with theory, since not too many aerodynamic tests have been conducted on soda straws and similar tubes. All we'll be able to do, then, is to establish the "order of magnitude" of the launch lug coefficient. The term "order of magnitude" is used by engineers when a precise answer cannot be obtained and it is necessary to know if the drag of a particular component will be an important factor. For example, will \( C_{DLL} \) be 1% of the total rocket drag or 50%? If \( C_{DLL} \) is just 1% of the total, then it doesn’t matter if we’re not too exact; if the \( C_{DLL} \) is 50%, then we’d better refine our analysis, since an error in \( C_{DLL} \) will certainly have a big influence on our \( C_{DO} \).

As we’ve done before, let’s think of the worst possible case in order to place an upper limit on the drag coefficient. If the launch lug were a solid disc standing at right angles to the flow, it would create the separated flow and high drag conditions shown in Fig. 53, Sketch (a). The drag coefficient for a disc at right angles to the flow is 1.2, based on the disc surface area. To be meaningful to us, we must convert this coefficient to the drag coefficient of the disc based on the body tube cross-sectional area that we’ve been using as a reference. Thus

$ C_{DLL_{MX}} = 1.2 \frac{S_{LL}}{S_{BT}} $

(23)

when we've added a subscript MX to indicate this is the maximum expected drag coefficient for the launch lug. For a launch lug that is 0.17” diameter (which gives an area of 0.0227 sq. in.) we may obtain the \( C_{DLL_{MX}} \) as the body tube diameter increases. On the other hand, with small body tube diameters the launch lug can be an appreciable fraction of the total rocket drag. (Looking back at Fig. 33 we find that the drag coefficient of the nose cone and body tube is less than 0.2 for a 1” diameter.)

Now let’s see if we can place a lower bound on the launch lug drag coefficient. In Sketch (b) of Fig. 53 we note the minimum possible drag would occur if we had made a ring by cutting an 0.15” diameter hole in the 0.17” flat disc. The drag would then be made up solely

![FIG. 53: Launch Lug Assumptions](image-placeholder)

1. **Launch Lug Assumed as Flat Solid Disc**

2. **Launch Lug Assumed as Flat Ring**

3. **Launch Lug with Length**

![FIG. 54: Estimate of Drag Coefficient of a Launch Lug](image-placeholder)



of pressure drag as before, but now the area of the ring is reduced to 0.00502 sq. in. Thus the minimum drag coefficient of the launch lug is given by Eq. (23) but with \( S_{LL} = 0.00502 \) instead of \( S_{LL} = 0.0228 \) as in the solid disc case. A curve of this minimum launch lug drag is also shown in Fig. 54.

We can now say that the launch lug drag coefficient lies somewhere between the two boundaries outlined on the figure. For a 1" diameter body tube, then, a \( C_{DLL} \) must be more than 0.008 and less than 0.035. That’s pretty good information, but can we improve upon it? Yes, since launch lugs are cylindrical, not discs, they will also have skin friction drag as shown in Sketch (c) of Fig. 53. We can make an estimate of this friction contribution and add it to the pressure drag for an improved value of \( C_{DLL}^* \).

We’ll assume a 1" long lug. The surface area is made up of the sum of the inner and outer surfaces:

$ S_{LLW} = \text{Surface Area} = \pi d_{\text{out}} \ell + \pi d_{\text{in}} \ell = \pi(.17)(1) + \pi(.15)(1) = 1.005 \text{ sq. in.} $

For simplicity, we’ll assume that the skin friction coefficient is constant over the range of speeds encountered by the launch lug, and we’ll let the value of \( C_f = 0.0045 \). Therefore, we can find a value for \( C_{DLL} \) from Eq. (24):

$ C_{DLL} = 1.2 \frac{S_{LL}}{S_{BT}} + 0.0045 S_{LLW} $

(24)

Results from Eq. (24) are presented in Fig. 54, along with another curve for a 2" long lug, included to indicate the effect of launch lug length upon the drag.

Figure 54, then, summarizes our theoretical treatment of launch lug drag. Perhaps you are concerned about a few fine points in this aerodynamic analysis -- for example, what about the fact that the launch lug is glued to a body tube which has a boundary layer growing on it? or the assumption that the friction drag is constant? or the influence of lug misalignment? These features certainly could alter the drag coefficient of the lug, but haven’t we done what we set out to do? We were looking for an “order of magnitude” for \( C_{DLL} \) and we have established reasonable bounds for its value. In the process, we’ve exercised our understanding of aerodynamic drag to determine the drag of a soda straw. (How many professional aerodynamicists have ever done that?) We’ll leave the refinements to this lug drag analysis to others; besides whenever good experimental data becomes available, we’ll be ready to employ that too.

### Combining Drag Coefficients

Let’s now examine the procedure to find the zero lift drag coefficient of a typical model rocket. A sport rocket, 1" in diameter, with four streamlined fins of \( AR = 2 \) is shown in Fig. 55. We’ll assume a flight speed of 100 ft/sec and find all the component drag coefficients for Eq. (7). To obtain the lower limit on the possible drag coefficient, we'll assume the boundary layer is laminar everywhere.

The first two coefficients are found from Fig. 28, which indicates

$ C_{DN} + C_{DBT} = 0.079 $

The base drag coefficient is then found from Fig. 30 to be \( C_{DB} = 0.104 \). For fins with a thickness to chord ratio \(\frac{t}{c} = 0.0397\), we find \( C_{DOF^*} = 0.010 \) from Fig. 41.

Now we’ll use Eq. 22 to determine the interference effect and \( C_{DOF} \) simultaneously:

$ C_{DOF} + C_{DINT} = C_{DOF^*} \frac{\text{Area} \times \text{No. of Fins}}{2 S_{BT}} $

$ = 0.010 \times \frac{3.15 \times 1.575}{2 \times .785} \times 4 $

$ = 0.1262^* $

With a launch lug drag coefficient for the two inch long lug bounded by the minimum value \( C_{DLL} = 0.02 \) and the maximum value of \( C_{DLL} = 0.035 \), let’s be conservative and try to account for any misalignments by picking a value of \( C_{DLL} = 0.035 \). Making these substitutions into Eq. (7):

$ C_{DO} = C_{DN} + C_{DBT} + C_{DB} + C_{DOF} + C_{DINT} + C_{DLL} $

$ = 0.079 + 0.104 + 0.1262 + 0.030 $

$ C_{DO} = 0.3392 $

![FIG. 55](image-placeholder)



Finally! We've just completed a calculation for the zero lift drag coefficient of a typical model rocket. This is what we've been trying to do since we started working on drag analysis. We have, of course, assumed a fully laminar boundary layer so this is the lowest possible drag coefficient and therefore the coefficient that we strive for.

What's the highest zero lift drag coefficient? This would be obtained if the boundary layer flow were everywhere turbulent. By repeating the procedure followed above, but using the turbulent value in all the charts we arrive at the following values:

- \( C_{DN} + C_{DBT} = 0.229 \)
- \( C_{DB} = 0.061 \)
- \( C_{DF} = 0.166 \)
- \( C_{DINT} = 0.0517 \)
- \( C_{DLL} = 0.030 \)

$ C_{DO} = 0.5377 $

Thus, a fully turbulent boundary layer will cause the drag coefficient to increase almost 60% above the laminar value — that’s certainly an important piece of information which we can use to improve rocket performance.

Other important information can be learned from all of this work if we go one step further. Let's point out these sources of drag in the next section.

## Analysis of Drag at Zero Lift

When we started our drag discussions we wanted to be able to identify the drag of each rocket component. In that way, we would be able to take corrective action to reduce the drag effect that each component we discovered. That’s why we’ve been so careful in our analysis to always define the drag coefficient of the fins, the body tube, the base, etc. We now have the opportunity to examine in detail the contributions of the basic components to the drag of the entire rocket.

The best way to do this is to consider an example. Since we’ve already started an analysis of the model shown in Fig. 55, let’s continue with this model at a flight speed of 100 ft./sec. To make the information easier to visualize, we’ll make use of bar charts to represent the contribution of each model rocket component. Figure 56 presents bar charts in two forms: the top set of bars gives the drag of each rocket component in coefficient form; the bottom set of bars gives the percentage contribution of each set.

Each of the five bars represents a different set of assumptions or techniques used to analyze our model rocket. For example, the first bar gives drag coefficients for the fully laminar boundary layer case examined in the last section. Directly below, the bottom bar shows the percentage of the total coefficient drag that the body, the base, the fins, the interference and the launch lug contribute to the zero lift value for \( C_{DO} = 0.339 \). The five bars, then, represent drag analyses based on the following five assumptions:

(i) a laminar boundary layer exists on all surfaces
(ii) a turbulent boundary layer exists on the body but the fins have a laminar layer
(iii) a turbulent boundary layer exists on all surfaces
(iv) a laminar boundary layer exists on the body, with unstreamlined, rectangular cross-section fins
(v) a turbulent boundary layer exists on the body, with unstreamlined, rectangular cross-section fins

Conditions (i) and (iii) have been worked out earlier, conditions (ii), (iv), and (v) are calculated in a similar manner.

![FIG. 56: Distribution of Drag Components for L/d = 12 Rocket with A.R. = 2 Fins at 100 ft./sec.](image-placeholder)

The bottom bar for condition (i), the all laminar case, indicates the rocket body contributes about 24% of the total drag, the base adds almost 30% and the fins about 25%; the remainder is due to launch lug and interference. Pretty interesting isn’t it? Each part contributes about the same amount of drag for the all laminar case. This bar chart shows how important base drag is. Maybe we should try to cut that part of the drag down if we can. In the next chapter we’ll take up drag reduction methods, so this type of information is particularly valuable.

What happens if we allow the body boundary layer to become turbulent? From the charts we note that the total drag coefficient increases to 0.446. This is more than a 30% increase over the previous value, all due to the extra drag of the body tube. As shown by the bottom bar, more than one half of the drag now comes from the



body tube. That's a dramatic way of showing how important it is to keep the boundary layer laminar for low drag.

If the boundary layer on the fins becomes turbulent, we get a further increase in drag. We’ve already shown \( C_{DO} = 0.538 \) for this condition (iii). The contribution of the fins to the total rises to near 28% from the 19% value of case (ii). Note that the percentage due to the turbulent body has dropped to 43%. Of course, the body drag coefficient is still the same (as shown in the upper bar graph), but the total drag coefficient, \( C_{DO} \) has become larger than the \( C_{DO} \) of condition (ii).

## Streamlining Effects

Another really interesting point is an illustration of the effect of streamlining the fins. Young model rocketeers often wonder if it’s worth taking the time to sand the fins to a proper airfoil shape; conditions (iv) and (v) show how much the drag is increased by leaving the front and rear edges of the fin flat. Because the pressure drag goes up so much, \( C_{DO} = 0.674 \), even with a laminar boundary layer on the rocket body. Compare this to \( C_{DO} = 0.339 \) when the fins are streamlined as in case (i). The rectangular fins account for 45% of the total drag - almost half the rocket drag is due to the fins! Finally, when the rocket body boundary layer is assumed to be turbulent, the drag coefficient increases to 0.781, so that the fin drag percentage drops under 40% while the body tube percentage increases to 29% of the total.

To sum up this part of our drag study, we see that the drag coefficients of the same basic model can vary from \( C_{DO} = 0.339 \) all the way up to \( C_{DO} = 0.781 \). That's quite a range of coefficients; how do we decide what the value is for our rocket?

Once we streamline the fins we know that the value will be somewhere between the fully laminar and fully turbulent case, so we may set these bounds. We can strive for the lowest value -- the all laminar case -- but a good reasonable drag coefficient will probably be greater. Without getting into some pretty complicated aerodynamics which will require prediction of boundary layer transition points, let’s say the drag of a streamliner (i), a turbulent boundary layer on the body but a laminar layer on the fins.

We must also remember that this drag coefficient is at zero lift (or zero angle to the wind) and for a constant velocity of 100 ft/sec. Since our analysis can examine both of these points, let’s consider next the effect of velocity on the distribution of drag coefficients and then the effect of angle of attack.

## Influence of Velocity

Most model rockets have three fins so, before considering the effect of velocity upon our rocket drag, let’s make a slight modification to the typical model rocket that we’ve been analyzing by changing it to a three-finned model. We’ll keep the same aspect ratio as before, with \( AR = 2 \), increase the size of each fin a little to 2 square inches. We’ll use the symbol \( S_{FS} \) to represent the area of a single fin. The three fins, therefore will total six square inches; we require about this much area to keep our model rocket stable. With a smaller area than previously used (total area, \( S_F = 6.75 \) sq. in. for the four finned rocket) we would expect the fin drag to be reduced. Don’t forget that the interference drag will also be cut, since only three fins will be causing flow disturbances.

At a rocket velocity of 100 ft/sec, we have already determined the drag of the nose cone and body tube, base and launch lug; only the fin and interference drag must be re-evaluated for the three fin model. The first task in the re-examination is to determine \( C_{DOF^*} \) which is shown in Fig. 41, depends upon the thickness ratio, \(\frac{t}{c}\). Because we’ve altered the surface area of each fin, the thickness ratio of our \( \frac{1}{16}" \) thick fins will be changed from the previous value. Referring back to Fig. 52, it is possible to perform a little algebraic manipulation to determine what the chord, \( C_R \), of the rectangular fin should be. In terms of the geometric properties of our fins which we know, \( S_{FS} \), \( AR \), and the body tube diameter, \( d \), the chord \( C_R \) turns out to be

$ C_R = \frac{1}{2} \left[ \frac{d}{AR} + \sqrt{\frac{d}{(AR)^2} + \frac{8S_{FS}}{AR}} \right] $

(25)

If you’ve had algebra and the quadratic formula, you might have some fun checking out this equation. Don’t worry if you haven’t; all we need do is substitute the numerical values into Eq. 25 to find the chord for our particular model rocket. When we do this we obtain

$ C_R = \frac{1}{2} \left[ \frac{1}{2} + \sqrt{\frac{1}{2} + \frac{8 \times 2}{2}} \right] = 1.687 \text{ inches} $

The thickness ratio for our fins becomes

$ \frac{t}{c} = \frac{0.0625}{1.687} = 0.037 $

According to Fig. 41, this thickness ratio results in

$ C_{DOF^*} = 0.0096 $

when the boundary layer is laminar and the velocity is 100 ft/sec. The fin and interference drag coefficient is determined from this value by Eq. 19 and Eq. 21:

$ C_{DOF} = C_{DOF^*} \frac{S_F}{S_{BT}} = 0.0096 \times \frac{6}{0.785} = 0.0734 $

$ C_{DINT} = C_{DOF} \frac{C_R}{S_{BT}} \times \frac{d}{2} \times 3 = 0.0096 \times \frac{1.687}{0.785} \times \frac{1}{2} \times 3 = 0.0309 $

Looking back to the four-finned case, we find \( C_{DOF} = 0.086 \) and \( C_{DINT} = 0.0402 \). By using three fins, then, we cut the sum of the fin and interference drag coefficients by 20%. Adding the contributions of the rest of the model rocket, the total zero lift drag coefficient becomes \( C_{DO} = 0.3173 \). Compared to the four-finned value obtained earlier, \( C_{DO} = 0.339 \), the zero lift drag at 100 ft/sec has been reduced about 7%.

To continue our examination of velocity effects we must repeat the calculations at other rocket flight speeds. This repetition is not difficult because we’ve been very careful to plot velocity on all our design charts. If you look at Fig. 32, you’ll find the sum of



the drag coefficients of the nose cone, body tube, and base — termed \( C_{DOB} \) — for both laminar and turbulent boundary layers plotted against rocket body length to diameter ratio for three specific speeds: \( V = 100, 300, \) and 500 ft/sec. All we have to do is select a rocket length to diameter ratio (we’ve been using \( L/d = 12 \), remember) and read off the value of \( C_{DOB} \) for the desired flight speed and boundary layer condition. Then we can use Fig. 41, as we’ve just illustrated, to find \( C_{DOF} \) which in turn determines the fin and interference drag coefficients. Adding these components along with the launch lug drag coefficient gives the zero lift drag coefficient, \( C_{DO} \).

Applying this procedure to the typical model rocket we’ve been using as an example, we obtain the drag coefficient distributions shown in Fig. 57. Once again, in order to place limits on the model rocket drag (and for simplicity), we’ve considered pure laminar flow to establish the lowest possible drag and entirely turbulent flow to find the maximum drag coefficient. With curves like these it’s easy to see the large effects of base drag for the laminar case, and how the much higher turbulent boundary layer increases the drag on the fins and body tube.

![FIG. 57: Distribution of Drag Coefficients vs Flight Velocity](image-placeholder)

## Boundary Layer

Perhaps now is the time to consider the boundary layer on our model rockets; we might be able to improve our drag predictions this way. In Chapter II we learned that boundary layers always start off smoothly with the laminar velocity profile existing over a surface, then after the air flows along the surface for a while the boundary layer “transitions” to the turbulent case. We found the proper index to describe the behavior of the boundary layer was not simply the speed of the air over the surface, or the length of the surface, but a multiplication of velocity and length combined with the density and viscosity of the air. This combination was termed the Reynolds number, and defined by Eq. 4.

A general rule which helps us determine what type of boundary layer exists on a surface — that is, a body tube or fin — is that laminar flow will always exist where RN is less than 100,000 and turbulent flow will always exist when RN is greater than 1,000,000. The question is, thus, what Reynolds numbers do we have on our model rockets? For our model rocket with a one foot long body tube, flying at 100 ft/sec at sea level, RN = 610,000; at 300 ft/sec, RN = 1,830,000, and at 500 ft/sec, RN = 3,050,000. Therefore, at these flight speeds, and certainly below 50 ft/sec, laminar flow will exist, but as the speed increases, a turbulent boundary layer will cover the body tube.

It’s important to note that the boundary layer does not become turbulent everywhere at one time; in fact, a little laminar boundary layer exists on all surfaces, even at very high Reynolds number. The turbulent layer begins at the rear of the tube and gradually spreads forward as speed increases. For our particular size model flying at 300 ft/sec, more than half of the body will have turbulent flow — with a good estimate being the region from the nose cone to the fin will transition to turbulent. The exact proportion of laminar and turbulent flow depends upon surface finish and nose shape and is no doubt, however, an important function of speed, just about all the surface will be turbulent.

What about the fin boundary layer? The fins project into the air stream in a manner that the air flows over them, but for a lot shorter distance than the body tube. We should use the fin chord, (1.687 inches) to find the appropriate Reynolds number for the fins. When we do this for the three flight velocities of 100, 300 and 500 ft/sec, the corresponding Reynolds numbers are RN = 86,000, 258,000 and 430,000. Based on our transition criteria, then, the fin boundary layer will certainly be laminar at 100 ft/sec and most probably be laminar at 300 ft/sec. In fact, there is a good chance to have a laminar fin boundary layer at 500 ft/sec if the fin is polished and has a streamline shape.

We can summarize this brief discussion of model rocket boundary layers with a few comments. At low speeds, say less than 100 ft/sec, the fully laminar curves of Fig. 57 should be used. At higher speeds, on the order of 300 ft/sec, portions of the body tube boundary layer will become turbulent, increasing the drag above the value predicted by the laminar curves. The drag will not be as great as that shown by the fully turbulent curves, since the fins and the front portion of body tube will remain laminar. At the highest speeds, that is, 500 ft/sec and above, the fully turbulent curves should be used.

The above observations suggest a simple modification to our drag prediction technique. Let us use fully turbulent flow for the body tube, but assume fully laminar flow for the fins. It is a simple matter to go back through our analyses and make these changes; when we do we obtain the curves of Fig. 58. The zero lift drag coefficient decreases from 0.424 at 100 ft/sec to 0.346 at 300 ft/sec and to 0.319 at 500 ft/sec. These values should be reasonable coefficients in the flight range of our model rockets; that is, between 100 and 400 ft/sec.



![FIG. 58: Distribution of Drag Coefficients vs Flight Velocity](image-placeholder)

## Drag Values

Before leaving this section on the influence of velocity, we have a few more points to make. We should look at the actual drag values in pounds or grams for our rocket for two reasons; to emphasize that although the drag coefficient decreases with velocity, the actual drag increases, and to find what level of drag is attained by a model rocket. All the difficult work has been done, so let us tabulate the values of \( C_{DO} \) and Drag for the three basic methods used to predict drag - fully laminar, fully turbulent, and the preferred method of turbulent body tube but laminar fins.

A more graphic display of this drag data is shown in Fig. 59. The rapid increase in drag that occurs as the rocket speed increases is clear. The dotted and dashed lines represent the fully turbulent and fully laminar boundary layer conditions to illustrate the maximum and minimum levels of drag for our typical model. The solid line represents the best prediction we can make about the drag, without going into a lengthy and complex analysis of the transition between laminar and turbulent flow.

This figure also points out the very small values of drag at low rocket flight speeds. Even at a velocity of 200 ft/sec the drag has reached no more than 0.115 lbs.

### Drag Summary for Typical Model Rocket

| Analysis                 | \( C_{DO} \) |  Drag in lbs.  | Drag in Gms  |
|--------------------------|--------------|----------------|--------------|
|                          | V = 100  | 300  | 500          | V = 100 | 300  | 500 |
| Fully Laminar            | .317    | .284 | .267         | .021  | .166 | .433 |
|                          |          |          |             | 9.3   | 75.2 | 196 |
| Fully Turbulent          | .500    | .421 | .396         | .032  | .246 | .642 |
|                          |          |          |             | 14.7 | 111.2 | 291 |
| Turbulent Body, Laminar Fins | .424 | .346 | .319         | .028  | .208 | .517 |
|                          |          |          |             | 12.5 | 94.4 | 234 |

![FIG. 59: Drag of Model Rocket vs Velocity](image-placeholder)

## Terminal Velocity

Ever wonder how fast a model rocket will come down if the nose cone doesn’t separate from the body tube? That information is contained in Fig. 59 also. After peak altitude, gravity pulls the model back to earth and



a recovery device, parachute or streamer, is usually deployed to lower the rocket safely to the ground. But suppose the recovery device does not deploy, or worse yet, the nose cone remains attached to the body tube and the rocket remains streamlined? The model will accelerate rapidly to high speed; as it does so, the aerodynamic resistance will increase, as shown in Fig. 59. This air resistance, drag, is now acting to “hold the model back”—to prevent it from going faster. When the aerodynamic drag builds up to equal the weight of the rocket, the gradual pull is exactly balanced by the air resistance and the rocket can no longer pick up speed. It must fall at a constant, maximum or “terminal” velocity.

Suppose our typical model, weighing 0.1 lbs., falls without a recovery device. When the terminal velocity is reached the drag will equal the weight, 0.1 lbs. By reading across the vertical scale of Fig. 59 at a drag of 0.1 lbs. to the dotted curve, then looking down to the velocity on the horizontal scale, the terminal velocity of 185 ft/sec is found. As you remember, the dotted curve represented the highest drag case. If we move further across to the dashed line representing laminar flow and the lowest drag, we find the terminal velocity is 225 ft/sec. It is obvious that low drag bodies will fall faster than high drag bodies of equivalent weight. In fact, the parachute and streamer are really devices to create high drag to produce a low terminal velocity.

The values quoted for terminal velocity were taken from Fig. 59, which was constructed from the drag coefficients of a typical model rocket. To be more general, since all bodies falling through the air will have a maximum velocity, we need a mathematical relation. All we need do is equate the weight of the body to the drag and then solve for the terminal velocity, \( V_T \), as shown below:

$ W = D = C_{DO} \frac{1}{2} dV_T^2 S_B $

$ V_T = \sqrt{\frac{2W}{C_{DO} d S_B}} $

(26)

Equation (26) is a general statement relating the terminal velocity to the weight, \( W \), of the body, its area, \( S_B \), and the zero lift drag coefficient, \( C_{DO} \). If we know numerical values for these three items and the air density, \( d \), it’s a simple matter to calculate the terminal velocity.

It’s interesting to note that the effect of decreasing air density is to raise the terminal velocity. The terminal velocity of a body is higher at altitude, therefore, than at sea level. In other words, a sky-diver leaving an airplane at a great height will slow down as he nears the ground! We can tell how much, now that we have Eq. 25. A 5’9" sky-diver weighing 165 pounds has a drag coefficient of \( C_D = 1.3 \) and a projected area of 7 ft² when he falls in a horizontal position. His terminal velocity at sea level, where \( d = 0.00238 \) slugs/ft³, will be 124 ft/sec, but at 20,000 feet, where \( p = 0.00172 \) slugs/ft³ his terminal velocity is 170 ft/sec. If the skydiver falls feet-first, his area is decreased to 1 ft² so \( V_T \) goes way up to 327 ft/sec. When his 28 foot diameter parachute opens, the drag coefficient becomes 1.2; you might check to find his descent rate as he lands (it's 17.1 ft/sec).

Let’s now move on to the last segment of this long chapter on model rocket drag and examine, briefly, the effects of angle of attack upon rocket drag.

## Influence of Angle of Attack

Although the complete calculation of the drag of a model rocket at angle of attack can be quite complicated, a good portion of the work has been accomplished in the preceding sections. Figure 34, for example, shows the drag coefficients for both laminar and turbulent boundary layers on typical model rocket bodies at angles of attack up to 10° (positive or negative). Further, Fig. 50 has indicated how the drag coefficient of a fin can change with angle of attack. Let’s examine the drag variation with angle of attack for one specific case - say when the rocket flight velocity is 300 ft/sec. We’ll assume that the boundary layer on the body tube is turbulent and that on the fins the boundary layer is laminar; we’ll also consider the launch lug drag coefficient to be constant with angle of attack.

Under these conditions, the results shown in Fig. 34 are directly applicable. The fin drag variant must be worked out again, because none of the fin configurations shown in Fig. 50 is applicable to the model we’ve been considering - that is, a rectangular fin planform with a streamline cross-section. That’s all right, though; in Chapter V we derived equations to allow a rapid computation of the fin drag coefficient at angle of attack. Now we’ll be able to review their use.

The fin drag coefficient, as expressed in Eq. 16, is the sum of the zero lift drag coefficient, \( C_{DOF}^* \), and the induced drag coefficient, \( C_{DiF}^* \). Equation 15 expresses \( C_{DiF}^* \) in terms of the lift coefficient; i.e.

$ C_{DiF}^* = \frac{C_L \times C_L}{\pi AR e_w} $

(15)

For our present purpose, a more convenient form of the induced drag coefficient is obtained by replacing the lift coefficient by the product of the lift coefficient per degree of angle of attack multiplied by the angle of attack. Making this change to Eq. 15, we arrive at the following equation:

$ C_{DiF}^* = \left(\frac{C_L}{\text{degree}}\right)^2 \times \alpha^2 \frac{1}{\pi AR e_w} $

(27)

*Equation (26) can be used to find descent rate of a model rocket being lowered by a parachute as well as the terminal velocity of a streamlined model rocket. Model rocket parachutes have \( C_{DO} \) values between 1.0 and 1.2; using this value for our 0.1 pound model equipped with a 16" diameter parachute we may find the descent rate (terminal velocity) at sea level to be:

$ V_T = \sqrt{\frac{2 \times 0.1}{1 \times 0.00238 \times 1.394}} = 60.2 = 7.8 \text{ ft/sec} $

where \( S_B = .785 (d^2) = .785 \times 16^2 = 201 \text{ in}^2 = 1.394 \text{ ft}^2 \) (remember to convert square inches to square feet!).




When we were studying the effects of planform on the ability of a wing to produce lift, we plotted \(\frac{C_L}{\text{degree}}\) for various wing shapes and aspect ratios in Fig. 46. For a rectangular planform with \( AR = 2 \), Fig. 46 gives \(\frac{C_L}{\text{degree}} = 0.046\). During that same study, we indicated the effect of planform on the wing efficiency factor, \( e_w \), in Fig. 44. From this figure, we find that the \( AR = 2 \) rectangular wing has an \( e_w = 0.75 \). Using these numerical values in Eq. (27), the induced drag coefficient becomes

$ C_{DiF}^* = \left(\frac{.046}{2}\right)^2 \times \alpha^2 \times \frac{1}{\pi \times 2 \times 0.75} = 4.49 \, \alpha^2 \times 10^{-4} $

The total fin drag coefficient is now obtained by adding the zero lift drag coefficient for the fin. For a thickness ratio of 0.037, Fig. 41 shows \( C_{DOF}^* = 0.0056 \) when \( V = 300 \) ft/sec. The fin drag coefficient, based on the body tube cross-section area, \( S_{BT} \), may be written finally, as below.

$ C_{DF} = [0.0056 + 4.49 \, \alpha^2 \times 10^{-4}] \frac{S_F}{S_{BT}} $

To find the total drag coefficient at each angle of attack, all we need do is to add \( C_{DB} \) from Fig. 34, \( C_{DF} \) from the relation above, \( C_{DINT} \) from Eq. 20 and \( C_{DLL} \). Performing this addition, we obtain the distribution of the drag coefficient shown in Fig. 60 for angles of attack at 300 ft/sec and for \( \pm 10^\circ \).

It is observed that conditions when the model rocket is at zero angle of attack; all other angles create higher drag coefficients. It is clear from Fig. 60 that the increase in \( C_D \) is due mainly to the induced drag caused by the fin trying to return the rocket to zero angle. At zero angle, for example, the fin and fin interference account for less than 20% of total drag coefficient; however, at a 5° angle of attack, these two factors contribute more than 33% of the total. When \( \alpha \) reaches 10°, fin and interference make up more than 50% of the total value of the coefficient, \( C_D = 0.78 \). Note that at this high angle, the drag coefficient has increased by a factor of 2 above the zero angle case. This is the reason that stable rockets attain greater altitudes than marginally stable birds—they fly closer to the zero lift, minimum drag coefficient case. Similarly, the same model flying under no-wind conditions will achieve higher altitudes than when flying under windy conditions—the wind causes an angle of attack that increases the drag (as well as causing the rocket to weathercock).

Figure 60 also summarizes our theoretical discussion on model rocket drag. We’ve completed our goal set out in Chapter III- to examine each drag producing component of a model rocket and to estimate its contribution to the total rocket drag. To be complete, our study of model rocket drag must take a further step and use this theoretical information to guide us in practical ways to reduce model rocket drag. We’ve made note of some of these methods in the past chapters—for example we know that using higher aspect ratio fins on the typical model of Fig. 60 would lower the fin induced drag and cut the drag at angle of attack. Let’s use the next chapter to examine these drag reduction techniques.

![FIG. 60: Effect of Angle of Attack on Drag Distribution for L/d = 12 Rocket with AR = 2 Fins at 300 ft./sec.](image-placeholder)

## VII. Drag Reduction Techniques

### Workmanship

Up to this point, we’ve learned that model rocket drag is felt in three forms: through an unbalance of pressure, through friction of the air over the model surface and as a penalty for producing the lift required to make our rockets stable. Our drag reduction procedure could quite naturally be to review these forms of drag and try to minimize each type of drag for a particular model. However, we have already spent considerable effort identifying the amount of drag caused by the different components of the rocket -- in fact, we’ve already found good nose cone shapes and preferred fin designs in Chapters IV and V — so we don’t have to repeat these observations now. Instead, we can concentrate on the other reducing features. For example, we’ve determined that the base of the rocket contributes a good portion of the total rocket drag. Is there any way to lower this base drag? Similarly, are there any steps we can take to reduce the interference drag? And what about construction features like fin misalignment and lack of air finish -- how important are these items? These are some very practical questions. If we can find answers to them we’ll have real help in our task of designing low drag model rockets.

Those last two questions above could be put into a broad category called model construction, or better yet, workmanship. Good workmanship is so important for low drag models that we’ll consider it first. It is no exag-



![FIG. 61: Fin Workmanship](image-placeholder)

**a) Amateur**

- Rough cut leading & trailing edges
- Unsanded unpainted surface

**b) Professional**

- Airfoil shape sanded into fin. Painted
- Clean sharp trailing edge
- Rounded leading edge

It is no exaggeration to state that a good looking bird will be a good performing model. But how do you judge the workmanship on a model? That’s not difficult; there are certain clues which tell how good a builder a model rocketeer is.

Pick up any model rocket. Look first at the fins. Are the edges cut clean or are they ragged looking? Sight along the fin from the tip to the root - is there an airfoil sanded into the fin? If there is, the leading edge should be rounded and the trailing edge nice and sharp as in Figure 61. We've already shown that "airfoiling" the fins makes a considerable difference in fin drag, but don’t forget that by sanding the fins you can reduce the weight of the fins by 50% and weight is also important in rocket performance.

Next, look down the front of the rocket. Are the fins aligned accurately? As shown in Figure 62, the fins should be spaced about the body tube in even increments (that is, exactly 120° apart for 3 finned models); when viewed from the side, the fins must also be lined up with the body tube centerline. Any misalignment will cause the model to spin during boost, decreasing the potential maximum altitude. The spiral ascent is caused by the air flowing past the misaligned fin at an angle, producing fin "lift" (even though the rocket body is at zero angle of attack as shown in Figure 63). The offset lift from this fin, in turn, causes the model to roll about the body axis on its way up. We learned that any time lift is generated, drag flight, therefore, is obtained when all the fins are lined up and the model ascends with no rotation. Make sure any model you build has perfectly aligned fins — that shows good workmanship and gives maximum performance.

![FIG. 62: Fin Misalignment](image-placeholder)

- **Note**: Flat surfaces of fins are visible because fins are cocked
- Angles are uneven
- Misalignment

**Front View - Misaligned Fins**

**Side View - Perfect Alignment**

The junction between the nose cone and body tube provides another clue to the quality of construction. The junction should be matched exactly as illustrated in Figure 64. To test the model under examination, run your finger along the model from the nose toward the fins. You'll hardly be able to feel the junction if a good match has been made. But, if you feel a step up or step down, you’ll know the workmanship can be improved. When the nose cone is too small, the step up at the body tube increases the pressure drag, forces the boundary layer to become turbulent and can cause flow separation if the step is too large. When the nose cone is too large, the step down at the body tube separates the flow, destroying the desired smooth airflow. In addition, the larger nose cone will have an extra amount of surface area, and hence, skin friction drag. It should be clear that the minimum drag occurs when the junction is perfectly matched. Once again good workmanship pays off.



![FIG. 63: Effect of Misaligned Fin](image-placeholder)

- **Oncoming Air Stream**
- **Lift from Fin Causes Roll** (Front View)
- **Lift from Fin**
- **Induced Drag** (Side View)

![FIG. 64: Airflow Past Nose Cone - Body Tube Junction](image-placeholder)

- Perfect match at nose cone - body tube junction for smooth airflow with minimum drag
- Nose cone too small - disrupted flow at shoulder, turbulent and possibly separated flow after junction
- Nose cone too large - flow separation at shoulder, excessive friction drag

Even little assembly details like launch lug attachment affect model performance. Check lug alignment by sighting down the front of the model and viewing from the side; the launch lug must be aligned with the body tube centerline and fastened near the model rocket center of gravity. If not attached properly, the model’s performance will suffer in two ways. First, as shown in the right sketch of Figure 65, the model can leave the launch rail at an angle of attack. In the last chapter, we noted that any model at an angle of attack will have higher drag than a model at zero angle. Second, even after the model has straightened out - as a stable rocket will do — the misaligned launch lug can cause the smooth airflow on the rocket body to separate in the manner shown in the left sketch of Figure 65. Flow separation, as always, will cause excessive drag. Details such as launch lug alignment must not be neglected in any drag reduction effort.

When you look at a model rocket, your first impression of the work that has gone into the bird comes from the kind of finish on the model. If there are no painted surfaces, or a heavy, sloppy paint job, you can bet the builder didn’t care too much about workmanship. On the other hand, a neatly painted model shows the modeler took pride in his work, making the model more attractive, easier to track, and a better performer. Although the surface finish is the last of the keys to judging workmanship that we will consider in this section, surface finish is by no means the least important. Quite the contrary, the surface finish has a very strong role in the drag reduction procedure.

![FIG. 65: Penalties for Launch Lug Misalignment](image-placeholder)

- **Angle of Attack at Launch Lug Misaligned**
- **Disturbed Flow about Misaligned Launch Lug Causes Excess Drag** (Shown Exaggerated for Clarity)

The surface finish affects the drag of our model rockets in two ways — first, it determines the point where the boundary layer makes the transition to turbulent flow and, second, finish determines the actual level of the drag due to the turbulent layer.

At the nose of the model, the boundary layer is always laminar. As the flow develops rearward, the tendency to move toward the turbulent layer increases. By keeping a smooth surface, the actual transition can be delayed considerably for the flight conditions encountered by our model rockets. With a glass-smooth surface, it is possible to maintain most of the flow laminar, thereby approaching the theoretical lower limit for the all-laminar \( C_{DO} \) that we set in the previous



chapter. If the surface of the model is unpainted and rough, an early transition to the turbulent layer will be guaranteed, and the high drag associated with this type of flow will exist on the model.

The surface finish also affects the level of this turbulent drag. All the calculations we made in the previous chapters were for smooth models in turbulent flow. We anticipated that all good modelers would take care to finish their rockets smoothly. However, if the model surface is rough, say it has little lumps and other roughness of wood grain which average 0.002 inches high, the drag coefficient will be 25% above the turbulent value for the smooth surface. Figure 66 illustrates these points on surface finish vividly. The figure shows that, for our typical model, the drag can be reduced by 25%, simply by finish alone! This increment would occur at 300 ft./sec. — at higher speeds the drag increase above the smooth surface is even greater, and at lower speeds, less.

![FIG. 66: Effect of Surface Finish on Model Rocket Drag](image-placeholder)

- **Boundary Layer Transition to Turbulent Flow Delayed by Smooth Finish**

- **Smooth Finish:**
  - Turbulent B.L. Over Smooth Surface
  - Model Boundary Layer Almost All Laminar So \( C_{DO} \) Approaches 0.28

- **Rough Finish:**
  - Early Boundary Layer Transition Due to Roughness
  - Increased Skin Friction Due to Surface Roughness
  - Boundary Layer on Body Tube All Turbulent So \( C_{DO} \) Greater Than 0.4
  - If Fins Also Turbulent, \( C_{DO} \) Greater Than 0.5

Clearly, the surface of a model should be as smooth as practical. A coat or two of filler on the nose cone and fins, then a coat or two of colored paint followed by a waxing with a soft rubbing will provide the kind of surface that will give long drag performance with a minimum weight penalty.

Attention to these concepts during construction — smooth nose cone, body tube junction, launch lug alignment, airfoiled and aligned fins, neat, lightweight finish — are evidence of quality workmanship. The flight characteristics of any model rocket will be improved by employing care as you build the model. Just try it and see.

## Boat-Tailing

Now that we have emphasized the importance of workmanship and surface finish for low drag model rockets, let’s return to one of the first questions we asked in this chapter. What can we do about the base drag of our rockets? When we discussed the contributions of the various rocket components to total drag, we learned that the blunt rocket base could account for as much as 30% of the drag of the model. With that much drag, the base region is a good target in any drag “clean-up” (as engineers call drag reduction techniques). Of course, we have to accept a certain amount of base drag—our rockets must have a blunt base, usually 0.7 inches in diameter, since that’s the diameter of most of our model rocket engines. In fact, if the rocket we are designing has a BT-20 body tube, there may not be too much we can do about base drag, as the rocket motor body tube diameter—a parachute duration bird that uses a BT-50 body tube to allow a large parachute obviously should use a boat-tail if we expect much reduction in base drag.

Base drag can be reduced by “boat-tailing” the rear of the rocket. As shown in Fig. 67, a “boat-tail” is simply a reduction of the diameter of the rocket from the size of the body tube to the engine diameter. Two common boat-tail shapes are shown in the figure; these are termed ogive and conical. The ogive boat-tail is a little more efficient, but the conical boat-tail is usually used because it is much easier to build. Fig. 67 illustrates other terms that help to describe boat-tails; these are: the ratio of base diameter to maximum body diameter, \(\frac{d_b}{d}\); the length of the boat-tail, \(x_{bt}\); and the boat-tail angle for conical boat-tails \(\Theta\).

![FIG. 67: Two Boat-Tail Designs](image-placeholder)

- **Ogive**
- **Conical**

  - \(d / d_b / x_{bt}\)



The flow patterns about three rocket bodies with different base designs are shown in Fig. 68. The flat-based first design has a large wake; the second sharp boat-tail design has a wake that differs a little from the first, but the third rocket body with the shallow boat-tail has a considerably different flow pattern. The air flows smoothly along the gentle boat-tail, reducing the size of the wake.

![FIG. 68: Air Flow Patterns About Three Base Designs](image-placeholder)

- **Blunt Base**
- **Sharp Boat-Tail (\(\Theta = 20^\circ\))**
- **Gentle Boat-Tail (\(\Theta = 5^\circ\))**

Remember that one of the first things we learned in our drag study was that the size of the wake was directly related to the pressure drag acting on any shape. What we are really doing when we reduce the size of the wake by a boat-tail is lowering the pressure drag of our model rockets.

The second rocket with the \(20^\circ\) boat-tail angle is shown to make a special point that the boat-tail angle, \(\Theta\), must be gentle for the boat-tail to be effective. Any time \(\Theta\) is greater than \(5^\circ\), the air flow will have difficulty following the boat-tail contour. Because of the viscosity of the air, sudden changes of direction cannot be made by the air flow and the flow separates from the rocket surface. When the flow separates, the size of the wake is increased and the boat-tail loses its drag reducing ability. You might look back to Fig. 8 of Chapter I where two egg-loft designs were shown that exhibited similar flow behavior at the transition section. For low drag designs, flow separation must be prevented on boat-tails and on transition sections by using gentle curves and shallow angles.

When we boat-tail a model we lower base drag because of three different, beneficial, aerodynamic effects, as illustrated in Fig. 69. First, the actual base area is reduced so that the low base pressure acts on a smaller surface; second, the base pressure with the boat-tail is higher than the base pressure without the boat-tail; and third, the pressure on the boat-tail surface has a component that acts like a thrust to oppose the rocket drag.*

![FIG. 69: How a Boat-Tail Reduces Pressure Drag](image-placeholder)

- **Blunt Base**
  - Note that the length of arrows indicates the magnitude of the pressure acting on model.
  - Outline of Pressure Field

- **Conical Boat-Tail**
  - Low Base Pressure on Large Area
  - Pressure Increases on Boat-Tail Surface
  - Higher Base Pressure on Smaller Area
  - See how pressure on a sloped surface pushes forward!

*This last effect is familiar to you. Just think what happens when you squeeze an orange seed between your thumb and first finger. The pressure exerted by your fingers on the sloped sides of the orange seed shoots the seed forward. That’s just what happens on the boat-tail surface except, of course, the air pressure isn’t as great as the pressure of your fingers so the thrust isn’t large enough to propel the rocket. It is enough, though, to cause a significant reduction in the base drag as shown above.



Now that we understand how the boat-tail helps, the next step is to find how much the drag can be reduced by using boat-tails on our models. An illustration of the percentage reduction of the base drag that is possible is shown in Fig. 70. The horizontal axis indicates the ratio of the base diameter divided by the maximum body diameter of the rocket, \(\frac{d_b}{d}\). That means when there is no boat-tail, \(\frac{d_b}{d} = 1\), and when we boat-tail the all the way to a point (that is, we have no blunt base) \(\frac{d_b}{d} = 0\). Between these limits, the percentage of the no boat-tail base drag is shown in the vertical axis, running from 100% when we have no boat-tail \((\frac{d_b}{d} = 1)\) down to 0% when we have a complete boat-tail \((\frac{d_b}{d} = 0)\). An important feature of this figure is that even at moderate boat-tail values of \(\frac{d_b}{d}\), we obtain sizable reductions in the base drag.

For example, simply by boat-tailing from a BT-50 to a BT-20 we can cut the base drag to 43% of the no boat-tail value. That’s a worthwhile decrease! You can check this result by noting

$$
\frac{d_b}{d} = 0.736 \text{ inch (for BT-20)} = 0.753
$$

then looking up from this value on the horizontal axis of Fig. 70 to the curve.

![FIG. 70: Effect of Boat-Tail on Base Drag](image-placeholder)

For convenience, a special table for boat-tailing has been made up. The table below gives the base drag reduction when boat-tailing a model from several standard body tubes sizes to the BT-20 and BT-50 tubes commonly used to hold model rocket engines. Included in this table is the minimum length of the boat-tail, \(X_{bt}\). This dimension is given to make sure any boat-tail you construct is gentle enough to be effective. The point about the shallow angle for proper boat-tails cannot be stressed too much. In fact, the shallow angles and the \(X_{bt}\) values shown in the table can be used for transition pieces for egg-lofters and other payload models as well to avoid flow separation. That means to reduce the base drag to 9% value shown in the table for the boat-tail (or transition section) made by going from a BT-60 to a BT-20 body tube, for example, the boat-tail length must be at least 5.15 inches long.

### Table for Boat-Tail Design for BT-20 Base Diameter

| From Body Tube | \(\frac{d_b}{d}\) | % No Boat-Tail Base Drag | Minimum Length \(X_{bt}\) |
|----------------|--------------------|--------------------------|--------------------------|
| BT-50          | 0.753              | 43                       | 1.38                     |
| BT-55          | 0.556              | 17                       | 3.37                     |
| BT-60          | 0.450              | 9                        | 5.15                     |
| BT-70          | 0.332              | 4                        | 8.49                     |

### For BT-50 Base Diameter

| From Body Tube | \(\frac{d_b}{d}\) | % No Boat-Tail Base Drag | Minimum Length \(X_{bt}\) |
|----------------|--------------------|--------------------------|--------------------------|
| BT-55          | 0.737              | 40                       | 2.00                     |
| BT-60          | 0.597              | 21                       | 3.80                     |
| BT-70          | 0.491              | 9                        | 7.10                     |

For other body combinations use \( X_{bt} = 11.43 - d_b \).

The table and Fig. 70 are based upon aerodynamic experiments with model rockets. These investigations were performed to find a generalized expression for the effect of the base to body tube diameter. In words we can write:

**Base Drag with Boat-Tail = Base Drag Without Boat-Tail x \(\left(\frac{d_b}{d}\right)^3\)**

This information can be used to modify the base drag equation we used earlier [ Eq (9) ] for the unboat-tailed model, to obtain a precise and useable equation for the base drag coefficient of the boat-tailed rocket, \( C_{DBbt} \):

$$
C_{DBbt} = \frac{0.029}{\sqrt{C_{DN} + C_{DBT}}} \left(\frac{d_b}{d}\right)^3
$$

(28)

It's now possible to go back through our drag analysis and incorporate this new expression to find how the total rocket drag is reduced by the boat-tail. We simply replace \( C_{DB} \) with \( C_{DBbt} \) from Eq. (28). As an example, we can replace Fig. 32, the zero lift drag coefficient of unboat-tailed rocket bodies as a function of length to diameter ratio, with Fig. 71, which shows \( C_{DO}\) for boat-tailed rocket bodies. It's clear from this figure that we get a good reduction in \( C_{DO} \) with a moderate amount of boat-tailing—say \( d_b = 0.7 \). Another feature is that the laminar boundary layer case is affected more by boat-tailing than the turbulent case. That’s because the base drag makes up a greater portion of the total rocket body drag for the laminar case and the base drag is the factor on which we're working.



![FIG. 71: Effect of Boat-Tail on Zero Lift Rocket Body Drag Coefficient](image-placeholder)

Further calculations of the boat-tail effect are left to the reader. We’ll close this section with the observation that boat-tailing has proved to be a very effective and simple technique for base drag reduction for both full scale missiles and model rockets. Anytime the required body diameter of a model is greater than the engine diameter, boat-tailing should be employed for top performance.

## Interference Drag

The last item in our discussion of drag reduction is interference drag. As we noted in Chapter III, this drag is the result of the air passing around the body tube interacting in an unfavorable manner with the air flowing over the fins. The interference between these two airflows causes additional drag that makes the resistance of the body and fins, when joined together, greater than the drag of the two parts taken separately. We called this drag increment "interference drag" and used the symbol, \( C_{Dint} \) to represent this drag in coefficient form. We learned earlier that this interference drag can account for as much as 10% of the total rocket drag. While this is not as large a contribution as the base drag, interference drag must, nevertheless, be considered for any refinements to a high performance model.

Having developed some physical feeling for the cause of interference drag, we can try to reduce its value. Recall that in Chapter VI we used Eq. 21 to find the level of \( C_{Dint} \). One of the terms in that equation was the number of fins, since that number determines the number of junctions causing interference. Clearly, we must minimize the number of fins to reduce the interference. What’s the minimum number? One fin can’t be used, neither can two fins; we need at least three fins if we are to obtain the stable non-rolling flight pattern of an efficient ascent. Certainly four, or even more fins, could be used to perform the same task, but these would provide more junctions for increased interference. Four fins, for example, have 33% more interference drag than three fins, even when the total fin area remains the same. It’s best, therefore, to use three fins for minimum interference on high performance models.

Now, is there any other technique we can employ to reduce the interference even more? Yes, we can prevent the air from flowing in the corners of the body tube-fin junctions, an area where the interference between the air flowing along the body and the air flowing past the fins is greatest. All we need do is fill in the corners of the fin-body junctions. This technique is shown on the low interference, three-finned model sketched in Fig. 72. For contrast, a four-finned, high interference drag model is also shown in the figure. The filled in corners, called fillets, work by smoothly bringing the two separate air flows together and then gently guiding the merged flows past the junctions.

![FIG. 72: High and Low Interference Drag](image-placeholder)

- **Dotted Lines Show Interference Zones**
- **Sharp Corner Increases Interference Between Body and Fin Air Flows**
- **Low Interference Drag**

Fillets can be made quite easily by running a bead of white glue along the fin-body junction as indicated in Fig. 73. Wipe out the excess glue with the tip of your finger to get the desired smooth contour. Another filleting technique employs a paste made by adding talcum powder to the paint to be used on the model. When this paste is applied to the joint and allowed to dry it can be sanded smooth and painted, making a professional looking low drag fillet.

The actual reduction of interference drag accomplished by such filleting techniques is difficult to assess. It is indeed possible, using fillets on streamlined fins, to reduce the interference drag to a negligible fraction of the value predicted by Eq. 21. However, there is a lack of precise information on interference drag—a lack incidentally, that also exists with full scale rockets—that prevents us from placing a numerical value on the reduction of interference drag possible by filleting. Such an evaluation of interference is another of those experimental projects that model rocketeers may wish to undertake to provide useful information for other model rocket designers. We’ll just have to make sure that any model that we want to attain the ultimate in performance has proper fillets.



![FIG. 73: Building Up a Fillet](image-placeholder)

- **Run Glue in Joint, Then Wipe Off Excess**
- **Note the Completed Fillet**

## Five Rules for Drag Reduction

As a summary of the main ideas we’ve picked up in this chapter, let’s list five basic rules to be used to reduce the drag of a model rocket. The first four rules can be applied to any model rocket you build; the last rule on boat-tailing may require a model modification.

**Rule 1:** **Use Good Workmanship**
From the beginning of model construction, take time to sand all parts for a good fit: match the nose cone - body tube junction carefully, round the leading edge and sharpen the trailing edge of the fins. This initial work is a great step toward the reduction of the pressure drag of the rocket.

**Rule 2:** **Align Fins and Launch Lugs Properly**
Correct fin alignment will keep the model from rolling during the ascent; this will eliminate unnecessary induced drag caused by the fins twisting through the air at angle of attack. Misalignment of the launch lugs can cause flow separation on the body tube and excessive pressure drag.

**Rule 3:** **Put a Smooth Finish on the Model**
Besides giving a high quality appearance to the model, the smooth finish delays transition of the flow to the high drag turbulent boundary layer condition. Even when the flow is turbulent, a slick surface will have less drag than a rough surface; so to reduce skin friction drag get the model rocket finish mirror-smooth.

**Rule 4:** **Fillet the Fins**
Reduce the interference drag by filling in the fin-body tube junction to guide the air smoothly past the fins.

**Rule 5:** **Boat-Tail Whenever Possible**
Anytime the body tube diameter is greater than the engine diameter because of some special design feature, boat-tail the model. The base drag, which contributes an appreciable fraction of the total drag will be cut drastically by the boat-tail.

Take any model you build, apply these basic rules, and watch the improved performance. The best proof of a theory or a concept is a test that you make for yourself, so test these rules in practice. In the next chapter we’ll apply these drag reduction ideas to an Alpha and measure the difference in performance.

## VIII. Putting It All Together

### Summary

We’ve completed our study of model rocket drag. We certainly have come a long way since we wondered “what drag is”. We now know that the resistance to our model’s flight, which we call “aerodynamic drag”, is caused both by a pressure unbalance and by the friction of the air sliding rapidly over the model’s surfaces. We’ve broken our model rockets into components and examined the drag of each; in this way we have found effective nose cone shapes and low drag fin designs. We’ve looked at the drag due to angle of attack, drag due to surface finish and drag due to the blunt base. Finally, we’ve set down practical rules to be followed to reduce the drag of our model rockets.

In addition to this, we’ve been able to describe the drag of our rockets in a completely theoretical manner. We can now predict the drag coefficient, and therefore the magnitude of the drag, just from the size and shape of the various rocket parts. That’s one of the tasks we set out to do. Just like full-scale rocket designers, we can sit down, sketch out a rocket configuration and, after using the charts and procedures presented in this report, determine the drag coefficient. To be sure, our analysis has regions which can be improved - we have had trouble estimating launch lug drag and the effects of interference - but we have put upper and lower bounds on these factors. Our analysis is ready to use improved techniques when this information becomes available; in the meantime we are assured that our “ballpark” values are good.

Or are we? Have any check to be certain that our theory is correct? Only what has been developed from long experience - and some rocketeers may challenge that fact. It seems that we have one last task to do before we can close this report and begin building low drag, high performance birds. We must check the theory. The best way would be to test the model in a wind tunnel and compare the experimental values of drag with the theory. But few of us have access to a high speed wind tunnel with the sensitive instrumentation necessary to perform the tests. An alternative approach is to conduct flight tests and infer the aerodynamic drag from the altitude reached by the model. Flight tests are more difficult than wind tunnel tests because factors like the wind, rocket thrust variations, tracking inaccuracies (and lost tracks) enter into the experiment to cloud the data. However, flight testing is certainly the real proof-of-the-pudding, and with care it is possible to obtain reasonable experimental results. Just such a careful flight program, aimed at testing the concepts and analysis developed in this report, has been flown,* it is reported below.

### Flight Test Program

**Program Outline**

The flight tests were conducted to evaluate the aerodynamic drag of a series of model rockets. The experimental technique employed in the program was quite simple: A series of models was constructed,

*The author is greatly indebted to the members of the Columbus Society for the Advancement of Rocketry, of the National Association of Rocketry, an experienced group of model rocketeers in the Columbus, Ohio area that built, launched and tracked the model rockets used in this test program.



![FIG. 74](image-placeholder)

weighted to identical weights, launched with engines of the same type, and tracked to altitude. Once this altitude data was in hand, it was processed in two ways. First, just by comparing the altitudes, the lowest drag models (which reached the highest altitudes) could be identified. Second by doing a little data analysis, using altitude charts from TR-10, the effective drag coefficient for each model could be obtained and compared with the predicted drag coefficient.

## Models

Twelve model rockets were used in the test program. These were basically Astron Alpha models but built in five different categories. Three models were constructed as though a beginner put them together. No paint or sanding was used during fabrication and the fins were left rough-cut with no airfoil. Three models were built as a more experienced builder would do, with a moderate amount of sanding and a light coat of paint, but with the fins left in a rectangular cross section. Three more models were built as though by experts, with a fine finish, fillets and good streamline airfoil fins. These nine models in the three classes — called A, B, and C respectively — form the basis of the test program. They were built to show how the drag reduction rules, good workmanship, smooth finish and airfoiled fins, could improve the altitude performance of the same model design.

Supplementing these three categories were two more special purpose classes of Alphas. One model, termed a “D” model, was built with one fin leading edge canted 1/16”. This was to cause the model to spin during the ascent to determine the loss of altitude due to spin. Another two Alphas were redesigned using all the drag reduction concepts suggested in this report, including boat-tailing and high aspect ratio, unswept fins. These two models were called Up-Rated Alphas and were to be the ultimate test of the drag reduction concepts under evaluation. A photograph of the models is shown in Fig. 74; the table below summarizes pertinent facts about the models.

### Models Used in Test Program

| Class | No. Built | Construction Technique     | Finish     | Fin Cross-Section | Purpose                           |
|-------|-----------|----------------------------|------------|------------------|-----------------------------------|
| A     | 3         | Beginning Modeler          | None       | Rectangular      | Highest Drag Model                |
| B     | 3         | Intermediate Modeler       | Good       | Rectangular      | Show Improvement With Finish      |
| C     | 3         | Expert Modeler             | Excellent  | Streamline       | Show Improvement With Airfoiled Fins |
| D     | 1         | Expert Modeler             | Excellent  | Streamline but Canted | Show Loss Due To Fin Misalignment |
| Up Rated | 2      | Expert Modeler             | Excellent  | Streamline       | Show Improvement With Boat-Tail and New Fins |



## Test Procedure

The procedures followed during the test program were tailored to minimize the pitfalls of flight test work. For example, although flights with "B" engines would emphasize the aerodynamic effects, the tests were conducted with A8-5 engines. These engines were used to reduce the trajectory dispersion due to the wind and to ease the tracking problem. Even more important, the engines were specially selected from a single production run at the Estes plant; in this way the engines used had less than ± 2% variation in total impulse from the 2.5 Newton-second mean value.

To maintain the data quality, the models were weighed on a precision balance prior to launch. The heaviest bird was a Type B model, scaling 21 grams. The weights of all the other models were brought up to this value by trimming the streamer size so that all the birds came in at 21 grams, ± 0.1 gram. This procedure did require some pretty long crepe paper streamers in the unpainted models - these models were the lightest, averaging about 17.5 grams without the streamer.

All the flights were conducted on the same day, to minimize any flight variations that could be introduced by the atmospheric conditions. The altitude tracking was performed by members of the CSAR using two optical theodolites on a 1000 foot base line. The two elevation and two azimuth angles were used to compute two altitudes in the accepted manner; a track was accepted as valid only if the two altitudes were within 10% of the average value of the two heights. Each tracking station was equipped with a stop watch to record the time the rocket took to reach the peak altitude. This data was to be used to find the average flight speed of the model rocket, using the simple rate equation:

$$
\text{average speed} = \frac{\text{peak altitude in feet}}{\text{time to peak altitude in seconds}}
$$

In an effort to average out individual variations in the flight data, all models were flown three times. Lost tracks that fell outside the 10% of average altitude reduced the number of accurate altitudes for each class of Alphas; however, at least six good tracks were obtained for the models of A, B, and C type, while the single D model had two closed tracks and the Up-Rated Alphas had three closed tracks (one of the Up-Rated birds was damaged after the first flight and was taken out of service). These several flights do offer some statistical basis for the averaging process as is shown by the flight results.

## Results

A table summarizing the flight test results is presented below. The maximum and minimum altitude attained by each type Alpha is listed, then the average altitude reached is given. The second to the last column presents the percentage increase above the lowest performing model, the Type A Alpha. The last column gives the average airspeed of the class.

### Flight Test Results

_Astron Alphas flown with A8-5 engines_

| Model Class | Max. Alt. (ft) | Min. Alt. (ft) | Ave. Alt. (ft) | % Above "A" | Ave. Speed (ft/sec) |
|-------------|----------------|----------------|----------------|-------------|---------------------|
| A           | 341            | 295            | 319            | 0           | 82                  |
| B           | 371            | 337            | 335            | 11          | 97                  |
| C           | 403            | 369            | 383            | 20          | 101                 |
| D           | 368            | 345            | 356            | 11          | 97                  |
| Up-Rated    | 450            | 443            | 446            | 40          | 112                 |

For a visual presentation of this tabulated data, Fig. 75 has been prepared. From the table and this graph, it is clear that the drag reduction techniques do, indeed, work. Simply by taking a little care during assembly and sanding the models before painting, the altitude of the Alpha was increased by 36 ft. or 11%. Then by sanding an airfoil into the fins, the altitude jumped another 28 ft., or 20% above the Type A Alpha. Therefore, simply by taking the time and effort to build a professional-looking model we were able to lower the rocket drag and achieve a 20% increase in performance.

![FIG. 75: Altitude Performance of the Alpha Models](image-placeholder)

You might note that the average speed of the Alpha increased from 82 ft/sec to 97 ft/sec to 101 ft/sec as the model construction technique varied from that of the beginner to expert. Incidentally, the top speed of the rocket must be twice the average speed, according to the definition of average speed, that is:

$$
\text{Average Speed} = \frac{\text{Top Speed + Minimum Speed}}{2}
$$

since the minimum speed at the peak altitude is exactly zero. This means the top speed of the rockets varies from about 160 ft/sec to 200 ft/sec, an interesting piece of information about our rockets.

Other information is also available to us from the test program. Spinning the model by canting the fin resulted in a 27 ft. loss in altitude. Misalignment of a single fin in an otherwise carefully built model can, evidently, cause a 7% decrease in performance. Maybe 27 ft. doesn’t sound like too much, but that’s only because we were using A engines. With B or larger engines, this decrease in performance would be much greater. Of course, even the 27 feet can be the difference between a winning performance and a modest flight.



The redesigned Alphas performed quite well, exceeding the expert built, basic Alphas by an average altitude of 63 feet. Further, the top speed of the Up-Rated Alpha was increased to near 225 ft/sec. This 40% increase above the beginner type Alpha demonstrates the validity of the basic design rules we stated in the last chapter.

Certainly, the concepts for drag reduction have been proven by these flight tests; it now remains for us to test the analysis to determine how closely the experimental drag coefficient matches the predicted value.

## Alpha Drag Analysis

Once more, let’s run through an analysis of the zero lift drag of a model rocket. This will be the last time we’ll perform this task, so let’s set down seven easy steps for this drag analysis. It is true that we’ve discussed many factors in our drag study, and it’s really possible to believe the problem is more complicated than it really is. Let’s try to simplify the job.

Whenever a drag analysis is to be made, the first item is to get the configuration of the model and list the important geometric properties. These are \( d \), \( d_b \), \( l \), \( S_{BT} \) for the body, \( C_T \), \( C_R \), \( S_F \) and \(\frac{t}{c}\) for the fins, and \( S_{LL} \) and \( S_{LLW} \) for the launch lug. Next, decide on the flight condition (this is usually just the velocity, \( V \)), and the type of boundary layer on the body and fins. Then just follow these seven simple steps:

### Step 1

**Find \( C_{DN} + C_{DBT} \)**
Use Fig. 28 for a model with a 3:1 ogive nose cone, use Eq. 8 for other nose cone designs. The value of \( C_{DN} + C_{DBT} \) depends upon \( l/d \), \( V \) and the type of boundary layer.

### Step 2

**Find \( C_{DB} \)**
Use result of Step 1 and \( d_b/d \)

$ C_{DB} = \frac{0.029}{\sqrt{C_{DN} + C_{DBT}}} \left(\frac{d_b}{d}\right)^3 $
 Eq. (28)

### Step 3

**Find \( C_{DOF}^* \)**
Use Fig. 40 if fins have rectangular section, or Fig. 41 if fins have streamline airfoil. \( C_{DOF}^* \) depends upon \(\frac{t}{c}\), \( V \), and type of boundary layer.

### Step 4

**Find \( C_{DOF} \)**
Use result of Step 3 and the fin and body tube areas

$ C_{DOF} = C_{DOF}^* \frac{S_F}{S_{BT}} $
Eq. (19)

### Step 5

**Find \( C_{Dint} \)**
Use result of Step 3, root chord \( C_R \) and the number of fins

$ C_{Dint} = C_{DOF}^* \frac{C_R}{S_{BT}} \frac{d}{2} \times \text{no. of fins} $
Eq. (21)

### Step 6

**Find \( C_{DLL} \)**

$ C_{DLL} = \frac{1.2 \, S_{LL} + 0.0045 \, S_{LLW}}{S_{BT}} $
Eq. (24)

### Step 7

**Add the basic component drag coefficients.**

$ C_{DOB} = C_{DN} + C_{DBT} + C_{DB} + C_{DOF} + C_{Dint} + C_{DLL} $

To demonstrate how rapidly this analysis may be used, consider the basic Alpha configuration in Fig. 76. From the figure, assemble the basic geometric parameters of the Alpha:

- **Body:**
  - \( d = 0.976 \text{ in.} \)
  - \( d_b = 0.976 \text{ in.} \)
  - \( l = 10.5 \text{ in.} \)
  - \( S_{BT} = 0.746 \text{ in}^2 \)
  - 3:1 ogive nose cone

- **Fin:**
  - \( C_R = 2.25 \text{ in.} \)
  - \( C_T = 1.50 \text{ in.} \)
  - \( S_F = 8.21 \text{ in}^2 \)
  - \(\frac{t}{c} = \frac{0.093}{2.25} = 0.041\)

- **Launch Lug:**
  - \( S_{LL} = .005 \text{ in}^2 \)
  - \( S_{LLW} = 1.5 \text{ in}^2 \)

![FIG. 76: Design of the Astron Alpha](image-placeholder)



Now, based on the flight test program which indicated that the average velocity was about 100 ft/sec, choose \( V = 100 \) ft/sec as the flight condition to be examined. As we noted earlier, select turbulent boundary layer conditions for the body and laminar for the fins. Following the seven steps for Type B Alpha we obtain:

### Step 1

**\( C_{DN} + C_{DBT} = 0.205 \)**
Using Fig. 28 with \( l/d = 10.5 \) at \( V = 100 \) ft/sec for the turbulent boundary layer case.

### Step 2

$ C_{DB} = \frac{0.029}{\sqrt{C_{DN} + C_{DBT}}} \left(\frac{d_b}{d}\right)^3 = 0.064 $

### Step 3

**\( C_{DOF}^* = 0.035 \)**
Using Fig. 40 and the \(\frac{t}{c} = 0.041\) for rectangular cross-section fins.

### Step 4

$ C_{DOF} = C_{DOF}^* \frac{S_F}{S_{BT}} = 0.035 \times \frac{8.21}{0.746} = 0.386 $
From Step 3 and the fin and body tube areas.

### Step 5

$ C_{Dint} = C_{DOF}^* \frac{C_R}{S_{BT}} \frac{d}{2} \times \text{no. of fins} $
$ = 0.035 \times \frac{2.25 \times 0.976 \times 3}{0.746} = 0.154 $

### Step 6

$ C_{DLL} = \frac{1.2 \times 0.005 + 0.0045 \times 1.5}{S_{BT}} $
$ C_{DLL} = \frac{1.2 \times 0.005 + 0.0045 \times 1.5}{0.746} = 0.017 $

### Step 7

$ C_{DOB} = C_{DN} + C_{DBT} + C_{DB} + C_{DOF} + C_{Dint} + C_{DLL} $
$ C_{DOB} = 0.205 + 0.064 + 0.386 + 0.154 + 0.017 = 0.826 $

This completes the zero lift drag coefficient analysis for the Type B Alpha with \( C_{DO} = 0.826 \). We did not consider the Type A Alpha at first because there is no direct way to account for poor workmanship in our theory. We do have one observation that is useful, however: We noted in Chapter VII that roughness can cause drag to increase by about 25 percent. Since the Type A and Type B Alphas are identical in configuration except for the rough workmanship, we can try to estimate the effects of this beginner’s quality workmanship for the Type A model by increasing the drag coefficient of the Type B Alpha by 25 percent. This gives \( C_{DO} = 1.03 \) for the Type A Alpha. For the Type C Alpha, we must repeat the seven steps in the drag analysis. We can expect a difference in \( C_{DO} \) because the streamlined fins will lower both \( C_{DOF} \) and \( C_{Dint} \).

Similarly, the Up-Rated Alpha has a completely different geometry, as shown in the design drawings of Fig. 77; the drag distribution will be altered accordingly. The theoretical distribution of drag and the zero lift coefficient of all the Alphas are presented in the following table.

When comparing the tabulated results for the Type B and Type C model Alphas, we encounter once again the benefits of streamlining the fins. Not only does the fin drag coefficient drop to \( C_{DOF} = 0.116 \) from \( C_{DOF} = 0.386 \), but the entire drag is reduced from \( C_{Dint} = 0.154 \) to \( C_{Dint} = 0.046 \). This double benefit lowers the zero lift drag coefficient to \( C_{DO} = 0.448 \), a significant reduction of 45 percent of the Type B value. Remember: that this is theory, still, and we must check with our flight experiments before we accept these numbers.

How does the analysis of the Up-Rated Alpha compare with the Type C Alpha? Base drag is greatly reduced by the boat-tail as expected, but this reduction in pressure drag was partially offset by the increased length of the rocket body. This is just another example of the compromises that must be faced by a designer of a flying machine--to reduce the base drag, we boat-tailed the Alpha, but the boat-tail lengthened the body, increasing the skin friction drag. The rocket body drag including base drag for the Up-Rated Alpha was still lower than for the Type C version, and besides, the added length was necessary to provide sufficient rocket stability with the new longer fins used on the Up-Rated model. The fin area for the Up-Rated Alpha was reduced to 75 percent of the standard Alpha fin area, so the zero lift fin drag was decreased proportionately. Fin thickness for the two versions was just about the same; 1/16" sheet balsa was used for the Up-Rated Alpha instead of the 3/32" sheet used for the Type C model. The net result

![FIG. 77: Design of the Up-Rated Alpha](image-placeholder)

- **Centering Rings for BT-20 Engine Tube**
- **Stiff Card 1/32" Balsa Boat-Tail**
- **Fin Stock: 1/16" Balsa**



| Component              | Drag Coefficient Symbol | A    | B    | C    | Up-Rated |
|------------------------|--------------------------|------|------|------|----------|
| Nose Cone & Body Tube  | \( C_{DN} + C_{DBT} \)   | 0.205| 0.205| 0.205| 0.233    |
| Base                   | \( C_{DB} \)             | 0.064| 0.064| 0.064| 0.026    |
| Fins                   | \( C_{DOF} \)            | 0.386| 0.386| 0.116| 0.086    |
| Interference           | \( C_{Dint} \)           | 0.154| 0.154| 0.046| 0.030    |
| Launch Lug             | \( C_{DLL} \)            | 0.017| 0.017| 0.017| 0.017    |
| Total Zero Lift        | \( C_{DO} \)             | 1.03*| 0.826| 0.448| 0.392    |

*Roughness factor applied as suggested in Chapter VII: \( C_{DO} \) for Type A = 1.25 times \( C_{DO} \) for Type B Alpha.

The drag reduction procedures is that the Up-Rated Alpha, with \( C_{DO} = 0.392 \), has 11 percent less drag at zero lift than the Type C Alpha. And, at any angle of attack this drag advantage of the Up-Rated Alpha will be increased because its unswept, high aspect ratio fins generate less induced drag than the unswept low aspect ratio fins of the standard Alpha.

## Flight Test Drag Coefficients

Let's reexamine the flight test data to see what additional information we can gather about the drag of our Alphas. Rockets reach a particular altitude because of an interplay between the rocket thrust, launch weight, and aerodynamic resistance. If we can understand precisely how these four factors of altitude, thrust, weight, and drag are related, we will be able to do more than predict altitude performance. We could, for example, turn the altitude problem around—instead of asking what altitude results from a given combination of thrust, weight, and drag, we could ask what air resistance must have been encountered for a given combination of thrust, weight, and achieved altitude. This air resistance leads, in turn, to the effective drag coefficient, the very factor we have been trying to predict in this report.

The physical law that governs the vertical ascent of a model rocket is called Newton's Second Law of Motion.* Although the law is not difficult to understand, the force required to accelerate a body depends upon its mass. This simple statement, written below in word equation form is the foundation of the field of Dynamics.

$$
\text{Force} = \text{Mass} \times \text{Acceleration}
$$

To demonstrate its importance, let's find the acceleration of a model rocket from the launch pad. Rearranging the Second Law in the form

$$
\text{Acceleration} = \frac{\text{Net Force Acting on Rocket}}{\text{Mass of Rocket}}
$$

we must make appropriate substitutions for a numerical answer. First, though, we must define the net force acting on a rocket. During vertical ascent this must be

$
\text{Net Force} = \text{Thrust of Rocket - Weight - Drag}$$

since both the weight and aerodynamic force act vertically downward as the rocket thrusts vertically upward. Assuming an A8 engine is used, thrust will average 8 Newtons or 1.8 lbs; the weight of the rocket is 1.36 ounces or 0.085 lbs. Recall that the mass of the rocket would be the weight divided by the acceleration of gravity—mass = 0.085/32.2 ft/sec² = 0.00264 slugs. With these values, initial acceleration becomes

$$
\text{Acceleration} = \frac{1.8 - 0.085 - 0}{0.00264} = 650 \text{ ft/sec}^2
$$

Drag was assumed to be zero in this case, because at the instant of ignition the velocity is zero so there is no aerodynamic drag. As soon as the model starts to move, drag appears, lowering the net force acting on the rocket and causing the acceleration to decrease. But then as the rocket consumes fuel its weight and mass will go down and that would cause the acceleration to increase. Not only that, but the thrust isn't really constant, but varies from instant to instant. No wonder we need a computer to keep track of all these changes in order to predict the actual model motion!

We'll close this footnote with the observation that three of the four factors of the altitude performance problem have been pointed out—thrust, weight and drag. Altitude comes from the acceleration, which is the change in velocity per unit time. Velocity is, in turn, the change in altitude with time. Therefore, by finding the acceleration, we can eventually determine the altitude. As an example, consider a special case that has zero drag. Then, by neglecting the small change in weight, we would have a constant acceleration during the rocket burn. At burnout, in 0.32 seconds for the A8 engine, the velocity has reached

$$
\text{Velocity} = \text{Acceleration} \times \text{Time}$$

$
= 650 \times 0.32 = 208 \text{ ft/sec}
$$

The average speed during boost, since we started from zero, is 104 ft/sec; the altitude at burnout for the no-drag case becomes

$
\text{Altitude} = \text{Ave. Speed} \times \text{Time}$$

$
= 104 \times 0.32 = 33.3 \text{ ft.}$$

Remember that these calculations could be made easily because we had assumed there was no drag. Drag would lower the burnout speed and altitude.

After burnout, the model will decelerate (the net force is negative in Newton's Second Law) until zero velocity at the peak altitude, then fall back to earth.



some advanced mathematics are required to manipulate this law to get practical altitude results. Luckily, we need not concern ourselves with these math details. In Estes Technical Report 10, "Model Rocket Altitude Prediction Charts," these complications have been removed and through extensive use of a computer the complex mathematical solutions have been reduced to a series of graphs. A chart is made up for each engine, and with knowledge of the launch weight, body cross-section area, and drag coefficient, the predicted altitude for a vertical, no-wind ascent can be obtained directly from the chart. Results from this Technical Report provide the special relations, in graphical form, that we need to obtain drag coefficients from our flight tests.

Using the information that is graphed in TR-10, one chart can be prepared that will give us the altitude performance of any model rocket built of a particular body tube size and powered by a specific engine. For the BT-50 body tube size and the A8 engines used in the Alpha flight test program, this special chart takes the form shown in Fig. 78. This single chart illustrates the influence of drag coefficient, launch weight and air density upon the height attained by a model rocket. Early in our drag study we learned that the air resistance depended upon the air density: therefore, the air density at the launch site must be considered in our altitude predictions.

In normal use of this chart, the launch weight is known, the drag coefficient is located (or calculated from this report) and the air density found from the curve in the Appendix or computed from the barometric pressure and temperature at the launch site. For example, consider a sea level launch of a model weighing 1.36 ounces with a \( C_{D} = 0.6 \). The pressure and temperature are entered into the formulas \( P = 29.92"\) Hg and the temperature = 59°F, then

$ \frac{P}{P_{SL}} = \frac{17.35 \times 29.92}{460 + 59} = 1.00 $

then \( C_{D}\frac{P}{P_{SL}} = 0.6 \times 1 = 0.6 \)

Locating the value 0.6 on the horizontal axis and moving up to the 1.36 ounce launch weight, then reading across to the altitude scale on the left, we find that the predicted performance of this bird is 390 feet. It’s worth noting that if the launch weight had been 2 ounces, the altitude would be down to 241 feet. That’s quite an altitude loss from the 0.64 ounce weight increase, so keep your models light. Suppose the launch site was located at an altitude other than sea level, where the barometer reading was 24.96” Hg and the temperature was 41°F?

First find

$ \frac{P}{P_{SL}} = \frac{17.35 \times 24.86}{460 + 41} = 0.861 $

then \( C_{D}\frac{P}{P_{SL}} = 0.6 \times 0.861 = 0.518 \)

with this value, Fig. 76 indicates the rocket would get to 412 feet. This model got 22 feet higher because the air density was less. Low air density comes with high elevations or with high air temperature, so those rocketeers flying from fields with these conditions have a clear advantage over the cold climate, sea-level flyers.

Well, although it’s nice to see precisely how much weight, drag coefficient and air density change the predicted altitude of a model rocket, that’s not the main purpose of Fig. 78 in this section. The reason the figure was constructed was to provide a means of converting our flight test results to drag coefficient form. All we need do is to work backward through the chart. Take a measured altitude, like the 355 foot height attained by the Type B Alpha for example, and move to the right from the vertical axis to the launch weight and read down to find \( C_{D}\frac{P}{P_{SL}} = 0.75 \). Then the effective drag coefficient for this flight was

$ C_{D} = \frac{0.75}{\frac{P}{P_{SL}}} $

During the flight test program, the weather station reported the temperature as 73°F and the barometric pressure as 29.45”Hg. Note that this barometer reading is the actual pressure at the field and is not corrected to the sea level value. Therefore

$ \frac{P}{P_{SL}} = \frac{17.35 \times 29.45}{460 + 73} = 0.96 $

This density ratio means that the effective drag coefficient for the Type B Alphas was

$ C_{D} = \frac{0.75}{0.96} = 0.781 $

Now, what was the predicted value for the Type B Alpha? \( C_{D} = 0.826 \). Very good! The difference between the coefficient obtained by flight test and by theory was less than 6 percent. Such results make any aerodynamicist glow—or make him suspicious because the results sound too good. Let’s check the other flight test drag coefficients in a similar manner. These results are presented in the table below and compared with the theoretical \( C_{DO} \) obtained from the drag analysis of the last section.

![FIG. 78: Predicted Altitude for A8 Powered Models with BT-50 Body Tubes](image-placeholder)



## Zero Lift Drag Coefficient Flight Test and Theory

| Model    | Ave Alt (ft) | \( C_D \) Flight Test | \( C_D \) Theory | % Deviation from Flight Test |
|----------|--------------|-----------------------|------------------|------------------------------|
| A        | 319          | 0.975                 | 1.03             | 6                            |
| B        | 355          | 0.781                 | 0.826            | 6                            |
| C        | 383          | 0.650                 | 0.448            | -31                          |
| Up-Rated | 446          | 0.421                 | 0.391            | -7                           |
| D        | 356          | 0.771                 | No Solution      |                              |

## Drag Coefficient Comparison

For the Type A and B Alphas, the agreement between the drag coefficients obtained from theory and experiment was excellent—differences less than 6 percent were recorded. Considering that we used only the geometry of the Alpha in our seven step drag analysis, the theory performed quite well. A question does arise, however, as to why the theoretical drag value, which is based on a zero angle of attack flight, was greater than the flight test drag coefficient. We would anticipate the opposite to be true. After all, during the ascent, the rocket would most likely encounter some disturbance resulting in an angle of attack and a corresponding increase in \( C_D \). It’s probable that a good agreement would be obtained between the two drag coefficients if this fin drag coefficient was lowered less than 10 percent. This 10-percent decrease in \( C_{DOF} \) is quite possible—just look back to Fig. 40. Rounding the fin leading and trailing edges can drop \( C_D \) by 40 percent. Although every effort was made to avoid this during fin construction, a very slightly rounded corner could have resulted from the light sanding, and this would reduce the flight test drag coefficient.

Another feature that should not be overlooked in the flight test results is the drag penalty caused by the beginner-type model construction. Flight test shows that the drag coefficient increased 24.8 percent. That surely verifies the estimate we made of a 25 percent variation due to model workmanship! This 25 percent rule should be passed on to beginners; it’ll show them good workmanship pays off.

Unhappily, the Type C Alphas do not exhibit the same agreement between the theoretical and the flight test \( C_D \) values that the Types A and B Alphas do. The 31 percent difference in the two drag coefficients is puzzling. When we examined the two Up-Rated Alpha drag coefficients, calculated and measured in the same manner, we again find excellent agreement (7 percent) between the analytic and flight test coefficients. What, then, caused the discrepancy in the case of the Type C Alpha?

We might look over the flight test program for clues to the apparent poor altitude performance of the Type C models. The three Type C Alphas were launched in a random sequence interspersed with A, B and Up-Rated Alpha launches. This randomizing procedure was used to average out the effects of wind and the trackers’ performance. The records show that two tracks were lost on one of the C Alphas and two of the other altitude tracks used were near the 10 percent limit. One point that could cause a lower than normal altitude would be a premature ejection of the streamer. The five second delay times were used to reduce this possibility and indeed, almost all of the flights went “over the top” of the trajectory before streamer deployment. A few ejections did occur, however, during ascent of the high performance birds. Unfortunately, the flights that experienced early ejection were not noted on the records. So, premature ejection might have caused the altitude deficit, but we cannot prove this.

It is entirely possible, of course, that the altitude achieved by the Type C Alpha was the correct value and the discrepancy is caused by a theoretical equation that just predicts too low a drag coefficient. Interference drag is always difficult to evaluate; maybe the interference drag for the swept streamlining fins is higher than the theoretical estimate. Possibly, the fin cross-sections were not made to the perfect streamline shape which would cause the actual fin drag coefficient to be greater than predicted. Or the surface may not have been perfectly smooth, causing extra drag. There is always this problem of constructing the model exactly as the mathematical representation. That’s why we build three models of each type—to get an average representation.

Now, when we consider the excellent comparisons of the flight test and theoretical drag coefficients for the Type A, B, and Up-Rated Alphas, and when we recognize the difficulties of any flight test program (as well as the limitations of any theory) we should not expound too much on the differences of a 31 percent discrepancy of the Type C Alpha.

Our confidence in the theory increased when we examine Fig. 79. The predicted drag coefficients and measured altitudes for our four Alpha models agree compared with the theoretical altitudes predicted with TR-10 for a 1.36 ounce launch weight. Once again the good agreement between the theoretical and actual performance of the Type A, B, and Up-Rated Alphas is shown. And, once again, the disconcerting disagreement of a Type C Alpha is pointed out. The 31 percent drag coefficient discrepancy causes a fifty foot altitude difference; we’ll just have to accept this difference. Hopefully, some rocketeer will conduct flight tests on the Type C Alpha (that’s the only kind we are supposed to build--professional looking) and help resolve our dilemma. We’re anxious to learn if the altitude achieved by our particular Type C Alphas was lower than the height that should have been attained or if the predicted drag coefficient was just too low.

One last comment on the flight test coefficients: The spinning of the Type D Alpha created a drag coefficient of \( C_D = 0.771 \). There is no theoretical solution worked out for this bird; that would require a prediction or measurement of the rate of roll with a calculation of the angle of attack of the fins and the corresponding induced drag. That calculation, as is often said, is beyond the scope of this report. It’s possible to observe the effect of the spin on the altitude, however, by comparing with the non-rolling Type C Alpha. Canting the one fin increased the flight test drag coefficient by almost 19 percent. That’s quite a penalty. In fact, the induced drag is about equivalent to the drag caused by



![FIG. 79](image-placeholder)

The rectangular fins instead of streamlined. The Type B Alphas, you might note, have a \( C_D = 0.781 \). This drag increase with roll is just another verification of our drag reduction concepts—keep the fins aligned to prevent roll for the lowest drag performance.

## The Last Word

The title of this last chapter was “Putting It All Together”. That's what we’ve done. We can now design a model rocket based on low drag concepts, and calculate the drag coefficient from the model geometry using the analysis of this report. We can then take this drag coefficient and the model rocket weight, employ TR-10 to predict the rocket performance and finally fly the bird to test the predictions, exactly like the aerospace designers and engineers. That’s what model rocketry is—aerospace engineering in miniature!


## Appendix I

As the drag analysis was developed in this report, this skin friction coefficient, \( C_f \), was brought up and used several times (Eq 8 and Eq 13 for example), but never discussed at any length. That procedure was followed to avoid clouding the topic of model rocket drag and to avoid becoming bogged down in too much detail. Using selected values for \( C_f \), we were able to draw curves for the rocket body and fin drag which did, indeed show magnitude of the drag coefficients of the various model rockets, but only for a specific class of rockets—those models that used a three-to-one ogive nose cone and traveled at speeds of 100, 300, and 500 ft/sec under standard atmospheric conditions. That was fine, and very instructive; however some model rocketeers may wish to use a two-to-one ogive nose cone or calculate the drag at 150 ft/sec and at some non-standard atmospheric conditions. What then? Well, in order to proceed with an analysis, we’d have to have the basic information—the data upon which the analysis was built. The data is provided in this Appendix by design charts.

The exact manner in which the skin friction coefficient \( C_f \) varies with Reynolds number is shown in Fig. A-1 for both the laminar and the turbulent boundary layer cases. The Reynolds number, in turn, can be found from Fig. A-2 when the air temperature and the altitude is known. Another important bit of data is presented in Fig. A-3. This figure gives the ratio of wetted surface area to body tube cross-section area, \( S_w/S_{BT} \), for conical and ogive nose cones of various length to diameter ratios. Finally, Fig. A-4 is included to allow calculation of the rocket drag at altitudes up to 20,000 feet by showing the density variation with altitude on a "standard day".

To illustrate the procedure to be used with these charts, let’s find the drag of a model that looks different from any we’ve examined so far. We’ll consider a model built from a BT-20 body tube and a two-to-one ogive nose cone and with rectangular fins of streamline cross-section. Then we’ll look at a non-standard flight condition, say an air temperature of 80°F and a field elevation of 2000 ft with the model moving at a flight speed of 150 ft/sec. The model is sketched below along with the geometric data.

- \( d = 0.736 \)
- \( d_b = 0.736 \)
- \( l = 14.0" \)
- \( S_{BT} = 0.415 \, \text{sq. in.} \)
- \(\frac{t}{c} = 0.0625\)

- \( C_T = 0 \)
- \( C_R = 2.0 \)
- \( S_{LL} = 0.005 \, \text{sq. in.} \)
- \( S_{LLW} = 2.00 \, \text{sq. in.} \)
- \( S_F = 4.5 \, \text{sq. in.} \)

Our seven step drag analysis starts with Eq. 8:

$$
C_{DN} + C_{DBT} = 1.02 \, C_f \left[ 1 + \frac{1.5}{(L/d)^{3/2}} \right] \frac{S_w}{S_{BT}}
$$

To use this equation we must find a value for \( C_f \) and \(\frac{S_w}{S_{BT}}\). We can obtain \( C_f \) from Fig. A-1 if we know the Reynolds number. We find the Reynolds number from Fig. A-2. Moving vertically up the 80°F line in Fig. A-2 to the 2000 ft field altitude we read 5410 on the vertical scale. This number is the Reynolds number of a one foot long model rocket body moving at 1 \(\nu\) ft/sec. To correct the Reynolds number for our case, we multiply by the length of our new model (in feet) and we multiply again by the flight velocity of our model:

$$
RN = 5410 \times \frac{14}{12} \times 150 = 945,000
$$

Using this Reynolds number in Fig A-1, we find \( C_f = 0.0045 \) for the turbulent boundary layer case.

Now, in order to find \(\frac{S_w}{S_{BT}}\), first observe that the rocket body is made up of two basic shapes, an ogive nose cone and a cylindrical body tube. The wetted surface to cross-section area ratio for each of these



![FIG. A-1: Skin Friction Coefficient \( C_f \) vs Reynolds Number RN for Laminar and Turbulent Boundary Layers](image-placeholder)

The basic shape is shown in Fig. A-3. To use the figure we need to know the length to diameter ratios, \(\frac{L_N}{d}\) of the nose cone and \(\frac{L_{BT}}{d}\) of the body tube. We obtain these from the sketch.

- \( \frac{L_N}{d} = \frac{1.47}{0.736} = 2.0 \)
- \( \frac{L_{BT}}{d} = \frac{12.53}{0.736} = 17.0 \)

The total value of \(\frac{S_w}{S_{BT}}\) is the sum of the two components; therefore adding the two values of the area ratio obtained from Fig. A-3 we arrive at

$ \frac{S_w}{S_{BT}} = \frac{S_w}{S_{BT}}_N + \frac{S_w}{S_{BT}}_{BT} = 5.4 + 68 = 73.4 $

You might note that the figure gives \(\frac{S_w}{S_{BT}}_{BT}\) only up to \(\frac{L_{BT}}{d} = 5\).

*That's because of the size of the chart. This is no handicap, though, since we see from the chart that the area ratio is exactly four times the length to diameter ratio.* All we have to do for the body tube, therefore, is use the equation:

$ \frac{S_w}{S_{BT}}_{BT} = 4 \frac{L_{BT}}{d} $

With this information, we can evaluate Eq. 8 for our new model rocket body:

$ C_{DN} + C_{DBT} = 1.02 \, C_f \left[ 1 + \frac{1.5}{(L/d)^{3/2}} \right] \frac{S_w}{S_{BT}} $

$ = 1.02 \times 0.0045 \left[ 1 + \frac{1.5}{(19)^{3/2}} \right] 73.4 $

$ C_{DN} + C_{DBT} = 0.343 $

The somewhat high figure for this drag coefficient (the Alphas had \( C_{DN} + C_{DBT} = 0.205 \)) is due to the high length to diameter of the rocket body. This does not mean that the drag is greater, as we shall see later when we use the smaller body tube diameter in our drag force calculation.

![FIG. A-2: Chart for Rocket Reynolds Number](image-placeholder)

### Example:

**Find RN for a 14 inch long model rocket moving at a speed of 150 ft/sec when \( T = 80°F \) at 2000 ft altitude, then:**

$ RN = 5410 \times \frac{14}{12} \times 150 = 945,000 $

*We can prove this statement from geometrical considerations:*

$$
\frac{S_w}{S_{BT}} = \text{Area of cylinder} = \frac{\pi \, d \, L_{BT}}{\frac{\pi \, d^2}{4}} = 4 \frac{L_{BT}}{d}
$$



The second step of our analysis is to find the base drag coefficient:

$$
C_{DB} = \frac{0.029}{\sqrt{C_{DN} + C_{DBT}}} = \frac{0.029}{\sqrt{0.343}} = 0.049
$$

![FIG. A-3: Ratio of Wetted Area to Cross-Section Area, \( S_w/S_{BT} \) for Conical and Ogive Nose Cones and for Cylinders](image-placeholder)

The third step is to find the fin drag coefficient \( C_{DOF}^* \). For this value we use Fig. A-1 again, once we determine the correct Reynolds number. We’ll use the average fin chord of one inch to calculate the effective Reynolds number on the fin. For our 80°F air with the 2000 foot altitude, we again find a value of 5410 from Fig. A-2. Correcting this for the one-inch chord and the 150 ft/sec flight velocity, we obtain the Reynolds numbers for the fin

$$
RN = 5410 \times \frac{1 \times 150}{12} = 57,600
$$

A smooth streamlined fin at this Reynolds number would have a laminar boundary layer. Looking up this Reynolds number to the laminar line in Fig. A-1 we read \( C_f = 0.00515 \). The fin drag coefficient is found from Eq. 14:

$$
C_{DOF}^* = 2 \, C_f \left[ 1 + 2 \frac{t}{c} \right]
$$

$$
= 2 \times 0.00515 \left[ 1 + 2 (0.0625) \right]
$$

$$
C_{DOF}^* = 0.0116
$$

It is necessary to reference the drag coefficient to the body cross-section area. Step four, then, is the application of Eq. 19:

$$
C_{DOF} = C_{DOF}^* \frac{S_F}{S_{BT}} = 0.0116 \times \frac{4.5}{0.425} = 0.123
$$

Step five is the evaluation of the interference drag coefficient from Eq. 21:

$$
C_{Dint} = C_{DOF}^* \frac{C_R}{S_{BT}} \frac{d}{2} \times \text{No. of fins}
$$

$$
= 0.0116 \times \frac{2}{0.425} \times \frac{0.736}{2} \times 3
$$

$$
C_{Dint} = 0.060
$$

The launch lug drag coefficient is the sixth step. Using a 1 1/2 inch lug in Eq. 24 we find:

$$
C_{DLL} = \frac{1.2 \, S_{LL} + 0.045 \, S_{LLW}}{S_{BT}}
$$

$$
= \frac{1.2 \times 0.005 + 0.0045 \times 1.5}{0.425}
$$

$$
C_{DLL} = 0.028
$$

This is a little higher than the earlier \( C_{DLL} \) but again that's because we are using a smaller body tube as the reference area, \( S_{BT} = 0.425 \) for the BT-20, instead of \( S_{BT} = 0.746 \) for the BT-50 body tube of the Alphas.

Summing all these components in the seventh and last step, we find the zero lift drag coefficient, \( C_{DO} \):

$$
C_{DO} = C_{DN} + C_{DBT} + C_{DB} + C_{DOF} + C_{Dint} + C_{DLL}
$$

$$
= 0.343 + 0.049 + 0.123 + 0.060 + 0.028
$$

$$
C_{DO} = 0.603
$$

This drag coefficient is significantly higher than the \( C_{DO} \) we calculated for either the Type C or Up-Rated Alpha. Before we jump to any conclusions, though, we’d better check the drag force on the new bird. We can use Eq. 17, modified slightly to allow us to find the effect of the 2000 foot altitude:

$$
D_0 = C_{DO} \frac{1}{2} p \, V^2 \, S_{BT} = C_{DO} \frac{1}{2} (0.002378) \frac{p}{P_{SL}} V^2 S_{BT}
$$

where we’ll use Fig. A-4 to find the density ratio \( p \) at 2000 feet. Inserting the appropriate values for the new bird:

$$
D_0 = 0.603 \times \frac{1/2 \times 0.002378 \times 0.942 \times (150)^2 \times 0.425}{144}
$$

$$
D_0 = 0.0453 \, \text{lbs or} \, 0.725 \, \text{ounces}
$$

We compare this drag with the drag of the Up-Rated Alpha flying at the same conditions. The Alpha’s \( C_{DO} = 0.364 \) for the 150 ft/sec flight velocity, down from \( C_{DO} = 0.392 \) we found at \( V = 100 \) ft/sec on a standard day. Using this value and the correct \( S_{BT} \) (remember to



divide the 0.746 square inch value by 144 to convert to area in square feet):

$$
D_0 = 0.364 \times \frac{1}{2} \times 0.002378 \times .942 \times (150)^2 \times \frac{0.746}{144}
$$

$
D_0 = 0.0475 \, \text{lbs or} \, 0.760 \, \text{ounces}
$$

Although \( C_{DO} \) was greater for the BT-20 rocket, the smaller body tube cross-section area more than made up for this drag coefficient increment; the actual drag force was about 5% below the drag force of the Up-Rated Alpha. That means that if both birds weighed the same, the new bird would go higher. But then you couldn’t pack as big a parachute in the BT-20 tube so this bird would descend faster. Hmm. We’re back to the compromises again and all we can say is that it’s the designer’s choice — and that’s the fun of model rocketry.

![FIG. A-4: Density Ratio Variation with Altitude for Standard Day Conditions](image-placeholder)

One last comment about Fig. A-4—besides giving needed information for calculating drag as in Eq. 17, it lets you have a little fun figuring the altitude effects on your model rocket performance. Launching the Up-Rated Alpha from Pike’s Peak in Colorado would give you a great performance. At the 14,110 foot altitude

$$
\frac{p}{P_{SL}} = 0.642
$$

Then

$$
C_{DO} \times \frac{p}{P_{SL}} = 0.392 \times 0.642 = 0.252
$$

When we look that value up in Fig. 76, we find the predicted altitude is over 510 ft. compared to a sea level launch which just clears 450 ft. Looks like Pike’s Peak would be a good place to set some altitude records!

## Appendix II

### Suggested Reading List

Model rocketeers who want to continue their studies of Drag and of Aerodynamics will find the books listed below quite helpful.

**SHAPE AND FLOW** by Ascher H. Shapiro, a Science Study Series Paperback ($21) published by Doubleday and Co. of Garden City, New York for $0.95, is an excellent text for all model rocketeers. The book reviews the fundamental ideas of concerning drag; i.e. air viscosity, Reynolds number, flow separation and laminar and turbulent boundary layers, in a stimulating and fresh approach to fluid dynamics.

**AERODYNAMICS** by Theodore Von Karman is another paperback put out by McGraw-Hill Book Company of New York for $2.45. Von Karman, the foremost aerodynamicist of the past half century writes a history of Aerodynamics, then goes beyond the basic concepts to include discussions of supersonic and hypersonic aerodynamics. The true genius of Von Karman is his ability to write of this complex field in a manner that all model rocketeers can understand.

**AIRPLANE AERODYNAMICS** by D.O. Dommasch, S. S. Sherby, and T. F. Connolly, published by Pitman Publishing Corporation of New York is a text for advanced rocketeers. It contains current information of supersonic as well as subsonic, aerodynamics and chapters on rockets and trajectories. The text is available in college bookstores and engineering libraries.

**AEROSPACE VEHICLE DESIGN** by K. D. Wood, published by Johnson Publishing Company of Boulder, Colorado. This is a design manual that contains practical charts and graphs of lift and drag of aerodynamic shapes, that can be used by model rocketeers to extend their design talents. Available in many college libraries.

**FLUID DYNAMIC DRAG** by Dr. Ing. S. F. Hoerner is published only by the author from 1438 Busted Drive, Midland, New Jersey. This is a classic collection of information on drag. Drag data ranging from flags to falling bodies, and from buildings to bullets are contained in this comprehensive book. Technical libraries have copies of the manual.



## Appendix III

### List of Symbols

- **A**: Area of an aerodynamic shape in square feet
- **\(C_D\)**: Drag coefficient of an aerodynamic shape, defined by Eq. (5)
- **\(C_{DB}\)**: Base drag coefficient defined by Eq. (9)
- **\(C_{DC}\)**: Drag coefficient of basic rocket parts, defined in Eq. (6)
- **\(C_{DF}\)**: Drag coefficient of fins
- **\(C_{DN}\)**: Drag coefficient of nose cone
- **\(C_{DO}\)**: Drag coefficient of complete rocket at zero angle of attack, defined by Eq. (7)
- **\(C_{DiF}\)**: Drag coefficient of fins due to induced drag, defined in Eq. (14)
- **\(C_{DOB}\)**: Drag coefficient of the rocket body at zero angle of attack, defined in Eq. (10)
- **\(C_{DOF}\)**: Drag coefficient of the fins at zero angle of attack
- **\(C_{DOF}^*\)**: Drag coefficient of the fins at zero angle of attack, based on fin surface area, \( S_F \)
- **\(C_{D}\)**: Drag coefficient of complete rocket at angle of attack
- **\(C_{D \lt 1/4 B}\)**: Drag coefficient of rocket body at angle of attack
- **\(C_{Dint}\)**: Drag coefficient due to fin and body interference, defined by Eq. (20)
- **\(C_{DLL}\)**: Drag coefficient of the launch lug, defined in Eq. (23)
- **\(C_f\)**: Skin friction coefficient due to boundary layer found from Fig. A-1 and used in Eq. (8)
- **\(C_L\)**: Lift coefficient generated by fins
- **D**: Drag force in pounds, defined in Eq. (1)
- **L**: Length of rocket
- **\(L_N\)**: Length of nose cone
- **\(L_{BT}\)**: Length of body tube
- **M**: Mass of any body, measured in slugs
- **R**: Radius of a body
- **S**: Surface area in square feet
- **\(S_F\)**: Surface area of all fins
- **\(S_w\)**: Wetted surface area of the entire rocket; Fig. A-3 has values for ogive and conical noses as well as cylinders.
- **\(S_{BT}\)**: Cross-sectional area of the body tube, this is the usual reference area for rocket drag coefficient calculations.
- **\(S_{SF}\)**: Surface area of a single fin
- **V**: Rocket velocity in ft/sec
- **\(V_T\)**: Terminal velocity, calculated with Eq. (25)
- **W**: Weight of a body in pounds
- **AR**: Aspect ratio of fins, defined in Eq. (11)
- **RN**: Reynolds number, used in Fig. A-1 to find the \( C_f \). Defined in Eq. (4)
- **SS**: Shearing stress, defined in Eq. (3), measured in pounds per sq. ft.
- **b**: Span of the fins of the rocket
- **d**: Diameter of the rocket
- **\(d_b\)**: Diameter of the base of the rocket
- **g**: Acceleration of a falling body due to gravitational force; equal to 32.2 ft/sec² on Earth at sea level.
- **\(t/c\)**: Thickness ratio of the fin; thickness divided by chord
- **\(\alpha\)**: Angle of attack of the fin or rocket body, that is the angle between the fin and the oncoming air stream.
- **\(\rho\)**: Density of the air
- **\(P_{SL}\)**: Density of air at sea level on a "standard day", \( SL = 0.002378 \) slugs/ft³
- **\(\Lambda\)**: Sweep angle of the fin
- **\(\Theta\)**: Boat-tail angle
- **\(\lambda\)**: Fin taper ratio, the ratio of the tip chord to the root chord
- **\(\mu\)**: Viscosity of the air
