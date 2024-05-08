% clc
clear
close all
set(0,'defaulttextinterpreter','latex')

%% Frame definitions:
% N: inertial frame
% H: HCW frame, based on a Keplerian body. Target starts at origin of HCW
% T: Servicer (Tug), also refered to as 1
% D: Debris, rotations between N and T given by DCM_NT, aka 2
% S: Spherical frame, servicer starts at origin of S frame


%% Orbits, trajectories and MSM models
mu = constants.muEarth;
params.mu = mu;

params.V = [25e3; -25e3]; %Craft voltages

params.chargeEstError = 'off';
params.voltageError = 0.0e3;
params.trackingOmega = 'true'; %just track body rates of target (perfect attitude tracking)
params.prescribedAtt = 'on';

%% Load MSM models and mass props

params.COM1 = [0; 0; 0];
params.COM2 = [0; 0; 0];

params.m1 = 500; % [kg] servicer mass
params.m2 = 2000; % [kg] debris mass

params.I1 = [1; 1; 1]*2/5*1^2*params.m1.*eye(3);
params.I2 = [1; 1; 1]*2/5*1^2*params.m2.*eye(3);

params.SPHS1 = [0 0 0 2]'; % sphere with radius 2m at origin of body frame
params.SPHS2 = [0 0 0 3]';

%% Make reference trajectory
Lr = 30; % reference separation
L0 = 20; % initial separation

r0_DT_N = [2; -L0; -3]/1000; % relative position between debris and servicer

%Define GEO reference orbit
a = 42164; % [km] semimajor axis

[r0_TN_N, v0_TN_N] = OE2SV(a,10,30,20,40,0); %( a,inclination,argPer,RAAN,anomaly,enorm,dT,type)
r0_DN_N = r0_TN_N + r0_DT_N;
v0_DN_N = v0_TN_N;
X0_TN_N = [r0_TN_N; v0_TN_N];
X0_DN_N = [r0_DN_N; v0_DN_N];

%% Attitdue ICs

omega_TH = [0 0 0]';
omega_DH = [0 0 0]';

beta_TN = angle2quat(0.4, -0.1, 0.2)';
beta_DN = [1 0 0 0]';

h_r = r0_TN_N/norm(r0_TN_N);
h_h = cross(r0_TN_N, v0_TN_N)/norm(cross(r0_TN_N, v0_TN_N));
h_theta = cross(h_h, h_r);
DCM_HN = [h_r'; h_theta'; h_h']; % DCM from N to H
DCM_TN = quat2dcm(beta_TN'); % DCM from N to T
DCM_DN = quat2dcm(beta_DN'); % DCM from N to D

DCM_TH = DCM_TN*DCM_HN';
beta_TH = dcm2quat(DCM_TH)';
DCM_DH = DCM_DN*DCM_HN';
beta_DH = dcm2quat(DCM_DH)';

%% UnitTest

Y0 = [X0_TN_N; beta_TN; omega_TH; X0_DN_N; beta_DN; omega_DH];

params.Lr = Lr;
params.X_r = [Lr; 0; 0];
params.Kgain = 4e-7;

[dy, u_H] = inertialDynamicsET(0.5, Y0, params);

u_N = DCM_HN'*u_H * 1000;
T_N = params.m1*u_N
u_T = DCM_TN*DCM_HN'*u_H * 1000;
T_T = params.m1*u_T

%% Integrated Test

% integration parameters
dT = 5; % step time
t = 5*3600; % simulation time
tSpan = 0:dT:t;
options = odeset('Maxstep', 10, 'RelTol', 1e-12, 'AbsTol', 1e-12);

% STRUCTURE FOR SV: 
% [1:3]-position 1          |   [14:16]-position 2
% [4:6]-velocity 1          |   [17:19]-Velocity 2
% [7:10]-quat 1             |   [20:23]-quat 2
% [11:13]-omega1            |   [24:26]-omega 2

dT = tSpan(2)-tSpan(1);

tic 

[~, statesPerturbed] = ode15s(@(t,y) inertialDynamicsET(t,y,params),tSpan,Y0,options);

pos1Pert = statesPerturbed(:,1:3)*1000;
pos2Pert = statesPerturbed(:,14:16)*1000;
relPos1to2Electro = pos1Pert-pos2Pert;
omegaTarget = statesPerturbed(:, 24:26);

fprintf('Final target rot rate (Charge Estimation Error): %f deg/s \n', vecnorm(omegaTarget(end,:)*180/pi) )
fprintf('Final position with Charge Estimation Error: %f meters\n',norm(pos1Pert(end,:)-pos2Pert(end,:)))

params.chargeEstError = 'off';

[~, statesUnpert] = ode15s(@(t,y) inertialDynamicsET(t,y,params),tSpan,Y0,options);%

pos1Unpert = statesUnpert(:,1:3)*1000;
pos2Unpert = statesUnpert(:,14:16)*1000;

fprintf('Final target rot rate (no Charge Estimation Error): %f deg/s \n', vecnorm(statesUnpert(end, 24:26)*180/pi) )
fprintf('Final position without Charge Estimation Error: %f meters\n',norm(pos1Unpert(end,:)-pos2Unpert(end,: )))
toc

figure
plot(tSpan/3600, vecnorm(pos1Pert-pos2Pert, 2, 2))

%% Functions

function [dy, uTot_H, aElec_H, F1_H, F2_H, L1_H, L2_H, qs] = inertialDynamicsET(t, u, params)
% STRUCTURE FOR U: 
% [1:3]-position 1          |   [14:16]-position 2
% [4:6]-velocity 1          |   [17:19]-Velocity 2
% [7:10]-quat 1             |   [20:23]-quat 2
% [11:13]-omega1            |   [24:26]-omega 2
% 1 IS SERVICER                 2 IS TARGET

mu = params.mu;

% ET control according to﻿http://arc.aiaa.org/doi/10.2514/1.56118

r_TN_N = u(1:3);
v_TN_N = u(4:6);
r_DN_N = u(14:16);
v_DN_N = u(17:19);

a = norm(r_TN_N);
n = sqrt(mu/a^3);      

h_r = r_TN_N/norm(r_TN_N);
h_h = cross(r_TN_N, v_TN_N)/norm(cross(r_TN_N, v_TN_N));
h_theta = cross(h_h, h_r);
DCM_HN = [h_r'; h_theta'; h_h'];

rho_N = (r_DN_N-r_TN_N)*1000; % position from servicer to the target        
rhoDot_N = (v_DN_N-v_TN_N)*1000; % inertial relative velocity

omega_HN = cross(r_TN_N, v_TN_N) / (norm(r_TN_N).^2); % angular velocity
rho_H = DCM_HN*rho_N;
rhoPrime_H = DCM_HN*(rhoDot_N - cross(omega_HN, rho_N));

L = norm(rho_H);
theta = atan2(rho_H(1),-rho_H(2));
phi = asin(-rho_H(3)/L);
X = [L theta phi]'; % in spherical (S) coordinate frame

XDot = [cos(phi)*sin(theta) -cos(theta)*cos(phi) -sin(phi);...
         (cos(theta)*sec(phi))/L (sec(phi)*sin(theta))/L 0;...
         -(sin(theta)*sin(phi))/L (cos(theta)*sin(phi))/L -cos(phi)/L]*rhoPrime_H;
Xsv = [X; XDot];

Fmtrx = [1/4*Xsv(1)*(n^2*(-6*cos(2*Xsv(2))*cos(Xsv(3))^2 + 5*cos(2*Xsv(3)) + 1) + 4*Xsv(5)*cos(Xsv(3))^2*(2*n+Xsv(5))+4*Xsv(6)^2);...
         (3*n^2*sin(Xsv(2))*cos(Xsv(2)) + 2*Xsv(6)*tan(Xsv(3))*(n+Xsv(5)))-2*Xsv(4)/Xsv(1)*(n+Xsv(5));...
         1/4*sin(2*Xsv(3))*(n^2*(3*cos(2*Xsv(2)) - 5) - 2*Xsv(5)*(2*n + Xsv(5))) - 2*Xsv(4)/Xsv(1)*Xsv(6)];
Gmtrx = [1 0 0; 0 1/(L*cos(phi)) 0; 0 0 -1/L];

Kgain = params.Kgain;
Pgain = 1.85*(Kgain)^0.5;
Kmtrx = Kgain*diag([1,1,1]);
Pmtrx = Pgain*diag([1,1,1]);

% DCM from Hill (H) frame to Spherical (S) frame
DCM_SH = [cos(Xsv(3))*sin(Xsv(2)) -cos(Xsv(2))*cos(Xsv(3)) -sin(Xsv(3));...
           cos(Xsv(2)) sin(Xsv(2)) 0;...
           sin(Xsv(2))*sin(Xsv(3)) -cos(Xsv(2))*sin(Xsv(3)) cos(Xsv(3))];

X_r = params.X_r;
u_S = inv(Gmtrx)*(-Pmtrx*XDot-Kmtrx*(X-X_r)-Fmtrx);
u_H = DCM_SH'*u_S/1000;

DCM_TH = quat2dcm(u(7:10)'); % DCM from Hill to Servicer
DCM_DH = quat2dcm(u(20:23)');
r_TD_H = DCM_HN*(u(1:3)-u(14:16))*1000; % distance from debris to servicer in Hill [m]
[F1_H, F2_H, L1_H, L2_H, qs] = multisphereFT( params.SPHS1, params.SPHS2, r_TD_H, params.V, DCM_TH', DCM_DH', params.COM1, params.COM2 ); %computes forces and torques

if isfield(params,'chargeEstError') && strcmp(params.chargeEstError, 'on')
    [ F1_H_est ] = multisphereFT( params.SPHS1, params.SPHS2, r_TD_H, params.V + [0; params.voltageError], DCM_TH', DCM_DH', params.COM1, params.COM2 ); %computes forces and torques
    uTot_H = - u_H - F1_H_est*(1/params.m1 + 1/params.m2)/1000;
else
    uTot_H = - u_H - F1_H*(1/params.m1 + 1/params.m2)/1000;
end
    
if norm(F1_H) > 1 || norm(F2_H) > 1  %check for higher forces than expected
    F1_H = F1_H/norm(F1_H);
    F2_H = F2_H/norm(F2_H);
    fprintf('Forces greater than threshold, likely singular case at t= %d \n',t)
    disp(F1_H)
end

aElec_H(1:3) = F1_H/(params.m1*1000);% Acceleration due to electrostatic perturbations in km/s
aElec_H(4:6) = F2_H/(params.m2*1000);
LElec_H(1:6) = [L1_H; L2_H]; % Add torque

uTot_N = DCM_HN'*uTot_H;
aElec_N(1:3) = DCM_HN'*aElec_H(1:3)';
aElec_N(4:6) = DCM_HN'*aElec_H(4:6)';

% SC1 translation-SERVICER
x = u(1); 
y = u(2);
z = u(3);
xdot = u(4); 
ydot = u(5); 
zdot = u(6);

r = (x^2 + y^2 + z^2)^0.5;
xddot = -mu*x/(r^3) + aElec_N(1) + uTot_N(1);
yddot = -mu*y/(r^3) + aElec_N(2) + uTot_N(2);
zddot = -mu*z/(r^3) + aElec_N(3) + uTot_N(3);

dX_T = [xdot; ydot; zdot; xddot; yddot; zddot];

% Find Target translation
x = u(14); 
y = u(15);
z = u(16);
xdot = u(17);
ydot = u(18);
zdot = u(19);

r = (x^2 + y^2 + z^2)^0.5;
xddot = -mu*x/(r^3) + aElec_N(4);
yddot = -mu*y/(r^3) + aElec_N(5);
zddot = -mu*z/(r^3) + aElec_N(6);

dX_D = [xdot; ydot; zdot; xddot; yddot; zddot];

% rotational dynamics
beta_TH = u(7:10); %EP of Servicer
beta_TH = beta_TH/norm(beta_TH);
beta_DH = u(20:23);
beta_DH = beta_DH/norm(beta_DH); %quaterions of SC2
% omega1 = u(11:13); %rotational rate of SC1
omega_D = u(24:26); %rotational rate of target

B =@(beta) [-beta(2) -beta(3) -beta(4);
            beta(1) -beta(4) beta(3);
            beta(4) beta(1) -beta(2);
            -beta(3) beta(2) beta(1)];
betaDotD = .5*B(beta_DH)*omega_D;

omegaTDot = params.I2\(cross(-omega_D, params.I2*omega_D) + LElec_H(4:6)') ;
    
if strcmp(params.trackingOmega,'true')
    omega_T = omega_D; %assuming att alg tracks target's rotational rates
    omegaSDot = omegaTDot;
else
    omega_T = u(11:13);
    omegaSDot = params.I1\(cross(-omega_T, params.I1*omega_T) + LElec_H(1:3)');
end
    betaDotT = .5*B(beta_TH)*omega_T;

if isfield(params,'prescribedAtt') && strcmp(params.prescribedAtt, 'on')
    betaDotT = betaDotT*0;
    omegaSDot = omegaSDot*0;
    betaDotD = betaDotD*0;
    omegaTDot = omegaTDot*0;
end

% derivative of states
dy = [dX_T; betaDotT; omegaSDot; dX_D; betaDotD; omegaTDot];

end

function [ r,v ] = OE2SV( a,inclination,argPer,RAAN,theta,enorm)
%orbital elements at initial time
% a = r1; %semi-major axis [kilometers] 
% enorm  = 0; %eccentricity. 
% inclination = 0; %inclination [degrees]
% RAAN = 0; %  right ascension of the ascending node [degrees]
% argPer = 0; % argument of perigee [degrees]
% theta = 0; %True anomaly [degrees]

if nargin == 1
    OE = a;
    a = OE(1);
    inclination = OE(2);
    argPer = OE(3);
    RAAN = OE(4);
    theta = OE(5);
    enorm = OE(6);
end

mu = constants.muEarth;

%converting orbital parameters to ECI cartsian coordinates 
p = a*(1-enorm^2); %intermediate variable
q = p/(1+enorm*cosd(theta));%intermediate variable

% Creating r vector in pqw coordinates
R_pqw(1,1) = q*cosd(theta);
R_pqw(2,1) = q*sind(theta);
R_pqw(3,1) = 0;
    
% Creating v vector in pqw coordinates    
V_pqw(1,1) = -(mu/p)^.5*sind(theta);
V_pqw(2,1) = ((mu/p)^.5)*(enorm + cosd(theta));
V_pqw(3,1) =   0;

%Solving for 313 rotation matrices
R1_i = [1 0 0; 0 cosd(inclination) sind(inclination); 0 -sind(inclination) cosd(inclination)];
R3_Om = [cosd(RAAN) sind(RAAN) 0; -sind(RAAN) cosd(RAAN) 0; 0 0 1];
R3_om = [cosd(argPer) sind(argPer) 0; -sind(argPer) cosd(argPer) 0; 0 0 1];

support_var = R3_om*R1_i*R3_Om; %Intermediate variable

r = support_var'*R_pqw; %Radius r [km] in ECI Cartesian
v = support_var'*V_pqw; %Velocity v [km/s] in ECI Cartesian


end

function [ F1, F2, L1, L2, qs, overlapFlag] = multisphereFT( SPHS1, SPHS2, r, V, C1, C2, COM1, COM2)
% Find force and torque between two MSM bodies
% Inputs:
% SPHS1 = [x1 x2 ...
%          y1 y2 ...
%          z1 z2 ...
%          R1 R2 ...]
%        (positions & radii of spheres composing body 1)
% SPHS2 = (same for body 2)
% C1/2 = DCM of body 1/2 rotation
% r = [xr;yr;zr] (position of body 2 origin relative to 1)
% V = [V1 V2] (voltages of bodies 1 and 2)
% COM1/2 = Center of mass to origin offset
%
% Outputs:
% F1, F2, L1, L2 = forces and torques (3x1 vectors) on body 1 and 2
% qs = vector of charges on each sphere

% assume zero rotations if not specified
if nargin < 5
    C1 = eye(3);
    C2 = eye(3);
    COM1 = [ 0 0 0 ]';   
    COM2 = [ 0 0 0 ]';
elseif nargin < 6
    C2 = eye(3);
    COM1 = [ 0 0 0 ]';   
    COM2 = [ 0 0 0 ]';
elseif nargin < 7
    COM1 = [ 0 0 0 ]';   
    COM2 = [ 0 0 0 ]';
end

% Coulomb's constant [Nm^2/C^2]
k = constants.Kc;

% number of spheres in body 1, body 2, total
n1 = size(SPHS1,2);
n2 = size(SPHS2,2);
n = n1 + n2; 

% rotate positions of spheres according to DCMs
SPHS1t = SPHS1;
SPHS1t(1:3,:) = C1*(SPHS1(1:3,:)) + r*ones(1,n1); 
SPHS2(1:3,:) = C2*(SPHS2(1:3,:)); %

% build matrix with all spheres
SPHS = [SPHS1t SPHS2];

% Find Cinv matrix
% needs to evaluate every sphere to find each charge
sph1 = repmat(SPHS(1:3,:),1,size(SPHS,2));
sph2 = repelem(SPHS(1:3,:),1,size(SPHS,2));

relPos = sum((sph1-sph2).^2,1).^0.5; %vecnorm, but 1/10 the time. Finds relative positions between each pair of spheres in the two bodies

relPos = reshape(relPos, n, n); %distance between each sphere: wil fill off-diagonal elements
Cinv = relPos + SPHS(4,:).*eye(n);

% check if any spheres are overlapping
lapCheck = relPos(n1+1:end, 1:n1) - 1.1*(SPHS1(4,:)+SPHS2(4,:)');
overlapFlag = 0;
if any(triu(lapCheck,1) < 0, 'all')
    overlapFlag = 1;
    %disp('Overlapping spheres')
%     figure; scatter3(SPHS1t(1,:),SPHS1t(2,:),SPHS1t(3,:)); hold on; scatter3(SPHS2(1,:),SPHS2(2,:),SPHS2(3,:),'filled');axis equal
% figure; makeSphsPicture_2craft( SPHS1t, SPHS2, [0 0 0], [0 0 0], V(1), eye(3), eye(3))
end


% figure; makeSphsPicture_2craft( SPHS1t, SPHS2, [0 0 0], [0 0 0], V(1), eye(3), eye(3)); view(3)

Cinv = Cinv.^-1;

% Find charge on each sphere
Vs = [V(1)*ones(n1,1); V(2)*ones(n2,1)];
qs = (k*Cinv)\Vs;

% seperate charges into body 1 & 2
qs1 = qs(1:n1);
qs2 = qs(n1+1:end);

% Find force
% F = k*q1*q2*r/r^3

%compute distances 
sph1 = repmat(SPHS1t(1:3,:), 1, size(SPHS2,2));
sph2 = repelem(SPHS2(1:3,:), 1, size(SPHS1t,2));

relPos = sum((sph1-sph2).^2,1).^0.5;
relVec = sph1-sph2;

Qs1 = repmat(qs1',1,length(qs2));
Qs2 = repelem(qs2',1,length(qs1));

Feach = k*Qs1.*Qs2./relPos.^3;
Feach = Feach.*relVec; %add direction

F1 = sum(Feach,2); %matches to eps preision
F2 = -F1;

if norm(F1) >1
    %disp('High fores')
end
% Find torque
% L = r X f
%compute distances RELATIVE TO CENTER OF MASSES 
sph1 = repmat(SPHS1t(1:3,:) + COM1, 1, size(SPHS2,2));
sph22 = repelem(SPHS2(1:3,:) + COM2, 1, size(SPHS1t,2));

L1 = sum(cross(sph1,Feach),2); %matches L1 to eps
L2 = -sum(cross(sph22,Feach),2); % matches L2 to eps-ish

if norm(abs(L1)-abs(L2)) < eps() && nnz(V)>1
    %disp('Torques are equal')
end

end






