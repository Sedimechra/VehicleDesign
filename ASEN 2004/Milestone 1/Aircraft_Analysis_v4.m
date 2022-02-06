%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%                   LAB 1 - ASEN 2004                     %
%                   Aircraft Analysis                     %
%                                                         %
%                                                         %
%        This script analyzes *SOMETHING* for the         %
%                    B747 and Tempest UAS.                %
%                                                         %
%                      01/20/2021                         %
%                                                         %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set Environment
clear
clc

%% Declare and Define the B747 Variables

B747_AR = 7;                        % Boeing 747 Aspect Ratio [unitless]
B747_Re = 1000000;                  % Boeing 747 Reynold's Number [unitless]
B747_e0 = -0.339;                   % Boeing 747 Oswald's Efficieny
B747_C_fe = 0.0030;                 % Civil Transport Skin Friction Coefficient
B747_S_wet = 3320.25;               % Boeing 747 Wet Planform Area [m^2]
B747_S_ref = 511;                   % Boeing 747 Reference Planform Area [m^2]
B747_d = 6.5;                       % Boeing 747 Fuselage Diameter [m]
B747_b = 59.64;                     % Boeing 747 Wing Span
B747_W = 3.559*10^6;                % Boeing 747 weight [N]
B747_h = 10500;                     % Boeing 747 cruise altitude [m]
B747_rho = 0.38857;                 % Air density STD ATM at B747_h [kg/m^3]

% Create the Variable Array for the B747
B747_vars = [B747_AR,B747_Re,B747_e0,B747_S_wet,B747_S_ref,B747_C_fe,B747_d,B747_b];

% Boeing 747  Table Data [AOA C_l C_d]
B747 = [-5 -0.3505 0.00953;
        -4 -0.2514 0.00852;
        -3 -0.1493 0.00741;
        -2 -0.0318 0.00584;
        -1 0.0912 0.0055;
         0 0.2133 0.00509;
         1 0.3048 0.00506;
         2 0.318  0.00743;
         3 0.3925 0.00948;
         4 0.4889 0.01033;
         5 0.5855 0.01132;
         6 0.6808 0.01256;
         7 0.7744 0.01441;
         8 0.8721 0.01634;
         9 0.9729 0.02052;
        10 1.061  0.02498;
        11 1.1263 0.03112;
        12 0.9165 0.07156;
        13 0.7781 0.14138];
    
% B747 CFD Drag Polar [AOA C_L C_D]
B747Act = [1 2];
    
%% Declare and Define the Tempest UAS Variables

Tempest_Re = 200000;                % Tempest UAS Reynold's Number [dimensionless]
Tempest_AR = 16.5;                  % Tempest UAS Aspect Ratio [dimensionless]
Tempest_e0 = 0.601;                 % Tempest UAS Oswald's Efficiency 
Tempest_C_fe = 0.0055;              % Civil Transport Skin Friction Coefficient
Tempest_S_wet = 1.87453;            % Tempest UAS Wet Planform Area [m^2]
Tempest_S_ref = 0.63;               % Tempest UAS Reference Planform Area [m]
Tempest_d = 0.16;                   % Tempest UAS Fuselage Diameter [m]
Tempest_b = 3.22;                   % Tempest UAS Wing Span [m^2]
Tempest_W = 62.78;                  % Tempest UAS weight [N]
Tempest_h = 1500;                   % Tempest UAS cruise altitude [m]
Tempest_rho = 1.0581;               % Air density STD ATM at Tempest_h [kg/m^3]

% Create the Variable Array for the Tempest
Tempest_vars = [Tempest_AR, Tempest_Re, Tempest_e0, Tempest_C_fe, Tempest_S_wet, Tempest_S_ref, Tempest_d, Tempest_b];

% Tempest UAS Table Data [AOA C_l C_d]
Tempest = [-5 -0.4166 0.04049;
            -4 -0.2734 0.02;
            -3 -0.125 0.01439;
            -2 0.0032 0.01054;
            -1 0.2136 0.00976;
             0 0.3312 0.00933;
             1 0.4263 0.00906;
             2 0.5241 0.00898;
             3 0.6236 0.00928;
             4 0.7217 0.0101;
             5 0.8165 0.01133;
             6 0.9059 0.01314;
             7 0.9889 0.01573;
             8 1.0582 0.02012;
             9 1.1042 0.02723;
            10 1.1555 0.03641;
            11.25 1.1303 0.05193;
            12 1.097 0.06243];

% Tempest CFD Drag Polar [AOA C_L C_D]
TempestAct = [-5	-0.32438	0.044251
              -4	-0.21503	0.033783
              -3	-0.10081	0.028627
              -2	0.010503	0.025864
              -1	0.12155	    0.024643
               0	0.24163	    0.025099
               1	0.34336	    0.025635
               2	0.45256	    0.02766
               3	0.56037	    0.030677
               4	0.66625	    0.034855
               5	0.76942	    0.040403
               6	0.86923	    0.04759
               7	0.96386	    0.057108
               8	1.0441	    0.070132
               9	1.0743	    0.090921
              10	1.0807	    0.11193
              11	1.0379	    0.13254
              12	1.034	    0.15645];

          
%% Boeing 747 Calculations

% Calculate the C_L and C_D for the B747 Finite Wing
[B747_C_L,B747_C_D_w] = finiteWingLift(B747,B747_vars,2,10);

% Calculate the Whole Drag Polar for the Boeing 747 using labDocument method
eLabDocument = labDocument(B747_vars);
B747_C_D_labDocument = wholeAircraftDragPolar(B747_C_L, B747_C_D_w, eLabDocument, B747_vars);

% Calculate the Whole Drag Polar for the Boeing 747 using MNita_DScholz method
eMNita_DScholz = MNita_DScholz(B747_C_D_w, 0.9, B747_vars);
B747_C_D_MNita_DScholz = wholeAircraftDragPolar(B747_C_L, B747_C_D_w, eMNita_DScholz, B747_vars);

% Calculate the Whole Drag Polar for the Boeing 747 using Oberts method
eOberts = oberts(B747_vars);
B747_C_D_Oberts = wholeAircraftDragPolar(B747_C_L, B747_C_D_w, eOberts, B747_vars);

% Calculate the Whole Drag Polar for the Boeing 747 using Kroo's method
eKroos = kroos(B747_C_D_w, 0.9, B747_vars);
B747_C_D_Kroos = wholeAircraftDragPolar(B747_C_L, B747_C_D_w, eKroos, B747_vars);

% Combine all Oswald's factors and C_Ds in one array for performance calcs
B747_e0 = [eKroos,eLabDocument,eMNita_DScholz,eOberts];
B747_C_D = [B747_C_D_Kroos,B747_C_D_labDocument,B747_C_D_MNita_DScholz,B747_C_D_Oberts];

figure;
hold on
plot(B747(:,1),B747_C_L);
plot(B747(:,1),B747(:,2));
title('Lift Curve Comparsion: B747-200');
xlabel('α (Angle of Attack)');
ylabel('C_L (Coefficient of Lift)');
legend('3-D Finite Wing Curve', '2-D Airfoil Curve', 'Location', 'best');
hold off

figure;
hold on
plot(B747_C_L,B747_C_D_labDocument);
plot(B747_C_L,B747_C_D_MNita_DScholz);
plot(B747_C_L,B747_C_D_Oberts);
plot(B747_C_L,B747_C_D_Kroos);
title('Whole Aircraft Drag Polar Comparsion: B747-200');
xlabel('C_L (Coefficient of Lift)');
ylabel('C_D (Coefficient of Drag)');
legend('C_D vs C_L using labDocument Oswalds', ...
       'C_D vs C_L using M. Nita D. Scholz Oswald', ...
       'C_D vs C_L using Oberts Oswald', ...
       'C_D vs C_L using Kroos Oswald', 'Location', 'best');
hold off

%% Tempest UAS Calculations

[Tempest_C_L,Tempest_C_D_w] = finiteWingLift(TempestAct,Tempest_vars,0,6);

% Calculate the Whole Drag Polar for the Tempest UAS using labDocument method
eLabDocument = labDocument(Tempest_vars);
Tempest_C_D_labDocument = wholeAircraftDragPolar(Tempest_C_L, Tempest_C_D_w, eLabDocument, Tempest_vars);

% Calculate the Whole Drag Polar for the Tempest UAS using MNita_DScholz method
eMNita_DScholz = MNita_DScholz(Tempest_C_D_w, 0.9, Tempest_vars);
Tempest_C_D_MNita_DScholz = wholeAircraftDragPolar(Tempest_C_L, Tempest_C_D_w, eMNita_DScholz, Tempest_vars);

% Calculate the Whole Drag Polar for the Tempest UAS using Oberts method
eOberts = oberts(Tempest_vars);
Tempest_C_D_Oberts = wholeAircraftDragPolar(Tempest_C_L, Tempest_C_D_w, eOberts, Tempest_vars);

% Calculate the Whole Drag Polar for the Tempest UAS using Kroo's method
eKroos = kroos(Tempest_C_D_w, 0.9, Tempest_vars);
Tempest_C_D_Kroos = wholeAircraftDragPolar(Tempest_C_L, Tempest_C_D_w, eKroos, Tempest_vars);

% Combine all Oswald's factors and C_Ds in one array for performance calcs
Tempest_e0 = [eKroos,eLabDocument,eMNita_DScholz,eOberts];
Tempest_C_D = [Tempest_C_D_Kroos,Tempest_C_D_labDocument,Tempest_C_D_MNita_DScholz,Tempest_C_D_Oberts];

figure;
hold on
plot(Tempest(:,1), Tempest_C_L);
plot(Tempest(:,1), Tempest(:,2));
plot(TempestAct(:,1), Tempest(:,2),'-.r*');
title('Lift Curve Comparsion: Tempest UAS');
xlabel('α (Angle of Attack)');
ylabel('C_L (Coefficient of Lift)');
legend('3-D Finite Wing Curve', '2-D Airfoil Curve', 'Tempest CFD Drag Polar', 'Location', 'best');
hold off

figure;
hold on
plot(Tempest_C_L,Tempest_C_D_labDocument);
plot(Tempest_C_L,Tempest_C_D_MNita_DScholz);
plot(Tempest_C_L,Tempest_C_D_Oberts);
plot(Tempest_C_L,Tempest_C_D_Kroos);
title('Whole Aircraft Drag Polar Comparsion: Tempest UAS');
xlabel('C_L (Coefficient of Lift)');
ylabel('C_D (Coefficient of Drag)');
legend('C_D vs C_L using labDocument Oswalds', ...
       'C_D vs C_L using M. Nita D. Scholz Oswald', ...
       'C_D vs C_L using Oberts Oswald', ...
       'C_D vs C_L using Kroos Oswald', 'Location', 'best');
hold off


%% Performace Flight Conditions Calculations

% Define empty arrays for max endurance glide ranges and velocities
Tempest_Glide_End_R = zeros(1,4);
Tempest_Glide_End_V = zeros(1,4);
B747_Glide_End_R = zeros(1,4);
B747_Glide_End_V = zeros(1,4);

% Define empty arrays for powered LDmaxes
Tempest_Glide_LDMax = zeros(1,4);
B747_Glide_LDMax = zeros(1,4);

% Define empty arrays for max range glide ranges and velocities
Tempest_Glide_Range_R = zeros(1,4);
Tempest_Glide_Range_V = zeros(1,4);
B747_Glide_Range_R = zeros(1,4);
B747_Glide_Range_V = zeros(1,4);

% Define empty arrays for max endurance powered velocities
Tempest_Powered_End_V = zeros(1,4); % maximum powered endurance
B747_Powered_End_V = zeros(1,4);

% Define empty arrays for max range powered velocities
Tempest_Powered_Range_V = zeros(1,4); % maximum powered range
B747_Powered_Range_V = zeros(1,4);

% Define empty arrays for powered LDmaxes
Tempest_Powered_LDmax = zeros(1,4);
B747_Powered_LDMax = zeros(1,4);

% Tempest performance calculations for each Oswald's using calculated data
for i=1:4
   [Tempest_Glide_LDMax(i),Tempest_Glide_End_R(i),Tempest_Glide_Range_R(i),...
       Tempest_Glide_End_V(i),Tempest_Glide_Range_V(i)] = ...
       glide(Tempest_h,Tempest_C_D(:,i),Tempest_C_L,Tempest_e0(i),...
       Tempest_AR,Tempest_rho,Tempest_W,Tempest_S_ref); 
   [Tempest_Powered_LDmax(i),Tempest_Powered_End_V(i),...
       Tempest_Powered_Range_V(i)] = poweredProp(Tempest_C_D(:,i),...
       Tempest_C_L,Tempest_e0(i),Tempest_AR,Tempest_rho,Tempest_W,...
       Tempest_S_ref);
end

% Tempest performance calculations for avg Oswald's using given data
[TempestAct_Glide_LDMax,TempestAct_Glide_End_R,TempestAct_Glide_Range_R,...
    TempestAct_Glide_End_V,TempestAct_Glide_Range_V] = glide(Tempest_h,...
    TempestAct(:,3),TempestAct(:,2),mean(Tempest_e0),Tempest_AR,...
    Tempest_rho,Tempest_W,Tempest_S_ref);
[TempestAct_Powered_LDMax,TempestAct_Powered_End_V,...
    TempestAct_Powered_Range_V] = poweredProp(TempestAct(:,3),...
    TempestAct(:,2),mean(Tempest_e0),Tempest_AR,Tempest_rho,Tempest_W,...
    Tempest_S_ref);

% B747 performance calculations for each Oswald's using calculated data
for i=1:4
    [B747_Glide_LDMax(i),B747_Glide_End_R(i),B747_Glide_Range_R(i),...
        B747_Glide_End_V(i),B747_Glide_Range_V(i)] = glide(B747_h,...
        B747_C_D(:,i),B747_C_L,B747_e0(i),B747_AR,B747_rho,B747_W,...
        B747_S_ref);
    [B747_Powered_LDMax(i),B747_Powered_End_V(i),...
        B747_Powered_Range_V(i)] = poweredJet(B747_C_D(:,i),B747_C_L,...
        B747_e0(i),B747_AR,B747_rho,B747_W,B747_S_ref);
end

% B747 performance calculations for avg Oswald's using given data
[B747Act_Glide_LDMax,B747Act_Glide_End_R,B747Act_Glide_Range_R,...
    B747Act_Glide_End_V,B747Act_Glide_Range_V] = glide(B747_h,...
    B747(:,3),B747(:,2),mean(B747_e0),B747_AR,B747_rho,B747_W,B747_S_ref);
[B747Act_Powered_LDMax,B747Act_Powered_End_V,B747Act_Powered_Range_V]...
    = poweredJet(B747(:,3),B747(:,2),mean(B747_e0),B747_AR,B747_rho,...
    B747_W,B747_S_ref);



%% Functions

% finiteWingLift
%
% Function that calculates the 3D finite wing 3D lift coefficient and 3D wing
% drag polar
%
% @param data   The table data that includes the AOA, C_l, and the C_d in
%                   that order.
% @param vars   Needed variables on a per aircraft process like Aspect ratio
%                   and oswald's number
% @param aoa_one The lowest AOA to evaluate at
% @param aoa_two The highest AOA to evaluate at
%
function [C_L,C_D_wing] = finiteWingLift(data,vars,aoa_one,aoa_two)
    % Define the variables
    AR = vars(1);       % Aspect Ratio
    
    % Get the slope of the linear portion of the 2D lift curve from airfoil data
    linearFit = polyfit(data(data(data(:,1)<aoa_two,1)>aoa_one,1),data(data(data(:,1)<aoa_two,1)>aoa_one,2),1);
    
    % Set the slope of 2D airfoil curve as a_0 according to convention
    a_0 = linearFit(1);
    
    % Define the Span Efficiency factor
    e = 0.9;
    
    % Define a the lift curve slope
    a = a_0/(1+((57.3*a_0)/(pi*e*AR)));
    
    % Get the C_l where the AOA equal 0
    alpha_aoa_0 = data(data(:,1)==0,2);
    
    % Calculate C_L for all AOA
    C_L = a*(data(:,1)-alpha_aoa_0)+linearFit(2);
    
    % Calculate C_D_wing
    C_D_wing = data(:,3) + (C_L.^2)/(pi*e*AR);
end

% wholeAircraftDragPolar
%
% Function that calculates the whole aircraft drag polar
%
% @param C_L        The calculated C_L values for a specific aircraft
% @param C_D_Wing   The calculated C_D_wing values for a specific aircraft
% @param e_0        The Oswald's Efficiency number for a specific model
% @param vars       Needed variables on a per aircraft process like Aspect ratio
%                       and oswald's number
%
function C_D = wholeAircraftDragPolar(C_L, C_D_wing, e_0, vars)
    % Define the variables
    AR = vars(1);       % Aspect Ratio
    S_wet = vars(4);    % Wet Planform Area
    S_ref = vars(5);    % Reference Planfrom Area
    C_f_e = vars(6);    % Coefficient of Skin Friction
    
    % Calculate K1
    k_1 = 1/(pi*e_0*AR);
    
    % Calculate C_D_min
    C_D_min = C_f_e * (S_wet/S_ref);
    
    % Calculate C_L_minD
    xmin = find(C_D_wing==min(C_D_wing));
    C_L_minD = C_L(xmin);
    
    % Calculate the Whole Aircraft Drag Polar
    C_D = C_D_min + (k_1*((C_L-C_L_minD).^2));
end

%% Oswald's Efficiency Calculation Functions

% labDocument
%
% Function that Oswald's number using the Lab Document method equation 12
%
% @param vars       Needed variables on a per aircraft process like Aspect Ratio
%                       and oswald's number
%
function e = labDocument(vars)
    % Define the variables
    AR = vars(1);       % Aspect Ratio
    
    % Calculate Oswald's factor
    e = (1.78*(1-(0.045*(AR^0.68)))) - 0.64;
end

% MNita_DScholz
%
% Function that Oswald's number using M. Nita D. Scholz method
%
% @param C_D        The Coefficient of Drag for Finite Wing
% @param e          The Span Efficieny Factor
% @param vars       Needed variables on a per aircraft process like Aspect ratio
%                       and oswald's number
%
function e = MNita_DScholz(C_D, e, vars)
    % Define the variables
    AR = vars(1);       % Aspect Ratio
    d = vars(7);        % Fuselage Diameter
    b = vars(8);        % Aircraft Wing Span
    K_e_m = 1;          % 1 because aircrafts are subsonic
    
    % Define k_e_f
    k_e_f = 1 - (2*((d/b)^2));
    
    % Find C_D_0
    C_D_0 = C_D(6); % AOA of 0 degrees
    
    % Define Q
    Q = 1 / (e*k_e_f);
    
    % Define P
    P = 0.38*C_D_0;
    
    % Calculate Oswald's factor
    e = K_e_m/(Q+(P*pi*AR));
end

% oberts
%
% Function that Oswald's number using Obert's method
%
% @param vars       Needed variables on a per aircraft process like Aspect ratio
%                       and oswald's number
%
function e = oberts(vars)
    % Define the variables
    AR = vars(1);       % Aspect Ratio

    % Calculate Oswald's factor
    e = 1/(1.05+(0.007*pi*AR));
end

% kroos
%
% Function that Oswald's number using Kroo's method
%
% @param C_D        The Coefficient of Drag for Finite Wing
% @param e          The Span Efficieny Factor
% @param vars       Needed variables on a per aircraft process like Aspect ratio
%                       and oswald's number
%
function e = kroos(C_D, e, vars)
    % Define the variables
    AR = vars(1);       % Aspect Ratio
    d = vars(7);        % Fuselage Diameter
    b = vars(8);        % Aircraft Wing Span
    K = 0.38;           % WTF is K?
    
    % Define u
    u = e;
    
    % Define S
    s = 1 - (2*((d/b)^2));
    
    % Define Q
    Q = 1/(u*s);
    
    % Find C_D_0
    C_D_0 = C_D(6); % AOA of 0 degrees
    
    % Define P
    P = K*C_D_0;
    
    % Calculate Oswald's Factor
    e = 1/((1/(u*s))+(P*pi*AR));
end

% glide

% Calculates L/Dmax, ranges and velocities for max endurance and max range
% glide

function [LDmax,R_end,R_max,V_end,V_range] = glide(h,C_D,C_L,e0,AR,rho,W,S)
% max endurance: at 3CD0 = kCL^2
% max range: at CD0 = kCL^2

    % define k
    k = 1/(AR*pi*e0);
    
    % find the C_L corresponding to 3CD0 = kCL^2 and CD0 = kCL^2 
    C_D0 = C_D - k*C_L.^2;
    max_C_L_end = sqrt(3*C_D0/k);
    max_C_L_range = sqrt(C_D0/k);
    [~,idx_end] = min(abs(C_L-max_C_L_end));
    [~,idx_range] = min(abs(C_L-max_C_L_range));
    C_L_end = C_L(idx_end);         % C_L for max endurance
    C_L_range = C_L(idx_range);     % C_L for max range
     
    % find corresponding coefficients of drag
    C_D_end = C_D(idx_end);
    C_D_range = C_D(idx_range);
    
    % calculate range
    R_end = h*C_L_end/C_D_end;
    R_max = h*C_L_range/C_D_range;
    
    % calculate glide angle
    theta_end = atan(h/R_end);
    theta_range = atan(h/R_max);
    
    % calculate L/D max:
    LDmax = 1/theta_range;  % L/Dmax at CD0 = kCL^2
    
    % combine two definitions for L to calculate freestream velocity V
    % L = Wcos(theta), L = C_L*q*S
    V_end = sqrt((2*cos(theta_end)*W)/(rho*C_L_end*S));
    V_range = sqrt((2*cos(theta_range)*W)/(rho*C_L_range*S));
end

% poweredJet

% Calculates L/Dmax and velocities for powered jet max endurance and max
% range

function [LDmax,V_end,V_range] = poweredJet(C_D,C_L,e0,AR,rho,W,S)
% max endurance: at CD0 = kCL^2
% max range: at CD0 = 3kCL^2

 % define k
    k = 1/(AR*pi*e0);
    
    % find the C_L corresponding to CD0 = 3kCL^2 and C_D0 = kCL^2
    C_D0 = C_D - k*C_L.^2;
    max_C_L_end = sqrt(C_D0/k);
    max_C_L_range = sqrt(C_D0/(k*3));
    [~,idx_end] = min(abs(C_L-max_C_L_end));
    [~,idx_range] = min(abs(C_L-max_C_L_range));
    C_L_end = C_L(idx_end);         % C_L for max endurance
    C_L_range = C_L(idx_range);     % C_L for max range
    
    % find corresponding coefficient of drag
    C_D_end = C_D(idx_end);
    
    % calculate LDmax for max range
    LDmax = C_L_end/C_D_end;        % L/Dmax at CD0 = kCL^2
        
    % combine two definitions for L to calculate freestream velocity V
    % L = W, L = C_L*q*S
    V_end = sqrt((2*W)/(rho*C_L_end*S));
    V_range = sqrt((2*W)/(rho*C_L_range*S));
end

% poweredProp

% Calculates L/Dmax and velocities for powered prop max endurance and max
% range

function [LDmax,V_end,V_range] = poweredProp(C_D,C_L,e0,AR,rho,W,S)
% max endurance: at 3CD0 = kCL^2
% max range: at CD0 = kCL^2

 % define k
    k = 1/(AR*pi*e0);
    
    % find the C_L corresponding to 3C_D0 = kCL^2 and CD0 = kCL^2
    C_D0 = C_D - k.*C_L.^2;
    max_C_L_end = sqrt(3*C_D0/k);
    max_C_L_range = sqrt(C_D0/k);
    [~,idx_end] = min(abs(C_L-max_C_L_end));
    [~,idx_range] = min(abs(C_L-max_C_L_range));
    C_L_end = C_L(idx_end);         % C_L for max endurance
    C_L_range = C_L(idx_range);     % C_L for max range
    
    % find corresponding coefficient of drag
    C_D_range = C_D(idx_range); 
    
    % calculate LDmax for max range
    LDmax = C_L_range/C_D_range;    % L/Dmax at CD0 = kCL^2
    
    % combine two definitions for L to calculate freestream velocity V
    % L = W, L = C_L*q*S
    V_end = sqrt((2*W)/(rho*C_L_end*S));
    V_range = sqrt((2*W)/(rho*C_L_range*S));
end

