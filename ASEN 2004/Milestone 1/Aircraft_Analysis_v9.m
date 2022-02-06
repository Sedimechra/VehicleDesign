%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                         %
%                   LAB 1 - ASEN 2004                     %
%                   Aircraft Analysis                     %
%                                                         %
%                                                         %
%        This script takes in 2-dimensional data for      %
%       the Boeing 747 and Tempest UAS and calculates     %
%        the 3-dimensional whole aircraft drag polar,     %
%        as well as performance analysis calculations     %
%       for maximum glide range, maximum powered range,   %
%             and maximum powered endurance.              %
%                                                         %
%                  Created: 01/20/2021                    %
%               Last Modified: 01/30/2021                 %
%                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set EnvironmentF
clear
clc
close all
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

% Boeing 747 Table Data [AOA C_l C_d]
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
    
% B747 CFD Drag Polar [C_L C_D]
B747Act = [0.5922	0.0342;
            0.5683	0.0323;
            0.5473	0.0306;
            0.5240	0.0290;
            0.4962	0.0273;
            0.4724	0.0259;
            0.4486	0.0247;
            0.4219	0.0232;
            0.3981	0.0224;
            0.3719	0.0212;
            0.3486	0.0203;
            0.3234	0.0194;
            0.3001	0.0186;
            0.2735	0.0182;
            0.2493	0.0174;
            0.2236	0.0171;
            0.2009	0.0169;
            0.1738	0.0165;
            0.1525	0.0165;
            0.1255	0.0163;
            0.1037	0.0165;
            0.0781	0.0166;
            0.0550	0.0168;
            0.0285	0.0173;
            0.0039	0.0180];
    
%% Declare and Define the Tempest UAS Variables

Tempest_Re = 200000;                % Tempest UAS Reynold's Number [dimensionless]
Tempest_AR = 16.5;                  % Tempest UAS Aspect Ratio [dimensionless]
Tempest_e0 = 0.601;                 % Tempest UAS Oswald's Efficiency 
Tempest_C_fe = 0.0055;              % Civil Transport Skin Friction Coefficient
Tempest_S_wet = 1.87453;            % Tempest UAS Wet Planform Area [m^2]
%Tempest_S_wet = 3.3;            % Tempest UAS Wet Planform Area [m^2]
Tempest_S_ref = 0.63;               % Tempest UAS Reference Planform Area [m]
Tempest_d = 0.16;                   % Tempest UAS Fuselage Diameter [m]
Tempest_b = 3.22;                   % Tempest UAS Wing Span [m^2]
Tempest_W = 62.78;                  % Tempest UAS weight [N]
Tempest_h = 1500;                   % Tempest UAS cruise altitude [m]
Tempest_rho = 1.0581;               % Air density STD ATM at Tempest_h [kg/m^3]

% Create the Variable Array for the Tempest
Tempest_vars = [Tempest_AR, Tempest_Re, Tempest_e0, Tempest_S_wet, Tempest_S_ref, Tempest_C_fe, Tempest_d, Tempest_b];

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
B747_eLabDocument = labDocument(B747_vars);
[B747_C_D_labDocument,B747_C_D_0_labDocument] = wholeAircraftDragPolar(B747_C_L, B747_C_D_w, B747_eLabDocument, B747_vars);

% Calculate the Whole Drag Polar for the Boeing 747 using MNita_DScholz method
B747_eMNita_DScholz = MNita_DScholz(B747_C_D_w, 0.9, B747_vars);
[B747_C_D_MNita_DScholz,B747_C_D_0_MNita_DScholz] = wholeAircraftDragPolar(B747_C_L, B747_C_D_w, B747_eMNita_DScholz, B747_vars);

% Calculate the Whole Drag Polar for the Boeing 747 using Oberts method
B747_eOberts = oberts(B747_vars);
[B747_C_D_Oberts,B747_C_D_0_Oberts] = wholeAircraftDragPolar(B747_C_L, B747_C_D_w, B747_eOberts, B747_vars);

% Calculate the Whole Drag Polar for the Boeing 747 using Kroo's method
B747_eKroos = kroos(B747_C_D_w, 0.9, B747_vars);
[B747_C_D_Kroos,B747_C_D_0_Kroos] = wholeAircraftDragPolar(B747_C_L, B747_C_D_w, B747_eKroos, B747_vars);

% Combine all Oswald's factors and C_Ds in one array for performance calcs
B747_e0 = [B747_eKroos,B747_eLabDocument,B747_eMNita_DScholz,B747_eOberts];
B747_C_D = [B747_C_D_Kroos,B747_C_D_labDocument,B747_C_D_MNita_DScholz,B747_C_D_Oberts];
B747_C_D_0 = mean([B747_C_D_0_Oberts,B747_C_D_0_labDocument,B747_C_D_0_MNita_DScholz,B747_C_D_0_Kroos]);


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
plot(B747_C_D_w,B747_C_L);
plot(B747Act(:,2), B747Act(:,1));
plot(B747_C_D_MNita_DScholz,B747_C_L);
title('Drag Polar Comparison: B747-200');
ylabel('C_L (Coefficient of Lift)');
xlabel('C_D (Coefficient of Drag)');
legend('3-D Finite Wing Drag Polar', 'B747 CFD Drag Polar - Truth','B747 CFD Drag Polar - Calculated','Location','best');
hold off


figure;
hold on
plot(B747_C_D_labDocument,B747_C_L);
plot(B747_C_D_MNita_DScholz,B747_C_L);
plot(B747_C_D_Oberts,B747_C_L);
plot(B747_C_D_Kroos,B747_C_L);
plot(B747Act(:,2), B747Act(:,1));
title('Whole Aircraft Drag Polar Comparsion: B747-200');
xlabel('C_D (Coefficient of Drag)');
ylabel('C_L (Coefficient of Lift)');
legend('C_L vs C_D using labDocument Oswalds', ...
       'C_L vs C_D using M. Nita D. Scholz Oswald', ...
       'C_L vs C_D using Oberts Oswald', ...
       'C_L vs C_D using Kroos Oswald', 'Boeing CFD Drag Polar','Location', 'best');
hold off

%% Tempest UAS Calculations

% Calculate the C_L and C_D for the Tempest UAS Finite Wing
[Tempest_C_L,Tempest_C_D_w] = finiteWingLift(TempestAct,Tempest_vars,0,6);

% Calculate the Whole Drag Polar for the Tempest UAS using labDocument method
Tempest_eLabDocument = labDocument(Tempest_vars);
[Tempest_C_D_labDocument,Tempest_C_D_0_labDocument] = wholeAircraftDragPolar(Tempest_C_L, Tempest_C_D_w, Tempest_eLabDocument, Tempest_vars);

% Calculate the Whole Drag Polar for the Tempest UAS using MNita_DScholz method
Tempest_eMNita_DScholz = MNita_DScholz(Tempest_C_D_w, 0.9, Tempest_vars);
[Tempest_C_D_MNita_DScholz,Tempest_C_D_0_MNita_DScholz] = wholeAircraftDragPolar(Tempest_C_L, Tempest_C_D_w, Tempest_eMNita_DScholz, Tempest_vars);

% Calculate the Whole Drag Polar for the Tempest UAS using Oberts method
Tempest_eOberts = oberts(Tempest_vars);
[Tempest_C_D_Oberts,Tempest_C_D_0_Oberts] = wholeAircraftDragPolar(Tempest_C_L, Tempest_C_D_w, Tempest_eOberts, Tempest_vars);

% Calculate the Whole Drag Polar for the Tempest UAS using Kroo's method
Tempest_eKroos = kroos(Tempest_C_D_w, 0.9, Tempest_vars);
[Tempest_C_D_Kroos,Tempest_C_D_0_Kroos] = wholeAircraftDragPolar(Tempest_C_L, Tempest_C_D_w, Tempest_eKroos, Tempest_vars);

% Combine all Oswald's factors and C_Ds in one array for performance calcs
Tempest_e0 = [Tempest_eKroos,Tempest_eLabDocument,Tempest_eMNita_DScholz,Tempest_eOberts];
Tempest_C_D = [Tempest_C_D_Kroos,Tempest_C_D_labDocument,Tempest_C_D_MNita_DScholz,Tempest_C_D_Oberts];
Tempest_C_D_0 = mean([Tempest_C_D_0_Oberts,Tempest_C_D_0_labDocument,Tempest_C_D_0_MNita_DScholz,Tempest_C_D_0_Kroos]);

figure;
hold on
plot(Tempest(:,1), Tempest_C_L);
plot(Tempest(:,1), Tempest(:,2));
plot(TempestAct(:,1), TempestAct(:,2),'-.r*');
title('Lift Curve Comparsion: Tempest UAS');
xlabel('α (Angle of Attack)');
ylabel('C_L (Coefficient of Lift)');
legend('3-D Finite Wing Curve', '2-D Airfoil Curve', 'Tempest CFD Drag Polar', 'Location', 'best');
hold off

figure;
hold on
plot(Tempest_C_D_w,Tempest_C_L);
plot(TempestAct(:,3), TempestAct(:,2));
plot(Tempest_C_D_MNita_DScholz,Tempest_C_L);
title('Drag Polar Comparison: Tempest UAS');
ylabel('C_L (Coefficient of Lift)');
xlabel('C_D (Coefficient of Drag)');
legend('3-D Finite Wing Drag Polar', 'Tempest CFD Drag Polar - Truth','Tempest CFD Drag Polar - Calculated','Location','best');
hold off

figure;
hold on
plot(Tempest_C_D_labDocument,Tempest_C_L);
plot(Tempest_C_D_MNita_DScholz,Tempest_C_L);
plot(Tempest_C_D_Oberts,Tempest_C_L);
plot(Tempest_C_D_Kroos,Tempest_C_L);
plot(TempestAct(:,3),TempestAct(:,2));
title('Whole Aircraft Drag Polar Comparsion: Tempest UAS');
xlabel('C_D (Coefficient of Drag)');
ylabel('C_L (Coefficient of Lift)');
legend('C_L vs C_D using labDocument Oswalds', ...
       'C_L vs C_D using M. Nita D. Scholz Oswald', ...
       'C_L vs C_D using Oberts Oswald', ...
       'C_L vs C_D using Kroos Oswald', 'Tempest CFD Drag Polar', 'Location', 'best');
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
Tempest_Glide_Theta = zeros(1,4);
B747_Glide_Range_R = zeros(1,4);
B747_Glide_Range_V = zeros(1,4);
B747_Glide_Theta = zeros(1,4);

% Define empty arrays for max endurance powered velocities
Tempest_Powered_End_V = zeros(1,4); % maximum powered endurance
B747_Powered_End_V = zeros(1,4);

% Define empty arrays for max range powered velocities
Tempest_Powered_Range_V = zeros(1,4); % maximum powered range
B747_Powered_Range_V = zeros(1,4);

% Define empty arrays for powered LDmaxes
Tempest_Powered_LDMax = zeros(1,4);
B747_Powered_LDMax = zeros(1,4);

% Tempest performance calculations for each Oswald's using calculated data
for i=1:4
   [Tempest_Glide_LDMax(i),Tempest_Glide_Range_V(i),Tempest_Glide_Theta(i)] = ...
       glide(Tempest_h,Tempest_C_D(:,i),Tempest_C_L,Tempest_e0(i),...
       Tempest_AR,Tempest_rho,Tempest_W,Tempest_S_ref,Tempest_C_D_0); 
   [Tempest_Powered_LDMax(i),Tempest_Powered_End_V(i),...
       Tempest_Powered_Range_V(i)] = poweredProp(Tempest_C_D(:,i),...
       Tempest_C_L,Tempest_e0(i),Tempest_AR,Tempest_rho,Tempest_W,...
       Tempest_S_ref,Tempest_C_D_0);
end

% Tempest performance calculations for avg Oswald's using given data
[TempestAct_Glide_LDMax,TempestAct_Glide_Range_V,TempestAct_Glide_Theta] = ...
    glide(Tempest_h,TempestAct(:,3),TempestAct(:,2),mean(Tempest_e0),...
    Tempest_AR,Tempest_rho,Tempest_W,Tempest_S_ref,Tempest_C_D_0);
[TempestAct_Powered_LDMax,TempestAct_Powered_End_V,...
    TempestAct_Powered_Range_V] = poweredProp(TempestAct(:,3),...
    TempestAct(:,2),mean(Tempest_e0),Tempest_AR,Tempest_rho,Tempest_W,...
    Tempest_S_ref,Tempest_C_D_0);

% B747 performance calculations for each Oswald's using calculated data
for i=1:4
    [B747_Glide_LDMax(i), B747_Glide_Range_V(i),B747_Glide_Theta(i)] = ...
        glide(B747_h,B747_C_D(:,i),B747_C_L,B747_e0(i),...
        B747_AR,B747_rho,B747_W,B747_S_ref,B747_C_D_0);
    [B747_Powered_LDMax(i),B747_Powered_End_V(i),...
        B747_Powered_Range_V(i)] = poweredJet(B747_C_D(:,i),B747_C_L,...
        B747_e0(i),B747_AR,B747_rho,B747_W,B747_S_ref,B747_C_D_0);
end

% B747 performance calculations for avg Oswald's using given data
[B747Act_Glide_LDMax, B747Act_Glide_Range_V,B747Act_Glide_Theta] = ...
    glide(B747_h,B747Act(:,2),B747Act(:,1),mean(B747_e0),...
    B747_AR,B747_rho,B747_W,B747_S_ref,B747_C_D_0);
[B747Act_Powered_LDMax,B747Act_Powered_End_V,B747Act_Powered_Range_V]...
    = poweredJet(B747Act(:,2),B747Act(:,1),mean(B747_e0),B747_AR,B747_rho,...
    B747_W,B747_S_ref,B747_C_D_0);

%% Print results out to terminal

B747_Results = table([B747_Glide_LDMax';mean(B747_Glide_LDMax);B747Act_Glide_LDMax],...
    [B747_Glide_Range_V';mean(B747_Glide_Range_V);B747Act_Glide_Range_V],...
    [B747_Glide_Theta';mean(B747_Glide_Theta);B747Act_Glide_Theta],...
    [B747_Powered_LDMax';mean(B747_Powered_LDMax);B747Act_Powered_LDMax],...
    [B747_Powered_Range_V';mean(B747_Powered_Range_V);B747Act_Powered_Range_V],...
    [B747_Powered_End_V';mean(B747_Powered_End_V);B747Act_Powered_End_V],...
    'RowNames',{'Kroos','Lab Doc','Scholz','Oberts','Mean','Actual'},...
    'VariableNames',{'Glide LDMax','Glide Max Range Velocity',...
    'Glide AoA','Powered LDMax','Powered Max Range Velocity',...
    'Powered Max Endurance Velocity'})

Tempest_Results = table([Tempest_Glide_LDMax';mean(Tempest_Glide_LDMax);TempestAct_Glide_LDMax],...
    [Tempest_Glide_Range_V';mean(Tempest_Glide_Range_V);TempestAct_Glide_Range_V],...
    [Tempest_Glide_Theta';mean(Tempest_Glide_Theta);TempestAct_Glide_Theta],...
    [Tempest_Powered_LDMax';mean(Tempest_Powered_LDMax);TempestAct_Powered_LDMax],...
    [Tempest_Powered_Range_V';mean(Tempest_Powered_Range_V);TempestAct_Powered_Range_V],...
    [Tempest_Powered_End_V';mean(Tempest_Powered_End_V);TempestAct_Powered_End_V],...
    'RowNames',{'Kroos','Lab Doc','Scholz','Oberts','Mean','Actual'},...
    'VariableNames',{'Glide LDMax','Glide Max Range Velocity',...
    'Glide AoA','Powered LDMax','Powered Max Range Velocity',...
    'Powered Max Endurance Velocity'})

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
    alpha_aoa_0 = interp1(data(:,2),data(:,1),0,'linear','extrap');
    
    % Calculate C_L for all AOA
    C_L = a*(data(:,1)-alpha_aoa_0);
    
    % Calculate C_D_wing
    C_D_wing = data(:,3) + (C_L.^2)./(pi*e*AR);
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
function [C_D,C_D_0] = wholeAircraftDragPolar(C_L, C_D_wing, e_0, vars)
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
    
    % Calculate C_D_0
    C_D_0 = C_D_min + (k_1*(C_L_minD^2));
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

%% Performance Calculations
% glide

% Calculates L/Dmax, ranges and velocities for max endurance and max range
% glide

function [LDmax,V_range,theta_range] = glide(h,C_D,C_L,e0,AR,rho,W,S,C_D_0)
% max endurance: at 3CD0 = kCL^2
% max range: at CD0 = kCL^2

    % define k
    k = 1/(AR*pi*e0);
    C_L_range = sqrt(C_D_0/k);
    [LDmax,idx] = max(C_L./C_D);
    R_max = h*(C_L(idx)/C_D(idx));
    
    % calculate glide angle
    theta_range = atan(h/R_max);
 
    V_range = sqrt((2*cos(theta_range)*W)/(rho*C_L_range*S));
    theta_range = rad2deg(theta_range);
end

% poweredJet

% Calculates L/Dmax and velocities for powered jet max endurance and max
% range

function [LDmax,V_end,V_range] = poweredJet(C_D,C_L,e0,AR,rho,W,S,C_D_0)
% max endurance: at CD0 = kCL^2
% max range: at CD0 = 3kCL^2

    % define k
    k = 1/(AR*pi*e0);
    
    % calculate LDmax for max range
    [LDmax,idx] = max(C_L./C_D);        % L/Dmax at CD0 = kCL^2
    
    C_L_end = sqrt(C_D_0/k);
    C_L_range = sqrt(C_D_0/(3*k));
    
    % combine two definitions for L to calculate freestream velocity V
    % L = W, L = C_L*q*S
    V_end = sqrt((2*W)/(rho*C_L_end*S));
    V_range = sqrt((2*W)/(rho*C_L_range*S));
end

% poweredProp

% Calculates L/Dmax and velocities for powered prop max endurance and max
% range

function [LDmax,V_end,V_range] = poweredProp(C_D,C_L,e0,AR,rho,W,S,C_D_0)
% max endurance: at 3CD0 = kCL^2
% max range: at CD0 = kCL^2

    % define k
    k = 1/(AR*pi*e0);
    
    % calculate LDmax for max range
    [LDmax,idx] = max(C_L./C_D);    % L/Dmax at CD0 = kCL^2
    
    C_L_end = sqrt(3*C_D_0/k);
    C_L_range = sqrt(C_D_0/k);
    
    % combine two definitions for L to calculate freestream velocity V
    % L = W, L = C_L*q*S
    V_end = sqrt((2*W)/(rho*C_L_end*S));
    V_range = sqrt((2*W)/(rho*C_L_range*S));
end