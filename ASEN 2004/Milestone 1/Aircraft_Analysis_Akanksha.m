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

B747_S = 511;                       % Boeing 747 Planform Area [m^2]
B747_AR = 7;                        % Boeing 747 Aspect Ratio [unitless]
B747_Re = 1000000;                  % Boeing 747 Reynold's Number [unitless]
B747_e0 = -0.339;
%B747_C_fe = ;
%B747_S_wet = ;
%B747_S_ref = ;

% Create the Variable Array for the B747
B747_vars = [B747_AR,B747_Re,B747_S,B747_e0];

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

    
%% Declare and Define the Tempest UAS Variables
% Tempest UAS Reynold's Number
Tempest_Re = 200000;
Tempest_S = 0.63; 
Tempest_AR = 16.5;
Tempest_e0 = 0.601;
%Tempest_C_fe = ;
%Tempest_S_wet = ;
%Tempest_S_ref = ;

% Create the Variable Array for the B747
Tempest_vars = [Tempest_AR, Tempest_Re, Tempest_S, Tempest_e0];

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

%% Calculate the C_L for the B747
[B747_C_L,B747_C_D] = finiteWingLift(B747,B747_vars,2,10);

figure;
hold on
plot(B747(:,1),B747_C_L);
plot(B747(:,1),B747(:,2));
title('Lift Curve Comparsion: B747-200');
xlabel('α (Angle of Attack)');
ylabel('C_L');
legend('3-D Finite Wing Curve', '2-D Airfoil Curve', 'Location', 'best');
hold off

%% Tempest Figures

[Tempest_C_L,Tempest_C_D] = finiteWingLift(Tempest,Tempest_vars,0,6);

figure;
hold on
plot(Tempest(:,1), Tempest_C_L);
plot(Tempest(:,1), Tempest(:,2));
plot(TempestAct(:,1), Tempest(:,2),'-.r*');
title('Lift Curve Comparsion: Tempest UAS');
xlabel('α (Angle of Attack)');
ylabel('C_L');
legend('3-D Finite Wing Curve', '2-D Airfoil Curve', 'Tempest CFD Drag Polar', 'Location', 'best');
hold off

figure;
hold on
plot(Tempest_C_L, Tempest_C_D);
plot(TempestAct(:,2), Tempest(:,3));
title('Drag Polar Comparison: Tempest UAS');
xlabel('C_L');
ylabel('C_D');
legend('3-D Finite Wing Drag Polar', 'Tempest CFD Drag Polar');
hold off

%% Functions

% finiteWingLift
%
% Function that calculates the 3D finite wing 3D lift coefficient and 3D wing
% drag polar
%
% @param data The table data that includes the AOA, C_l, and the C_d in
%               that order.
%
function [C_L,C_D_wing] = finiteWingLift(data,vars,aoa_one,aoa_two)
    % Define the variables
    AR = vars(1);       % Aspect Ratio
    e0 = vars(4);
    
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
    
    %xmin=find(C_D_wing==min(C_D_wing));
    %C_LminD = C_L(xmin);
    %C_Dmin = ;
    
    %k1 = 1/(pi*e0*AR);
    
    % Calculate C_D_aircraft
    %C_D_aircraft = C_Dmin + (k1 * (C_L - C_LminD).^2);
end
