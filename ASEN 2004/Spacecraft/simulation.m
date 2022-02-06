function [] = simulation(N, v0,endpoint)
%SIMULATION Performs a monte carlo simulation with a given number of trials

%gravity
g = 9.81;
%mass
mp = 1.001 + 0.005*randn(N,1); %[kg] propellant mass (fuel mass)
ms = .128 + 0.005*randn(N,1); %[kg] dry mass
mf = ms; %[kg] post-flight mass
m0 = mp + ms; %[kg] inital mass (wet mass)

%bottle and geometry
bottle_diameter = .105 + 0.001*randn(N,1); %[m] bottle diameter
Ar = (pi/4)*bottle_diameter.^2; %[m^2] refrence area

%launch
theta = 45+randn(N,1); %[deg] launch pad angle
heading = 40*ones(N,1); %[deg] heading measured degrees from noth, measured clockwise
L = .5; %[m] length of the launch stand

%environment (wind direction reported from where it is blowing)
wind_ground = convvel(0, 'mph', 'm/s'); %[m/s] winds at ground level
wind_aloft = convvel(3, 'mph', 'm/s'); %[m/s] winds in flight
wind_angle_aloft = dir2angle('NE') + 22.5*randn(N,1); %[deg] heading measured degrees from noth, measured clockwise
wind_angle_ground = dir2angle('N') + 22.5*randn(N,1); %[deg] heading measured degrees from noth, measured clockwise
[~,~,~,p_air] = atmoscoesa(1624); %[kg/m^3]
CD = 0.38; %coefficient of drag

%% Set Up Initial Conditions
e1 = zeros();
e2 = zeros();
figure();
for i = 1:N
    x0 = 0; %[m]
    y0 = 0; %[m]
    z0 = 0; %[m]
    v0x = v0*cosd(theta(i)); %[m/s]
    v0y = 0; %[m/s]
    v0z = v0*sind(theta(i)); %[m/s]

    IC = [x0,y0,z0,v0x,v0y,v0z];
    consts = [g,mf(i),Ar(i),heading(i),L,wind_ground,wind_aloft,...
        wind_angle_aloft(i),wind_angle_ground(i),p_air,CD];

%% Perform Simulations

    tspan = [0 10];
    options = odeset('Events', @stop);
    [t,X] = ode45(@(t,X) rocket(t,X,consts), tspan, IC, options);
    plot3(X(:,1),X(:,2),X(:,3))
    hold on
    xlabel('Downrange [m]')
    ylabel('Crossrange [m]')
    zlabel('Height [m]')
    xlim([-1 70])
    ylim([-5 5])
    zlim([0 25])
    grid on;

    e1(i) = X(end,1);
    e2(i) = X(end,2);
end

%% Plot Error Ellipses

figure; plot(e1,e2,'k.','markersize',6)
axis equal; grid on; xlabel('x [m]'); ylabel('y [m]'); hold on;
plot(endpoint(1),endpoint(2),'r*','markersize',6)
 
% Calculate covariance matrix
P = cov(e1,e2);
mean_x = mean(e1);
mean_y = mean(e2);
 
% Calculate the define the error ellipses
n=100; % Number of points around ellipse
p=0:pi/n:2*pi; % angles around a circle
 
[eigvec,eigval] = eig(P); % Compute eigen-stuff
xy_vect = [cos(p'),sin(p')] * sqrt(eigval) * eigvec'; % Transformation
x_vect = xy_vect(:,1);
y_vect = xy_vect(:,2);
 
% Plot the error ellipses overlaid on the same figure
plot(1*x_vect+mean_x, 1*y_vect+mean_y, 'b')
plot(2*x_vect+mean_x, 2*y_vect+mean_y, 'g')
plot(3*x_vect+mean_x, 3*y_vect+mean_y, 'r')
end

