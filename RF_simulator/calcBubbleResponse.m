function [response, eqparam] = calcBubbleResponse(liquid, gas, ...
    shell, bubble, pulse)
% Compute the radial response and the scattered pressure of a microbubble.
% Nathan Blanken, University of Twente, 2020

%% Timer 
time_out = 30;                  % Stop integration after time_out seconds
assignin('base','timeout_reached',0);   % Time-out status
timer_inst = timer('TimerFcn', ...
    'timeout_reached = 1; disp("timeout reached")',...
    'StartDelay', time_out);

%% Display progress settings
dispProgress = false;           % Show ODE solver progress
assignin('base','tplo',0)       % Time to display progress (s)
assignin('base','dtplo',5e-7)   % Increment time to display progress (s)

%% Equation parameters (damping parameters and polytropic exponent)
eqparam = getEqParam(liquid, gas, shell, bubble, pulse);

%% Initial conditions and nondimensionalization
x0dot = 0; x0 = 0;                              % Initial conditions
x0v = [x0; x0dot];

T = sqrt(liquid.rho*bubble.R0^2/liquid.P0);     % Characteristic time scale
tau = pulse.t/T;                                % Nondimensional time

%% ODE options
InitialSte = 1e-12;
options = odeset('BDF','on','AbsTol',1e-6,'RelTol',1e-6, ...
    'InitialStep',InitialSte,...
    'OutputFcn',@(tau,y,flag) odeOutputFcn(tau,y,flag,T,dispProgress));

%% Run the ODE solver:
RP_handle = @(tau,vec) simple_RP(tau,vec,...
    liquid,shell,eqparam,bubble, pulse, T);

start(timer_inst)
[tau,X]=ode45(RP_handle,tau,x0v,options);
stop(timer_inst)

%% Return to dimensional variables
response.R = bubble.R0.*(1+X(:,1));     % Radius (m)
response.Rdot = bubble.R0.*X(:,2)/T;    % Radial velocity (m/s)
response.t = tau*T;

end








