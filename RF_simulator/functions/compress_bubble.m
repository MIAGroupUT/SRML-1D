function [liquid, gas, bubble, shell] = ...
    compress_bubble(liquid, gas, bubble, shell, P1)
% Compute new ambient pressure, new bubble equilibrium radius, new bubble
% equilibrium surface tension, and new gas density (at new ambient 
% pressure) after application of overpressure P1 (Pa). Negative P1 
% corresponds to underpressure.
%
% Nathan Blanken, University of Twente, 2021

kappa = 1;                  % Polytropic exponent (isothermal)

R0 = bubble.R0;             % Original bubble equilibrium radius
P0 =    liquid.P0;          % Original ambient pressure (Pa)
sig_0 = shell.sig_0;        % Original equilibrium surface tension

N = 1e4;
R = R0*linspace(0.5,2,N);	% Search array radius (m)

sig = calc_surface_tension(R,shell);

P_in = (P0 + 2*sig_0/R0)*(R/R0).^(-3*kappa);    % Pressure inside bubble
P_eq = P0 + 2*sig./R + P1;                      % Equilibrium pressure

% Find index I in array for which P_in = P_eq
[~,I] = min(abs(P_in-P_eq));

% Check if solution was found:
if I == N
    error('Overpressure too low to find solution')
elseif I == 1
    error('Overpressure too high to find solution')
end

% Find radius R_new for which P_in = P_eq
R_new = interp1((P_in(I-1:I+1)-P_eq(I-1:I+1)),R(I-1:I+1),0);           

% New equilibrium surface tension:
sig_new = calc_surface_tension(R_new,shell);

% Isothermal compression of the gas:
rho_new = gas.rho*(P0 + P1)/P0;

liquid.P0   = P0 + P1;   	% New ambient pressure (Pa)
bubble.R0   = R_new;      	% New equilibrium bubble radius (m)
shell.sig_0 = sig_new;      % New equilibrium surface tension (N/m)
gas.rho     = rho_new;      % New gas density (outside bubble) (kg/m^3)

end
