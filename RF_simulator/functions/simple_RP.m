function [dvec] = simple_RP(tau, vec, ...
    liquid,shell,eqparam,bubble, pulse, T)
% Rayleigh-Plesset model according to:
% Marmottant et al, J. Acoust. Soc. Am, 118, 2005. (Eq. 3)
% 
% Nathan Blanken, University of Twente, 2020

% Microbubble and shell properties
R0 =    bubble.R0;      % Initial microbubble radius (m)
Ks =    shell.Ks;       % Shell stiffness (N/m)  
sig_0 = shell.sig_0;    % Initial surface tension (N/m)

% Liquid properties
rhol =  liquid.rho;     % Density (kg/m^3)
P0 =    liquid.P0;      % Ambient pressure (Pa)
c =     liquid.c;       % speed of sound (m/s)

kappa = eqparam.kappa;  % Polytropic exponent
nu =    eqparam.nu;     % Effective viscosity (Pa.s)

% Convert nondimensional time to dimensional time:
t = tau*T;

% Get acoustic pressure at this time:
Pacc = interp1( pulse.t, pulse.p, t, 'pchip' );

x = vec(1);
xdot = vec(2);

% Get surface tension for this bubble radius:
R = R0*(1+x);

% Compute the surface tension of the shell at radius R:
sig = calc_surface_tension(R,shell);

% Nondimensional RP equation:
xdotdot = 1/(1+x)*(...
    -3/2*xdot^2 ...
    + (1 + 2*sig_0/(R0*P0))*(1+x)^(-3*kappa)*...
    (1-3*kappa/c*sqrt(P0/rhol)*xdot)...
    -  1 - 2*sig/(R0*P0)*(1+x)^(-1) ...
    - 4*nu/sqrt(rhol*R0^2*P0)*xdot/(1+x) ...
    - 4*Ks/sqrt(rhol*R0^4*P0)*xdot/(1+x)^2 ...
    - Pacc/P0...
    );

dvec(1,1) = vec(2);
dvec(2,1) = xdotdot;

end



