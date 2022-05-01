function eqparam = getEqParam(liquid, gas, shell, bubble, pulse)
% Compute the damping constants and the polytropic exponent.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATERIAL AND PULSE PROPERTIES
nu_l    = liquid.nu;    % Dynamic viscosity liquid (Pa.s)
rhol    = liquid.rho;   % Density liquid (kg/m^3)
c       = liquid.c;     % Speed of sound (m/s)

gam     = gas.gam;      % Heat capacity ratio gas
R0      = bubble.R0;    % Initial microbubble radius (m)

w       = pulse.w;      % Angular frequency pulse (Hz)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THERMAL MODEL

% The thermal model by Prosperetti only works for an oscillating field:
if strcmp(liquid.ThermalModel,'Prosperetti') && w == 0
  error('Prosperetti model only valid for w>0')
end
    
% Thermal damping and polytropic exponent
switch liquid.ThermalModel
    case 'Prosperetti'
        % Thermal model, Prosperetti, JASA, 61, 1977
        [eqparam.nu_th, eqparam.kappa] = calc_thermal_damp(...
            liquid,gas,bubble,shell,w);

    case 'Adiabatic'
        eqparam.nu_th =0;
        eqparam.kappa = gam;% Polytropic exponent 

    case 'Isothermal'
        eqparam.nu_th=0;
        eqparam.kappa = 1;% Polytropic exponent
        
    otherwise
        error('Thermal model not recognized')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VISCOUS DAMPING AND THERMAL DAMPING
eqparam.nu_vis = nu_l;   

% Effective viscosity linear model  	
eqparam.nu = eqparam.nu_vis + eqparam.nu_th;  % Effective viscosity (Pa.s)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RADIATION DAMPING
% Only set RadiationDamping = true, if the Rayleigh-Plesset equation does
% not account for reradiation itself:
RadiationDamping = false;

% Radiation damping, Prosperetti, JASA, 61, 1977?p.18, Eq. 10 - 12
x = w*R0/c;
eqparam.nu_rad = rhol*R0^2/4*(x/(1+x^2))*w;

if RadiationDamping==true
    eqparam.nu = eqparam.nu + eqparam.nu_rad;
end
    
end