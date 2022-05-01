%% RF line simulator
%
% Simulates the element RF echo data coming from a single microbubble in a
% rectangular domain. The simulator simulates a plane, 
% homogeneous transmit wave. Bubble responses are computed with a
% Marmottant-type Rayleigh-Plesset equation, which takes viscous,
% radiation, shell, and thermal damping into account. Bubble-bubble
% interactions are neglected. The pulse shape is based on the pulse from
% the P4-1 transducer.
%
% Nathan Blanken, University of Twente, 2021

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INITIALIZE

clear
clc
close all

dispFig = false;            % Display figures
single_element = true;      % Only compute RF data centre element

% Add the functions folder to path
addpath './functions'

% Get the transducer transfer functions of the P4-1 transducer
Tfit = load('TransmitTransferFunctionFit.mat');
Hfit = load('ReceiveTransferFunctionFit.mat','Hfit');

%% Pulse properties
Ncy = 1;                    % Total number of cycles
f = 1.7e6;              	% Centre frequency (Hz)
Fs = Tfit.Fs;              	% Simulation sampling rate (Hz)
downsample = 4;             % Measurement sampling rate: Fs/downsample
Tresp = 4e-6;               % Echo receive time after pulse (s)
PA = 125;                   % Acoustic pressure amplitude (kPa)
PA = PA*1000;               % Acoustic pressure amplitude (Pa)

pulse = getPulse(f,Ncy,PA,Fs,Tresp,dispFig,Tfit.Tfit);

%% material properties and environmental conditions
[liquid, gas] = getMaterialProperties();

% Select a thermal model: 'Adiabatic', 'Isothermal', or 'Propsperetti':
liquid.ThermalModel = 'Prosperetti';

%% Scan domain and transducer properties
depth = 0.1;                % Imaging depth (m)
width = 0.028;              % Transducer width (m) (P4-1 transducer)
c = liquid.c;             	% Speed of sound in the medium (m/s)
d1 = 0.0037;                % left boundary bubble locations (m)
d2 = depth - d1;            % right boundary bubble locations (m)
N_elem = 96;                % Number of transducer elements
                 
% Element positions (m):
if single_element == true
    x_el = 0;               
else
    x_el = linspace(-width/2,width/2,N_elem);
end

domain = struct('depth',depth,'width',width,'c',c,'d1',d1,'d2',d2,...
    'x_el',x_el);
clear depth width c d1 d2 x_el N_elem


%% Bubble and shell properties
R0 = 0.5;                   % Mean bubble radius (um)

bubble.z  = 0.05;           % Axial coordinate bubble (m)
bubble.x  = 0;             	% Lateral coordinate bubble (m)
bubble.R0 = R0*1e-6;      	% Bubble radius (m)
bubble.r0 = 0.001;      	% Scattered pressure sensor distance to centre

shell.model = 'Segers';     % Marmottant, Segers, or SegersTable
shell.sig_0 = 10e-3;     	% Equilibrium surface tension bubble (N/m).       
shell = getShellProperties(bubble,shell,liquid);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SIMULATE THE RF SIGNALS

%% Bubble response

% Simulate the wave propagation from transducer to bubble
pulseLocal = pulsePropagation(pulse, liquid, bubble.z);

% Compute the radial response of the bubble:
[response, ~] = calcBubbleResponse(liquid, gas, ...
    shell, bubble, pulseLocal); 

% Do not compute the rapidly decaying r^(-3) term:
nearfield = false;        
% Compute the scattered pressure:
scatter = calc_scatter(response,liquid,bubble,pulse,nearfield);
scattercell{1} = scatter;

%% Wave propgation and RF signal construction  
RF = constructRF(scattercell,bubble,liquid,pulse,domain,Fs, ...
    downsample,Hfit.Hfit);

%% Compute the pressure field
pfield = computepressurefield(pulse,liquid,domain,Fs,downsample);

%% Save results   
% Remove redundant fields to save disk space:
% 
% filename = 'SimulationSingleBubble125kPa';
% save(strcat(filename,'.mat'),...
%     'domain', 'liquid','gas','shell','pulse',...
%     'pfield', 'bubble','RF')
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the results and clear variables
T = 2*domain.depth/domain.c;    % Total receive time
RF(1).t = 0:1/RF(1).fs:T;   	% time vector

plot(RF(1).t,RF(1).V)
hold on
xlabel('t (s)')
ylabel('V (V)')

clear NSIM dispFig delim filedir Tfit Hfit
clear Ncy f Fs downsample Tresp Pmin Pmax 
clear Nmax Nmin R0 sR0 r0 shell_model sig_0
clear PA Nb PAlocal pulseLocal N_elem RFc scatter_elem scatter pulse
clear k m n M
clear filename
clear T z x_el scattercell single_element
