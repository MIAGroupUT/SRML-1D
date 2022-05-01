function scatterOUT = pulsePropagationSpherical(...
    scatterIN, liquid, bubble, x_el)
% Simulate the waveform at a distance d from the transducer for spherical
% wave transmission. Nonlinear propagation is neglected, considering the
% rapid decay of a spherical wave.
%
% Nathan Blanken, University of Twente, 2020

% d0 = 0.001;           % reference distance to bubble centre (m)
% x_el                  % element position

% MATERIAL PROPERTIES

x  = bubble.x;          % Lateral coordinates bubble (m)
z  = bubble.z;          % Axial coordinate bubble (m)
r0 = bubble.r0;         % Distance scattered pressure sensor (m)

r  = sqrt((x-x_el)^2 + z^2);    % Distance to centre of transducer (m)

% acoustic attenuation parameters
a   = liquid.a;
b   = liquid.b;
c0  = liquid.c;        % speed of sound in the medium

% TIME AXIS
% Convert retarded time to absolute time
scatterOUT.t = scatterIN.t + (r-r0)/c0;

% ACOUSTIC ATTENUATION

% Set up frequency axis
Fs = scatterIN.fs;
N = length(scatterIN.t);
f = (0:(N-1))/N*Fs;
% Convert to MHz
f = f/10^6;     

% Compute attenuation as function of frequency
alpha   = a*abs(f).^b;      % Attenuation coefficient (dB/cm)   
alpha = alpha';

% Make symmetryic around Fs/2 (real signal has symmetric real part Fourier
% transform).
alpha(ceil(N/2+1):N) = alpha(floor(1+N/2):-1:2);

% Compute amplitude spectrum
scatterFFT = fft(scatterIN.ps);
% Compute attenuated amplitude spectrum
FFTatt = scatterFFT.*10.^(-alpha*r*100/20);
% Convert back to time domain
scatterOUT.ps = real(ifft(FFTatt))*r0/r;

end


