function pulseOUT = pulsePropagation(pulseIN, liquid, d)
% Simulate the waveform at a distance d from the transducer for plane wave
% transmission
%
% Assumptions:
% - Small amplitude.
% - For computation second harmonic p2, assume p1 = p0*exp(-alpha1*d).
%
% Nathan Blanken, University of Twente, 2020

% Make sure this function is only used for media with a frequency that
% depends quadratically on frequency:
if ~liquid.b==2
    error(['This function is only accurate for media with an'...
        ' attenuation that scales quadratically with frequency.'])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FREQUENCIES
f0 = pulseIN.f/1e6;         % centre frequency (MHz)

% Set up frequency axis
Fs = pulseIN.fs;            % Sampling rate (Hz)
N = length(pulseIN.t);      % Signal length
f = (0:(N-1))/N*Fs;         % Frequency array (Hz)
f = f/10^6;                 % Frequency array (MHz)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATERIAL PROPERTIES

beta = liquid.beta;     % nonlinear parameter

rho0 = liquid.rho;      % density of the medium
c0   = liquid.c;        % speed of sound in the medium

% acoustic attenuation(dB/cm)
a = liquid.a;
b = liquid.b;
alpha1   = a*f0^b;      % Acoustic attenuation coefficient at f0 (dB/cm)   

% Acoustic attenuation as a function of frequency:
alpha    = a*f.^b;      % Acoustic attenuation coefficient at f (dB/cm)
% Make symmetryic around Fs/2 (real signal has symmetric real part Fourier
% transform).
alpha(ceil(N/2+1):N) = alpha(floor(1+N/2):-1:2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TIME AXIS
% Convert retarded time to absolute time
pulseOUT.t = pulseIN.t + d/c0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNDAMENTAL COMPONENT
% No assumptions on bandwidth of signal.
% Compute Fourier transform
pulseFFT = fft(pulseIN.p);
% Compute attenuated amplitude spectrum
p1FFT = pulseFFT.*10.^(-alpha*d*100/20);
% Convert back to time domain
p1 = real(ifft(p1FFT));  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SECOND HARMONIC GENERATION
% Second harmonic generation, calculated from the Burgers' equation, 
% assuming p2 << p1. Assuming p1 = p0*exp(-alpha1*d).

Beta = beta/(rho0*c0^3)*(pulseIN.p).*(pulseIN.dp); % Source term

a1 = alpha1*log(10)*100/20; % Acoustic attenuation at f0 (Np/m)
aa = alpha *log(10)*100/20; % Acoustic attenuation at f (Np/m)

BetaFFT = fft(Beta);
p2FFT = BetaFFT.*(exp(-2*a1*d) - exp(-aa*d))./(aa-2*a1);
p2FFT(aa==2*a1) = BetaFFT(aa==2*a1)*d*exp(-2*a1*d); % Limit aa -> 2*a1
p2 = real(ifft(p2FFT));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADD FUNDAMENTAL AND SECOND HARMONIC COMPONENT
pulseOUT.p = p1 + p2;
pulseOUT.p1 = p1;
pulseOUT.p2 = p2;

pulseOUT.f = pulseIN.f;
pulseOUT.w = pulseIN.w;

end

