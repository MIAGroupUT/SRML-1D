function RF = constructRF(scattercell,bubble,liquid,pulse,domain,Fs,...
    downsample,Hfit)
% Construct radio-frequency data by adding up microbubble echoes.
%
% Nathan Blanken, University of Twente, 2021

x_el = domain.x_el;     % Transducer element positions
N_elem = length(x_el);  % Number of transducer elements

% Preallocate RF lines
RF = preallocaterf(domain,N_elem,Fs);

% Add responses to RF line
for k = 1:length(bubble)

    scatter = scattercell{k};   % Cell with microbubble echo data
    scatter.fs = pulse.fs;      % Sampling rate of the echoes

    for m = 1:N_elem 
        % Simulate the wave propagation from bubble to transducer
        scatter_elem = pulsePropagationSpherical(scatter, liquid, ...
            bubble(k), x_el(m));

        % Add the bubble response to the RF line
        RF(m).p = addToRF(RF(m).t, RF(m).p, scatter_elem);
    end

end

for m = 1:N_elem
    % Convert pressure signal to voltage signal:
    RF(m).V = receiveTransferFunction(RF(m).p,Hfit,RF(m).fs);
    
    % Resample the RF signals
    RF(m) = resampleRF(RF(m),downsample);
end

end


function RF = preallocaterf(domain,N_elem,Fs)
% Preallocate structure for radio-frequency (RF) data.

T = 2*domain.depth/domain.c;                 	% Total receive time
RF(N_elem).t = 0:1/Fs:T;                        % Time vector
RF(N_elem).p = zeros(1,length(RF(N_elem).t));   % Receive buffer
RF(N_elem).fs = Fs;

for m = 1:N_elem
    RF(m).t = 0:1/Fs:T;                         % Time vector
    RF(m).p = zeros(1,length(RF(m).t));         % Receive buffer
    RF(m).fs = Fs;
end

end

function V = receiveTransferFunction(p,H,Fs)
% Convert received pressure to a voltage signal.

% Check sample rate
if Fs ~= 250e6
    error('This function only works for a sampling frequency of 250 MHz')
end

% Resample transfer function
N = length(p);
M = length(H);
H_r = resample(H,N,M);

% Apply transfer function
pfft = fft(p);
Vfft = pfft.*H_r;
V = real(ifft(Vfft));

end

function RF = addToRF(t,RF,signal)
% Add individual scatterer signal to the RF data

t0 = signal.t(1);               % start time scattered signal
t1 = signal.t(end);             % end time signal
tq = t((t>=t0)&(t<=t1));        % interpolation time points

% Find values on grid by interpolation
if isfield(signal,'ps')
    signal_int = interp1(signal.t,signal.ps',tq);
elseif isfield(signal,'V')
    signal_int = interp1(signal.t,signal.p,tq);
else
    msg = 'No field ps or V found.';
    error(msg)
end

% Add the interpolated signal to the RF line
RF((t>=t0)&(t<=t1)) = RF((t>=t0)&(t<=t1)) + signal_int;
    

end

function RF = resampleRF(RF,n)
% Resample signal with downsampling factor n

RF.t = RF.t(1:n:end);
RF.p = RF.p(1:n:end);
if isfield(RF,'V')
    RF.V = RF.V(1:n:end);
end
RF.fs = RF.fs/n;

end
