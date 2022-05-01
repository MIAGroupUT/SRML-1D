% Delay-and-sum reconstruction of the bubble image from the element data
% simulated with RF_simulatorFINAL.
%
% Nathan Blanken, University of Twente, 2021

clear
clc

%% LOAD DATA AND METADATA
filename        = 'RFDATA2D00001.mat';

% Load the metadata from the original MATLAB files:
load(filename)

% Domain and transducer properties
width = domain.width;       % domain width (m)
depth = domain.depth;       % domain depth (m)
N = length(RF(1).p);        % number of samples per RF line
Nelem = length(RF);         % number of transducer elements
Fs = RF(1).fs;              % sample rate (Hz)
t = (0:(N-1))/Fs;           % time axis (s)

x_el = linspace(-width/2,width/2,Nelem); % Element positions (m)

% Convert RF struct to matrix
RF_matrix = [RF.V];

RF_matrix = reshape(RF_matrix, N, Nelem)';  % matrix of element RF data

%% WAVE PROPERTIES
c = liquid.c;               % speed of sound in the medium (m/s)
f0 = pulse.f;               % centre frequency (Hz)
lambda = c/f0;              % wavelength (m)

%% TIME GAIN COMPENSATION (TGC)
% Apply a linear TGC to compensate for the 1/r decay of scattered pressure:
TGC = t;                

RF_TGC = RF_matrix;
for i = 1:Nelem
    RF_TGC(i,:) = RF_matrix(i,:).* TGC ;
end
clear RF_matrix

%% DELAY-SUM RECONSTRUCTION
% Compute the approximate duration of the pulse in number of samples:
[~,I] = max(abs(hilbert(pulse.p)));     % Find maximum of pressure pulse
sig_dur = I/pulse.fs*RF(1).fs*2;        % Signal duration (samples)
clear RF

% Dimensions of the reconstructed image:
IM_WI = width*1.5;          % Width of the reconstructed image (m)
IM_DE = depth;              % Depth of the reconstructed image (m)
pix_siz = lambda/50;        % Pixel size for image reconstruction
 
tic
% Delay and sum reconstruction:
img = delay_and_sum(RF_TGC, IM_WI, IM_DE, pix_siz, x_el, c, sig_dur, Fs);
toc

%% SAVE THE RESULTS      
x = -IM_WI/2:pix_siz:IM_WI/2;       % Lateral coordinates (m)
z = 0:pix_siz:IM_DE;                % Axial coordinates (m)

save([filename(1:end-4) '_DAS.mat'],'img','x','z')

