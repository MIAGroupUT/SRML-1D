% Plot the frequency spectrum of a single bubble as a function of acoustic
% pressure. Driven by a short pulse.

filedir  = '.\SingleBubbles';
delim    = '\';

load('ColorMapMaker\CustomColorMap')

% List files in source directory
filelist = dir(filedir);
filelist = filelist(3:end);

filename = filelist(1).name;
load([filedir delim filename])

N_PA = length(filelist);        % Number of acoustic pressures
PA_list = zeros(1,N_PA);        % List with acoustic pressures

cmap = parula(N_PA);


p_s_matrix = zeros(length(filelist),length(RF.V));  % scattered pressures
FFT_matrix = zeros(length(filelist),length(RF.V));  % FFT scatter

for k = 1:length(filelist)
   	% Load the file
    filename = filelist(k).name;
    load([filedir delim filename])
    
    PA = pulse.A;               % Acoustic pressure amplitude
    PA_list(k) = PA;  
    Y = abs(fft(RF.V));         % Fourier transform of signal  
    FFT_matrix(k,:) = Y/PA;     % Fourier transform normalized by PA
    p_s_matrix(k,:) = RF.V/PA;  % Signal normalized by PA
end


%%

N = length(RF.V);               % Signal length
Fs = RF.fs;                     % Sampling rate (Hz)
f = (0:(N-1))/N*Fs;             % Frequency vector (Hz)
f = f/1e6;                      % Frequency vector (MHz)

cmap = CustomColorMap;

figure
FFT_0 = max(FFT_matrix,[],'all');
imagesc(PA_list/1000,f,20*log10(FFT_matrix'/FFT_0))
set(gca,'YDir','normal')
caxis([-35 0])
colormap(parula)
ylabel('Frequency (MHz)')
xlabel('Acoustic pressure (kPa)')
ylim([0 5])
colormap(cmap)
a = colorbar;
a.Label.String = 'dB';

hold on
plot([1 250],[1.70 1.70],'w') % Driving frequency 
plot([1 250],[1.04 1.04],'w--') % -12 dB points
plot([1 250],[4.12 4.12],'w--') % -12 dB points

set(gcf, 'Position', [50 50 350 230]);