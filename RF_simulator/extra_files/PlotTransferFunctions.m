% Get the transducer transfer functions of the P4-1 transducer
Tfit = load('TransmitTransferFunctionFit.mat');
Hfit = load('ReceiveTransferFunctionFit.mat','Hfit');

T = Tfit.Tfit;                      % Trasmit transfer function
TdB = 20*log10(abs(T)/max(abs(T))); % Transmit transfer function in dB
H = Hfit.Hfit;                      % Trasmit transfer function
HdB = 20*log10(abs(H)/max(abs(H))); % Transmit transfer function in dB
Fs = Tfit.Fs;                       % Sampling rate (Hz)

N = length(T);
f = (0:(N-1))/N*Fs;         % Frequency vector

figure
plot(f/1e6,TdB)
hold on
plot(f/1e6,HdB)
xlim([0 6])
ylim([-35 5])
ylabel('Response (dB)')
xlabel('Frequency (MHz)')

plot(f(TdB>-6)/1e6,TdB(TdB>-6),'.')
plot(f(HdB>-6)/1e6,HdB(HdB>-6),'.')

t = (0:(N-1))/Fs;
T_IR = real(ifft(T));
H_IR = real(ifft(H));

grid on
legend('Transmit transfer function', 'Receive transfer function',...
    'Location','southeast')
set(gcf, 'Units', 'inches')
set(gcf, 'Position', [1 1 3.5 2.5])

figure
plot(t*1e6,T_IR/max(T_IR));
hold on
plot(t*1e6,H_IR/max(H_IR));
xlabel('time (us)')
ylabel('Normalized response')
ylim([-2 2])
xlim([0 2.5])

grid on
legend('Transmit impulse response', 'Receive impulse response',...
    'Location','southeast')
set(gcf, 'Units', 'inches')
set(gcf, 'Position', [1 1 3.5 2.5])
