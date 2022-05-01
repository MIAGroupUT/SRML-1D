function img = delay_and_sum(RF_data, IM_WI, IM_DE, pix_siz, ...
    x_el, c, sig_dur, Fs)
% Delay and sum reconstruction
%
% RF_data:  matrix of RF data
% IM_WI:    image width (m)
% IM_DE:    image depth (m)
% pix_siz:  image pixel size (m)
% x_el:     transducer element coordinates (m)
% c:        speed of sound (m/s)
% sig_dur:  signal duration (s)
% t:        time vector

% Grid of the new image
% Define here the vector defining the x and y axis of te new image.
x_rec = -IM_WI/2:pix_siz:IM_WI/2;
y_rec = 0:pix_siz:IM_DE;

Nx_rec = length(x_rec);         % number of pixels
Ny_rec = length(y_rec);         % number of pixels
Nelem  = length(x_el);        	% number of elements
N      = size(RF_data,2);       % number of samples per RF line

% Add zeros to represent time delays out of range:
RF_data = [RF_data zeros(Nelem,1)];  

% Matrix to store the reconstructed image pixels:
img = zeros(Nx_rec,Ny_rec); 

for i = 1:size(img,1)

    % Display progress:
    clc
    recon_progress = floor((i-1)/(size(img,1)-1)*1000)/10;
    disp([num2str(recon_progress) ' % of image reconstructed']); 

    for j=1:size(img,2)
        
        X = x_rec(i);               % target point x-coordinate
        Y = y_rec(j);               % target point y-coordinate

        % Time delay for point (X,Y)
        t_del = Y/c + sqrt((X-x_el).^2 + Y^2)/c;

        % Corresponding sample points:
        Ntimedel = round(t_del*Fs + sig_dur);
        % NOTE: for a linear scatterer, we should add sig_dur/2 to the time
        % delay. However, it appears that the maximum in the scattered
        % pressure of the bubble lags behind the maximum in the drive
        % pulse. Adding sig_dur seems to work pretty well.
        
        % Time delays out of range do not contribute:
        Ntimedel(Ntimedel > N) = N+1;   
        Ntimedel(Ntimedel < 1) = N+1;

        % Add signal contributions per element
        val = 0;
        for g = 1:Nelem
            val = val+RF_data(g,Ntimedel(g));
        end
        img(i,j) = val;

    end
end

end