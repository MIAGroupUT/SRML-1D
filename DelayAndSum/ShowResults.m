% Script to create the delay-and-sum figure in the manuscript.

close all

% Load the metadata from the original MATLAB files:
filename = 'RFDATA2D00001.mat';
load(filename)

% SELECT REGIONS OF INTEREST:
% Near-field region:
x1_1 = 8;
x1_2 = 18;
y1_1 = 0;
y1_2 = 10;

% Far-field region:
x2_1 = 85;
x2_2 = 95;
y2_1 = 5;
y2_2 = 15;

%% SHOW THE DIFFRACION-LIMITED RECONSTRUCTION:
load('RFDATA2D00001_DAS.mat')

super_resolved = false;
showbubbles = false;

% DEMODULATION
% Demodulation of the signals
img_demod = zeros(size(img));
for k = 1:size(img_demod,1)
    img_demod(k,:) = abs(hilbert(img(k,:)));
end

% LOG COMPRESSION
maximg = max(max(img_demod));
img_log = 20*log10(img_demod/maximg);

% SHOW RECONSTRUCTION
figure(1)
show_reconstruction(z,x,img_log,super_resolved,bubble,showbubbles)

% Highlight the regions of interest:
hold on
rectangle('Position',[x1_1 y1_1 (x1_2-x1_1) (y1_2-y1_1)],'EdgeColor','r')
rectangle('Position',[x2_1 y2_1 (x2_2-x2_1) (y2_2-y2_1)],'EdgeColor','r')
ylim([-20 20])

% SHOW ZOOMED RECONSTRUCTION IN THE NEAR-FIELD
figure(2);
show_reconstruction(z,x,img_log,super_resolved,bubble,showbubbles)
xlim([x1_1 x1_2])
ylim([y1_1 y1_2])
set(gcf, 'Position', [50 50 350 350]);

% SHOW ZOOMED RECONSTRUCTION IN THE FAR-FIELD
figure(3);
show_reconstruction(z,x,img_log,super_resolved,bubble,showbubbles)
xlim([x2_1 x2_2])
ylim([y2_1 y2_2])

set(gcf, 'Position', [50 50 350 350]);

% SHOW CROSS-SECTION
I  = 1714;
J1 = 600;
J2 = 4850;
y = img_demod(I,J1:J2);

figure(7)
plot(z(J1:J2)*1000,y/max(y))
hold on


%% SHOW THE SUPER-RESOLVED RECONSTRUCTION

load('RFDATA2D00001_DAS_sr.mat')
img_demod = abs(img);
super_resolved = true;

% SHOW RECONSTRUCTION
figure(4);
show_reconstruction(z,x,img_demod,super_resolved,bubble,showbubbles)

% Highlight the regions of interest:
hold on
rectangle('Position',[x1_1 y1_1 (x1_2-x1_1) (y1_2-y1_1)],'EdgeColor','r')
rectangle('Position',[x2_1 y2_1 (x2_2-x2_1) (y2_2-y2_1)],'EdgeColor','r')
ylim([-20 20])

% SHOW ZOOMED RECONSTRUCTION IN THE NEAR-FIELD
figure(5);
showbubbles = true;
show_reconstruction(z,x,img_demod,super_resolved,bubble,showbubbles)
xlim([x1_1 x1_2])
ylim([y1_1 y1_2])
set(gcf, 'Position', [50 50 350 350])

% SHOW ZOOMED RECONSTRUCTION IN THE FAR-FIELD
figure(6);
show_reconstruction(z,x,img_demod,super_resolved,bubble,showbubbles)
xlim([x2_1 x2_2])
ylim([y2_1 y2_2])
set(gcf, 'Position', [50 50 350 350])

% Show cross-section
y = img_demod(I,J1:J2);
figure(7)
plot(z(J1:J2)*1000,y/max(y))
xlim([11 14])
hold on

%% FUNCTIONS

function show_reconstruction(z,x,img,super_resolved,bubble,showbubbles)

if super_resolved == false
    dyn_range = -30;                        % Dynamic range in dB
    fig_title = 'Normal resolution';        % Title for the figure
    colob_title = 'image intensity (dB)';   % Title for the colorbar
    cmap = 'gray';                          % Colormap
    
	% Colorbar limits:
    c1 = dyn_range;                         % Lower limit colorbar
    c2 = 0;                                 % Upper limit colorbar
else
    cmap = 'gray';
    cmap = colormap(cmap);
    cmap = colormap(flipud(cmap));          % Invert colormap
    
    fig_title = 'Super-resolved';           % Title for the figure
    colob_title = ['image intensity,'...    % Title for the colorbar
        newline 'linear scale (a.u.)'];
    
    % Colorbar limits:
    c1 = 20;                                % Lower limit colorbar
    c2 = 30;                                % Upper limit colorbar
end


imagesc(z.*1e3,x.*1e3,img);

if showbubbles
    % Show the bubbles
    hold on
    for k = 1:length(bubble)
       plot(bubble(k).z*1e3, bubble(k).x*1e3, 'ro') 
    end

    legend('Bubbles')
end

caxis([c1 c2]);
colob = colorbar;
ylabel('lateral distance (mm)','interpreter', 'latex','fontsize',16)
xlabel('axial distance (mm)','interpreter', 'latex','fontsize',16)
ylabel(colob,colob_title,'interpreter', 'latex','fontsize',16);
title(fig_title)
axis equal
drawnow
colormap(cmap);

end
