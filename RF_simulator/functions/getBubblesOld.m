function [bubble, shell] = getBubblesOld(bubble,R0,sR0,liquid,...
    shell_model,sig_0,r0)
% Get bubble locations from file (reload from earlier random distribution).
% Add a random radius to the bubbles. The bubble have a mean radius R0 with
% a standard deviation of sR0. Based on the radius of each bubble, compute
% the shell properties. 
%
% Nathan Blanken, University of Twente, 2020

zb = [bubble.z];  	% Bubble axial coordinates (m)            
xb = [bubble.x];   	% Bubble lateral coordinates (m)

Nb = length(bubble);                % Number of bubbles

R0v = R0 + sR0*randn(1,Nb);     	% Bubble radii (um);
R0v(R0v<0.5*R0) = 0.5*R0;           % Keep radii within limits
R0v(R0v>1.5*R0) = 1.5*R0;
R0v = R0v/1e6;                      % Bubble radii (m);

r0v = r0*ones(1,Nb);                % Distance bubble to pressure sensor

bubble = struct('z',num2cell(zb),'x',num2cell(xb),'R0',num2cell(R0v),...
    'r0',num2cell(r0v));

% shell_model: Marmottant, Segers, or SegersTable:
shell.model = shell_model;

% Equilibrium surface tension bubble (N/m):
shell.sig_0 = sig_0; 

% Preallocate shell struct:
shell = getShellProperties(bubble(Nb),shell,liquid);
shell(Nb) = shell;

% Typical value intial surface tension: Sijl et al., J. Acoust. Soc.
% Am., 129, 1729 (2011)

for k = 1:length(bubble)
    shell(k).model = shell_model;
    shell(k).sig_0 = sig_0;        
    shell(k) = getShellProperties(bubble(k),shell(k),liquid);
end

end

