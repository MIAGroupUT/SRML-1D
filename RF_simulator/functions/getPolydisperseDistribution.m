function [R,P] = getPolydisperseDistribution()
% Polydisperse size distribution of SonoVue, see:
%
% Segers, Tim, et al. "Monodisperse versus polydisperse ultrasound contrast
% agents: Non-linear response, sensitivity, and deep tissue imaging 
% potential." Ultrasound Med. Biol. 44(7), 2018, 1482-1492.
%
% Only consider microbubbles between 0.5 um and 6 um.
%
% Nathan Blanken, University of Twente, 2022

N = 1e3;
R = linspace(0.5,6,N+1);    	% Radii (um)
a = 2.19;
P = R.^2.*exp(-a*R);        	% Size distribution
P = P/sum(P);                   % Normalize size distribution

end