function pfield = computepressurefield(pulse,liquid,domain,Fs,downsample)
% Compute the acoustic pressure amplitude as a function of the depth z.

    z = 0:domain.c/(2*Fs/downsample):domain.depth;  % Axial coordinate (m)

    M = length(z);
    PAlocal = zeros(1,M);

    for k = 1:M
        pulseLocal = pulsePropagation(pulse, liquid, z(k));
        PAlocal(k) = max(abs(pulseLocal.p));
    end

    pfield.PA = PAlocal;
    pfield.z = z;
    
end