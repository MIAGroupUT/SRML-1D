function X = getRandomRadius(P,R,N)
% Draw N random numbers from a probability density function P(R).
% N: number of microbubbles (an integer)
% P: probability distribution
% R: list of radii

% Nathan Blanken, University of Twente, 2022

% Normalize the distribution if necessary
if sum(P)>1.01
    warning(['Sum of all probabilities greater than 1. '...
        'Normalizing probability density function.'])
    P = P/sum(P);
end

% Cumulative distribution
Pcdf = cumsum(P);

% Make cumulative probabilites on either side of the pdf equal, if 
% necessary:
if sum(P) < 0.99
    Pcdf = Pcdf + (1-sum(P))/2;
    warning(['Sum of all probabilities less than 1. '...
        'Assuming equal probability on each side of the probability '...
        'density function.'])
end


% Get N uniformly distributed numbers between 0 and 1
Y = rand(1,N);

% Convert random number to random number from probability density function
% by interpolation of the cdf
X = zeros(1,length(Y));

for k = 1:length(Y)
        
    if Y(k) <= min(Pcdf)
        X(k) = R(1);
    elseif Y(k) >= max(Pcdf)
        X(k) = R(end);
    else

        X(k) = interp1(Pcdf,R,Y(k));

    end

end

end
