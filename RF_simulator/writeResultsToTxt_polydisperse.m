% Write arrays from RF simulation .mat files to a .txt file. 
% Two column version (acoustic pressure field constant)
%
% Column 1: bubble count (number of bubbles at each sample point)
% Column 2: RF voltage values
% Column 3: pressure field distribution
%
% Nathan Blanken, University of Twente, 2020

NTRN = 3000;    % Number of training data files
NVAL = 1000;    % Number of validation data files

% source directory:
filedir_src = ['/home/blankenn/SuperResolutionProjectREVISION/'...
    'RF_simulatorPOLYDISPERSE_RESULTS'];

% destination directory
filedir_dst = ['/home/blankenn/SuperResolutionProjectREVISION/'...
    'DATA_POLYDISPERSE']; 

% List files in source directory
filelist = dir(filedir_src);
filelist = filelist(3:end);

N = length(filelist);

for n = 1:N
    
    clc
    disp(n)
    
    % Source file
    filename_src = filelist(n).name;
    filenumber = str2double(filename_src(7:end-4));
    
    % Create name destination file
    filename_dst = strcat(filename_src(1:end-4),'.txt');

    % Load data
    load(strcat(filedir_src,'/',filename_src))
    
    % Transform data
    RFvoltage = RF.V;  
    bubbleCount = getBubbleCount(bubble,RF,domain);
    pressureField = pfield.PA;  % (in Pa)
    
    Nb = length(bubble);   % Total number of bubbles
    PA = pulse.A/1e3;       % Acoustic pressure (kPa)
    
    % Sort data in training, validation, and test data:
    subfolder = '/TRAINING';
    if filenumber > NTRN
        subfolder = '/VALIDATION';
    end
    if filenumber > NTRN + NVAL
        subfolder = '/TESTING';
    end

    % Write data to text file
    A = [bubbleCount; RFvoltage; pressureField];
    fileID = fopen(strcat(filedir_dst,subfolder,'/',filename_dst),'w');
    
    % Write header:
    fprintf(fileID,['"Generated with writeResultsToTxt.m from ' ...
        filename_src '"\n']); 
    
    fprintf(fileID,'"Number of bubbles:",%d\n',Nb);
    fprintf(fileID,'"Acoustic pressure (kPa):",%.2f\n',PA);
    
    fprintf(fileID,['"Bubble count","Voltage (a.u)",'...
        '"Pressure field (Pa)"\n']);
    
    % Write data:
    fprintf(fileID,'%d,%.10f,%.0f\n',A);
    fclose(fileID);

end

function bubbleCount = getBubbleCount(bubble,RF,domain)
% Get number of bubbles at each sample point

    Nb = length(bubble);        % number of bubbles
    N  = length(RF.V);          % number of sample points

    % Get arrays of bubble locations
    z = [bubble.z];             % Axial coordinates (m)
    x = [bubble.x];             % Lateral coordinates (m)
    r = sqrt(x.^2 + z.^2);      % Distance from centre element (m)

    % Convert to RF sample number
    t = (z+r)/domain.c;         % Echo arrival time (s)
    I = t*RF.fs;                % Convert to sample number
    I = round(I)+1;
    
    T = 2*domain.depth/domain.c;    % Total receive time
    RF.t = 0:1/RF.fs:T;          	% Time vector

    % Construct bubble count array
    bubbleCount = zeros(1,N);
    for n = 1:Nb
        bubbleCount(I(n)) = bubbleCount(I(n)) + 1;
    end

end