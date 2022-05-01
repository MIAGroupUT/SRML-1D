function A = read_rf_txt(filepath,NumColumns)
% Import RF data from text file

% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", NumColumns);

% Specify range and delimiter

opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableTypes = repmat("double",1,NumColumns);

% Import the data
A = readtable(filepath, opts);

% Convert to output type
A = table2array(A);


end