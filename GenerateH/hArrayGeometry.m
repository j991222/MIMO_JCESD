%hArrayGeometry antenna array geometry for CDL channel model

%   Copyright 2018-2020 The MathWorks, Inc.

function [txArrayGeometry,rxArrayGeometry] = hArrayGeometry(nTxAnts,nRxAnts,varargin)
    
    if (nargin==2)
        linkDirection = 'downlink';
    else
        linkDirection = varargin{1};
    end
    if (strcmpi(linkDirection,'downlink'))
        txArrayGeometry = bsArrayGeometry(nTxAnts);
        rxArrayGeometry = ueArrayGeometry(nRxAnts);
    else % uplink
        txArrayGeometry = ueArrayGeometry(nTxAnts);
        rxArrayGeometry = bsArrayGeometry(nRxAnts);
    end
    
end

function bsArray = bsArrayGeometry(nBsAnts)

    % Setup the base station antenna geometry
    % Table of antenna panel array configurations
    % M:  no. of rows in each antenna panel
    % N:  no. of columns in each antenna panel
    % P:  no. of polarizations (1 or 2)
    % Mg: no. of rows in the array of panels
    % Ng: no. of columns in the array of panels
    % Row format: [M  N   P   Mg  Ng]
    antarrays = ...
       [1   1   1   1   1;   % 1 ants
        1   1   2   1   1;   % 2 ants
        2   1   2   1   1;   % 4 ants
        2   2   2   1   1;   % 8 ants
        2   4   2   1   1;   % 16 ants
        4   4   2   1   1;   % 32 ants
        4   4   2   1   2;   % 64 ants
        4   8   2   1   2;   % 128 ants
        4   8   2   2   2;   % 256 ants
        8   8   2   2   2;   % 512 ants
        8  16   2   2   2];  % 1024 ants
    antselected = min(1+ceil(log2(nBsAnts)),size(antarrays,1));
    bsArray = antarrays(antselected,:);
        
    if prod(bsArray) ~= nBsAnts
        pwarningstr = ['The number of transmit antenna elements configured '...
                       '(NTxAnts = %d) is not one of the set '...
                       '(1,2,4,8,16,32,64,128,256,512,1024). '...
                       'Using NTxAnts = %d instead.'];
        warning('nr5g:hArrayGeometry:nBsAnts',pwarningstr, nBsAnts, prod(bsArray))
    end
end

function ueArray = ueArrayGeometry(nUeAnts)

    % Setup the UE antenna geometry
    if nUeAnts == 1
        % In the following settings, the number of rows in antenna array, 
        % columns in antenna array, polarizations, row array panels and the
        % columns array panels are all 1
        ueArray = ones(1,5);
    else
        % In the following settings, the no. of rows in antenna array is
        % nRxAntennas/2, the no. of columns in antenna array is 1, the no.
        % of polarizations is 2, the no. of row array panels is 1 and the
        % no. of column array panels is 1. The values can be changed to
        % create alternative antenna setups
        ueArray = [ceil(nUeAnts/2),1,2,1,1];
    end
    
    if prod(ueArray) ~= nUeAnts
        pwarningstr = ['The number of receive antenna elements configured '...
                       '(NRxAnts = %d) is not even. '...
                       'Using NRxAnts = %d instead.'];
        warning('nr5g:hArrayGeometry:nUeAnts',pwarningstr, nUeAnts, prod(ueArray))
    end
    
end
