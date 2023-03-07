%hPRGPrecode precoding for PDSCH Precoding Resource Block Group (PRG) bundling
%   [ANTSYM,ANTIND] = hPRGPrecode(SIZ,NSTARTGRID,PORTSYM,PORTIND,F)
%   performs precoding for PDSCH PRG bundling according to TS 38.214
%   Section 5.1.2.3, to produce precoded symbols ANTSYM and corresponding
%   indices ANTIND. ANTSYM is a matrix of size NRE-by-P where NRE is number
%   of PDSCH resource elements, and P is the number of antennas. ANTIND is
%   also of size NRE-by-P.
%
%   SIZ is the size of the carrier resource grid to which ANTIND applies,
%   [K L P] where K is the number of subcarriers and L is the number of
%   OFDM symbols. 
%
%   NSTARTGRID is the starting RB of the carrier resource grid, relative to 
%   CRB 0. This is required because Precoding Resource Block Groups (PRGs)
%   are aligned with CRB 0. 
%
%   PORTSYM is a matrix of symbols to be precoded, of size NRE-by-NLAYERS
%   where NRE is the number of resource elements per layer and NLAYERS is
%   the number of layers.
% 
%   PORTIND is a matrix of the same size as PORTSYM, NRE-by-NLAYERS,
%   containing the 1-based linear indices of the symbols in PORTSYM. The
%   indices address a K-by-L-by-NLAYERS resource array, where K and L are
%   given by SIZ. The precoding performed by this function assumes that TS
%   38.211 Section 7.3.1.4 simply maps layers to ports, i.e. layers
%   0...NLAYERS-1 correspond to ports 0...NLAYERS-1 and therefore the
%   columns of PORTIND should address resource array planes 0...NLAYERS-1.
%
%   F is an array of size NLAYERS-by-P-by-NPRG, where NPRG is the number of
%   PRGs in the carrier (NRB = K/12 resource blocks). F defines a different
%   precoding matrix of size NLAYERS-by-P for each PRG. The effective PRG
%   bundle size (precoder granularity) is Pd_BWP = ceil(NRB / NPRG). Valid
%   PRG bundle sizes given in TS 38.214 Section 5.1.2.3, and the
%   corresponding values of NPRG, are as follows:
%   Pd_BWP = 2 (NPRG = ceil(NRB / 2))
%   Pd_BWP = 4 (NPRG = ceil(NRB / 4))
%   Pd_BWP = 'wideband' (NPRG = 1)
%
%   Note that although the 2nd dimension size of PORTSYM and PORTIND is
%   described in terms of number of layers NLAYERS, PORTSYM and PORTIND can
%   also contain dimensions corresponding to other planes, provided that
%   the final dimension size matches the first dimension size of F. A
%   typical example would be precoding of an NRE-by-R-by-P channel estimate
%   using P-by-NLAYERS matrices for each PRG bundle (the transpose of the
%   transmit precoding matrices) to yield an NRE-by-R-by-NLAYERS channel
%   estimate giving the "effective channel" between receive antennas and
%   transmit layers.
%   
%   Example:
%   % Perform PDSCH precoding using a bundle size of 4 PRBs.
%   
%   % Configure carrier and number of transmit antennas
%   carrier = nrCarrierConfig;
%   carrier.NSizeGrid = 18;
%   carrier.NStartGrid = 1;
%   ntxants = 4;
%
%   % Create carrier grid
%   grid = nrResourceGrid(carrier,ntxants);
%
%   % Configure PDSCH parameters
%   pdsch = nrPDSCHConfig;
%   pdsch.Modulation = 'QPSK';
%   pdsch.NumLayers = 2;
%   pdsch.DMRS.NumCDMGroupsWithoutData = 1;
%
%   % Create PDSCH indices for a PDSCH allocated in RBs 2 to 16 (0-based),
%   % and 7 OFDM symbols starting at symbol 1 (0-based)
%   pdsch.PRBSet = 2:16;
%   pdsch.SymbolAllocation = [1 7];
%   [portind,indinfo] = nrPDSCHIndices(carrier,pdsch);
%
%   % Create PDSCH codeword and perform PDSCH modulation
%   cw = randi([0 1],indinfo.G,1);
%   portsym = nrPDSCH(carrier,pdsch,cw);
%
%   % Create precoding matrices (identity matrices with different scaling
%   % per PRG, for illustration purposes)
%   prgBundleSize = 4;
%   NPRG = ceil(carrier.NSizeGrid / prgBundleSize);
%   F = eye(pdsch.NumLayers,ntxants);
%   F = reshape(kron(2:(NPRG+1),F),pdsch.NumLayers,ntxants,[]);
%
%   % Perform PRG precoding
%   [antsym,antind] = hPRGPrecode(size(grid),carrier.NStartGrid,portsym,portind,F);
%
%   % Map the precoded PDSCH to the carrier grid
%   grid(antind) = antsym;
%   
%   % Plot the carrier grid to illustrate the PRGs (nominally 4 RBs each).
%   % Note that the PRGs are aligned to CRB 0, so only the last PRB of the
%   % first PRG overlaps with the PDSCH allocation - the first CRB (0) is
%   % outside the carrier grid because carrier.NStartGrid=1, and the next 
%   % two CRBs (1 and 2) are outside the PDSCH allocation because 
%   % pdsch.PRBSet starts as 2. Note also that the last PRG in the 
%   % allocation can be partial, depending on the allocation size and 
%   % alignment
%   figure;
%   imagesc(abs(grid(:,:,1)));
%   axis xy;
%   title('PDSCH PRG bundling');
%   xlabel('OFDM symbols');
%   ylabel('Subcarriers');
%   hold on;
%   for i = 1:NPRG
%      patch([-2 -3 -3 -2],[-2 -2 -3 -3],i+1);
%   end
%   legend("PRG " + (1:NPRG));
%
%   See also nrExtractResources.

%   Copyright 2020-2021 The MathWorks, Inc.

function [antsym,antind] = hPRGPrecode(siz,nstartgrid,portsym,portind,F)
    
    % Get the number of precoder resource groups NPRG from the precoder
    % array F
    NPRG = size(F,3);
    
    % Get the number of resource blocks from the size vector
    NRB = siz(1) / 12;
    
    % Get the PRG numbers (1-based) for each CRB in the whole carrier
    prgset = getPRGSet(NRB,nstartgrid,NPRG);
    
    % Establish the dimensionality of the grid of port indices
    portsiz = siz;
    ndims = max(length(portsiz),3);
    portsiz(end) = size(F,1);
    
    % Calculate 1-based RE subscripts from port indices
    [subs{1:ndims}] = ind2sub(portsiz,portind);
    resubs = subs{1};
    
    % Calculate 0-based CRB subscripts from RE subscripts
    crbsubs = floor((resubs - 1) / 12);
    
    % Calculate 1-based PRG subscripts from CRB subscripts
    prgsubs = prgset(crbsubs + 1);

    % Perform precoding to produce antenna symbols and antenna indices
    [antsym,antind] = hPrecode(siz,portsym,portind,F,prgsubs);

end

% Get PRG numbers (1-based) for a carrier with NRB resource blocks,
% starting CRB 'nstartgrid' and NPRG precoder resource groups
function prgset = getPRGSet(NRB,nstartgrid,NPRG)

    Pd_BWP = ceil((NRB + nstartgrid) / NPRG);
    prgset = repmat(1:NPRG,[Pd_BWP 1]);
    prgset = reshape(prgset(nstartgrid + (1:NRB).'),[],1);

end

% Precoding of symbols and projection of the corresponding indices
function [symout,indout] = hPrecode(siz,symin,indin,F,prgsubs)
    
    % Dimensionality information
    ndims = max(length(siz),3);
    outplanedim = size(F,2);
    NPRG = size(F,3);
    outdims = [siz(1:(ndims-1)) outplanedim];
    
    % Create empty antenna grid
    antgrid = zeros(outdims);
    
    % For each PRG
    for prg = 1:NPRG
        
        % Create array of logical indices indicating which indices and 
        % symbols correspond to this PRG
        thisprg = (prgsubs==prg);
        
        % Create empty port grid
        portgrid = zeros([prod(outdims(1:end-1)) size(F,1)]);
        
        % Assign symbols for this PRG into the port grid
        portgrid(indin(thisprg)) = symin(thisprg);
        
        % Beamform the port grid with the precoder for this PRG
        bfgrid = portgrid * F(:,:,prg);
        
        % Reshape the beamformed grid and add to the antenna grid
        antgrid = antgrid + reshape(bfgrid,outdims);
        
    end
    
    % Extract antenna symbols and antenna indices
    [symout,indout] = nrExtractResources(indin,antgrid);

end
