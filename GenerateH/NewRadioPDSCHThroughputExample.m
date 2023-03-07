addpath('/opt/software/MATLAB/R2019a/examples/5g/main/')
%% NR PDSCH Throughput 
% This example shows how to measure the physical downlink shared channel
% (PDSCH) throughput of a 5G New Radio (NR) link, as defined by the 3GPP NR
% standard. The example implements the PDSCH and downlink shared channel
% (DL-SCH). The transmitter model includes PDSCH demodulation reference
% signals (DM-RS), PDSCH phase tracking reference signals (PT-RS), and
% synchronization signal (SS) bursts. The example supports both clustered
% delay line (CDL) and tapped delay line (TDL) propagation channels. You
% can perform perfect or practical synchronization and channel estimation.
% To reduce the total simulation time, you can execute the SNR points in
% the SNR loop in parallel by using the Parallel Computing Toolbox(TM). 

% Copyright 2017-2021 The MathWorks, Inc.

%% Introduction
% This example measures the PDSCH throughput of a 5G link, as defined by
% the 3GPP NR standard [ <#14 1> ], [ <#14 2> ], [ <#14 3> ], [ <#14 4> ].
% 
% The example models these 5G NR features:
% 
% * DL-SCH transport channel coding
% * Multiple codewords, dependent on the number of layers
% * PDSCH, PDSCH DM-RS, and PDSCH PT-RS generation
% * SS burst generation (PSS/SSS/PBCH/PBCH DM-RS)
% * Variable subcarrier spacing and frame numerologies (2^n * 15 kHz) for
% normal and extended cyclic prefix
% * TDL and CDL propagation channel models
% 
% Other features of the simulation are:
% 
% * PDSCH precoding using SVD
% * CP-OFDM modulation
% * Slot wise and non slot wise PDSCH and DM-RS mapping
% * SS burst generation (cases A-E, SS/PBCH block bitmap control)
% * Perfect or practical synchronization and channel estimation
% * HARQ operation with 16 processes
% * The example uses a single bandwidth part across the whole carrier
% 
% The figure shows the implemented processing chain. For clarity, the 
% DM-RS, PT-RS, and SS burst generation are omitted. 
% 
% <<../PDSCHLinkExampleProcessingChain.png>>
% 
% This example supports both wideband and subband precoding. The precoding
% matrix is determined using SVD by averaging the channel estimate across
% all PDSCH PRBs in the allocation (wideband case) or in the subband.
% There is no beamforming on any SS/PBCH blocks in the SS burst.
% 
% To reduce the total simulation time, you can use the Parallel Computing
% Toolbox to execute the SNR points of the SNR loop in parallel.

%% Simulation Length and SNR Points
% Set the length of the simulation in terms of the number of 10ms frames.
% A large number of NFrames should be used to produce meaningful throughput
% results. Set the SNR points to simulate. The SNR for each layer is
% defined per RE, and it includes the effect of signal and noise across all
% antennas.

simParameters = struct();       % Clear simParameters variable to contain all key simulation parameters 
simParameters.NFrames = 2;      % Number of 10 ms frames
simParameters.SNRIn = [-5 0 5]; % SNR range (dB)

%% Channel Estimator Configuration
% The logical variable |PerfectChannelEstimator| controls channel
% estimation and synchronization behavior. When set to |true|, perfect
% channel estimation and synchronization is used. Otherwise, practical
% channel estimation and synchronization is used, based on the values of
% the received PDSCH DM-RS.

simParameters.PerfectChannelEstimator = true;

%% Simulation Diagnostics
% The simulation always displays the CRC pass/fail result of the PDSCH 
% transmission for the HARQ process used in each slot. This includes the 
% RV value used and the instantaneous code rate. Note that the code rate
% may differ from the process target code rate, especially in a slot 
% containing SS blocks when there is a loss of physical resources available
% to the PDSCH. This increase in code rate may require further transport
% block retransmissions for successful reception, even at high SNR.
%
% The |DisplayDiagnostics| flag enables the plotting of the EVM per layer.
% This plot monitors the quality of the received signal after equalization.
% The EVM per layer figure shows:
%
% * The EVM per layer per slot, which shows the EVM evolving with time.
% * The EVM per layer per resource block, which shows the EVM in frequency.
%
% This figure evolves with the simulation and is updated with each slot. 
% Typically, low SNR or channel fades can result in decreased signal 
% quality (high EVM). The channel affects each layer differently,
% therefore, the EVM values may differ across layers.
%
% In some cases, some layers can have a much higher EVM than others. These
% low-quality layers can result in CRC errors. This behavior may be caused
% by low SNR or by using too many layers for the channel conditions. You
% can avoid this situation by a combination of higher SNR, lower number
% of layers, higher number of antennas, and more robust transmission 
% (lower modulation scheme and target code rate).

simParameters.DisplayDiagnostics = false;

%% Carrier and PDSCH Configuration
% Set the key parameters of the simulation. These include:
% 
% * The bandwidth in resource blocks (12 subcarriers per resource block).
% * Subcarrier spacing: 15, 30, 60, 120, 240 (kHz)
% * Cyclic prefix length: normal or extended
% * Cell ID
% * Number of transmit and receive antennas
% 
% A substructure containing the DL-SCH and PDSCH parameters is also
% specified. This includes:
% 
% * Target code rate
% * Allocated resource blocks (PRBSet)
% * Modulation scheme: 'QPSK', '16QAM', '64QAM', '256QAM'
% * Number of layers
% * PDSCH mapping type
% * DM-RS configuration parameters
% * PT-RS configuration parameters
% 
% Other simulation wide parameters are:
% 
% * Propagation channel model: 'TDL' or 'CDL'
% * SS burst configuration parameters. Note that the SS burst generation
% can be disabled by setting the |SSBTransmitted| field to [0 0 0 0].

% Set waveform type and PDSCH numerology (SCS and CP type)

Ideal_H = zeros(20000,12,24,4,2);

simParameters.Carrier = nrCarrierConfig;
simParameters.Carrier.NSizeGrid = 51;            % Bandwidth in number of resource blocks (51 RBs at 30 kHz SCS for 20 MHz BW)
simParameters.Carrier.SubcarrierSpacing = 30;    % 15, 30, 60, 120, 240 (kHz)
simParameters.Carrier.CyclicPrefix = 'Normal';   % 'Normal' or 'Extended' (Extended CP is relevant for 60 kHz SCS only)
simParameters.Carrier.NCellID = 1;               % Cell identity

% SS burst configuration
% The burst can be disabled by setting the SSBTransmitted field to all zeros
simParameters.SSBurst = struct();
simParameters.SSBurst.BlockPattern = 'Case B';    % 30 kHz subcarrier spacing
simParameters.SSBurst.SSBTransmitted = [0 1 0 1]; % Bitmap indicating blocks transmitted in the burst
simParameters.SSBurst.SSBPeriodicity = 20;        % SS burst set periodicity in ms (5, 10, 20, 40, 80, 160)

% PDSCH/DL-SCH parameters
simParameters.PDSCH = nrPDSCHConfig;      % This PDSCH definition is the basis for all PDSCH transmissions in the BLER simulation
simParameters.PDSCHExtension = struct();  % This structure is to hold additional simulation parameters for the DL-SCH and PDSCH

% Define PDSCH time-frequency resource allocation per slot to be full grid (single full grid BWP)
simParameters.PDSCH.PRBSet = 0:simParameters.Carrier.NSizeGrid-1;                 % PDSCH PRB allocation
simParameters.PDSCH.SymbolAllocation = [0,simParameters.Carrier.SymbolsPerSlot];  % Starting symbol and number of symbols of each PDSCH allocation
simParameters.PDSCH.MappingType = 'A';     % PDSCH mapping type ('A'(slot-wise),'B'(non slot-wise))

% Scrambling identifiers
simParameters.PDSCH.NID = simParameters.Carrier.NCellID;
simParameters.PDSCH.RNTI = 1;

% PDSCH resource block mapping (TS 38.211 Section 7.3.1.6)
simParameters.PDSCH.VRBToPRBInterleaving = 0; % Disable interleaved resource mapping
simParameters.PDSCH.VRBBundleSize = 4;

% Define the number of transmission layers to be used
simParameters.PDSCH.NumLayers = 1;            % Number of PDSCH transmission layers

% Define codeword modulation and target coding rate
% The number of codewords is directly dependent on the number of layers so ensure that 
% layers are set first before getting the codeword number
if simParameters.PDSCH.NumCodewords > 1                             % Multicodeword transmission (when number of layers being > 4)
    simParameters.PDSCH.Modulation = {'16QAM','16QAM'};             % 'QPSK', '16QAM', '64QAM', '256QAM'
    simParameters.PDSCHExtension.TargetCodeRate = [490 490]/1024;   % Code rate used to calculate transport block sizes
else
    simParameters.PDSCH.Modulation = 'QPSK';                       % 'QPSK', '16QAM', '64QAM', '256QAM'
    simParameters.PDSCHExtension.TargetCodeRate = 490/1024;         % Code rate used to calculate transport block sizes
end

% DM-RS and antenna port configuration (TS 38.211 Section 7.4.1.1)
simParameters.PDSCH.DMRS.DMRSPortSet = 0:simParameters.PDSCH.NumLayers-1; % DM-RS ports to use for the layers
simParameters.PDSCH.DMRS.DMRSTypeAPosition = 2;      % Mapping type A only. First DM-RS symbol position (2,3)
simParameters.PDSCH.DMRS.DMRSLength = 1;             % Number of front-loaded DM-RS symbols (1(single symbol),2(double symbol))
simParameters.PDSCH.DMRS.DMRSAdditionalPosition = 0; % Additional DM-RS symbol positions (max range 0...3)
simParameters.PDSCH.DMRS.DMRSConfigurationType = 2;  % DM-RS configuration type (1,2)
simParameters.PDSCH.DMRS.NumCDMGroupsWithoutData = 1;% Number of CDM groups without data
simParameters.PDSCH.DMRS.NIDNSCID = 1;               % Scrambling identity (0...65535)
simParameters.PDSCH.DMRS.NSCID = 0;                  % Scrambling initialization (0,1)

% PT-RS configuration (TS 38.211 Section 7.4.1.2)
simParameters.PDSCH.EnablePTRS = 0;                  % Enable or disable PT-RS (1 or 0)
simParameters.PDSCH.PTRS.TimeDensity = 1;            % PT-RS time density (L_PT-RS) (1, 2, 4)
simParameters.PDSCH.PTRS.FrequencyDensity = 2;       % PT-RS frequency density (K_PT-RS) (2 or 4)
simParameters.PDSCH.PTRS.REOffset = '00';            % PT-RS resource element offset ('00', '01', '10', '11')
simParameters.PDSCH.PTRS.PTRSPortSet = [];           % PT-RS antenna port, subset of DM-RS port set. Empty corresponds to lower DM-RS port number

% Reserved PRB patterns, if required (for CORESETs, forward compatibility etc)
simParameters.PDSCH.ReservedPRB{1}.SymbolSet = [];   % Reserved PDSCH symbols
simParameters.PDSCH.ReservedPRB{1}.PRBSet = [];      % Reserved PDSCH PRBs
simParameters.PDSCH.ReservedPRB{1}.Period = [];      % Periodicity of reserved resources

% Additional simulation and DL-SCH related parameters
%
% PDSCH PRB bundling (TS 38.214 Section 5.1.2.3)
simParameters.PDSCHExtension.PRGBundleSize = [];      % 2, 4, or [] to signify "wideband"
% HARQ process and rate matching/TBS parameters
simParameters.PDSCHExtension.XOverhead = 6*simParameters.PDSCH.EnablePTRS; % Set PDSCH rate matching overhead for TBS (Xoh) to 6 when PT-RS is enabled, otherwise 0
simParameters.PDSCHExtension.NHARQProcesses = 16;     % Number of parallel HARQ processes to use
simParameters.PDSCHExtension.EnableHARQ = true;       % Enable retransmissions for each process, using RV sequence [0,2,3,1]

% LDPC decoder parameters
% Available algorithms: 'Belief propagation', 'Layered belief propagation', 'Normalized min-sum', 'Offset min-sum'
simParameters.PDSCHExtension.LDPCDecodingAlgorithm = 'Layered belief propagation';
simParameters.PDSCHExtension.MaximumLDPCIterationCount = 6;

% Define the overall transmission antenna geometry at end-points
% If using a CDL propagation channel then the integer number of antenna elements is
% turned into an antenna panel configured when the channel model object is created
simParameters.NTxAnts = 1;                        % Number of PDSCH transmission antennas (1,2,4,8,16,32,64,128,256,512,1024) >= NumLayers
if simParameters.PDSCH.NumCodewords > 1           % Multi-codeword transmission
    simParameters.NRxAnts = 4;                    % Number of UE receive antennas (even number >= NumLayers)
else
    simParameters.NRxAnts = 4;                    % Number of UE receive antennas (1 or even number >= NumLayers)
end

% Define the general CDL/TDL propagation channel parameters
simParameters.DelayProfile = 'CDL-A';   % Use CDL-C model (Urban macrocell model)
simParameters.DelaySpread = 30e-9;
simParameters.MaximumDopplerShift = 5; 

% Cross-check the PDSCH layering against the channel geometry 
validateNumLayers(simParameters);

%% 
% The simulation relies on various pieces of information about the baseband 
% waveform, such as sample rate.

waveformInfo = nrOFDMInfo(simParameters.Carrier); % Get information about the baseband waveform after OFDM modulation step

%% Propagation Channel Model Construction
% Create the channel model object for the simulation. Both CDL and TDL channel 
% models are supported [ <#14 5> ].

% Constructed the CDL or TDL channel model object
if contains(simParameters.DelayProfile,'CDL','IgnoreCase',true)
    
    channel = nrCDLChannel; % CDL channel object
    
    
    % Turn the overall number of antennas into a specific antenna panel
    % array geometry. The number of antennas configured is updated when
    % nTxAnts is not one of (1,2,4,8,16,32,64,128,256,512,1024) or nRxAnts
    % is not 1 or even.
    [channel.TransmitAntennaArray.Size, channel.ReceiveAntennaArray.Size] = ...
        hArrayGeometry(simParameters.NTxAnts,simParameters.NRxAnts);
    nTxAnts = prod(channel.TransmitAntennaArray.Size);
    nRxAnts = prod(channel.ReceiveAntennaArray.Size);
    simParameters.NTxAnts = nTxAnts;
    simParameters.NRxAnts = nRxAnts;
else
    channel = nrTDLChannel; % TDL channel object
    
    % Set the channel geometry
    channel.NumTransmitAntennas = simParameters.NTxAnts;
    channel.NumReceiveAntennas = simParameters.NRxAnts;
end

% Assign simulation channel parameters and waveform sample rate to the object
channel.DelayProfile = simParameters.DelayProfile;
channel.DelaySpread = simParameters.DelaySpread;
channel.MaximumDopplerShift = simParameters.MaximumDopplerShift;
channel.SampleRate = waveformInfo.SampleRate;

%% 
% Get the maximum number of delayed samples by a channel multipath
% component. This is calculated from the channel path with the largest
% delay and the implementation delay of the channel filter. This is
% required later to flush the channel filter to obtain the received signal.

chInfo = info(channel);
maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate)) + chInfo.ChannelFilterDelay;

%% Reserve PDSCH Resources Corresponding to SS burst
% This section shows how to reserve resources for the transmission of the 
% SS burst.

% Get information about the SS burst configuration
% Some dependent parameter assignments are required first
simParameters.SSBurst.NCellID = simParameters.Carrier.NCellID;        
simParameters.SSBurst.SampleRate = waveformInfo.SampleRate;
ssbInfo = hSSBurstInfo(simParameters.SSBurst);

% Map the occupied subcarriers and transmitted symbols of the SS burst
% (defined in the SS burst numerology) to PDSCH PRBs and symbols in the
% PDSCH BWP/carrier numerology
[mappedPRB,mappedSymbols] = mapNumerology(ssbInfo.OccupiedSubcarriers,ssbInfo.OccupiedSymbols,ssbInfo.NRB,simParameters.Carrier.NSizeGrid,ssbInfo.SubcarrierSpacing,simParameters.Carrier.SubcarrierSpacing);
% Configure the PDSCH to reserve these resources so that the PDSCH
% transmission does not overlap the SS burst
reservation = nrPDSCHReservedConfig;
reservation.SymbolSet = mappedSymbols;
reservation.PRBSet = mappedPRB;
reservation.Period = simParameters.SSBurst.SSBPeriodicity * (simParameters.Carrier.SubcarrierSpacing/15); % Period in slots
simParameters.PDSCH.ReservedPRB{end+1} = reservation;

%% Processing Loop
% To determine the throughput at each SNR point, analyze the PDSCH data 
% per transmission instance using the following steps:
% 
% * _Update current HARQ process._ Check the CRC of the previous
% transmission for the given HARQ process. Determine whether a
% retransmission is required. If retransmission is not required, generate
% new data.
% * _Resource grid generation._ Perform channel coding by calling the
% <docid:5g_ref#mw_sysobj_nrDLSCH nrDLSCH> System object. The object
% operates on the input transport block and keeps an internal copy of the
% transport block in case a retransmission is required. Modulate the coded
% bits on the PDSCH by using the <docid:5g_ref#mw_function_nrPDSCH nrPDSCH>
% function. Then apply precoding to the resulting signal.
% * _Waveform generation._ OFDM modulate the generated grid.
% * _Noisy channel modeling._ Pass the waveform through a CDL or TDL
% fading channel. Add AWGN. The SNR is defined per RE at each UE
% antenna. For an SNR of 0 dB the signal and noise contribute equally to
% the energy per PDSCH RE per receive antenna.
% * _Perform synchronization and OFDM demodulation._ For perfect
% synchronization, reconstruct the channel impulse response to synchronize
% the received waveform. For practical synchronization, correlate the
% received waveform with the PDSCH DM-RS. Then OFDM demodulate the
% synchronized signal.
% * _Perform channel estimation._ For perfect channel estimation,
% reconstruct the channel impulse response and perform OFDM demodulation.
% For practical channel estimation, use the PDSCH DM-RS.
% * _Perform equalization and CPE compensation._ MMSE equalize the
% estimated channel. Estimate the common phase error (CPE) by using the
% PT-RS symbols, then correct the error in each OFDM symbol within the
% range of reference PT-RS OFDM symbols.
% * _Precoding matrix calculation._ Generate the precoding matrix W for the
% next transmission by using singular value decomposition (SVD).
% * _Decode the PDSCH._ To obtain an estimate of the received codewords,
% demodulate and descramble the recovered PDSCH symbols for all transmit
% and receive antenna pairs, along with a noise estimate, by using the
% <docid:5g_ref#mw_function_nrPDSCHDecode nrPDSCHDecode> function.
% * _Decode DL-SCH and store the block CRC error for a HARQ process._ Pass
% the vector of decoded soft bits to the
% <docid:5g_ref#mw_sysobj_nrDLSCHDecoder nrDLSCHDecoder> System object. The
% object decodes the codeword and returns the block CRC error used to
% determine the throughput of the system.

% Array to store the maximum throughput for all SNR points
maxThroughput = zeros(length(simParameters.SNRIn),1); 
% Array to store the simulation throughput for all SNR points
simThroughput = zeros(length(simParameters.SNRIn),1);

% Set up Redundancy Version (RV) sequence to be used, according to the HARQ configuration 
if simParameters.PDSCHExtension.EnableHARQ
    % In the final report of RAN WG1 meeting #91 (R1-1719301), it was
    % observed in R1-1717405 that if performance is the priority, [0 2 3 1]
    % should be used. If self-decodability is the priority, it should be
    % taken into account that the upper limit of the code rate at which
    % each RV is self-decodable is in the following order: 0>3>2>1
    rvSeq = [0 2 3 1];
else
    % HARQ disabled - single transmission with RV=0, no retransmissions
    rvSeq = 0; 
end

% Create DL-SCH encoder system object to perform transport channel encoding
encodeDLSCH = nrDLSCH;
encodeDLSCH.MultipleHARQProcesses = true;
encodeDLSCH.TargetCodeRate = simParameters.PDSCHExtension.TargetCodeRate;

% Create DL-SCH decoder system object to perform transport channel decoding
% Use layered belief propagation for LDPC decoding, with half the number of
% iterations as compared to the default for belief propagation decoding
decodeDLSCH = nrDLSCHDecoder;
decodeDLSCH.MultipleHARQProcesses = true;
decodeDLSCH.TargetCodeRate = simParameters.PDSCHExtension.TargetCodeRate;
decodeDLSCH.LDPCDecodingAlgorithm = simParameters.PDSCHExtension.LDPCDecodingAlgorithm;
decodeDLSCH.MaximumLDPCIterationCount = simParameters.PDSCHExtension.MaximumLDPCIterationCount;

for snrIdx = 1:numel(simParameters.SNRIn)      % comment out for parallel computing
% parfor snrIdx = 1:numel(simParameters.SNRIn) % uncomment for parallel computing
% To reduce the total simulation time, you can execute this loop in
% parallel by using the Parallel Computing Toolbox. Comment out the 'for'
% statement and uncomment the 'parfor' statement. If the Parallel Computing
% Toolbox is not installed, 'parfor' defaults to normal 'for' statement

    % Set the random number generator settings to default values
    rng('default');
    
    % Take full copies of the simulation-level parameter structures so that they are not 
    % PCT broadcast variables when using parfor 
    simLocal = simParameters;
    waveinfoLocal = waveformInfo;
    
    % Take copies of channel-level parameters to simply subsequent parameter referencing 
    carrier = simLocal.Carrier;
    pdsch = simLocal.PDSCH;
    pdschextra = simLocal.PDSCHExtension;
    ssburst = simLocal.SSBurst;
    decodeDLSCHLocal = decodeDLSCH;  % Copy of the decoder handle to help PCT classification of variable
    decodeDLSCHLocal.reset();        % Reset decoder at the start of each SNR point
    pathFilters = [];
    ssbWaveform = [];
     
    % Prepare simulation for new SNR point
    SNRdB = simLocal.SNRIn(snrIdx);
    fprintf('\nSimulating transmission scheme 1 (%dx%d) and SCS=%dkHz with %s channel at %gdB SNR for %d 10ms frame(s)\n',...
        simParameters.NTxAnts,simParameters.NRxAnts,carrier.SubcarrierSpacing, ...
        simLocal.DelayProfile,SNRdB,simLocal.NFrames); 
        
    % Initialize variables used in the simulation and analysis
    bitTput = [];           % Number of successfully received bits per transmission
    txedTrBlkSizes = [];    % Number of transmitted info bits per transmission
    
    % Specify the order in which we cycle through the HARQ processes
    harqSequence = 1:pdschextra.NHARQProcesses;

    % Initialize the state of all HARQ processes
    harqProcesses = hNewHARQProcesses(pdschextra.NHARQProcesses,rvSeq,pdsch.NumCodewords);
    harqProcCntr = 0; % HARQ process counter
        
    % Reset the channel so that each SNR point will experience the same
    % channel realization
    reset(channel);
    
    % Total number of slots in the simulation period
    NSlots = simLocal.NFrames * carrier.SlotsPerFrame;

    % Index to the start of the current set of SS burst samples to be
    % transmitted
    ssbSampleIndex = 1;
    
    % Obtain a precoding matrix (wtx) to be used in the transmission of the
    % first transport block
    estChannelGrid = getInitialChannelEstimate(carrier,simLocal.NTxAnts,channel);    
    newWtx = getPrecodingMatrix(carrier,pdsch,estChannelGrid);
    
    % Timing offset, updated in every slot for perfect synchronization and
    % when the correlation is strong for practical synchronization
    offset = 0;

    % Loop over the entire waveform length
    for nslot = 0:800-1
        
        % Update the carrier slot numbers for new slot
        carrier.NSlot = nslot;
        
        % Generate a new SS burst when necessary
        if (ssbSampleIndex==1)
            nSubframe = carrier.NSlot / carrier.SlotsPerSubframe;
            ssburst.NFrame = floor(nSubframe / 10);
            ssburst.NHalfFrame = mod(nSubframe / 5,2);
            [ssbWaveform,~,ssbInfo] = hSSBurst(ssburst);
        end
        
        % Get HARQ process index for the current PDSCH from HARQ index table
        harqProcIdx = harqSequence(mod(harqProcCntr,length(harqSequence))+1);
        
        % Update current HARQ process information (this updates the RV
        % depending on CRC pass or fail in the previous transmission for
        % this HARQ process)
        harqProcesses(harqProcIdx) = hUpdateHARQProcess(harqProcesses(harqProcIdx),pdsch.NumCodewords);
        
        % Calculate the transport block sizes for the codewords in the slot
        [pdschIndices,pdschIndicesInfo] = nrPDSCHIndices(carrier,pdsch);
        trBlkSizes = nrTBS(pdsch.Modulation,pdsch.NumLayers,numel(pdsch.PRBSet),pdschIndicesInfo.NREPerPRB,pdschextra.TargetCodeRate,pdschextra.XOverhead);
        
        % HARQ processing
        % Check CRC from previous transmission per codeword, i.e. is a retransmission required?
        for cwIdx = 1:pdsch.NumCodewords
            newdata = false;
            if harqProcesses(harqProcIdx).blkerr(cwIdx) % Error for last recorded decoding
                if (harqProcesses(harqProcIdx).RVIdx(cwIdx)==1) % Signals the start of the RV sequence
                    resetSoftBuffer(decodeDLSCHLocal,cwIdx-1,harqProcIdx-1);  % Explicit reset required in this case
                    newdata = true;
                end
            else    % No error
                newdata = true;
            end
            if newdata 
                trBlk = randi([0 1],trBlkSizes(cwIdx),1);
                setTransportBlock(encodeDLSCH,trBlk,cwIdx-1,harqProcIdx-1);
            end
        end
                                
        % Encode the DL-SCH transport blocks
        codedTrBlocks = encodeDLSCH(pdsch.Modulation,pdsch.NumLayers,...
            pdschIndicesInfo.G,harqProcesses(harqProcIdx).RV,harqProcIdx-1);
    
        % Get precoding matrix (wtx) calculated in previous slot
        wtx = newWtx;
        
        % Resource grid array
        pdschGrid = nrResourceGrid(carrier,simLocal.NTxAnts);
        
        % PDSCH modulation and precoding
        pdschSymbols = nrPDSCH(carrier,pdsch,codedTrBlocks);
        [pdschAntSymbols,pdschAntIndices] = hPRGPrecode(size(pdschGrid),carrier.NStartGrid,pdschSymbols,pdschIndices,wtx);
        
        % PDSCH mapping in grid associated with PDSCH transmission period
        pdschGrid(pdschAntIndices) = pdschAntSymbols;
        
        % PDSCH DM-RS precoding and mapping
        dmrsSymbols = nrPDSCHDMRS(carrier,pdsch);
        dmrsIndices = nrPDSCHDMRSIndices(carrier,pdsch);
        [dmrsAntSymbols,dmrsAntIndices] = hPRGPrecode(size(pdschGrid),carrier.NStartGrid,dmrsSymbols,dmrsIndices,wtx);        
        pdschGrid(dmrsAntIndices) = dmrsAntSymbols;

        % PDSCH PT-RS precoding and mapping
        ptrsSymbols = nrPDSCHPTRS(carrier,pdsch);
        ptrsIndices = nrPDSCHPTRSIndices(carrier,pdsch);
        [ptrsAntSymbols,ptrsAntIndices] = hPRGPrecode(size(pdschGrid),carrier.NStartGrid,ptrsSymbols,ptrsIndices,wtx);
        pdschGrid(ptrsAntIndices) = ptrsAntSymbols;        

        % OFDM modulation of associated resource elements
        txWaveform = nrOFDMModulate(carrier, pdschGrid);
        
        % Add the appropriate portion of SS burst waveform to the
        % transmitted waveform
        Nt = size(txWaveform,1);
        txWaveform = txWaveform + ssbWaveform(ssbSampleIndex + (0:Nt-1),:);
        ssbSampleIndex = mod(ssbSampleIndex + Nt,size(ssbWaveform,1));

        % Pass data through channel model. Append zeros at the end of the
        % transmitted waveform to flush channel content. These zeros take
        % into account any delay introduced in the channel. This is a mix
        % of multipath delay and implementation delay. This value may 
        % change depending on the sampling rate, delay profile and delay
        % spread
        txWaveform = [txWaveform; zeros(maxChDelay, size(txWaveform,2))];
        [rxWaveform,pathGains,sampleTimes] = channel(txWaveform);
        
        % Add AWGN to the received time domain waveform
        % Normalize noise power by the IFFT size used in OFDM modulation,
        % as the OFDM modulator applies this normalization to the
        % transmitted waveform. Also normalize by the number of receive
        % antennas, as the channel model applies this normalization to the
        % received waveform by default
        SNR = 10^(SNRdB/20); % Calculate linear noise gain
        N0 = 1/(sqrt(2.0*simLocal.NRxAnts*double(waveinfoLocal.Nfft))*SNR);
        noise = N0*complex(randn(size(rxWaveform)),randn(size(rxWaveform)));
        rxWaveform = rxWaveform + noise;

        if (simLocal.PerfectChannelEstimator)
            % Perfect synchronization. Use information provided by the channel
            % to find the strongest multipath component
            pathFilters = getPathFilters(channel); % get path filters for perfect channel estimation
            [offset,mag] = nrPerfectTimingEstimate(pathGains,pathFilters);
        else
            % Practical synchronization. Correlate the received waveform 
            % with the PDSCH DM-RS to give timing offset estimate 't' and 
            % correlation magnitude 'mag'. The function hSkipWeakTimingOffset
            % is used to update the receiver timing offset. If the 
            % correlation peak in 'mag' is weak, the current timing
            % estimate 't' is ignored and the previous estimate 'offset'
            % is used
            [t,mag] = nrTimingEstimate(carrier,rxWaveform,dmrsIndices,dmrsSymbols); 
            offset = hSkipWeakTimingOffset(offset,t,mag);
            % Display a warning if the estimated timing offset exceeds the
            % maximum channel delay
            if offset > maxChDelay
                warning(['Estimated timing offset (%d) is greater than the maximum channel delay (%d).' ...
                    ' This will result in a decoding failure. This may be caused by low SNR,' ...
                    ' or not enough DM-RS symbols to synchronize successfully.'],offset,maxChDelay);
            end
        end
        rxWaveform = rxWaveform(1+offset:end, :);

        % Perform OFDM demodulation on the received data to recreate the
        % resource grid, including padding in the event that practical
        % synchronization results in an incomplete slot being demodulated
        rxGrid = nrOFDMDemodulate(carrier, rxWaveform);
        [K,L,R] = size(rxGrid);
        if (L < carrier.SymbolsPerSlot)
            rxGrid = cat(2,rxGrid,zeros(K,carrier.SymbolsPerSlot-L,R));
        end

        if (simLocal.PerfectChannelEstimator)
            % Perfect channel estimation, using the value of the path gains
            % provided by the channel. This channel estimate does not
            % include the effect of transmitter precoding
            estChannelGrid = nrPerfectChannelEstimate(carrier,pathGains,pathFilters,offset,sampleTimes);
            estChannel = estChannelGrid(1:600,3:14,:);
            Ideal_H(nslot*25+1:(nslot+1)*25,:,:,:,1) = real(permute(reshape(estChannel,24,25,12,4),[2,3,1,4]));
            Ideal_H(nslot*25+1:(nslot+1)*25,:,:,:,2) = imag(permute(reshape(estChannel,24,25,12,4),[2,3,1,4]));
            %disp(size(estChannelGrid));
            disp(nslot);

            % Get perfect noise estimate (from the noise realization)
%             noiseGrid = nrOFDMDemodulate(carrier,noise(1+offset:end ,:));
%             noiseEst = var(noiseGrid(:));
% 
%             % Get precoding matrix for next slot
%             newWtx = getPrecodingMatrix(carrier,pdsch,estChannelGrid);
% 
%             % Get PDSCH resource elements from the received grid and 
%             % channel estimate
%             [pdschRx,pdschHest,~,pdschHestIndices] = nrExtractResources(pdschIndices,rxGrid,estChannelGrid);
% 
%             % Apply precoding to channel estimate
%             pdschHest = hPRGPrecode(size(estChannelGrid),carrier.NStartGrid,pdschHest,pdschHestIndices,permute(wtx,[2 1 3]));
        else
            % Practical channel estimation between the received grid and
            % each transmission layer, using the PDSCH DM-RS for each
            % layer. This channel estimate includes the effect of
            % transmitter precoding
            [estChannelGrid,noiseEst] = nrChannelEstimate(carrier,rxGrid,dmrsIndices,dmrsSymbols,'CDMLengths',pdsch.DMRS.CDMLengths); 
            
            % Get PDSCH resource elements from the received grid and
            % channel estimate
            [pdschRx,pdschHest] = nrExtractResources(pdschIndices,rxGrid,estChannelGrid);

            % Remove precoding from estChannelGrid prior to precoding
            % matrix calculation
            estChannelGridPorts = precodeChannelEstimate(carrier,estChannelGrid,conj(wtx));
            
            % Get precoding matrix for next slot
            newWtx = getPrecodingMatrix(carrier,pdsch,estChannelGridPorts);
        end

        % Equalization
%         [pdschEq,csi] = nrEqualizeMMSE(pdschRx,pdschHest,noiseEst);
% 
%         % Common phase error (CPE) compensation
%         if ~isempty(ptrsIndices)
%             % Initialize temporary grid to store equalized symbols
%             tempGrid = nrResourceGrid(carrier,pdsch.NumLayers);
% 
%             % Extract PT-RS symbols from received grid and estimated
%             % channel grid
%             [ptrsRx,ptrsHest,~,~,ptrsHestIndices,ptrsLayerIndices] = nrExtractResources(ptrsIndices,rxGrid,estChannelGrid,tempGrid);
%             
%             if (simLocal.PerfectChannelEstimator)
%                 % Apply precoding to channel estimate
%                 ptrsHest = hPRGPrecode(size(estChannelGrid),carrier.NStartGrid,ptrsHest,ptrsHestIndices,permute(wtx,[2 1 3]));
%             end
% 
%             % Equalize PT-RS symbols and map them to tempGrid
%             ptrsEq = nrEqualizeMMSE(ptrsRx,ptrsHest,noiseEst);
%             tempGrid(ptrsLayerIndices) = ptrsEq;
% 
%             % Estimate the residual channel at the PT-RS locations in
%             % tempGrid
%             cpe = nrChannelEstimate(tempGrid,ptrsIndices,ptrsSymbols);
% 
%             % Sum estimates across subcarriers, receive antennas, and
%             % layers. Then, get the CPE by taking the angle of the
%             % resultant sum
%             cpe = angle(sum(cpe,[1 3 4]));
% 
%             % Map the equalized PDSCH symbols to tempGrid
%             tempGrid(pdschIndices) = pdschEq;
% 
%             % Correct CPE in each OFDM symbol within the range of reference
%             % PT-RS OFDM symbols
%             symLoc = pdschIndicesInfo.PTRSSymbolSet(1)+1:pdschIndicesInfo.PTRSSymbolSet(end)+1;
%             tempGrid(:,symLoc,:) = tempGrid(:,symLoc,:).*exp(-1i*cpe(symLoc));
% 
%             % Extract PDSCH symbols
%             pdschEq = tempGrid(pdschIndices);
%         end
% 
%         % Decode PDSCH physical channel
%         [dlschLLRs,rxSymbols] = nrPDSCHDecode(carrier,pdsch,pdschEq,noiseEst);
%         
%         % Display EVM per layer, per slot and per RB
%         if (simLocal.DisplayDiagnostics)
%             plotLayerEVM(NSlots,nslot,pdsch,size(pdschGrid),pdschIndices,pdschSymbols,pdschEq);
%         end
%         
%         % Scale LLRs by CSI
%         csi = nrLayerDemap(csi); % CSI layer demapping
%         for cwIdx = 1:pdsch.NumCodewords
%             Qm = length(dlschLLRs{cwIdx})/length(rxSymbols{cwIdx}); % bits per symbol
%             csi{cwIdx} = repmat(csi{cwIdx}.',Qm,1);                 % expand by each bit per symbol
%             dlschLLRs{cwIdx} = dlschLLRs{cwIdx} .* csi{cwIdx}(:);   % scale by CSI
%         end
%         
%         % Decode the DL-SCH transport channel
%         % Write the decoding CRC error back into the HARQ process state structure
%         decodeDLSCHLocal.TransportBlockLength = trBlkSizes;
%         [decbits,harqProcesses(harqProcIdx).blkerr] = decodeDLSCHLocal(dlschLLRs,pdsch.Modulation,pdsch.NumLayers,harqProcesses(harqProcIdx).RV,harqProcIdx-1);
%         
%         % Store values to calculate throughput (only for active PDSCH instances)
%         if(any(trBlkSizes ~= 0))
%             bitTput = [bitTput trBlkSizes.*(1-harqProcesses(harqProcIdx).blkerr)];
%             txedTrBlkSizes = [txedTrBlkSizes trBlkSizes];                         
%         end
%         
%         % Update HARQ process counter
%         harqProcCntr = harqProcCntr + 1;
%                 
%         % Display transport block CRC error information per codeword managed by current HARQ process
%         icr = trBlkSizes./pdschIndicesInfo.G;    % Instantaneous code rate
%         csn = mod(nslot,carrier.SlotsPerFrame);  % Slot number in frame
%         fprintf('\n(%3.2f%%) HARQ Proc %d: ',100*(nslot+1)/NSlots,harqProcIdx);
%         estrings = {'passed','failed'};
%         rvi = harqProcesses(harqProcIdx).RVIdx; 
%         for cw=1:length(rvi)
%             cwrvi = rvi(cw);
%             % Create a report on the RV state given position in RV sequence and decoding error
%             if cwrvi == 1
%                 ts = sprintf('Initial transmission (NSlot=%d,RV=%d,CR=%f)',csn,rvSeq(cwrvi),icr(cw));
%             else
%                 ts = sprintf('Retransmission #%d (NSlot=%d,RV=%d,CR=%f)',cwrvi-1,csn,rvSeq(cwrvi),icr(cw));
%             end
%             fprintf('CW%d:%s %s. ',cw-1,ts,estrings{1+harqProcesses(harqProcIdx).blkerr(cw)}); 
%         end
         
     end
    save('/public/share/hcju/MIMO/Ideal_H/Ideal_H_CDLA_5Hz_30ns.mat','Ideal_H');
    % Calculate maximum and simulated throughput
    maxThroughput(snrIdx) = sum(txedTrBlkSizes); % Max possible throughput
    simThroughput(snrIdx) = sum(bitTput,2);      % Simulated throughput
    
    % Display the results dynamically in the command window
    fprintf([['\n\nThroughput(Mbps) for ', num2str(simLocal.NFrames) ' frame(s) '],...
        '= %.4f\n'], 1e-6*simThroughput(snrIdx)/(simLocal.NFrames*10e-3));
    fprintf(['Throughput(%%) for ', num2str(simLocal.NFrames) ' frame(s) = %.4f\n'],...
        simThroughput(snrIdx)*100/maxThroughput(snrIdx));
     
end

%% Results
% Display the measured throughput. This is calculated as the percentage of
% the maximum possible throughput of the link given the available resources
% for data transmission.

figure;
plot(simParameters.SNRIn,simThroughput*100./maxThroughput,'o-.')
xlabel('SNR (dB)'); ylabel('Throughput (%)'); grid on;
title(sprintf('%s (%dx%d) / NRB=%d / SCS=%dkHz',...
              simParameters.DelayProfile,simParameters.NTxAnts,simParameters.NRxAnts,...
              simParameters.Carrier.NSizeGrid,simParameters.Carrier.SubcarrierSpacing));

% Bundle key parameters and results into a combined structure for recording
simResults.simParameters = simParameters;
simResults.simThroughput = simThroughput;

%%
% The figure below shows throughput results obtained simulating 10000
% subframes (|NFrames = 1000|, |SNRIn = -18:2:16|).
%
% <<../longRunThroughput.png>>
%

%% Appendix
% This example uses the following helper functions:
% 
% * <matlab:edit('hArrayGeometry.m') hArrayGeometry.m>
% * <matlab:edit('hNewHARQProcesses.m') hNewHARQProcesses.m>
% * <matlab:edit('hPRGPrecode.m') hPRGPrecode.m>
% * <matlab:edit('hSkipWeakTimingOffset.m') hSkipWeakTimingOffset.m>
% * <matlab:edit('hSSBurst.m') hSSBurst.m>
% * <matlab:edit('hUpdateHARQProcess.m') hUpdateHARQProcess.m>

%% Selected Bibliography
% # 3GPP TS 38.211. "NR; Physical channels and modulation."
% 3rd Generation Partnership Project; Technical Specification Group Radio
% Access Network.
% # 3GPP TS 38.212. "NR; Multiplexing and channel coding." 
% 3rd Generation Partnership Project; Technical Specification Group Radio
% Access Network.
% # 3GPP TS 38.213. "NR; Physical layer procedures for control." 
% 3rd Generation Partnership Project; Technical Specification Group Radio
% Access Network.
% # 3GPP TS 38.214. "NR; Physical layer procedures for data."
% 3rd Generation Partnership Project; Technical Specification Group Radio
% Access Network.
% # 3GPP TR 38.901. "Study on channel model for frequencies from 0.5 to 100
% GHz." 3rd Generation Partnership Project; Technical
% Specification Group Radio Access Network.

%% Local Functions

function validateNumLayers(simParameters)
% Validate the number of layers, relative to the antenna geometry

    nlayers = simParameters.PDSCH.NumLayers;
    ntxants = simParameters.NTxAnts;
    nrxants = simParameters.NRxAnts;
    antennaDescription = sprintf('min(NTxAnts,NRxAnts) = min(%d,%d) = %d',ntxants,nrxants,min(ntxants,nrxants));
    if nlayers > min(ntxants,nrxants)
        error('The number of layers (%d) must satisfy NLayers <= %s',...
            nlayers,antennaDescription);
    end
    
    % Display a warning if the maximum possible rank of the channel equals
    % the number of layers
    if (nlayers > 2) && (nlayers == min(ntxants,nrxants))
        warning(['The maximum possible rank of the channel, given by %s, is equal to NLayers (%d).' ...
            ' This may result in a decoding failure under some channel conditions.' ...
            ' Try decreasing the number of layers or increasing the channel rank' ...
            ' (use more transmit or receive antennas).'],antennaDescription,nlayers); %#ok<SPWRN>
    end

end

function estChannelGrid = getInitialChannelEstimate(carrier,nTxAnts,propchannel)
% Obtain channel estimate before first transmission. This can be used to
% obtain a precoding matrix for the first slot.

    ofdmInfo = nrOFDMInfo(carrier);
    
    chInfo = info(propchannel);
    maxChDelay = ceil(max(chInfo.PathDelays*propchannel.SampleRate)) + chInfo.ChannelFilterDelay;
    
    % Temporary waveform (only needed for the sizes)
    tmpWaveform = zeros((ofdmInfo.SampleRate/1000/carrier.SlotsPerSubframe)+maxChDelay,nTxAnts);
    
    % Filter through channel    
    [~,pathGains,sampleTimes] = propchannel(tmpWaveform);
    
    % Perfect timing synch    
    pathFilters = getPathFilters(propchannel);
    offset = nrPerfectTimingEstimate(pathGains,pathFilters);
    
    % Perfect channel estimate
    estChannelGrid = nrPerfectChannelEstimate(carrier,pathGains,pathFilters,offset,sampleTimes);
    
end

function wtx = getPrecodingMatrix(carrier,pdsch,hestGrid)
% Calculate precoding matrices for all PRGs in the carrier that overlap
% with the PDSCH allocation
    
    % Maximum CRB addressed by carrier grid
    maxCRB = carrier.NStartGrid + carrier.NSizeGrid - 1;
    
    % PRG size
    if (isfield(pdsch,'PRGBundleSize') && ~isempty(pdsch.PRGBundleSize))
        Pd_BWP = pdsch.PRGBundleSize;
    else
        Pd_BWP = maxCRB + 1;
    end
    
    % PRG numbers (1-based) for each RB in the carrier grid
    NPRG = ceil((maxCRB + 1) / Pd_BWP);
    prgset = repmat((1:NPRG),Pd_BWP,1);
    prgset = prgset(carrier.NStartGrid + (1:carrier.NSizeGrid).');
    
    [~,~,R,P] = size(hestGrid);
    wtx = zeros([pdsch.NumLayers P NPRG]);
    for i = 1:NPRG
    
        % Subcarrier indices within current PRG and within the PDSCH
        % allocation
        thisPRG = find(prgset==i) - 1;
        thisPRG = intersect(thisPRG,pdsch.PRBSet(:) + carrier.NStartGrid,'rows');
        prgSc = (1:12)' + 12*thisPRG';
        prgSc = prgSc(:);
       
        if (~isempty(prgSc))
        
            % Average channel estimate in PRG
            estAllocGrid = hestGrid(prgSc,:,:,:);
            Hest = permute(mean(reshape(estAllocGrid,[],R,P)),[2 3 1]);

            % SVD decomposition
            [~,~,V] = svd(Hest);
            wtx(:,:,i) = V(:,1:pdsch.NumLayers).';

        end
    
    end
    
    wtx = wtx / sqrt(pdsch.NumLayers); % Normalize by NumLayers

end

function estChannelGrid = precodeChannelEstimate(carrier,estChannelGrid,W)
% Apply precoding matrix W to the last dimension of the channel estimate

    [K,L,R,P] = size(estChannelGrid);
    estChannelGrid = reshape(estChannelGrid,[K*L R P]);
    estChannelGrid = hPRGPrecode([K L R P],carrier.NStartGrid,estChannelGrid,reshape(1:numel(estChannelGrid),[K*L R P]),W);
    estChannelGrid = reshape(estChannelGrid,K,L,R,[]);

end

function [mappedPRB,mappedSymbols] = mapNumerology(subcarriers,symbols,nrbs,nrbt,fs,ft)
% Map the SSBurst numerology to PDSCH numerology. The outputs are:
%   - mappedPRB: 0-based PRB indices for carrier resource grid (arranged in a column)
%   - mappedSymbols: 0-based OFDM symbol indices in a slot for carrier resource grid (arranged in a row)
% The input parameters are:
%   - subcarriers: 1-based row subscripts for SSB resource grid (arranged in a column)
%   - symbols: 1-based column subscripts for SSB resource grid (arranged in an N-by-4 matrix, 4 symbols for each transmitted burst in a row, N transmitted bursts)
%     SSB resource grid is sized using ssbInfo.NRB, normal CP, spanning 5 subframes
%   - nrbs: source (SSB) NRB
%   - nrbt: target (carrier) NRB
%   - fs: source (SSB) SCS
%   - ft: target (carrier) SCS

    mappedPRB = unique(fix((subcarriers-(nrbs*6) - 1)*fs/(ft*12) + nrbt/2),'stable');
    
    symbols = symbols.';
    symbols = symbols(:).' - 1;

    if (ft < fs)
        % If ft/fs < 1, reduction
        mappedSymbols = unique(fix(symbols*ft/fs),'stable');
    else
        % Else, repetition by ft/fs
        mappedSymbols = reshape((0:(ft/fs-1))' + symbols(:)'*ft/fs,1,[]);
    end
    
end

function plotLayerEVM(NSlots,nslot,pdsch,siz,pdschIndices,pdschSymbols,pdschEq)
% Plot EVM information

    persistent slotEVM;
    persistent rbEVM
    persistent evmPerSlot;
    
    if (nslot==0)
        slotEVM = comm.EVM;
        rbEVM = comm.EVM;
        evmPerSlot = NaN(NSlots,pdsch.NumLayers);
        figure;
    end
    evmPerSlot(nslot+1,:) = slotEVM(pdschSymbols,pdschEq);
    subplot(2,1,1);
    plot(0:(NSlots-1),evmPerSlot);
    xlabel('Slot number');
    ylabel('EVM (%)');
    legend("layer " + (1:pdsch.NumLayers),'Location','EastOutside');
    title('EVM per layer per slot');

    subplot(2,1,2);
    [k,~,p] = ind2sub(siz,pdschIndices);
    rbsubs = floor((k-1) / 12);
    NRB = siz(1) / 12;
    evmPerRB = NaN(NRB,pdsch.NumLayers);
    for nu = 1:pdsch.NumLayers
        for rb = unique(rbsubs).'
            this = (rbsubs==rb & p==nu);
            evmPerRB(rb+1,nu) = rbEVM(pdschSymbols(this),pdschEq(this));
        end
    end
    plot(0:(NRB-1),evmPerRB);
    xlabel('Resource block');
    ylabel('EVM (%)');
    legend("layer " + (1:pdsch.NumLayers),'Location','EastOutside');
    title(['EVM per layer per resource block, slot #' num2str(nslot)]);
    
    drawnow;
    
end