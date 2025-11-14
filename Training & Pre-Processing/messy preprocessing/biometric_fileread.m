% This script will read the .dat and .hea files downloaded locally to
% to your hard drive. The downloaded files are organzied in 3 folders
% 'Session 1','Session 2' and 'Session 3'. Each folder contains 119 data
% files (.dat) and 119 header files (.hea).
%
% signal properties
% total channels = 28 (16 forearm + 12 wrist)
% sampling frequency = 2048 Hz
% bandpass filtering (hardware) = 10Hz-500Hz
%
%
% In order to run this script make sure the above three folders are and
% a fileconversion function 'rdwfdb.m' are in the same directory
%
%
% output %%%%%%%%%
% Main Folder: 'Output BM'
% Folders: 'Session 1_converted','Session 2_converted', 'Session 3_converted',
% Each folder: 43 .mat files
% VarOut: DATA_FOREARM, DATA_WRIST (7x17 cell matrices)
% DATA_FOREARM: each cell: 5secs*sampfreq x Nchannels numeric array
% DATA_WRIST: each cell: 5secs*sampfreq x Nchannels numeric array
%
% Forearm Electrode Configuration %%%%%%%%%
%  1  2  3  4  5  6  7  8 
%  9 10 11 12 13 14 15 16 
%
% Wrist Electrode Configuration %%%%%%%%%
%  1  2  3  4  5  6   
%  7  8  9 10 11 12  
%
% Written by Ashirbad Pradhan
% email: ashirbad.pradhan@uwaterloo.ca



clear 
addpath(genpath([pwd filesep 'Session 1']))
addpath(genpath([pwd filesep 'Session 2']))
addpath(genpath([pwd filesep 'Session 3']))

%obtain the total number of subject. Note the total subjects should be same
% in all the three folders
NSUB = length(dir([pwd filesep 'Session 1']))-2; %please make sure the number 
                        %of subjects in the three session folders are the
                        %same. Default value is 43
NSESSION = 3;
NGESTURE = 16;              %total number of gestures
NTRIALS = 7;                %total number of trials

%% Define output folder
    if ~exist('Output BM', 'dir')
        mkdir('Output BM')
    else
        while(1)
            display(['Found exisiting folder in: ' pwd])
            cont = upper(input('Overwrite it (Y/N)?','s'));
            if(strcmp(cont,'Y') || strcmp(cont,'N'))
                if(strcmp(cont,'Y'))
                    disp('Overwriting')
                    rmdir('Output BM','s')
                    mkdir('Output BM')
                    break;
                else
                    flag=1;
                    disp('Exiting Script !')
                    return;
                end
            end
        end
    end

%% read all files
foldername=[];
filename=[];
flag=0;count=0;
for isession = 1:NSESSION              %total number of sessions per participants
    converted_folder=['Session',num2str(isession),'_converted'];
    mkdir(['Output BM' filesep converted_folder])    
    for isub = 1:NSUB
        foldername = ['session',num2str(isession),'_participant',num2str(isub)];
        for igesture = 1:NGESTURE+1 % +1 to include rest gesture
            for itrial = 1:NTRIALS
                filename = ['session',num2str(isession),'_participant',num2str(isub),'_gesture',num2str(igesture),'_trial',num2str(itrial)];
                filepath = fullfile(pwd,['Session ',num2str(isession)],foldername, filename);
                [tm,data_emg,fs,siginfo]=rdwfdb(filename);
                % the channel numbers for forearm are 1-8 and 9-16
                % the channel numbers for the wrist are 18-23 and 26-31
                forearm_channels=[ones(1,16) zeros(1,16)];
                wrist_channels=[zeros(1,16) 0 ones(1,6) 0 0 ones(1,6) 0];
                DATA_FOREARM{itrial,igesture}=data_emg(:,logical(forearm_channels));
                DATA_WRIST{itrial,igesture}=data_emg(:,logical(wrist_channels));
            end
        end
        count=count+1;        
        disp(['converted: ',num2str(count),' of ', num2str(NSUB*NSESSION),' files'])
        save(['Output BM' filesep converted_folder filesep foldername '.mat'],'DATA_FOREARM','DATA_WRIST')
    end
end


display(['Next: EMG Signal Processing: ' pwd])
cont = upper(input('Proceed (Y/N)?','s'));
if(strcmp(cont,'Y') || strcmp(cont,'N'))
    if(strcmp(cont,'Y'))
        biometric_feature_extraction;        
    else
        flag=1;
        disp('Exiting Script !')
        return;
    end
end


