clear

% Load any channel. All the trial times and parameters are the same foe all the channels.

%Condition and orientation codes: left (l)= 0 , right (r)= 1; vertical (v)= 3 , horizontal (h)= 1 : lv=03 , rv=13 , lh=01 , rh=11
lv=03; rv=4; lh=01; rh=2;
% sessionName = '2015_03_05_R';

% sessionName = '20141014_G';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2014_10_14_G\gdayREC32_B14_10_14\gdayREC32_B14_10_14_Ch01.mat')
% trialList = [1:25 26:51 52:77 78:103];% 105:129 130:154 155:179 180:205];
% CondOrien=[lv*ones(1,25), rv*ones(1,51-26+1), lh*ones(1,77-52+1), rh*ones(1,103-78+1)]; 
% 
% sessionName = '20141009_G';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2014_10_09_G\gdayREC30_A14_10_09\gdayREC30_A14_10_09_Ch01.mat')
% trialList = [1:25 27:51 53:77 78:102];
% CondOrien=[lv*ones(1,25), rv*ones(1,51-27+1), lh*ones(1,77-53+1), rh*ones(1,102-78+1)]; 

% sessionName = '20141021_G';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2014_10_21_G\gdayREC36_A14_10_21\gdayREC36_A14_10_21_Ch01.mat')
% trialList = [1:25 26:50 51:75 76:100];% 102:126 127:152 153:178 179:203];
% CondOrien=[lv*ones(1,25), rv*ones(1,50-26+1), lh*ones(1,75-51+1), rh*ones(1,100-76+1)]; 
% 
% sessionName = '20141020_G';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2014_10_20_G\gdayREC35_A14_10_20\gdayREC35_A14_10_20_Ch01.mat')
% trialList = [1:27 28:52 53:77 78:105];
% CondOrien=[lv*ones(1,27), rv*ones(1,52-28+1), lh*ones(1,77-53+1), rh*ones(1,105-78+1)]; 
% % 
% sessionName = '20141028_G';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2014_10_28_G\gdayREC40_B14_10_28\gdayREC40_B14_10_28_Ch01.mat')
% trialList = [1:25 26:50 51:75 76:100];% 101:125 126:151 152:176];
% CondOrien=[lv*ones(1,25), rv*ones(1,50-26+1), lh*ones(1,75-51+1), rh*ones(1,100-76+1)]; 

% sessionName = '20141203_G';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2014_12_03_G\gdayREC55_A14_12_03\gdayREC55_A14_12_03_Ch01.mat')
% trialList = [1:25 26:50 51:75 76:100 ];%102:126 127:151 152:176 177:201];
% CondOrien=[lv*ones(1,25), rv*ones(1,50-26+1), lh*ones(1,75-51+1), rh*ones(1,100-76+1)]; 
% 
% sessionName = '20141216_G';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2014_12_16_G\gdayREC57_A14_12_16\gdayREC57_A14_12_16_Ch01.mat')
%  trialList = [1:25 26:50 51:75 76:100 ];%101:125 126:150 151:175 176:200];
% CondOrien=[lv*ones(1,25), rv*ones(1,50-26+1), lh*ones(1,75-51+1), rh*ones(1,100-76+1)];  

% sessionName = '20141017_G';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2014_10_17_G\gdayREC34_A14_10_17\gdayREC34_A14_10_17_Ch01.mat')
%  trialList = [1:25 26:50 51:75 76:100 ];%101:125 126:150 151:175 176:200];
% CondOrien=[lv*ones(1,25), rv*ones(1,50-26+1), lh*ones(1,75-51+1), rh*ones(1,100-76+1)]; 
% 
% sessionName = '20150226_R';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2015_02_26_R\rdayREC14_A15_02_26\rdayREC14_A15_02_26_Ch01.mat')
% trialList = [1:25 26:50 51:75 76:100];% 101:125 126:150 151:175 180:204];
% CondOrien=[lv*ones(1,25), rv*ones(1,50-26+1), lh*ones(1,75-51+1), rh*ones(1,100-76+1)]; 

% sessionName = '20150219_R';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2015_02_19_R\rdayREC11_A15_02_19\rdayREC11_A15_02_19_Ch01.mat')
% trialList = [1:25 26:50 51:75 76:100];% 101:125 126:132 136:160];
% CondOrien=[lv*ones(1,25), rv*ones(1,50-26+1), lh*ones(1,75-51+1), rh*ones(1,100-76+1)]; 

% sessionName = '20150310_R';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2015_03_10_R\rdayREC19_A15_03_10\rdayREC19_A15_03_10_Ch01.mat')
% trialList = [1:25 26:50 56:75 76:100];% 101:125 127:151 156:176 177:194];
% CondOrien=[lv*ones(1,25), rv*ones(1,50-26+1), lh*ones(1,75-56+1), rh*ones(1,100-76+1)]; 

% sessionName = '20150306_R';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2015_03_06_R\rdayREC18_A15_03_06\rdayREC18_A15_03_06_Ch01.mat')
% trialList = [1:25 26:50 51:75 76:100];% 101:125 126:150 151:175 176:200];
% CondOrien=[lv*ones(1,25), rv*ones(1,50-26+1), lh*ones(1,75-51+1), rh*ones(1,100-76+1)]; 

% sessionName = '20150324_R';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2015_03_24_R\rdayREC26_A15_03_24\rdayREC26_A15_03_24_Ch01.mat')
% trialList = [1:25 26:50 56:80 81:105];% 106:130 132:140 141:160 162:166];
% CondOrien=[lv*ones(1,25), rv*ones(1,50-26+1), lh*ones(1,80-56+1), rh*ones(1,105-81+1)]; 

% sessionName = '20150313_R';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2015_03_13_R\rdayREC21_A15_03_13\rdayREC21_A15_03_13_Ch01.mat')
% trialList = [1:25 26:50 51:75 76:100];% 101:125 126:150 151:175 176:200];
% CondOrien=[lv*ones(1,25), rv*ones(1,50-26+1), lh*ones(1,80-56+1), rh*ones(1,105-81+1)]; 

% sessionName = '20150317_R';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2015_03_17_R\rdayREC23_A15_03_17\rdayREC23_A15_03_17_Ch01.mat')
% trialList = [51:77 26:50 103:127 78:102];
%  CondOrien=[lv*ones(1,77-51+1), rv*ones(1,50-26+1), lh*ones(1,127-103+1), rh*ones(1,102-78+1)]; 
% % 
% sessionName = '20150206_R';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2015_02_06_R\rdayREC02_A15_02_06\rdayREC02_A15_02_06_Ch01')
% trialList = [1:25 51:75 102:126 134:158];
% CondOrien=[lv*ones(1,25), rv*ones(1,75-51+1), lh*ones(1,126-102+1), rh*ones(1,158-134+1)]; 

sessionName = '20150305_R';
load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2015_03_05_R\rdayREC17_A15_03_05\rdayREC17_A15_03_05_Ch01.mat')
trialList = [1:25 26:50 52:76 77:101]; %101:125 126:150 151:175 176:200];
CondOrien=[lv*ones(1,25), rv*ones(1,50-26+1), lh*ones(1,76-52+1), rh*ones(1,101-77+1)]; 

% sessionName = '20150211_R';
% load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2015_02_11_R\rdayREC05_C15_02_11\rdayREC05_C15_02_11_Ch01.mat')
% trialList = [1:25 26:50 51:75 77:101];
% CondOrien=[lv*ones(1,25), rv*ones(1,50-26+1), lh*ones(1,75-51+1), rh*ones(1,101-77+1)]; 

% % sessionName = '20141017_G';
% % load('\\LaboDancauseDS\LabData\MacaquesData\InactivationData\inactivationMacaquesSorting\2014_10_17_G\gdayREC34_A14_10_17\gdayREC34_A14_10_17_Ch01.mat')
% % trialList = [1:26 27:51 52:76 77:101]; %105:130 131:155 156:180 180:205];
% % CondOrien=[lv*ones(1,25), rv*ones(1,50-26+1), lh*ones(1,80-56+1), rh*ones(1,105-81+1)]; 

% list some trial duration information to determine good trials
emgC = struct2cell(File.Trials);

% Extracting relevant parameters
relTrialTimes.paramDef = 'trialNumbers | originalTrialNumber | isRightHandUsed | handOrien | trialStartTimes | trialRewardDrop | trialReachOn | trialGraspOn | trialGraspOff';
trialNumbers = double( cell2mat(emgC(1,trialList))); % Getting trial numbers as set at the time of data collection
isRightHandUsed = contains(emgC(2,trialList),{'Right'});
trialStartTimes = double(cell2mat(emgC(15,trialList))); 
trialRewardDrop = double(cell2mat(emgC(17,trialList))); % Using pellet release (TimeSevGranuleReleased) as the basis for start of a trial
trialReachOn = double(cell2mat(emgC(18,trialList))); % Start of reach (TimeSevHandExitHP)
trialGraspOn = double(cell2mat(emgC(19,trialList))); % Start of grasp (TimeSevGetGranule)
trialGraspOff = double(cell2mat(emgC(20,trialList))); % Using end of grasp (TimeSevEatGranule) as the basis for end of a trial
relTrialTimes.handUsed = emgC(2,trialList)';

% interTrialTimes = [0 diff(trialStartTimes)]';
% condChangeAUTO = find(interTrialTimes > 20000);
nTrials = numel(trialNumbers);
disp(relTrialTimes.paramDef)
relTrialTimes.data = [(1:nTrials)' trialNumbers' isRightHandUsed' CondOrien' (trialStartTimes)' (trialRewardDrop)' (trialReachOn)' (trialGraspOn)' (trialGraspOff)'];


% Times used and other times that will be useful in subsequent steps

if ~isfolder('relTrialTimes')
    mkdir('relTrialTimes')
end

save(['relTrialTimes\relTrialTimes_' sessionName],'relTrialTimes')






