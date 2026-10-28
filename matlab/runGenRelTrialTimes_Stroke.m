clear
clc

% Load any channel. All the trial times and parameters are the same for all the channels.

% Session name


sessionName = '20180531_Y';
load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2018_05_31_Y.mat')
genRelTrialTimes_Stroke(sessionName,TrialsParameters)
% 
sessionName = '20180601_Y';
load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2018_06_01_Y.mat')
genRelTrialTimes_Stroke(sessionName,TrialsParameters)


sessionName = '20180606_Y';
load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2018_06_06_Y.mat')
genRelTrialTimes_Stroke(sessionName,TrialsParameters)


sessionName = '20180607_Y';
load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2018_06_07_Y.mat')
genRelTrialTimes_Stroke(sessionName,TrialsParameters)


sessionName = '20180608_Y';
load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2018_06_08_Y.mat')
genRelTrialTimes_Stroke(sessionName,TrialsParameters)


sessionName = '20180612_Y';
load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2018_06_12_Y.mat')
genRelTrialTimes_Stroke(sessionName,TrialsParameters)


% sessionName = '20180613_Y';
% load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2018_06_13_Y.mat')
% genRelTrialTimes_Stroke(sessionName,TrialsParameters)
% 
% 
% sessionName = '20180615_Y';
% load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2018_06_15_Y.mat')
% genRelTrialTimes_Stroke(sessionName,TrialsParameters)
% 
% sessionName = '20180618_Y';
% load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2018_06_18_Y.mat')
% genRelTrialTimes_Stroke(sessionName,TrialsParameters)

% 
% 
% sessionName = '20190429_B';
% load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2019_04_29_B.mat')
% genRelTrialTimes_Stroke(sessionName,TrialsParameters) 
% 
% sessionName = '20190514_B';
% load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2019_05_14_B.mat')
% genRelTrialTimes_Stroke(sessionName,TrialsParameters)
% 
sessionName = '20190516_B';
load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2019_05_16_B.mat')
genRelTrialTimes_Stroke(sessionName,TrialsParameters)

sessionName = '20190517_B';
load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2019_05_17_B.mat')
genRelTrialTimes_Stroke(sessionName,TrialsParameters)
% 
% sessionName = '20190523_B';
% load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2019_05_23_B.mat')
% genRelTrialTimes_Stroke(sessionName,TrialsParameters)
% 
% sessionName = '20190524_B';
% load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2019_05_24_B.mat')
% genRelTrialTimes_Stroke(sessionName,TrialsParameters)
% 
sessionName = '20190527_B';
load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2019_05_27_B.mat')
genRelTrialTimes_Stroke(sessionName,TrialsParameters)

sessionName = '20190528_B';
load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2019_05_28_B.mat')
genRelTrialTimes_Stroke(sessionName,TrialsParameters)
% 
sessionName = '20190531_B';
load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2019_05_31_B.mat')
genRelTrialTimes_Stroke(sessionName,TrialsParameters)
% 
sessionName = '20190603_B';
load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2019_06_03_B.mat')
genRelTrialTimes_Stroke(sessionName,TrialsParameters)
% 
sessionName = '20190530_B';
load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2019_05_30_B.mat')
genRelTrialTimes_Stroke(sessionName,TrialsParameters)
% 
sessionName = '20190604_B';
load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2019_06_04_B.mat')
genRelTrialTimes_Stroke(sessionName,TrialsParameters)

% 
% sessionName = '20190510_B';
% load('\\LaboDancauseDS\LabData\MacaquesData\StrokeData\allTrialsParameters\Params_2019_05_10_B.mat')
% genRelTrialTimes_Stroke(sessionName,TrialsParameters)

% % 




% 
% % TrialsParameters = spikesPlotted.trialsParameters; 
% % clear spikesPlotted
% 
% % Select the trials to be kept
% isPrecision = TrialsParameters.isPrecision == 1;
% isOneOrThree = TrialsParameters.CondCode == 1 | TrialsParameters.CondCode == 3;
% keepTrial = isPrecision & isOneOrThree;
% 
% % Extracting relevant parameters
% trialNumbersOld = TrialsParameters.Number(keepTrial); % Getting trial numbers as set at the time of data collection
% trialStartTimes = TrialsParameters.TimeCueOn(keepTrial) - 500; % Set as 500 ms before TimeCueOn / LED on
% trialGoCue = TrialsParameters.TimeCueOff(keepTrial); 
% trialReachOn = TrialsParameters.TimeExitHP(keepTrial);
% trialGraspOn = TrialsParameters.TimeGraspStart(keepTrial);
% trialGraspOff = TrialsParameters.TimeGraspEnd(keepTrial);
% handUsed = TrialsParameters.isRightHand(keepTrial);
% 
% newTrialNumbers = TrialsParameters.newTrialNumber(keepTrial);
% 
% % generate OFD code
% relStartTimes = trialStartTimes / 1000;
% relEndTimes = trialGraspOff / 1000;
% 
% % % Set the segments where the waveforms will be invalidated
% % % Timing based on epochs in Ian's paper
% % segDeletionStart = [0 ; relEndTimes' + 0.150];  % start invalidating 100 ms (0.1 S) after end of grasp
% % segDeletionEnd = [relStartTimes'; 13000]; % end invalidating  500 ms (0.5 S) before TimeCueOn / LED on
% % 
% % % Check folder for Online Sorter Batch script
% % if ~isfolder('ofbScript')
% %     mkdir('ofbScript')
% % end
% %   
% % % % Writing Online Sorter Batch script
% % % 
% % ofbScriptDir = 'C:\Users\Stephan\Documents\MatlabAnalysis\inactivationMacaqueData2020\segmentInvalidationScripts\ofbScripts\';
% % fid = fopen([ofbScriptDir 'invalidateSegments_' sessionName '.ofb'],'wt');
% % 
% % fprintf(fid, 'ForEachChannel InvalidateWfsBetweenTimes\n');
% % fprintf(fid, '\n');
% % 
% % for i = 1:length(segDeletionStart)
% %     
% %     fprintf(fid, 'set TimeStart ');
% %     fprintf(fid, [num2str(segDeletionStart(i)) '\n']);
% %     
% %     fprintf(fid, 'set TimeEnd ');
% %     fprintf(fid, [num2str(segDeletionEnd(i)) '\n']);
% %     
% %     fprintf(fid, 'Process\n');
% %     
% %     fprintf(fid, '\n');
% %     
% % end
% % 
% % fprintf(fid, 'ForEachFile ExportToPLX\n');
% % 
% % fprintf(fid, '\n');
% % 
% % fprintf(fid, 'set SaveInvalidated 0\n');
% % 
% % fprintf(fid, '\n');
% % 
% % fprintf(fid, 'Process\n');
% % 
% % fclose(fid);
% 
% if ~isfolder('relTrialTimes')
%     mkdir('relTrialTimes')
% end
% 
% % Times used and other times that will be useful in subsequent steps
% relTrialTimes.paramDef = 'newTrialNumbers | oldTrialNumbers | handUsed | trialStartTimes | trialGoCue | trialReachOn | trialGraspOn | trialGraspOff';
% relTrialTimes.data = [(newTrialNumbers)' (trialNumbersOld)' (handUsed)' (trialStartTimes / 1000)' (trialGoCue / 1000)' (trialReachOn / 1000)' (trialGraspOn / 1000)' (trialGraspOff / 1000)'];
% 
% save(['relTrialTimes\relTrialTimes_' sessionName],'relTrialTimes')
% 
% 




