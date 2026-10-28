function genRelTrialTimes_Stroke(sessionName,TrialsParameters)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% TrialsParameters = spikesPlotted.trialsParameters; 
% clear spikesPlotted

%Condition and orientation codes: left (l)= 0 , right (r)= 1; vertical (v)= 3 , horizontal (h)= 1 : lv=03 , rv=13 , lh=01 , rh=11
%Condition and orientation codes: left (l)= 0 , right (r)= 1; vertical (v)= 3 , horizontal (h)= 1 : lv=03 , rv=13 , lh=01 , rh=11
%lv=0+3 = 3; rv=1+3=4; lh=0+1=1; rh=1+1=2;
% Select the trials to be kept
isPrecision = TrialsParameters.isPrecision == 2;
isOneOrThree = TrialsParameters.CondCode == 1 | TrialsParameters.CondCode == 3;
keepTrial = isPrecision & isOneOrThree;

% Extracting relevant parameters
trialNumbersOld = TrialsParameters.Number(keepTrial); % Getting trial numbers as set at the time of data collection
trialbaseline = TrialsParameters.TimeCueOn(keepTrial) - 250; % Set as 250 ms before TimeCueOn / LED ontrialCueOn = TrialsParameters.TimeCueOn(keepTrial); 
trialCueOn = TrialsParameters.TimeCueOn(keepTrial);
trialGoCue = TrialsParameters.TimeCueOff(keepTrial); 
trialReachOn = TrialsParameters.TimeExitHP(keepTrial);
trialGraspOn = TrialsParameters.TimeGraspStart(keepTrial);
trialGraspOff = TrialsParameters.TimeGraspEnd(keepTrial);
handUsed = TrialsParameters.isRightHand(keepTrial);
orientation=TrialsParameters.CondCode(keepTrial);
newTrialNumbers = TrialsParameters.newTrialNumber(keepTrial);

% generate OFD code
% relStartTimes = trialCueOn / 1000;
% relEndTimes = trialGraspOff / 1000;

% % Set the segments where the waveforms will be invalidated
% % Timing based on epochs in Ian's paper
% segDeletionStart = [0 ; relEndTimes' + 0.150];  % start invalidating 100 ms (0.1 S) after end of grasp
% segDeletionEnd = [relStartTimes'; 13000]; % end invalidating  500 ms (0.5 S) before TimeCueOn / LED on
% 
% % Check folder for Online Sorter Batch script
% if ~isfolder('ofbScript')
%     mkdir('ofbScript')
% end
%   
% % % Writing Online Sorter Batch script
% % 
% ofbScriptDir = 'C:\Users\Stephan\Documents\MatlabAnalysis\inactivationMacaqueData2020\segmentInvalidationScripts\ofbScripts\';
% fid = fopen([ofbScriptDir 'invalidateSegments_' sessionName '.ofb'],'wt');
% 
% fprintf(fid, 'ForEachChannel InvalidateWfsBetweenTimes\n');
% fprintf(fid, '\n');
% 
% for i = 1:length(segDeletionStart)
%     
%     fprintf(fid, 'set TimeStart ');
%     fprintf(fid, [num2str(segDeletionStart(i)) '\n']);
%     
%     fprintf(fid, 'set TimeEnd ');
%     fprintf(fid, [num2str(segDeletionEnd(i)) '\n']);
%     
%     fprintf(fid, 'Process\n');
%     
%     fprintf(fid, '\n');
%     
% end
% 
% fprintf(fid, 'ForEachFile ExportToPLX\n');
% 
% fprintf(fid, '\n');
% 
% fprintf(fid, 'set SaveInvalidated 0\n');
% 
% fprintf(fid, '\n');
% 
% fprintf(fid, 'Process\n');
% 
% fclose(fid);

if ~isfolder('relTrialTimes')
    mkdir('relTrialTimes')
end

% Times used and other times that will be useful in subsequent steps
relTrialTimes.paramDef = 'newTrialNumbers | oldTrialNumbers | isRightHandUsed | angle | trialbaseline | trialCueOn | trialGoCue | trialReachOn | trialGraspOn | trialGraspOff';
relTrialTimes.data = [(newTrialNumbers)' (trialNumbersOld)' (handUsed)' (orientation)' (trialbaseline / 1000)' (trialCueOn / 1000)' (trialGoCue / 1000)' (trialReachOn / 1000)' (trialGraspOn / 1000)' (trialGraspOff / 1000)'];

save(['relTrialTimes\relTrialTimes_' sessionName,'-power'],'relTrialTimes')



end