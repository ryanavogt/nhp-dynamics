function psth=PSTH(spiketimes,binsize,window)

T=length(window)-1;
% spiketimes=ArmOri;
% binsize=1;

psth = zeros(T/binsize+1,2);                          % one extra chan for timebase
% psth (:,1) = (-(T/2):binsize:T/2);                  % time base
psth (:,1) = (window(1):binsize:window(end));         % time base
% if dataset==1
    spxtimes=sort(cell2mat(spiketimes))+abs(window(1));        %%%%%%%%%%
% else
%     spxtimes=(cell2mat(spiketimes));        %%%%%%%%%%
% end

for j=0:1:T/binsize
    psth(j+1,2)=length(spxtimes((binsize*j)<=spxtimes & spxtimes<(binsize*(j+1))));
end

end