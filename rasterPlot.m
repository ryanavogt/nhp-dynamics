function rasterPlot(window,spiketimes)
numtrial=size(spiketimes,1);

hold on
for trial=1:numtrial
    scatter(spiketimes{trial}, trial * ones(size(spiketimes{trial})), 10, 'k', 'filled');
end
hold off
t1=window(1);t2=window(end);
xlim([t1 t2])
xlabel('Time (ms)');
ylabel('Trial Number');
end
