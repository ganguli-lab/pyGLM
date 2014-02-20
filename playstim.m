load stim;

fig(1);

for j = 1:size(stim,1)
    imgsc(squeeze(stim(j,:,:)));
    drawnow;
end
