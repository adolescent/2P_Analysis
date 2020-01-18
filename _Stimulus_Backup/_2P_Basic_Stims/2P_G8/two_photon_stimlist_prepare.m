function [stimlist,logname] = two_photon_stimlist_prepare(Nstim,Ncond,Nblock,stimname)
% input the Number of stim and Number of condition, Number of block
% output the stlimlist
logname=strcat(stimname,num2str(rem(now,100)),'.txt');
stimlist = zeros(Nblock,Ncond);
for i = 1:Nblock
    seq_temp = randperm(Ncond)-1;
    stimlist(i,:) = seq_temp;
end
save([logname(1:end-4),'_list.mat'],'stimlist');