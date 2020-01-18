% HDRDSub02StimParaPrep;   % a section of code for preparation of stimulus parameters (separated just for clearity)
% for RD

% prepare stim parameter arrays
% vectorized following (each parameter should have size(x,1) either =1 or =Nstim), note: repeats always in column dimension 
%   fixcolor, rddensity1, esize1, velocity1, direction1, wincenter1, winsize1, whitecolor1, blackcolor1, elength1
%             rddensity2, esize2, velocity2, direction2, wincenter2, winsize2, whitecolor2, blackcolor2, elength2

if length(fixcolor)==1
    fixcolor=repmat(fixcolor, Nstim, 1);
elseif length(fixcolor)~=Nstim
    fprintf('Error: <length of fixcolor>\r');
end

if length(rddensity1)==1
    rddensity1=repmat(rddensity1, Nstim, 1);
elseif length(rddensity1)~=Nstim
    fprintf('Error: <length of rddensity1>\r');
end

if length(esize1)==1
    esize1=repmat(esize1, Nstim, 1);
elseif length(esize1)~=Nstim
    fprintf('Error: <length of esize1>\r');
end

if length(velocity1)==1
    velocity1=repmat(velocity1, Nstim, 1);
elseif length(velocity1)~=Nstim
    fprintf('Error: <length of velocity1>\r');
end

if length(velocity1pix)==1
    velocity1pix=repmat(velocity1pix, Nstim, 1);
elseif length(velocity1pix)~=Nstim
    fprintf('Error: <length of velocity1pix1>\r');
end

if length(direction1)==1
    direction1=repmat(direction1, Nstim, 1);
elseif length(direction1)~=Nstim
    fprintf('Error: <length of direction1>\r');
end

if size(wincenter1, 1)==1
    wincenter1=repmat(wincenter1, Nstim, 1);
elseif size(wincenter1, 1)~=Nstim
    fprintf('Error: <length of wincenter1>\r');
end

if size(wincenter1pix, 1)==1
    wincenter1pix=repmat(wincenter1pix, Nstim, 1);
elseif size(wincenter1pix, 1)~=Nstim
    fprintf('Error: <length of wincenter1pix>\r');
end

if size(winsize1, 1)==1
    winsize1=repmat(winsize1, Nstim, 1);
elseif size(winsize1, 1)~=Nstim
    fprintf('Error: <length of winsize1>\r');
end

if size(whitecolor1, 1)==1
    whitecolor1=repmat(whitecolor1, Nstim, 1);
elseif size(whitecolor1, 1)~=Nstim
    fprintf('Error: <length of whitecolor1>\r');
end

if size(blackcolor1, 1)==1
    blackcolor1=repmat(blackcolor1, Nstim, 1);
elseif size(blackcolor1, 1)~=Nstim
    fprintf('Error: <length of blackcolor1>\r');
end

if length(elength1)==1
    elength1=repmat(elength1, Nstim, 1);
elseif length(elength1)~=Nstim
    fprintf('Error: <length of elength1>\r');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% patch 2
if length(rddensity2)==1
    rddensity2=repmat(rddensity2, Nstim, 1);
elseif length(rddensity2)~=Nstim
    fprintf('Error: <length of rddensity2>\r');
end

if length(esize2)==1
    esize2=repmat(esize2, Nstim, 1);
elseif length(esize2)~=Nstim
    fprintf('Error: <length of esize1>\r');
end

if length(velocity2)==1
    velocity2=repmat(velocity2, Nstim, 1);
elseif length(velocity2)~=Nstim
    fprintf('Error: <length of velocity2>\r');
end

if length(velocity2pix)==1
    velocity2pix=repmat(velocity2pix, Nstim, 1);
elseif length(velocity2pix)~=Nstim
    fprintf('Error: <length of velocity2pix>\r');
end

if length(direction2)==1
    direction2=repmat(direction2, Nstim, 1);
elseif length(direction2)~=Nstim
    fprintf('Error: <length of direction2>\r');
end

if size(wincenter2, 1)==1
    wincenter2=repmat(wincenter2, Nstim, 1);
elseif size(wincenter2, 1)~=Nstim
    fprintf('Error: <length of wincenter2>\r');
end

if size(wincenter2pix, 1)==1
    wincenter2pix=repmat(wincenter2pix, Nstim, 1);
elseif size(wincenter2pix, 1)~=Nstim
    fprintf('Error: <length of wincenter2pix>\r');
end

if size(winsize2, 1)==1
    winsize2=repmat(winsize2, Nstim, 1);
elseif size(winsize2, 1)~=Nstim
    fprintf('Error: <length of winsize2>\r');
end

if length(elength2)==1
    elength2=repmat(elength2, Nstim, 1);
elseif length(elength2)~=Nstim
    fprintf('Error: <length of elength2>\r');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% mask
if length(flagBiMask)==1
    flagBiMask=repmat(flagBiMask, Nstim, 1);
elseif length(flagBiMask)~=Nstim
    fprintf('Error: <length of flagBiMask>\r');
end

if length(orientation0)==1
    orientation0=repmat(orientation0, Nstim, 1);
elseif length(orientation0)~=Nstim
    fprintf('Error: <length of orientation0>\r');
end

if length(SF0)==1
    SF0=repmat(SF0, Nstim, 1);
elseif length(SF0)~=Nstim
    fprintf('Error: <length of SF0>\r');
end

if length(flagMaskLine)==1
    flagMaskLine=repmat(flagMaskLine, Nstim, 1);
elseif length(flagMaskLine)~=Nstim
    fprintf('Error: <length of flagMaskLine>\r');
end

if length(lineDirection)==1
    lineDirection=repmat(lineDirection, Nstim, 1);
elseif length(lineDirection)~=Nstim
    fprintf('Error: <length of lineDirection>\r');
end

if length(lineVelocity)==1
    lineVelocity=repmat(lineVelocity, Nstim, 1);
elseif length(lineVelocity)~=Nstim
    fprintf('Error: <length of lineVelocity>\r');
end

if length(lineVelocityPixel)==1
    lineVelocityPixel=repmat(lineVelocityPixel, Nstim, 1);
elseif length(lineVelocityPixel)~=Nstim
    fprintf('Error: <length of lineVelocityPixel>\r');
end

if length(linecolor3)==1
    linecolor3=repmat(linecolor3, Nstim, 1);
elseif length(linecolor3)~=Nstim
    fprintf('Error: <length of linecolor3>\r');
end

if isiPageType < 100 && (isiPageType < 0 || isiPageType > Nstim)
    fprintf('Error: <isiPageType out of range>\r');
end
        
