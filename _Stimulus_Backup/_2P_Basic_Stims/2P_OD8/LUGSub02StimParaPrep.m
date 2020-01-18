% HDGSub02StimParaPrep;   % a section of code for preparation of stimulus parameters (separated just for clearity)

% prepare stim parameter arrays
% vectorized following (each parameter should have size(x,1) either =1 or =Nstim), note: repeats always in column dimension 
%   fixcolor,
%   gratingtype1, wincenter1, size1, orientation1, direction1, velocity1, whitecolor1, blackcolor1, dutycycle1, SF1, Sphase1, TF1, Tphase1
%   gratingtype2, wincenter2, size2, orientation2, direction2, velocity2, whitecolor2, blackcolor2, dytycycle2, SF2, Sphase2, TF2, Tphase2
%   gratingtype1b, SF1b, TF1b, orientation1b,
%   gratingtype2b, SF2b, TF2b, orientation2b
% Sphase, Tphase, direction can be randomnized. 

if length(fixcolor)==1
    fixcolor=repmat(fixcolor, Nstim, 1);
elseif length(fixcolor)~=Nstim
    fprintf('Error: <length of fixcolor>\r');
end
if length(gratingtype1)==1
    gratingtype1=repmat(gratingtype1, Nstim, 1);
elseif length(gratingtype1)~=Nstim
    fprintf('Error: <length of gratingtype1>\r');
end
if length(gratingtype2)==1
    gratingtype2=repmat(gratingtype2, Nstim, 1);
elseif length(gratingtype2)~=Nstim
    fprintf('Error: <length of gratingtype2>\r');
end
if length(velocity1)==1
    velocity1=repmat(velocity1, Nstim, 1);
elseif length(velocity1)~=Nstim
    fprintf('Error: <length of velocity1>\r');
end
if length(velocity2)==1
    velocity2=repmat(velocity2, Nstim, 1);
elseif length(velocity2)~=Nstim
    fprintf('Error: <length of velocity2>\r');
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
if size(whitecolor2, 1)==1
    whitecolor2=repmat(whitecolor2, Nstim, 1);
elseif size(whitecolor2, 1)~=Nstim
    fprintf('Error: <length of whitecolor2>\r');
end
if size(blackcolor2, 1)==1
    blackcolor2=repmat(blackcolor2, Nstim, 1);
elseif size(blackcolor2, 1)~=Nstim
    fprintf('Error: <length of blackcolor2>\r');
end
if size(wincenter1, 1)==1
    wincenter1=repmat(wincenter1, Nstim, 1);
elseif size(wincenter1, 1)~=Nstim
    fprintf('Error: <length of wincenter1>\r');
end
if size(wincenter2, 1)==1
    wincenter2=repmat(wincenter2, Nstim, 1);
elseif size(wincenter2, 1)~=Nstim
    fprintf('Error: <length of wincenter2>\r');
end
if size(winsize1, 1)==1
    winsize1=repmat(winsize1, Nstim, 1);
elseif size(winsize1, 1)~=Nstim
    fprintf('Error: <length of winsize1>\r');
end
if size(winsize2, 1)==1
    winsize2=repmat(winsize2, Nstim, 1);
elseif size(winsize2, 1)~=Nstim
    fprintf('Error: <length of winsize2>\r');
end
if length(orientation1)==1
    orientation1=repmat(orientation1, Nstim, 1);
elseif length(orientation1)~=Nstim
    fprintf('Error: <length of orientation1>\r');
end
if length(orientation2)==1
    orientation2=repmat(orientation2, Nstim, 1);
elseif length(orientation2)~=Nstim
    fprintf('Error: <length of orientation2>\r');
end
if length(dutycycle1)==1
    dutycycle1=repmat(dutycycle1, Nstim, 1);
elseif length(dutycycle1)~=Nstim
    fprintf('Error: <length of dutycycle1>\r');
end
if length(dutycycle2)==1
    dutycycle2=repmat(dutycycle2, Nstim, 1);
elseif length(dutycycle2)~=Nstim
    fprintf('Error: <length of dutycycle2>\r');
end
if length(SF1)==1
    SF1=repmat(SF1, Nstim, 1);
elseif length(SF1)~=Nstim
    fprintf('Error: <length of SF1>\r');
end
if length(SF2)==1
    SF2=repmat(SF2, Nstim, 1);
elseif length(SF2)~=Nstim
    fprintf('Error: <length of SF2>\r');
end
if length(TF1)==1
    TF1=repmat(TF1, Nstim, 1);
elseif length(TF1)~=Nstim
    fprintf('Error: <length of TF1>\r');
end
if length(TF2)==1
    TF2=repmat(TF2, Nstim, 1);
elseif length(TF2)~=Nstim
    fprintf('Error: <length of TF2>\r');
end
if length(direction1)==1
    direction1=repmat(direction1, Nstim, 1);
elseif length(direction1)~=Nstim
    fprintf('Error: <length of direction1>\r');
end
if length(direction2)==1
    direction2=repmat(direction2, Nstim, 1);
elseif length(direction2)~=Nstim
    fprintf('Error: <length of direction2>\r');
end

if length(Sphase1)==1
    Sphase1=repmat(Sphase1, Nstim, 1);
elseif length(Sphase1)~=Nstim
    fprintf('Error: <length of Sphase1>\r');
end
if length(Sphase2)==1
    Sphase2=repmat(Sphase2, Nstim, 1);
elseif length(Sphase2)~=Nstim
    fprintf('Error: <length of Sphase2>\r');
end
if length(TFtype1)==1
    TFtype1=repmat(TFtype1, Nstim, 1);
elseif length(TFtype1)~=Nstim
    fprintf('Error: <length of TFtype1>\r');
end
if length(TFtype2)==1
    TFtype2=repmat(TFtype2, Nstim, 1);
elseif length(TFtype2)~=Nstim
    fprintf('Error: <length of TFtype2>\r');
end
if length(Tphase1)==1
    Tphase1=repmat(Tphase1, Nstim, 1);
elseif length(Tphase1)~=Nstim
    fprintf('Error: <length of Tphase1>\r');
end
if length(Tphase2)==1
    Tphase2=repmat(Tphase2, Nstim, 1);
elseif length(Tphase2)~=Nstim
    fprintf('Error: <length of Tphase2>\r');
end

%   SF1b, TF1b, orientation1b, Sphase1b, Tphase1b
%   SF2b, TF2b, orientation2b, Sphase2b, Tphase2b
if size(gratingtype1b, 1)==1
    gratingtype1b=repmat(gratingtype1b, Nstim, 1);
elseif size(gratingtype1b, 1)~=Nstim
    fprintf('Error: <length of gratingtype1b>\r');
end
if size(gratingtype2b, 1)==1
    gratingtype2b=repmat(gratingtype2b, Nstim, 1);
elseif size(gratingtype2b, 1)~=Nstim
    fprintf('Error: <length of gratingtype2b>\r');
end
if size(SF1b, 1)==1
    SF1b=repmat(SF1b, Nstim, 1);
elseif size(SF1b, 1)~=Nstim
    fprintf('Error: <length of SF1b>\r');
end
if size(SF2b, 1)==1
    SF2b=repmat(SF2b, Nstim, 1);
elseif size(SF2b, 1)~=Nstim
    fprintf('Error: <length of SF2b>\r');
end
if size(TF1b, 1)==1
    TF1b=repmat(TF1b, Nstim, 1);
elseif size(TF1b, 1)~=Nstim
    fprintf('Error: <length of TF1b>\r');
end
if size(TF2b, 1)==1
    TF2b=repmat(TF2b, Nstim, 1);
elseif size(TF2b, 1)~=Nstim
    fprintf('Error: <length of TF2b>\r');
end
if size(orientation1b, 1)==1
    orientation1b=repmat(orientation1b, Nstim, 1);
elseif size(orientation1b, 1)~=Nstim
    fprintf('Error: <length of orientation1b>\r');
end
if size(orientation2b, 1)==1
    orientation2b=repmat(orientation2b, Nstim, 1);
elseif size(orientation2b, 1)~=Nstim
    fprintf('Error: <length of orientation2b>\r');
end
if size(Sphase1b, 1)==1
    Sphase1b=repmat(Sphase1b, Nstim, 1);
elseif size(Sphase1b, 1)~=Nstim
    fprintf('Error: <length of Sphase1b>\r');
end
if size(Sphase2b, 1)==1
    Sphase2b=repmat(Sphase2b, Nstim, 1);
elseif size(Sphase2b, 1)~=Nstim
    fprintf('Error: <length of Sphase2b>\r');
end
if size(Tphase1b, 1)==0     % noet here use 9999 for same value as gratinb 1
    Tphase1b=9999;
end
if size(Tphase1b, 1)==1
    Tphase1b=repmat(Tphase1b, Nstim, 1);
elseif size(Tphase1b, 1)~=Nstim
    fprintf('Error: <length of Tphase1b>\r');
end
if size(Tphase2b, 1)==0     % noet here use 9999 for same value as gratinb 1
    Tphase2b=9999;
end
if size(Tphase2b, 1)==1
    Tphase2b=repmat(Tphase2b, Nstim, 1);
elseif size(Tphase2b, 1)~=Nstim
    fprintf('Error: <length of Tphase2b>\r');
end

% if size(, 1)==1
%     =repmat(, Nstim, 1);
% elseif size(, 1)~=Nstim
%     fprintf('Error: <length of >\r');
% end
% if length()==1
%     =repmat(, Nstim, 1);
% elseif length()~=Nstim
%     fprintf('Error: <length of >\r');
% end

% note here when orientation1 and direction1 are radomnized, orientation1b and direction1b will be the same value. didn't implement randomnization here. 