%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all;
global CRS;
vsgInit;
Scan_mode=1;
% CheckCard = crsInit('Config.vsg'); % Note: Config is located at..., calibrated 090421 Monitor # ...
% if (CheckCard < 0)
%     return;
% end;
crsIOWriteDAC(0,'Volts5');  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nstim=8;        % total number of stimulus, used for generate stim parameter vectors (e.g. fixcolor, orientation...)
Ncond=9;        % Ncond = Nstim + number of blanks
Tpre=0.5;       % second, pre-stimulus blcrsinitank time, only for anesthetized imaging (runmode=3)
Tcue=3;      % Grating 1/2 duration in second (if Tcueb ~= 0, Tcue1 usually for adaption)
ISI = 2.5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nblock = 100;
stimname = 'Runxx_G8_';
[stimlist,logname] = two_photon_stimlist_prepare(Nstim,Ncond,Nblock,stimname);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% grating parameter
wincenter1=[0 0];    % [x0, y0] of the grating window, middle of the screen is [0 0]
winsize1= [38 30];	 % window size in x and y in deg. For 20 inch screen at 570mm, [40, 30] is about full screen
gratingtype1=[1];  % 1: square wave, 2: sinewave, 3: load&display image
orientation1=[0 45 90 135 180 225 270 315]';   % 0-180, use 999 for random, orientation 0 is H gratings move up, 90 is Vertical gratings moving left
SF1=[1.5];           % Spatial frequency
velocity1=[8]';          % in cycle/sec
dutycycle1=[0.2]';          % white line is thiner if dutycycle < 0.5
whitecolor1=[0.5 0.5 0.5];            % [R1 G1 B1; R2 G2 B2]  so whitecolor(1, :)=[R1 G1 B1]
blackcolor1=[0 0 0];
backgroundcolor=[0.1 0.1 0.1];  % <not implemented yet, need vectorize>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% parameters donot change
viewdist=570;   % in mm
fixsize=0;      % in deg, set to 0 for no fixation spot (anesthetized exp)
sccadex=4;      % in deg, for both left and right
gratingtype2=gratingtype1; 
gratingtype1b=gratingtype1;  % the grating showed after the first one (when 1st one is for adaption)
gratingtype2b=gratingtype1; 
wincenter2=wincenter1;    
winsize2 =winsize1;    % [0 0] to disable 2nd grating
SF2=SF1;                      
SF1b=SF1; 
SF2b=SF1;                      
TF1=[0]';              % temporal modulation frequency    
TF2=0;                       
TF1b=[0]'; 
TF2b=0;                       
TFtype1=[0];            % temporal envelope: 1=square wave, 2=sinewave; (if no temporal modulation is needed, set TF zero)
TFtype2=[0];
Tphase1=999;            % temporal modulation phase <need more test>
Tphase2=999;
Tphase1b=[999];       % use 999 for random, use 9999 for same value as Tphase1 (when Tphase1 is random), use 99999 for reversed phase to Tphase1 (360-Tphase1), other values are treated as degrees 
Tphase2b=[999];
velocity2=velocity1;
orientation2=orientation1;   
orientation1b=[0]';  
orientation2b=[0]';   
direction1=[1]';            % 3 values allowed: 1, -1, 999; -1 is reverse direction (+180), 999 is random (either 1 or -1).
direction2=[1]';			% dir only use ful for random dir, other cases can be handeled by ori with range (0-360)
direction1b=[1];
direction2b=[1];
Sphase1=999;                % 0-360 degree, 999 for random.
Sphase2=999;
Sphase1b=99999;            % use 999 for random, use 9999 for same value as Sphase1 (when Sphase1 is random), use 99999 for reversed phase to Sphase1 (360-Sphase1), other values are treated as degrees 
Sphase2b=99999;
dutycycle2=dutycycle1;
whitecolor2=whitecolor1;            
blackcolor2=blackcolor1;
whitecolor1b=whitecolor1;            
blackcolor1b=blackcolor1;
MyFixationColor1=[0.5, 0, 0];   % red fixation, no saccade
MyFixationColor2=[0,   0, 1];     % blue fixation saccade
fixcolor=[4]';           % index to above myfixationcolors (4 for fixcolor1, 5 for fix color2) useing [4] or [4 4 4 5 5 5] (i.e. length is either 1 or Nstim).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% deg2pix=19.685; 
deg2pix=33.7838;% this value is calculated for 570mm distance, Sony (screen width 16") and 800 pixel horizontal resolution; 090716
pix2deg=0.03;  % this value is calculated for 570mm distance, Sony (screen width 16") and 800 pixel horizontal resolution; 090716
NBitIn=7;              % maximum 8, currently use 5 input bits for stim ID (Labview may use higher bit 7 8 was for something else). 
BitStimPrep=8;  % use 10th bit as indication of stim preparation (high is preparation, low is present) (in mode3 (anes) it should be 8 for Go bit)
fid=fopen(logname, 'w');
pixtablelow1 = 11;   % pixel levels for grating animation, total 252 levels available, need devide if draw 2 or more obj, see "setPixLevel" for detail
pixtablelow2 = 61;
pixtablelow1b = 111;   
pixtablelow2b = 161;
npixlevel = 50;     % currently use 50 levels for each gratings, so grating 1 is 11:60, grating 2 is 61:110, grating 1b is 111:161, grating 2b is 161:210
if isempty(backgroundcolor)
    if gratingtype1(1)==1
        backgroundcolor=whitecolor1(1,:).*dutycycle1(1)+blackcolor1(1,:)*(1-dutycycle1(1))    % note: this is a easy way to calculate background gray
    elseif gratingtype1(1)==2
        backgroundcolor=(whitecolor1(1,:)+blackcolor1(1,:))./2;    % note: this is a easy way to calculate background gray
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LUGSub01KeyPrep;        % a section of code prepare keyboard input
crsSetViewDistMM(viewdist);
crsSetSpatialUnits(CRS.DEGREEUNIT);     % all units in degree of angles
crsSetDrawMode(CRS.CENTREXY);    % center is 0
winsize1(1)=min(crsGetScreenWidthDegrees, winsize1(1));  % it will cause trouble if winsize(1) is larger than screen
winsize2(1)=min(crsGetScreenWidthDegrees, winsize2(1));
winsize1(2)=min(crsGetScreenHeightDegrees, winsize1(1));
winsize2(2)=min(crsGetScreenHeightDegrees, winsize2(1));
crsSetCommand(CRS.PALETTECLEAR);
Palette=zeros(256, 3);
crsPaletteSet(Palette);
crsPaletteSetPixelLevel(1, backgroundcolor);
crsPaletteSetPixelLevel(2, [1, 1, 1]);          % white
crsPaletteSetPixelLevel(3, [0, 0, 0]);          % black
crsPaletteSetPixelLevel(4, MyFixationColor1);   % red fixation
crsPaletteSetPixelLevel(5, MyFixationColor2);   % blue  fixation
LUGSub02StimParaPrep;   % a section of code for preparation of stimulus parameters (separated just for clearity)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fill all page (page 1-5) with background gray
for i=1:5
    crsSetDrawPage(CRS.VIDEOPAGE, i, 1);     
end
crsSetDisplayPage(1);                           % page 1: blank page

newstim=1;  
testrun=0;                                      % if testrun=1, show stimulus and pause (press any key to continue)
% initiallization of video pages 1-4 (note, all parameters (e.g. color, orientation...) use the first one in the array)
LUGSub03PagePrep;       % a section of code for preparation 3 pages (fix, cue, saccade) (separated from main program just for clearity)
fprintf('\r Ready...\r');
pause;

% starting loop
testrun=0;
newstim=0;
oldstim=0;
newstimprep=0; 
oldstimprep=0;
oldstimbitall=0;
LEyeOpen=0;
REyeOpen=0;
digtemplet = '0000000000';      % to combine with digbitcheck 
%crsIOWriteDigitalOut(1,bin2dec('0000000001'));  % set ready to high (ready), bin2dec('0000000001') to specify the lowest bit, basicly visage is always ready except when prepare stim (so vdaqs need wait a little bit to listen to ready line)
crsIOWriteDigitalOut(0,bin2dec('1111111111'));  % set ready 'low' (not ready) all the time, and high after prepared stim.
totaltrial=0;

 % 2-photon
totalblock = 1;

while (1)   % will continue loop, press "ctrl C" to stop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    newstimprep='1' ;
    if newstimprep=='1'     % for preparation only
        flagstimprep=1;
        flagstimpres=0;
    else                    % for presentation only
        flagstimprep=0;
        flagstimpres=1;
    end
    cnd = mod(totaltrial, Ncond)+1;
    newstim = stimlist(totalblock,cnd);
    oldstim=newstim;    % make current stim old so the program will go back to while loop searching for new stim
    oldstimprep=newstimprep;
%% prepare
    if flagstimprep   % this is an preparation, not for presentation
        fprintf('block No. %d ...', totalblock);
        fprintf('stim %d ...', newstim);
        testrun=0;
        if newstim==0 % just ignor these impossible situation (early version Labview has difficulties to bypass this), these condition is treated as no change on stim lines
            totaltrial=totaltrial+1;
            fprintf(fid, '%d \t', newstim);
            if mod(totaltrial, Ncond)==0
                totalblock = totalblock+1;
                fprintf(fid, '\r\n');
                fclose(fid);
                fid=fopen(logname, 'a');                
            end
        elseif newstim>0 & newstim<=Nstim
            crsObjDestroy(objhandle1);
            crsObjDestroy(objhandle2);
            crsObjDestroy(objhandle1b);
            crsObjDestroy(objhandle2b);
            LUGSub03PagePrep;                    % prepare stim pages
            totaltrial=totaltrial+1;
            fprintf(fid, '%d \t', newstim);
            if mod(totaltrial, Ncond)==0
                        
                % 2-photon: 
                totalblock = totalblock+1;
                
                fprintf(fid, '\r\n');
                fclose(fid);
                fid=fopen(logname, 'a');                
            end
        else
            fprintf('stim ID out of range (1-%d)...', Nstim);
        end
    else
    end   %if flagstimprep
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    newstimprep='0' ;
    if newstimprep=='1'     % for preparation only
        flagstimprep=1;
        flagstimpres=0;
    else                    % for presentation only
        flagstimprep=0;
        flagstimpres=1;
    end
%% present
   if Scan_mode==1;  % GA²ÅÐè¶ÔÆë
        aa1=0; aa2=0;
        while 1;
            aa1=aa2; aa2=crsIOReadADC(0);  %disp([aa1,aa2]);pause(0.1);
            if aa2<aa1  && aa2>9000 && aa2<10000; %
                break;
            end
        end
    end
    crsIOWriteDAC(5,'Volts5');
    if flagstimpres 
        if newstim==0                       % blank page
            pause(Tpre);
            crsSetDisplayPage(1);
            tic;
            pause(Tcue);
        elseif newstim>0
          pause(Tpre);
           if Tcue>0
               crsSetDrawPage(3);
               tic
               crsPresent;        % crsPresent command will activate the last drawing page
           end
           pause(Tcue);
        else
            fprintf('stim ID (%d) out of range (1 - %d)\r', newstim, Nstim);
            crsSetDisplayPage(1);
        end % if
    end % if newstimprep==1
    crsIOWriteDAC(0,'Volts5');  
    toc;
    crsSetDisplayPage(1);
    pause(ISI);            
    oldstim=newstim;    % make current stim old so the program will go back to while loop searching for new stim
    oldstimprep=newstimprep;
    if newstim~=0 & newstimprep~=1
%        fprintf('\n');
    end
end
fclose(fid);