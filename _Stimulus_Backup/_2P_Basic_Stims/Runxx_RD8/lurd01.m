%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all;
global CRS;
vsgInit;
Scan_mode=2;
% CheckCard = crsInit('Config.vsg'); % Note: Config is located at..., calibrated 090421 Monitor # ...
% if (CheckCard < 0)
%     return;
% end;
crsIOWriteDAC(0,'Volts5');  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

viewdist=570;  % in mm (1400mm for awake room 2)
crsSetViewDistMM(viewdist);

screenSize_width_height= [crsGetScreenWidthDegrees, crsGetScreenHeightDegrees] ;	 % screen size in x and y in degrees
screenSizePixel_width_height= [crsGetScreenWidthPixels, crsGetScreenHeightPixels] ;	 % screen size in x and y in pixels. 

runmode=3;      % 1: key mode,  2: auto, 3: Anes (VDAQ) 5: Awake (Labview)
Nstim=8;       % total number of stimulus, used for generate stim parameter vectors (e.g. fixcolor, orientation...)
Ncond=9;        % Ncond = Nstim + number of blanks
fixsize=0;   % in deg, set to 0 for no fixation spot (anesthetized exp)
sccadex=4;      % in deg, for both left and right
Tpre=0.5;       % second, pre-stimulus blank time, only for anesthetized imaging (runmode=3)
Tcue=3.5;       % second, the time to run each stimilus.
ISI = 1;
% Direction definition ----- 0: stable, 1: right, 2: up right, 3: up, 4: up left
%                                       5: left, 6: down left, 7: down, 8: down right
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nblock = 100;
stimname = 'Runxx_RD8_';
[stimlist,logname] = two_photon_stimlist_prepare(Nstim,Ncond,Nblock,stimname);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RD contour parameters
flagBiMask=[0]';   % 0: both mask are 1's, 1: use following 3 parameters (orientation0, SF0, phase0) to create two masks.
orientation0=[0]';
SF0=[0.4]';     % do not put 0, if no contour mask is needed, set flagmask as 0; and, don't use it as vector, otherwise preparing mask will fail.
lineDirection=[0]'; 
lineVelocity=[0]'; % in degree
randomizeFrameCount=0;  %NOT avilable now. if >0, ask for randomize map1 and map2

flagMaskLine=[0]';  % draw mask line on the final map (defined below), it is treated as a mask and prepared in 'LURDSub04Mask', use same orientation as contour (orientation0)
linethickness3=[0]';	% this is a temporary solution for drawing a line grating on RD, set to 0 if use SF0
linecolor3=[1];					% this is a temporary solution goes with linethickness

% RD patch one 
esize1=[2];        % RD element size, note: pix2deg=0.0208
elength1=[1]';     % start version 04, RD elements are bars of elength1 long orthogonal to the moving direction, note: the real length is a multiple of esize (e.g. esize=3, elength=4, then the real length is 6), just for easy coding 
rddensity1=[0.03]';  % RD density
velocity1= [8]';     % Speed of RD patch one, Note: for visage, the max speed is 37 deg/sec for 3 sec, which needs a 5429x5429 pixel scrach page. 
direction1=[1 2 3 4 5 6 7 8]';  
wincenter1=[    % [x0, y0] of the grating window
0,0
];   

winsize1=  [	 % window size in x and y in deg. 
30, 38
];   

whitecolor1=[            % [R1 G1 B1; R2 G2 B2]  so whitecolor(1, :)=[R1 G1 B1]
1, 1, 1
];
blackcolor1=[
0, 0, 0
];   

%Stimulus 0 -- Blank page definition
blankPageType=1; % 0: blank page; 1: random dot page;
dotSizeOfBlankPage=2;        % RD element size, in pixel
dotDensityOfBlankPage = 0.03;  % RD density: the percentage of area covered by random dots.

%interval page definition
isiPageType = 102; % ISI inter-stimulus interval display : [1-100] stim ID fist frame, 101: black screen, 102: last frame, 103: random dot page ....
isiPage = 4; % vedio page index for ISI page
isiPageMaskLine = 0; %display mask line ? 0:not; 1:based on definition of stimulus
isiHostPage = 0;
dotSizeOfIntervalPage = 2;        % RD element size, in pixel
dotDensityOfIntervalPage = 0.03;  % RD density: the percentage of area covered by random dots.

% 2nd patch added in v05 070901, currently only deal with overlap case (i.e. RD center/size are the same)
esize2=[2];
elength2=[1]';
rddensity2=[0]';  
velocity2 =[2]'; 
direction2=[6]';
wincenter2=wincenter1;  
winsize2=winsize1; 

MyFixationColor1=[1, 0, 0];   % red fixation, no saccade
MyFixationColor2=[0, 0, 1];     % blue fixation saccade
fixcolor=5;           % index to above myfixationcolors useing [4] or [4 4 4 5 5 5]' (i.e. length is either 1 or Nstim).
backgroundcolor=[0 0 0];  % <not implemented yet, need vectorize>
filtering=0;            % filter RD dots by gaussian
bktype=0;               % 0: no background (black) 1: RD background

%logname='log.txt';

%% transfer all variable from degree to pixel
wincenter1pix = round(crsUnitToUnit(CRS.DEGREEUNIT, wincenter1, CRS.PIXELUNIT)); %window center in x and y in pixels
winsize1pix = round(crsUnitToUnit(CRS.DEGREEUNIT, winsize1, CRS.PIXELUNIT)); %window size in x and y in pixels
wincenter2pix = round(crsUnitToUnit(CRS.DEGREEUNIT, wincenter2, CRS.PIXELUNIT)); %window center in x and y in pixels
winsize2pix = round(crsUnitToUnit(CRS.DEGREEUNIT, winsize2, CRS.PIXELUNIT)); %window size in x and y in pixels
velocity1pix = round(crsUnitToUnit(CRS.DEGREEUNIT, velocity1, CRS.PIXELUNIT));
velocity2pix = round(crsUnitToUnit(CRS.DEGREEUNIT, velocity2, CRS.PIXELUNIT));
SF0pix =  round(crsUnitToUnit(CRS.DEGREEUNIT, SF0, CRS.PIXELUNIT));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%deg2pix = crsUnitToUnit(CRS.DEGREEUNIT, 1, CRS.PIXELUNIT);
deg2pix = 33.7838;
framerate= crsGetSystemAttribute(CRS.FRAMERATE);  % framerate of current system
mapsize=round(max([velocity1pix; velocity2pix])*Tcue+max(max(winsize1pix(:), winsize2pix(:)))+10);     % scratch map that the window moves on
mapsize=round(mapsize / 2) * 2; % make sure it's a even number
totalframe=round(framerate*Tcue);
fprintf('mapsize=%d, totalframe=%d\r', mapsize, totalframe);

lineVelocityPixel = lineVelocity * deg2pix / framerate; % Mask line movement velocity in pixel.

hostpages=zeros(totalframe,1); % hostpages for store images prepared.
mapPages=zeros(2,1); % mapPages for store map1 and map2.
prePage = 0; % a scratch page for image preparation.

NBitIn=5;       % maximum 8, currently use 5 input bits for stim ID (Labview may use higher bit 7 8 was for something else). 
BitStimPrep=8;  % use 10th bit as indication of stim preparation (high is preparation, low is present)
fid=fopen(logname, 'w');
if isempty(backgroundcolor)
    backgroundcolor=(whitecolor1(1,:)+blackcolor1(1,:))./2;    % note: this is a easy way to calculate background gray
end

%hdKeyPreparation; NO KEYs

crsSetSpatialUnits(CRS.DEGREEUNIT);     % all units in degree of angles
crsSetDrawMode(CRS.CENTREXY);    % center is [0 0]

winsize1(1)=min(crsGetScreenHeightDegrees, winsize1(1));  % otherwise (winsize > screen size) will cause trouble
winsize1(2)=min(crsGetScreenWidthDegrees, winsize1(2));  % otherwise (winsize > screen size) will cause trouble
winsize2(1)=min(crsGetScreenHeightDegrees, winsize2(1));  % otherwise (winsize > screen size) will cause trouble
winsize2(2)=min(crsGetScreenWidthDegrees, winsize2(2));  % otherwise (winsize > screen size) will cause trouble

%some functions need [width height] as parameter.
winsize1pix_width_height = [winsize1pix(2) winsize1pix(1)];
winsize1_width_height = [winsize1(2) winsize1(1)];

crsSetCommand(CRS.PALETTECLEAR);
Palette=zeros(256, 3);
crsPaletteSet(Palette);
crsPaletteSetPixelLevel(1, backgroundcolor);
crsPaletteSetPixelLevel(2, [1, 1, 1]);          % white
crsPaletteSetPixelLevel(3, [0, 0, 0]);          % black
crsPaletteSetPixelLevel(4, MyFixationColor1);   % red fixation
crsPaletteSetPixelLevel(5, MyFixationColor2);   % blue  fixation
crsPaletteSetPixelLevel(6, whitecolor1(1,:));   % RD dot color
crsPaletteSetPixelLevel(7, blackcolor1(1,:));   % no use here
crsPaletteSetPixelLevel(8, (whitecolor1(1,:) + blackcolor1(1,:)) / 2);   % Grey color

% Definition for hdRandomDot which will be used as control stimulus
controlRandomDot = hdRandomDot(wincenter1pix(1), wincenter1pix(2), winsize1pix_width_height(1), winsize1pix_width_height(2));
controlRandomDot.DotNumber = 100;
controlRandomDot.DotSize = [4 4];
controlRandomDot.DotPen = 6;
controlRandomDot.BackgroundPen = 7;
controlRandomDot.FrameCount = totalframe;
controlRandomDot.Speed = deg2pix * 4;
controlRandomDot.MoveLength = deg2pix * 6;

% prepare stim parameter arrays
% vectorized following (each parameter should have size(x,1) either =1 or =Nstim), note: repeats always in column dimension 
%   fixcolor,
%   wincenter1, size1, rddensity1, esize1, direction1, whitecolor1, blackcolor1
LURDSub02StimParaPrep;   % a section of code for preparation of stimulus parameters (separated just for clearity)

% temp testing code start here
% mask1=imread('01map.bmp');
% mask1n=1-mask1;
LURDSub04Mask;      % create all contour masks
% temp testing code finish here

newstim=1;  
testrun=0;   % if testrun=1, show stimulus and pause (press any key to continue)
% initiallization of video pages 1-4 (note, all parameters (e.g. color, orientation...) use the first one in the array)

LURDSub03StimPrep;       % a section of code for preparation 3 pages (fix, cue, saccade) (separated from main program just for clearity)
fprintf('\r Ready...\r');
pause;


%% starting loop
testrun=0;
newstim=-1;
oldstim=-1;
newstimprep=0; 
oldstimprep=0;
oldstimbitall=0;
digtemplet = '0000000000';      % to combine with digbitcheck 
%crsIOWriteDigitalOut(1,bin2dec('0000000001'));  % set ready to high (ready), bin2dec('0000000001') to specify the lowest bit, basicly visage is always ready except when prepare stim (so vdaqs need wait a little bit to listen to ready line)
crsIOWriteDigitalOut(0,bin2dec('0000000001'));  % try to set ready 'low' all the time, and high after prepare stim.

flagstimprep=0;
flagstimpres=1;      

totaltrial=0;
bstoped=0;
 % 2-photon
totalblock = 1;

while (1)   % will continue loop, press "ctrl C" to stop
%% Stim ID Generation    
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
%% Stimulus preperation
    if flagstimprep   % this is an preparation, not for presentation
        fprintf('block No. %d ...', totalblock);
        fprintf('stim %d ...', newstim);
        testrun=0;
        if newstim==0 
            totaltrial=totaltrial+1;
            fprintf(fid, '%d \t', newstim);
            if(blankPageType ~=0)%blank的时候不放黑屏，还是放随机点
                hdPrepareOneSideDotPage(CRS.VIDEOPAGE, 1, wincenter1pix(1,:), 6,7,winsize1pix, dotSizeOfBlankPage, dotDensityOfBlankPage);          
            end
            if mod(totaltrial, Ncond)==0
                        
                % 2-photon: 
                totalblock = totalblock+1;
                
                fprintf(fid, '\r\n');
                fclose(fid);
                fid=fopen(logname, 'a');                
            end
            fprintf('blank: prepared.\r');
            tic
        elseif newstim>0 %&& newstim<=Nstim
            LURDSub03StimPrep;                    % prepare stim presentation 
            
            totaltrial=totaltrial+1;
            fprintf(fid, '%d \t', newstim);
            if mod(totaltrial, Ncond)==0
                
                % 2-photon: 
                totalblock = totalblock+1;
                
                fprintf(fid, '\r\n');
                fclose(fid);
                fid=fopen(logname, 'a');                
            end
            tic
        else
            fprintf('stim ID out of range (1-%d)...', Nstim);
        end
    else
        fprintf('(%d)\r', newstim);
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    newstimprep='0' ;
    if newstimprep=='1'     % for preparation only
        flagstimprep=1;
        flagstimpres=0;
    else                    % for presentation only
        flagstimprep=0;
        flagstimpres=1;
    end
%% presentation
   if Scan_mode==1;  % GA才需对齐
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
        if newstim==0 % blank page
            pause(Tpre+Tcue);
            crsSetDisplayPage(1);
%             crsSetDisplayPage(isiPage);
            if(isiPageType ~= 102)
                pause(Tcue+Tpre);
                if(isiPageType == 103)
                    hdPrepareOneSideDotPage(CRS.VIDEOPAGE, isiPage, wincenter1pix(1,:), 6,7,winsize1pix, dotSizeOfIntervalPage, dotDensityOfIntervalPage);          
                end
                crsSetDisplayPage(isiPage);
                pause(Tcue+Tpre);
            end
            
        elseif newstim<=Nstim               % cue page
            displayPages = [5 6]; %The vedio pages used for display.
            pause(Tpre);
            t0=clock;
            for n = 1:totalframe
                displayPage = displayPages(mod(n, 2)+1); %switch display page
                
                crsSetDrawPage(CRS.VIDEOPAGE, displayPage, 1);
                crsSetDrawMode(CRS.CENTREXY);
                
                %copy pre-generated image from host page to vedio page.
                crsDrawMoveRect(CRS.HOSTPAGE, hostpages(n),[0 0],winsize1_width_height,wincenter1(newstim, :),winsize1_width_height);
                
                %display current page.
                crsSetDisplayPage(displayPage);
            end
            fprintf(' display movie cost %3.2f sec...\r', etime(clock, t0));
            
            if(isiPageType ~= 102)
                if(isiPageType == 103)
                    hdPrepareOneSideDotPage(CRS.VIDEOPAGE, isiPage, wincenter1pix(1,:), 6,7,winsize1pix, dotSizeOfIntervalPage, dotDensityOfIntervalPage);          
                end
                crsSetDisplayPage(isiPage);
            end
        else
            fprintf('stim ID (%d) out of range (1 - %d)\r', newstim, Nstim);
        end
    end
    crsIOWriteDAC(0,'Volts5');  
    toc;
%     crsSetDisplayPage(1);
    pause(ISI);            
    oldstim=newstim;    % make current stim old so the program will go back to while loop searching for new stim
    oldstimprep=newstimprep;
    if newstim~=0 & newstimprep~=1
%        fprintf('\n');
    end
end
fclose(fid);


% notes:
% when use masking, winsize should not change (since mask is pre-drawn)
