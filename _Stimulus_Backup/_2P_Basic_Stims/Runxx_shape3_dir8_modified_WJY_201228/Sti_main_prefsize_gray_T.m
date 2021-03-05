% Tang Rendong 20170317
clc; clear all;
addpath(genpath([pwd,'\stimprep']));

global CRS;
vsgInit;
crsIOWriteDAC(0,'Volts5');  
deg2pix=34.13;
% deg2pix=20.7263;
% deg2pix=crsUnitToUnit(CRS.DEGREEUNIT, 1, CRS.PIXELUNIT);
framestep=1;

% Stimulus Parameters
CenXY = [10.3,0.75] *deg2pix; % Degrees of visual angle 
sizep=4;
sizem=sizep*1.5;
SF = 1;           % Spatial frequency  % Cycles per degree of visual angle
TF = sizep *SF*framestep;

Scan_mode=2;  % 1 GA  2 RG
Nstim=24;     %  68
Ntrials=25;                  %repeat all stim 100 times; 
zoomrate=1.15;          % zoomrate of hdMatrixGrating

Orientation1=-90:30:240;   % 0-180, use 999 for random, orientation 0 is H gratings move up, 90 is Vertical gratings moving left
Direction1=0:30:330;   % 0-180, use 999 for random, orientation 0 is H gratings move up, 90 is Vertical gratings moving left

Orientation=-90:45:225;   % 0-180, use 999 for random, orientation 0 is H gratings move up, 90 is Vertical gratings moving left
Direction=0:45:315;   % 0-180, use 999 for random, orientation 0 is H gratings move up, 90 is Vertical gratings moving left

gratingtype =1;              %1.square wave 0.sine wave
dutycycle = 0.2;              % white line is thiner if dutycycle > 0.5
DotSize=3;            % RD element size, note: pix2deg=0.0208
DotDensity=0.2;      % RD density: the percentage of area covered by random dots.
Velocity=TF/SF;
curFrameRate= crsGetSystemAttribute(CRS.FRAMERATE);  % framerate of current system
Vpixl=TF/SF *deg2pix/curFrameRate; % =2 

time_ISI = 3;                 % Seconds.
time_on =  2;                  % Seconds.
cmSizep = round(sizep*deg2pix);
cmSizem = round(sizem*deg2pix);
totalframe=round(curFrameRate*time_on/framestep)+1;

Nsize=ceil(Vpixl*totalframe/cmSizem)+1;
% stimuli prepare
load_mat;

% Colour Definitions
bgl=0.12;  % background level=0.1
bgs=0.098;
crsPaletteSetPixelLevel(1, [bgl, bgl, bgl]);
crsPaletteSetPixelLevel(10, [0, 0, 0]);
crsPaletteSetPixelLevel(240, [1, 1, 1]);

% mask
hs0=cmSizep/2;
hs1=cmSizem/2;
x=1:cmSizem;  y=x;
rfx=repmat(x,length(x),1); rfx=rfx'; rfy=rfx';
mask0=sqrt((rfx-hs1).^2+(rfy-hs1).^2); 
mask1=mask0;
mask0(mask0<=hs0)=1; mask0(mask0>hs0)=0;  % prefsize
mask1(mask1<=hs1)=1; mask1(mask1>hs1)=0;  % prefsize*1.2

maskb=zeros(cmSizem,cmSizem);
maskb=maskb+bgs;

% 空间单位定义
viewdist=570;                 % in mm
crsSetViewDistMM(viewdist);
crsSetSpatialUnits(CRS.PIXELUNIT);     % all units in degree of angles

%draw prepare
crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
crsSetDrawPage(CRS.VIDEOPAGE, 2, 1);

% hostpages
hostpages=zeros(totalframe,1);
for n=1:totalframe
    hostpages(n) = crsPAGECreate(CRS.HOSTPAGE,[cmSizem,cmSizem], CRS.EIGHTBITPALETTEMODE)+1;    
    crsSetDrawPage(CRS.HOSTPAGE, hostpages(n));
end

% create condition file name
N_time=fix(clock);
logname=strcat('Sti main_gray_',num2str(N_time(4)),'_',num2str(N_time(5)),'_',num2str(N_time(6)),'.txt');

fprintf('press any key to start...\n');
pause;

% stimuli loop
for j=1:Ntrials  % trials
    stimindex=randperm(Nstim)';    % create and save condition file
%     stimindex=(52:Nstim)';    % create and save condition file

    fid=fopen(logname, 'a');
    fprintf(fid,'%2.0f\t',stimindex);  
    fclose(fid); 
    
    for i=1:Nstim     %  
        disp(['Ntrials= ', num2str(j)]);       
        stimID=stimindex(i);          

       if stimID<9                                
           stimid=stimID;       stimprep_G12; % 201228-WJY-for-G8

       elseif  stimID>8  && stimID<17            
           stimid=stimID-8;    stimprep_triangle8;

       elseif  stimID>16  && stimID<25            
           stimid=stimID-16;    stimprep_circle8;
       end 

    % stimuli present
    if Scan_mode==1;  % GA才需对齐
    aa1=0; aa2=0;
    while 1;
        aa1=aa2; aa2=crsIOReadADC(0);  %disp([aa1,aa2]);pause(0.1);
        if aa2<aa1 && aa2>9000 && aa2<10000;
            break;
        end
    end
    end
    crsIOWriteDAC(5,'Volts5');

    start_step = clock; % Record reference time stamp at beginning of each loop.
    for n=1:totalframe-1
        crsSetDrawPage(2);
        crsSetDrawMode(CRS.CENTREXY);
        %copy pre-generated image from host page to vedio page.
        crsDrawMoveRect(CRS.HOSTPAGE, hostpages(n),[0,0],[cmSizem,cmSizem],CenXY,[cmSizem,cmSizem]);                
        %display current page.
        crsSetDisplayPage(2);
%         crsSetDisplayPage(2); 
    end

    crsIOWriteDAC(0,'Volts5');  
    disp(etime(clock,start_step));
    crsSetDisplayPage(1); % Display background screen
    end
end
pause(time_ISI);  

