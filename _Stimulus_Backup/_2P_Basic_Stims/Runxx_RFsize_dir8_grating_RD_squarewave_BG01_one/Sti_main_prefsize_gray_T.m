% Tang Rendong 20170317
clc; clear all;
global CRS;
vsgInit;
crsIOWriteDAC(0,'Volts5'); 
deg2pix=32;    % 1280/40deg
% deg2pix=20.7263;
% deg2pix=crsUnitToUnit(CRS.DEGREEUNIT, 1, CRS.PIXELUNIT);
framestep=2;

% Stimulus Parameters
Scan_mode=2;                    % 1 GA  2 RG
CenXY = round([-1,-3] *deg2pix);         
sizep=2;                        % 0.5 * prefsize of RF
SF = 1.5;                         % Cycles per degree of visual angle
TF = 8;
Direction=0:45:315;             % 0:45:315, orientation 0 is H gratings move up, 90 is Vertical gratings moving left
sizemode=1;                     % 1: 6 sizes   2: 6+3 sur

Ntrials=7;                      
sizem=sizep*6;
if sizemode==1
    Nstim=6*length(Direction);   
    SizeC = [0.5 1 2 3 4 6];
    SizeS = [0.5 1 2 3 4 6];
    DirS=[0,0,0,0,0,0];
elseif sizemode==2
    Nstim=9*length(Direction);   
    SizeC = [sizep/4,sizep/2,sizep,sizep*2,sizep*4,sizep*6,sizep*2,sizep*2,sizep*2];
    SizeS = [sizep/4,sizep/2,sizep,sizep*2,sizep*4,sizep*6,sizep*6,sizep*6,sizep*6];
    DirS=[0,0,0,0,0,0,180,0,180];
end

gratingtype =1;                  %1.square wave 0.sine wave
dutycycle = 0.2;                 % white line is thiner if dutycycle > 0.5
DotSize=round(deg2pix*0.25);     % RD element size;  pixel_num * pixel_num
DotDensity=0.2;                  % RD density: the percentage of area covered by random dots.
Velocity=TF/SF;
curFrameRate= crsGetSystemAttribute(CRS.FRAMERATE);  % framerate of current system
Vpixl=TF/SF *deg2pix/curFrameRate;

time_ISI = 3;                    % Seconds.
time_on = 2;                     % Seconds.
cmSizep = round(sizep*deg2pix);  % Spatial extent of stimulus
cmSizem = round(sizem*deg2pix);  % 4_+8; % Spatial extent of stimulus
cmSizeC = round(SizeC*deg2pix);  % Spatial extent of stimulus
cmSizeS = round(SizeS*deg2pix);  % Spatial extent of stimulus
totalframe=round(curFrameRate*time_on/framestep)+1;

SizeP=sizep*6+Velocity/framestep*time_on*2+1;
SizeP=ceil(SizeP*deg2pix);

% Colour Definitions
bg1=0.12;  % background level=0.1
bg2=0.098;
crsPaletteSetPixelLevel(1, [bg1, bg1, bg1]);
crsPaletteSetPixelLevel(10, [0, 0, 0]);
crsPaletteSetPixelLevel(240, [1, 1, 1]);

% 空间单位定义
viewdist=570;               
crsSetViewDistMM(viewdist);
crsSetSpatialUnits(CRS.PIXELUNIT);     % all units in degree of angles
%draw prepare
crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
crsSetDrawPage(CRS.VIDEOPAGE, 2, 1);

% hostpages
hostpages=zeros(totalframe,1);
for n=1:totalframe
    hostpages(n) = crsPAGECreate(CRS.HOSTPAGE,[cmSizem,cmSizem], CRS.EIGHTBITPALETTEMODE)+1;    
    crsSetDrawPage(CRS.HOSTPAGE, hostpages(n),1);
end

% create condition file name
N_time=fix(clock);
logname=strcat('Sti main1_gray_',num2str(N_time(4)),'_',num2str(N_time(5)),'_',num2str(N_time(6)),'.txt');

fprintf('press any key to start...\n');
pause;

% stimuli loop
for j=1:Ntrials  % trials
    stimindex=randperm(Nstim)';    % create and save condition file
%     stimindex=(1:Nstim)';        % create and save condition file
    
    fid=fopen(logname, 'a');
    fprintf(fid,'%2.0f\t',stimindex);  
    fclose(fid); 
    
    for i=1:Nstim        
        disp(['Ntrials= ', num2str(j)]);
        disp(['CurrentID= ', num2str(stimindex(i))]);
        stimID=stimindex(i);          

        stimid=stimID;       stimprep_RFsize;

    % stimuli present
    if Scan_mode==1;                % GA才需对齐
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
        crsSetDisplayPage(2); 
    end

    crsIOWriteDAC(0,'Volts5');  
    disp(etime(clock,start_step));
    crsSetDisplayPage(1);              % Display background screen
    end
end
pause(time_ISI);  

