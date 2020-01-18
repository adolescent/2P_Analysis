% Tang Rendong 20170317
clc; clear all;
global CRS;
vsgInit;
deg2pix=33.6102;
% deg2pix=20.7263;
% deg2pix=crsUnitToUnit(CRS.DEGREEUNIT, 1, CRS.PIXELUNIT);
framestep=1;
crsIOWriteDAC(0,'Volts5'); 
    
% Stimulus Parameters
Scan_mode=2;  % 1 GA  2 RG
CenXY =round([0,0]*deg2pix); % Degrees of visual angle                         
stimsize=3;  % 3 or 2
stepsize=3; %  3 or 1   Spatial extent of stimulus

time_ISI = 0.4;                 % Seconds.
time_on = 1.5;                  % Seconds.
Ntrials=1;                  %repeat all stim 100 times; 
Nstim=25*3*4; % 3 shape 4 dir
grid=5; % how man flash in x/y axis
Direction=45:90:315;
load circle.mat;

% 一般下面不用改
SF=1; % cycle/degree
TF = 4 *framestep;
gratingtype =1;              %1.square wave 2.sine wave
dutycycle = 0.2;              % white line is thiner if dutycycle > 0.5
DotSize=2;            % RD element size, note: pix2deg=0.0208
DotDensity=0.2;      % RD density: the percentage of area covered by random dots.
Velocity=TF/SF;

% 定义坐标矩阵
stepsize=round(stepsize*deg2pix);
xloca=CenXY(1)-stepsize*floor(grid/2):stepsize:CenXY(1)+stepsize*floor(grid/2);
yloca=CenXY(2)-stepsize*floor(grid/2):stepsize:CenXY(2)+stepsize*floor(grid/2);

Size =[stimsize,stimsize];
cmSize =round(Size*deg2pix); % Spatial extent of stimulus

curFrameRate=crsGetSystemAttribute(CRS.FRAMERATE);  % framerate of current system
V_pixls=TF/SF*deg2pix/curFrameRate;  
totalframe=round(curFrameRate*time_on/framestep)+1;
Nsize=ceil(V_pixls*totalframe/cmSize(1))+1;

% Colour Definitions
bgl=0.12;  % background level=0.1
crsPaletteSetPixelLevel(1, [bgl, bgl, bgl]);
crsPaletteSetPixelLevel(20, [0, 0, 0]);
crsPaletteSetPixelLevel(200, [1, 1, 1]);
% crsPaletteSetPixelLevel(250, [1, 1, 1]);

viewdist=570;                 % in mm
crsSetViewDistMM(viewdist);
crsSetSpatialUnits(CRS.PIXELUNIT);     % all units in degree of angles

%draw prepare
crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
crsSetDrawPage(CRS.VIDEOPAGE, 2, 1);

% hostpages
hostpages=zeros(totalframe,1);
for n=1:totalframe
    hostpages(n) = crsPAGECreate(CRS.HOSTPAGE,cmSize, CRS.EIGHTBITPALETTEMODE)+1;    
    crsSetDrawPage(CRS.HOSTPAGE, hostpages(n));
end

hs=cmSize(1)/2;
x=1:cmSize(1);  y=x;
rfx=repmat(x,length(x),1); rfx=rfx'; rfy=rfx';
mask0=sqrt((rfx-hs).^2+(rfy-hs).^2);
mask0(mask0<=hs)=1; mask0(mask0>hs)=0;
maskb=zeros(cmSize)+0.098;

% create condition file name
N_time=fix(clock);
logname=strcat('RFloca25_',num2str(N_time(4)),'_',num2str(N_time(5)),'_',num2str(N_time(6)),'.txt');

fprintf('press any key to start...\n');
pause;

% stimuli loop
for j=1:Ntrials  % trials   
    stimindex=randperm(Nstim)';    % create and save condition file
%     stimindex=(1:Nstim)';    % create and save condition file
    fid=fopen(logname, 'a');
    fprintf(fid,'%2.0f\t',stimindex);  
    fclose(fid); 
 
for i=1:Nstim     % 
   stimID=stimindex(i);          
   [xid,yid,dirid,shapeid]=ind2sub([grid,grid,4,3],stimID);
   disp(['xid=',num2str(xid),'  yid=',num2str(yid)]);
   if shapeid==1                                
       stimprep_Grat;

   elseif  shapeid==2              
       stimprep_RD;

   elseif  shapeid==3            
       stimprep_circle;
   end
       
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

t1=tic;  
for n=1:totalframe-1
    crsSetDrawPage(CRS.VIDEOPAGE, 2,1);
    crsSetDrawMode(CRS.CENTREXY);
    %copy pre-generated image from host page to vedio page.
    crsDrawMoveRect(CRS.HOSTPAGE, hostpages(n),[0 0],cmSize,[xloca(xid),yloca(yid)],cmSize);                
    %display current page.
    crsSetDisplayPage(2);
%     crsSetDisplayPage(2);
end

crsIOWriteDAC(0,'Volts5');  
crsSetDisplayPage(1);
t_disp=toc(t1);
disp(['Present time= ',num2str(t_disp)]);

end
end
pause(2);


