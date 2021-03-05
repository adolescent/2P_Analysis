% stimprep_RD_rot2
t1=tic;  t2=clock; 
for n=1:totalframe-1
    crsPAGEDelete(hostpages(n));
end

%准备随机点
if stimid==1
    dir=-1;   % clock rot
elseif stimid==2
    dir=1;   % anti-clock rot
end
fprintf('Direction=%8.3f\n',dir);
        
c1 = hdMatrixRandomDot(sizem*2,sizem*2);
c1.DotSize = DotSize;
c1.DotDensity = DotDensity;
c1.Velocity = Velocity;
c1.Direction = 0;
c1.FrameCount = totalframe + 1;
c1.Prepare();
c1.GetNextFrame(1);

curFrame0=c1.GetNextFrame(1);
anglestep=TFw*360*framestep/curFrameRate;

hostpages=zeros(totalframe,1);
for n=1:totalframe
    if hostpages(n) == 0 
    hostpages(n) = crsPAGECreate(CRS.HOSTPAGE,[cmSizem,cmSizem], CRS.EIGHTBITPALETTEMODE) + 1;
    end
    crsSetDrawMode(CRS.CENTREXY);
    crsSetPen1(240);
    crsSetPen2(10);
    curFrame1=imrotate(curFrame0,(n-1)*anglestep*dir,'bilinear','crop');
    curFrame2=curFrame1(round(cmSizem/2)+1:round(cmSizem/2)+cmSizem,round(cmSizem/2)+1:round(cmSizem/2)+cmSizem);
    curFrame = curFrame2.*mask0+maskb;
    curFrame(curFrame>1)=1;
    crsDrawMatrix(curFrame);     
end    
t_prepare=toc(t1);
disp(t_prepare);
while (etime(clock, t2) < time_ISI); end   

 


