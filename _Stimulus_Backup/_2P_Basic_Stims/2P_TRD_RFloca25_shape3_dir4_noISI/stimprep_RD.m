% stimprep_RD
t1=tic;  t2=clock; 
for n=1:totalframe-1
    crsPAGEDelete(hostpages(n));
end

%准备随机点
fprintf('Direction=%8.3f\n',Direction(dirid));

c1 = hdMatrixRandomDot(Size(1),Size(2));
c1.DotSize = DotSize;
c1.DotDensity = DotDensity;
c1.Velocity = Velocity;
c1.Direction = Direction(dirid);
c1.FrameCount = totalframe + 1;
c1.Prepare();
c1.GetNextFrame(1);

hostpages=zeros(totalframe,1);
for n=1:totalframe
    if hostpages(n) == 0 
    hostpages(n) = crsPAGECreate(CRS.HOSTPAGE,cmSize, CRS.EIGHTBITPALETTEMODE) + 1;
    end
    crsSetDrawMode(CRS.CENTREXY);
    crsSetPen1(240);
    crsSetPen2(10);
    curFrame = c1.GetNextFrame(1); curFrame=curFrame(1:cmSize(1),1:cmSize(1));
    curFrame = curFrame.*mask0+maskb;
    curFrame(curFrame>1)=1;
    crsDrawMatrix(curFrame);     
end    
t_prepare=toc(t1);
disp(t_prepare);
while (etime(clock, t2) < time_ISI); end   

 


