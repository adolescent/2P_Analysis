% stimprep_Plaid8
t1=tic;  t2=clock; 
for n=1:totalframe-1
    crsPAGEDelete(hostpages(n));
end

%准备刺激
oriid=stimid; % 1-8 ori 1-8, location不变
fprintf('Direction=%8.3f\n',Orientation1(oriid)+90);

c1 = hdMatrixGrating(gratingtype,[sizem*zoomrate,sizem*zoomrate]);
c1.SpatialFrequency = SF;
c1.SpatialPhase = fix(rand(1)*360);
c1.Angle = Orientation(oriid);
c1.DriftVelocity=TF;
c1.DutyCycle=dutycycle;
c1.Prepare();
c1.GetFrame(1);

hostpages=zeros(totalframe,1);
for n=1:totalframe
    if hostpages(n) == 0 
        hostpages(n) = crsPAGECreate(CRS.HOSTPAGE,[cmSizem,cmSizem], CRS.EIGHTBITPALETTEMODE) + 1;
    end
    crsSetDrawMode(CRS.CENTREXY);
    crsSetPen1(240);
    crsSetPen2(10);
    curFrame0 = c1.GetFrame(n);
    curFrame=curFrame0(1:cmSizem,1:cmSizem);
    curFrame = curFrame.*mask0+maskb;
    curFrame(curFrame>1)=1;
    crsDrawMatrix(curFrame);     
end    
t_prepare=toc(t1);
disp(t_prepare);
while (etime(clock, t2) < time_ISI); end       

        


