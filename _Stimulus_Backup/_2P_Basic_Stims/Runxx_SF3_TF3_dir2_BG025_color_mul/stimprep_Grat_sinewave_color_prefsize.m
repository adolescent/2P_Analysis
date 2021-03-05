% stimprep_Plaid8
t1=tic;  t2=clock; 
for n=1:totalframe-1
    crsPAGEDelete(hostpages(n));
end

%×¼±¸´Ì¼¤
fprintf('SF=%8.3f\n',SF(sfid));
fprintf('TF=%8.3f\n',TF(tfid)/framestep);
fprintf('Dir=%8.3f\n',Direction(dirid));

c1 = hdMatrixGrating(gratingtype,[sizem*1.15,sizem*1.15]);
c1.SpatialFrequency = SF(sfid);
c1.SpatialPhase = fix(rand(1)*360);
c1.Angle = Direction(dirid)-90;
c1.DriftVelocity=TF(tfid);
c1.DutyCycle=dutycycle;
c1.Prepare();
c1.GetFrame(1);

hostpages=zeros(totalframe,1);
for n=1:totalframe
    curFrame0 = c1.GetFrame(n);
    curFrame0=curFrame0(1:cmSizem,1:cmSizem);
    curFrame1 = (curFrame0*bg*2).*mask0+maskb;
    curFrame=cat(3,curFrame1,curFrame1,curFrame1);
%     curFrame(curFrame>1)=1;

    if hostpages(n) == 0 
        hostpages(n) = crsPAGECreate(CRS.HOSTPAGE,[cmSizem,cmSizem], CRS.TRUECOLOURMODE) + 1;
    end
%     crsSetDrawPage(CRS.HOSTPAGE, hostpages(n),1);
    crsSetDrawMode(CRS.CENTREXY);
    crsDrawMatrix24bitColour(curFrame);     
end    

t_prepare=toc(t1);
disp(t_prepare);
while (etime(clock, t2) < time_ISI); end       

        


