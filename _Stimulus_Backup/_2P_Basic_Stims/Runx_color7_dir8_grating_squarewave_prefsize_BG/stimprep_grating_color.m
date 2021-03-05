% stimprep_color7
t1=tic;  t2=clock; 
for n=1:totalframe-1
    crsPAGEDelete(hostpages(n));
end

%准备刺激
[dirid,colorid]=ind2sub([length(Direction),size(color,1)],stimid); % 1-8 ori 1-8, location不变
disp(['Direction= ',num2str(Direction(dirid))]);        
disp(['Color_id= ',num2str(colorid)]);

dir_r=Direction(dirid)/180*pi;

c = hdMatrixGrating(gratingtype,[SizeP*1.15,SizeP*1.15]);
c.SpatialFrequency = SF;
c.SpatialPhase = fix(rand(1)*360);
c.Angle = Orientation(dirid);
c.DriftVelocity=TF;
c.DutyCycle=dutycycle;
c.Prepare();
c.GetFrame(1);
stimuB=c.GetFrame(1);

sizeB=size(stimuB,1);
x_st=round((sizeB-cmSizem)/2);
y_st=round((sizeB-cmSizem)/2);

hostpages=zeros(totalframe,1);
for n=1:totalframe
    x0 = x_st+round((n-1)*Vpixl*sin(dir_r));
    y0 = y_st-round((n-1)*Vpixl*cos(dir_r));
    x1 = x0+cmSizem-1;
    y1 = y0+cmSizem-1;
    
    curFrame = stimuB(x0:x1,y0:y1);
    curFrame=curFrame.*mask0;    
    curFrameb=(1-curFrame)*bg;
    
    stimu(:,:,1)=curFrame.*color(colorid,1)+curFrameb;
    stimu(:,:,2)=curFrame.*color(colorid,2)+curFrameb;
    stimu(:,:,3)=curFrame.*color(colorid,3)+curFrameb; 
    
    if hostpages(n) == 0 
        hostpages(n) = crsPAGECreate(CRS.HOSTPAGE,[cmSizem,cmSizem], CRS.TRUECOLOURMODE) + 1;
    end
    crsSetDrawPage(CRS.HOSTPAGE, hostpages(n));
    crsSetDrawMode(CRS.CENTREXY);
    crsDrawMatrix24bitColour(stimu); 
end    
t_prepare=toc(t1);
disp(t_prepare);
while (etime(clock, t2) < time_ISI); end   

  


