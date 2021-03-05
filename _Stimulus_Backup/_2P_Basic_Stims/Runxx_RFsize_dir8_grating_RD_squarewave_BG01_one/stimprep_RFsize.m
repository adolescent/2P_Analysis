% stimprep_RFsize
t1=tic;  t2=clock; 
for n=1:totalframe-1
    crsPAGEDelete(hostpages(n));
end

%准备刺激
[dirid,sizeid]=ind2sub([length(Direction),length(SizeC)],stimid); % 1-8 ori 1-8, location不变
fprintf('DirectionC=%8.3f\n',Direction(dirid));
disp(['SizeC=',num2str(SizeC(sizeid))]);

dir_c=Direction(dirid)*pi/180;
dir_s=(Direction(dirid)+DirS(sizeid))*pi/180;

% mask
hsc=cmSizeC(sizeid)/2;
hss=cmSizeS(sizeid)/2;
x=1:cmSizem;  y=x;
rfx=repmat(x,length(x),1); rfx=rfx'; rfy=rfx';
mask0=sqrt((rfx-cmSizem/2).^2+(rfy-cmSizem/2).^2);  
mask1=mask0; 
mask0(mask0<=hsc)=1; mask0(mask0>hsc)=0;
mask1(mask1<=hsc)=0; mask1(mask1>hss)=0; mask1(mask1>0)=1;
maskb=zeros(cmSizem,cmSizem)+bg2;

c1 = hdMatrixGrating(gratingtype,[SizeP/deg2pix*1.15,SizeP/deg2pix*1.15]);
c1.SpatialFrequency = SF;
c1.SpatialPhase = fix(rand(1)*360);
c1.Angle = Direction(dirid)-90;
c1.DriftVelocity=TF;
c1.DutyCycle=dutycycle;
c1.Prepare();
c1.GetFrame(1);
stimuB1=c1.GetFrame(1);

% 1-6 size tuning;  7 S-grat dir+180;  % 8 S-RD dir+0;   9 S-RD dir+180;
if sizeid==7
    c2 = hdMatrixGrating(gratingtype,[SizeP/deg2pix*1.15,SizeP/deg2pix*1.15]);
    c2.SpatialFrequency = SF;
    c2.SpatialPhase = fix(rand(1)*360);
    c2.Angle = Direction(dirid)-90+DirS(sizeid);
    c2.DriftVelocity=TF;
    c2.DutyCycle=dutycycle;
    c2.Prepare();
    c2.GetFrame(1);
    stimuB2=c2.GetFrame(1);
elseif sizeid>7 
    c2 = hdMatrixRandomDot(SizeP/deg2pix*1.15,SizeP/deg2pix*1.15);
    c2.DotSize = DotSize;
    c2.DotDensity = DotDensity;
    c2.Velocity = Velocity;
    c2.Direction = Direction(dirid)+DirS(sizeid);
    c2.FrameCount = totalframe + 1;
    c2.Prepare();
    c2.GetNextFrame(1);
    stimuB2=c2.GetNextFrame(1);
end

x_st=round(SizeP/2)-cmSizem/2;%+round(rand(1)*shapesize);
y_st=round(SizeP/2)-cmSizem/2;%+round(rand(1)*shapesize);

hostpages=zeros(totalframe,1);
for n=1:totalframe
    x0 = x_st+round((n-1)*Vpixl*sin(dir_c));
    y0 = y_st-round((n-1)*Vpixl*cos(dir_c));
    x1 = x0+cmSizem-1;
    y1 = y0+cmSizem-1;

    x2 = x_st+round((n-1)*Vpixl*sin(dir_s));
    y2 = y_st-round((n-1)*Vpixl*cos(dir_s));
    x3 = x2+cmSizem-1;
    y3 = y2+cmSizem-1;
    
     if sizeid<7
         curFrame1=stimuB1(x0:x1,y0:y1);         
         curFrame = curFrame1.*mask0 +maskb;
     else
         curFrame1=stimuB1(x0:x1,y0:y1);
         curFrame2=stimuB2(x2:x3,y2:y3);
         curFrame = curFrame1.*mask0 + curFrame2.*mask1 +maskb;
     end     
    curFrame(curFrame>1)=1; 
    
    if hostpages(n) == 0 
        hostpages(n) = crsPAGECreate(CRS.HOSTPAGE,[cmSizem,cmSizem], CRS.EIGHTBITPALETTEMODE) + 1;
    end
    crsSetDrawMode(CRS.CENTREXY);
    crsSetPen1(240);
    crsSetPen2(10);
    crsDrawMatrix(curFrame);     
end    

t_prepare=toc(t1);
disp(t_prepare);
while (etime(clock, t2) < time_ISI); end  


