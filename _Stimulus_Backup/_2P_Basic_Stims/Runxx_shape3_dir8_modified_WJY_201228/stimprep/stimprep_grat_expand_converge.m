% stimprep_
t1=tic;  t2=clock; 
for n=1:totalframe-1
    crsPAGEDelete(hostpages(n));
end

if stimid==1
    dir=-1;   % expand
elseif stimid==2
    dir=1;   % converge
end
fprintf('Direction=%8.3f\n',dir);

phasestep=TF*pi*framestep/curFrameRate/2;
x=1:cmSizem;  y=x;
rfx=repmat(x,length(x),1); rfx=rfx'; rfy=rfx';
curFrame0=sqrt((rfx-cmSizem/2).^2+(rfy-cmSizem/2).^2);  
curFrame0=curFrame0/deg2pix;

hostpages=zeros(totalframe,1);
for n=1:totalframe
    curFrame1=sin(SF*2*pi*curFrame0+dir*n*phasestep);
    if gratingtype ==1
        curFrame1(curFrame1<sin(pi*(0.5-dutycycle)))=0; curFrame1(curFrame1>=sin(pi*(0.5-dutycycle)))=1;
    end
    curFrame = curFrame1.*mask0+maskb;
    curFrame(curFrame>1)=1;
    
    if hostpages(n) == 0 
        hostpages(n) = crsPAGECreate(CRS.HOSTPAGE,[cmSizem,cmSizem], CRS.EIGHTBITPALETTEMODE) + 1;
    end
    crsSetDrawPage(CRS.HOSTPAGE, hostpages(n), 1);
    crsSetDrawMode(CRS.CENTREXY);
    crsSetPen1(240);
    crsSetPen2(10);
    crsDrawMatrix(curFrame);     
end    
t_prepare=toc(t1);
disp(t_prepare);
while (etime(clock, t2) < time_ISI); end   

        


