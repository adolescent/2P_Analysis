% stimprep_Shape3
t1=tic;  t2=clock; 
for n=1:totalframe-1
    crsPAGEDelete(hostpages(n));
end

fprintf('Direction=%8.3f\n',Direction(dirid));
Direction1=Direction *pi/180;   % 0-180, use 999 for random, orientation 0 is H gratings move up, 90 is Vertical gratings moving left
dir_r=Direction1(dirid);
        
stimuB=zeros(cmSize*(Nsize*2+1));
stimu=picture;
stimu1=imresize(stimu,cmSize(1)/size(stimu,1));
stimu1=stimu1.*mask0;
stimu1(stimu1>1)=1; stimu1(stimu1<0)=0;        
for id=1:Nsize
    xi=round((id-1)*cmSize(1)*sin(dir_r));
    yi=round((id-1)*cmSize(1)*cos(dir_r));
    stimuB((cmSize(1)*Nsize+1:cmSize(1)*(Nsize+1))+xi,(cmSize(1)*Nsize+1:cmSize(1)*(Nsize+1))-yi)=...
        stimuB((cmSize(1)*Nsize+1:cmSize(1)*(Nsize+1))+xi,(cmSize(1)*Nsize+1:cmSize(1)*(Nsize+1))-yi)+stimu1;
end

hostpages=zeros(totalframe,1);
for n=1:totalframe
    xi=round((n-1)*V_pixls*sin(dir_r));
    yi=round((n-1)*V_pixls*cos(dir_r));
    curFrame=stimuB((cmSize(1)*Nsize+1:cmSize(1)*(Nsize+1))+xi,(cmSize(1)*Nsize+1:cmSize(1)*(Nsize+1))-yi).*mask0+maskb;
    curFrame(curFrame>1)=1;
    
    if hostpages(n) == 0 
    hostpages(n) = crsPAGECreate(CRS.HOSTPAGE,cmSize, CRS.EIGHTBITPALETTEMODE) + 1;
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




