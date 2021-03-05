% stimprep_Shape3
t1=tic;  t2=clock; 
for n=1:totalframe-1
    crsPAGEDelete(hostpages(n));
end

Direction1=Directions *pi/180;   % 0-180, use 999 for random, orientation 0 is H gratings move up, 90 is Vertical gratings moving left

dirid=stimid; % 1-8 ori 1-8, location²»±ä
fprintf('Direction=%8.3f\n',Directions(dirid));
dir_r=Direction1(dirid);
        
stimuB=zeros(cmSizep*(Nsize*2+1),cmSizep*(Nsize*2+1));
stimu=stim_circle_bright;
for id=1:Nsize
    xi=round((id-1)*cmSizep*sin(dir_r));
    yi=round((id-1)*cmSizep*cos(dir_r));
    stimuB((cmSizep*Nsize+1:cmSizep*(Nsize+1))+xi,(cmSizep*Nsize+1:cmSizep*(Nsize+1))-yi)=...
        stimuB((cmSizep*Nsize+1:cmSizep*(Nsize+1))+xi,(cmSizep*Nsize+1:cmSizep*(Nsize+1))-yi)+stimu;
end

hostpages=zeros(totalframe,1);
for n=1:totalframe
    xi=round((n-1)*Vpixl*sin(dir_r)-cmSizem/2+cmSizep/2);
    yi=round((n-1)*Vpixl*cos(dir_r)+cmSizem/2-cmSizep/2);
    curFrame=stimuB((cmSizep*Nsize+1:cmSizep*Nsize+cmSizem)+xi,(cmSizep*Nsize+1:cmSizep*Nsize+cmSizem)-yi).*mask1+maskb;
    curFrame(curFrame>0.5)=0;
    
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




