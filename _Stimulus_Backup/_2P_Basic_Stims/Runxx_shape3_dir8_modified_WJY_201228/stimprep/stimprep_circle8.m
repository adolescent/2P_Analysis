% stimprep_Shape3
t1=tic;  t2=clock; 
for n=1:totalframe-1
    crsPAGEDelete(hostpages(n));
end

Direction_r=Direction *pi/180;   % 0-180, use 999 for random, orientation 0 is H gratings move up, 90 is Vertical gratings moving left

dirid=stimid; % 1-8 ori 1-8, location²»±ä
fprintf('Direction=%8.3f\n',Direction(dirid));
dir_r=Direction_r(dirid);

stimu=stim_circle_bright;        
stimuB=zeros(cmSizem*(Nsize*2+1),cmSizem*(Nsize*2+1));
for id=1:Nsize
    xi=round((id-0.8)*cmSizem*sin(dir_r));
    yi=round((id-0.8)*cmSizem*cos(dir_r));
    stimuB((cmSizem*Nsize+1:cmSizem*Nsize+cmSizep)+xi,(cmSizem*Nsize+1:cmSizem*Nsize+cmSizep)-yi)=...
        stimuB((cmSizem*Nsize+1:cmSizem*Nsize+cmSizep)+xi,(cmSizem*Nsize+1:cmSizem*Nsize+cmSizep)-yi)+stimu;
end

hostpages=zeros(totalframe,1);
for n=1:totalframe
    xi=round((n-1)*Vpixl*sin(dir_r)-cmSizem/2+cmSizep/2);
    yi=round((n-1)*Vpixl*cos(dir_r)+cmSizem/2-cmSizep/2);
    curFrame=stimuB((cmSizem*Nsize+1:cmSizem*Nsize+cmSizem)+xi,(cmSizem*Nsize+1:cmSizem*Nsize+cmSizem)-yi).*mask1+maskb;
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




