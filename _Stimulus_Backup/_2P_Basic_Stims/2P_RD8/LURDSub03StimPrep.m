% HDGSub03PagePrep.m
% Prepare Stimulus: draw a big map, generate a sequence of coordinates for cuting and pasting
% only need specify: newstim (stim ID), testrun, which indicates following specifications:
%   fixcolor, rddensity1, esize1, velocity1, direction1, wincenter1, winsize1, whitecolor1, blackcolor1, elength1
%             rddensity2, esize2, velocity2, direction2, wincenter2, winsize2, whitecolor2, blackcolor2, elength2

% Give values to parameters that is changing for each trial (random)
if direction1(newstim)==999
    Cdirection1=double(round(rand(1))*2-1); % value 1 or -1
else
    Cdirection1=direction1(newstim);
end
if direction2(newstim)==999
    Cdirection2=double(round(rand(1))*2-1); 
else
    Cdirection2=direction2(newstim);
end

%________________________________________________ Prepare a new background
RDBK=zeros(1024, 1280);
if flagBiMask(newstim)==1
    RDBK=double(rand(round([1024, 1280]/esize1(newstim)))<rddensity1(newstim));
elseif flagBiMask(newstim)==0
    RDBK=double(rand(round([1024, 1280]/esize1(newstim)))<(rddensity1(newstim)) | rand(round([1024, 1280]/esize1(newstim)))<(rddensity1(newstim)));
end
if esize1(newstim)~=1
    RDBK=imresize(RDBK, esize1(newstim));
    if filtering==1
        RDBK = conv2(RDBK, fspecial('gaussian', round(esize1(newstim)/2), floor(round(esize1(newstim)/2))), 'same');
        RDBK=double(RDBK>0.5);    
    end
end
if elength1(newstim)~=esize1(newstim)
    switch Cdirection1
        case {1, 5}
        case {2, 6}            
        case {3, 7}
        case {4, 8}            
    end
end
crsPaletteSetPixelLevel(6, whitecolor1(newstim,:));        % RD dot color changes each time
crsPaletteSetPixelLevel(7, blackcolor1(newstim,:));        

%________________________________________________ Prepare page 3 and 5 first, and use it's initial RD patch for other pages
%% map1
t1=clock;

map1 =  GenerateRandomDotsZeroToOneMap(mapsize, esize1(newstim), rddensity1(newstim), elength1(newstim) , Cdirection1, filtering);

% coordinates
xcenter1=round(mapsize/2);
ycenter1=round(mapsize/2);
halfdist1=round(velocity1pix(newstim)*Tcue/2);        % half of the travel distance
angle1=(Cdirection1-1)*45*pi/180;         % dir 1 is move toward right, dir2 is move toward top right. 
xstart1=xcenter1-round(halfdist1*cos(angle1));
xstop1 =xcenter1+round(halfdist1*cos(angle1));
ystart1=ycenter1+round(halfdist1*sin(angle1));  % note since Y axis in matlab is reversed, reversed to make direction change counder-clock wise
ystop1 =ycenter1-round(halfdist1*sin(angle1));
xx1=round(linspace(xstop1, xstart1, totalframe)); % Note: "window move direction" and "RD movd direction" are reversed
yy1=round(linspace(ystop1, ystart1, totalframe));
halfwinwidthpix1=round(winsize1pix(2)/2);
halfwinheightpix1=round(winsize1pix(1)/2);    
winleft1=round(xx1-halfwinwidthpix1);
winright1=round(xx1+halfwinwidthpix1);
wintop1=round(yy1-halfwinheightpix1);
winbottom1=round(yy1+halfwinheightpix1);

fprintf('map1 cost %3.2f sec... ', etime(clock, t1));

%% map2
t2=clock;

map2 = GenerateRandomDotsZeroToOneMap(mapsize, esize2(newstim), rddensity2(newstim), elength2(newstim),  Cdirection2, filtering);

% coordinates
xcenter2=round(mapsize/2);
ycenter2=round(mapsize/2);
halfdist2=round(velocity2pix(newstim)*Tcue/2);    
angle2=(Cdirection2-1)*45*pi/180;         
xstart2=xcenter2-round(halfdist2*cos(angle2));
xstop2 =xcenter2+round(halfdist2*cos(angle2));
ystart2=ycenter2+round(halfdist2*sin(angle2)); 
ystop2 =ycenter2-round(halfdist2*sin(angle2));
xx2=round(linspace(xstop2, xstart2, totalframe)); 
yy2=round(linspace(ystop2, ystart2, totalframe));
halfwinwidthpix2=round(winsize2pix(2)/2);
halfwinheightpix2=round(winsize2pix(1)/2);    
winleft2=round(xx2-halfwinwidthpix2);
winright2=round(xx2+halfwinwidthpix2);
wintop2=round(yy2-halfwinheightpix2);
winbottom2=round(yy2+halfwinheightpix2);

xOrigin=startxy(1,newstim)+ round(rand(1)* 2 * periodxy(1, newstim));
yOrigin=startxy(2,newstim)+ round(rand(1)* 2 * periodxy(2, newstim));
x0 = xOrigin;
y0 = yOrigin;
% if flagmask(newstim)==1
%     tmpmap1 = map1(wintop1(1):winbottom1(1), winleft1(1):winright1(1));
%     tmpmasks1 = masks(y0:y0+winheightpix1, x0:x0+winwidthpix1, newstim);
%     tmpmap2 = map2(wintop2(1):winbottom2(1), winleft2(1):winright2(1));
%     tmpmask2 = masksn(y0:y0+winheightpix1, x0:x0+winwidthpix1, newstim);
%     centerpatch12=tmpmap1.*tmpmasks1 + tmpmap2.*tmpmask2;       % this method can keep frame rate 100hz, but can't deal with graylevel dots
% elseif flagmask(newstim)==0 % for all 1 masks, adding two maps will cause pixel value > 1
%     centerpatch12=map1(wintop1(1):winbottom1(1), winleft1(1):winright1(1))| map2(wintop2(1):winbottom2(1), winleft2(1):winright2(1));       % this method can keep frame rate 100hz, but can't deal with graylevel dots
% end

fprintf(' map2 cost %3.2f sec... ', etime(clock, t2));

%% Prepare hostpages
t3 = clock;

crsSetPen1(6);
crsSetPen2(7);

firstMap = zeros(winsize1pix);

askForMask = flagBiMask(newstim) ~= 0;
curFlagMask = askForMask;   %&& (~(velocity1pix(newstim) == velocity2pix(newstim)&& direction1(newstim) == direction2(newstim))); %%%  Determind when need the mask, just FlagBimask=1 or both FlagBimask=1 and the opposite of direction & velocity are the same.
askForMaskLine = flagMaskLine(newstim) ~= 0;

periodx2 = 2 * periodxy(1, newstim);
periody2 = 2 * periodxy(2, newstim);

%draw map1 matrix to host page
if mapPages(1) == 0
  mapPages(1) = crsPAGECreate(CRS.HOSTPAGE, [mapsize mapsize], CRS.EIGHTBITPALETTEMODE) + 1;
end
crsSetDrawPage(CRS.HOSTPAGE, mapPages(1), 1); 
crsSetDrawOrigin(0,0);
crsSetDrawMode(0);
crsSetPen1(6);
crsSetPen2(7);
crsDrawMatrix(map1);

%draw map2 matrix to host page
if mapPages(2) == 0
  mapPages(2) = crsPAGECreate(CRS.HOSTPAGE, [mapsize mapsize], CRS.EIGHTBITPALETTEMODE) + 1;
end
crsSetDrawPage(CRS.HOSTPAGE, mapPages(2), 1);
crsSetDrawOrigin(0,0);
crsSetDrawMode(0);
crsSetPen1(6);
crsSetPen2(7);
crsDrawMatrix(map2);

if prePage == 0
  prePage = crsPAGECreate(CRS.HOSTPAGE, winsize1pix_width_height, CRS.EIGHTBITPALETTEMODE) + 1;
end;

debug=0;

%It's easy to calculate location and copy range in pixel mode.
crsSetSpatialUnits(CRS.PIXELUNIT);     % all units in pixel

%prepare pages
for n = 1:totalframe
    if hostpages(n) == 0 %allocate host pages used to store final images
        hostpages(n) = crsPAGECreate(CRS.HOSTPAGE, winsize1pix_width_height, CRS.EIGHTBITPALETTEMODE) + 1;
    end
    
    if(randomizeFrameCount > 0) %if ask for randomize map1 and map2
        if(mod(n, randomizeFrame) == 0)
            map1=double(rand(ceil(mapsize/esize1(newstim)))<rddensity1(newstim));
            if esize1(newstim)~=1
                map1=imresize(map1, esize1(newstim), 'nearest');
                if filtering==1
                    map1=conv2(map1, fspecial('gaussian', round(esize1(newstim)/2), floor(round(esize1(newstim)/2))), 'same');
                    map1=double(map1>0.5);
                end
            end

            crsSetDrawPage(CRS.HOSTPAGE, mapPages(1), 1); 
            crsSetDrawOrigin(0,0);
            crsSetDrawMode(0);
            crsSetPen1(6);
            crsSetPen2(7);
            crsDrawMatrix(map1);

            map2=double(rand(ceil(mapsize/esize2(newstim)))<rddensity2(newstim));
            if esize2(newstim)~=1
                map2=imresize(map2, esize2(newstim),'nearest');
                if filtering==1
                    map2=conv2(map2, fspecial('gaussian', round(esize2(newstim)/2), floor(round(esize2(newstim)/2))), 'same');
                    map2=double(map2>0.5);
                end
            end

            crsSetDrawPage(CRS.HOSTPAGE, mapPages(2), 1);
            crsSetDrawOrigin(0,0);
            crsSetDrawMode(0);
            crsSetPen1(6);
            crsSetPen2(7);
            crsDrawMatrix(map2);
        end
    end;
    
    if(askForMask || askForMaskLine)
        curVelocity = round(n * lineVelocityPixel(newstim));
        sqrtCurVelocity = round(sqrt(2) * n * lineVelocityPixel(newstim));
        switch lineDirection(newstim) 
            case 0
            case 1 %right
                if(0 ~= periodx2)
                    x0 = xOrigin - round(curVelocity);
                    while(x0 <= 0)
                        x0 = x0 + periodx2;
                    end
                end
            case 2 %up right
                if(0 ~= periodx2)
                    x0 = xOrigin - round(sqrtCurVelocity);
                    while(x0 <= 0)
                        x0 = x0 + periodx2;
                    end
                else
                    if(0 ~= periody2)
                        y0 = yOrigin + round(sqrtCurVelocity);
                        while(y0 > periody2)
                            y0 = y0 - periody2;
                        end
                    end
                end
            case 3 %up
                if(0 ~= periody2)
                    y0 = yOrigin + round(curVelocity);
                    while(y0 > periody2)
                        y0 = y0 - periody2;
                    end
                end
            case 4 %up left
                if(0 ~= periodx2)
                    x0 = xOrigin + round(sqrtCurVelocity);
                    while(x0 > periodx2)
                        x0 = x0 - periodx2;
                    end
                else
                    if(0 ~= periody2)
                        y0 = yOrigin + round(sqrtCurVelocity);
                        while(y0 > periody2)
                            y0 = y0 - periody2;
                        end
                    end
                end
            case 5 %left
                if(0 ~= periodx2)
                    x0 = xOrigin + round(curVelocity);
                    while(x0 > periodx2)
                        x0 = x0 - periodx2;
                    end
                end
            case 6 %left down
                if(0 ~= periodx2)
                    x0 = xOrigin + round(sqrtCurVelocity);
                    while(x0 > periodx2)
                        x0 = x0 - periodx2;
                    end
                else
                    if(0 ~= periody2)
                        y0 = yOrigin - round(sqrtCurVelocity);
                        while(y0 <= 0)
                            y0 = y0 + periody2;
                        end
                    end
                end
            case 7 %down
                if(0 ~= periody2)
                    y0 = yOrigin - round(curVelocity);
                    while(y0 <= 0)
                        y0 = y0 + periody2;
                    end
                end
            case 8 %right down
                if(0 ~= periodx2)
                    x0 = xOrigin - round(sqrtCurVelocity);
                    while(x0 <= 0)
                        x0 = x0 + periodx2;
                    end
                else
                    if(0 ~= periody2)
                        y0 = yOrigin - round(sqrtCurVelocity);
                        while(y0 <= 0)
                            y0 = y0 + periody2;
                        end
                    end
                end
        end;
    end;
    
    if askForMask
        if curFlagMask
            crsSetDrawPage(CRS.HOSTPAGE, hostpages(n), 1);
            crsSetDrawOrigin(0,0);
            crsSetDrawMode(0);
            crsDrawMoveRect(CRS.HOSTPAGE, maskPages(1, newstim),[x0 y0],winsize1pix_width_height,[0 0],winsize1pix_width_height);

            if(debug)
                crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
                crsSetDrawOrigin(0,0);
                crsSetDrawMode(0);
                crsDrawMoveRect(CRS.HOSTPAGE, hostpages(n),[0 0],winsize1pix_width_height,[0 0],winsize1pix_width_height);
                crsSetDisplayPage(1);
                fprintf('Prepare : mask... press any key to continue.\r');   
                pause;
            end;

            crsSetDrawPage(CRS.HOSTPAGE, hostpages(n), CRS.NOCLEAR);
            crsSetDrawMode(CRS.TRANSONHIGHER);
            crsDrawMoveRect(CRS.HOSTPAGE, mapPages(1),[winleft1(n) wintop1(n)],winsize1pix_width_height,[0 0],winsize1pix_width_height);

            if(debug)
                crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
                crsSetDrawOrigin(0,0);
                crsSetDrawMode(0);
                crsDrawMoveRect(CRS.HOSTPAGE, hostpages(n),[0 0],winsize1pix_width_height,[0 0],winsize1pix_width_height);
                crsSetDisplayPage(1);
                fprintf('Prepare : mask+map1... press any key to continue.\r');   
                pause;
            end;

            crsSetDrawPage(CRS.HOSTPAGE, prePage, 1);
            crsSetDrawOrigin(0,0);
            crsSetDrawMode(0);
            crsDrawMoveRect(CRS.HOSTPAGE, maskPages(2, newstim),[x0 y0],winsize1pix_width_height,[0 0],winsize1pix_width_height);

            if(debug)
                crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
                crsSetDrawOrigin(0,0);
                crsSetDrawMode(0);
                crsDrawMoveRect(CRS.HOSTPAGE, prePage,[0 0],winsize1pix_width_height,[0 0],winsize1pix_width_height);
                crsSetDisplayPage(1);
                fprintf('Prepare : maskn... press any key to continue.\r');   
                pause;
            end;

            crsSetDrawPage(CRS.HOSTPAGE, prePage, CRS.NOCLEAR);
            crsSetDrawMode(CRS.TRANSONHIGHER);
            crsDrawMoveRect(CRS.HOSTPAGE, mapPages(2),[winleft2(n) wintop2(n)],winsize1pix_width_height,[0 0],winsize1pix_width_height);

            if(debug)
                crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
                crsSetDrawOrigin(0,0);
                crsSetDrawMode(0);
                crsDrawMoveRect(CRS.HOSTPAGE, prePage,[0 0],winsize1pix_width_height,[0 0],winsize1pix_width_height);
                crsSetDisplayPage(1);
                fprintf('Prepare : maskn+map2... press any key to continue.\r');   
                pause;
            end;

            crsSetDrawPage(CRS.HOSTPAGE, hostpages(n), CRS.NOCLEAR);
            crsSetDrawMode(CRS.TRANSONLOWER);
            crsDrawMoveRect(CRS.HOSTPAGE, prePage,[0 0],winsize1pix_width_height,[0 0],winsize1pix_width_height);

            if(debug)
                crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
                crsSetDrawOrigin(0,0);
                crsSetDrawMode(0);
                crsDrawMoveRect(CRS.HOSTPAGE, hostpages(n),[0 0],winsize1pix_width_height,[0 0],winsize1pix_width_height);
                crsSetDisplayPage(1);
                fprintf('Prepare : mask+map1+maskn+map2... press any key to continue.\r');   
                pause;
            end;
        else %just display map1
            crsSetDrawPage(CRS.HOSTPAGE, hostpages(n), 1);
            crsSetDrawOrigin(0,0);
            crsSetPen1(6);
            crsSetPen2(7);
            crsSetDrawMode(0);
            crsDrawMoveRect(CRS.HOSTPAGE, mapPages(1),[winleft1(n) wintop1(n)],winsize1pix_width_height,[0 0],winsize1pix_width_height);
        end;
    else % for all 1 masks
        crsSetDrawPage(CRS.HOSTPAGE, hostpages(n), 1);
        crsSetDrawOrigin(0,0);
        crsSetPen1(6);
        crsSetPen2(7);
        crsSetDrawMode(0);
        crsDrawMoveRect(CRS.HOSTPAGE, mapPages(1),[winleft1(n) wintop1(n)],winsize1pix_width_height,[0 0],winsize1pix_width_height);
        crsSetDrawMode(CRS.TRANSONHIGHER);
        crsDrawMoveRect(CRS.HOSTPAGE, mapPages(2),[winleft2(n) wintop2(n)],winsize1pix_width_height,[0 0],winsize1pix_width_height);
    end
    
    if askForMaskLine
        crsSetDrawPage(CRS.HOSTPAGE, hostpages(n), CRS.NOCLEAR);
        crsSetDrawMode(CRS.TRANSONHIGHER);
        crsDrawMoveRect(CRS.HOSTPAGE, maskPages(3, newstim),[x0 y0],winsize1pix_width_height,[0 0],winsize1pix_width_height);
        
        if(debug)
            crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
            crsSetDrawOrigin(0,0);
            crsSetDrawMode(0);
            crsDrawMoveRect(CRS.HOSTPAGE, hostpages(n),[0 0],winsize1pix_width_height,[0 0],winsize1pix_width_height);
            crsSetDisplayPage(1);
            fprintf('Prepare : mask+map1+maskn+map2+maskline... press any key to continue.\r');   
            pause;
        end;
    end

    crsSetDrawPage(CRS.HOSTPAGE, hostpages(n), CRS.NOCLEAR);
    crsSetDrawOrigin(winsize1pix_width_height / 2); %Each host page has its own draw origin.
    
    debug = 0;
end

crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
crsSetDrawOrigin(screenSizePixel_width_height / 2);

crsSetSpatialUnits(CRS.DEGREEUNIT);     % all units in degree of angles

fprintf(' hostpages cost %3.2f sec...\r', etime(clock, t2));

if testrun
    crsSetDrawPage(CRS.VIDEOPAGE, 4, 1);  
    crsSetDrawMode(CRS.CENTREXY);
    crsSetPen1(6);
    crsSetPen2(7);

    if bktype==1
        crsDrawMatrix(0, 0, RDBK);
    end

    crsDrawMoveRect(CRS.HOSTPAGE, hostpages(1),[0 0],winsize1_width_height,wincenter1(newstim, :),winsize1_width_height);
    
    crsSetDisplayPage(4);
    fprintf('Video page # 4: stimpage... press any key to continue.\r');   
    pause;
end

%________________________________________________ page 1: blank (RD background) 

if testrun
    if(blankPageType == 0)
        crsSetDrawPage(CRS.VIDEOPAGE, 1, 1); 
        crsSetDrawMode(CRS.CENTREXY);
        crsSetPen1(6);
        crsSetPen2(7);
        if bktype==1
            crsDrawMatrix(0, 0, RDBK);
        end;
    else
        hdPrepareOneSideDotPage(CRS.VIDEOPAGE, 1, wincenter1pix(1,:), 6,7,winsize1pix, dotSizeOfBlankPage, dotDensityOfBlankPage);          
    end;
    
    crsSetDisplayPage(1);
    fprintf('Video page # 1: blank... press any key to continue.\r');
    pause;
end

%________________________________________________ page 2: fixation dot 

if testrun
    crsSetDrawPage(CRS.VIDEOPAGE, 2, 1); 
    crsSetDrawMode(CRS.CENTREXY);
    crsSetPen1(6);
    crsSetPen2(7);
    if bktype==1
        crsDrawMatrix(0, 0, RDBK);
    end;

    crsDrawMoveRect(CRS.HOSTPAGE, hostpages(1),[0 0],winsize1_width_height,wincenter1(newstim, :),winsize1_width_height);
    crsSetDisplayPage(2);
    fprintf('Video page # 2: fixation dot... press any key to continue.\r');
    pause;
end

%________________________________________________ page 3: Saccade page
if testrun
    crsSetDrawPage(CRS.VIDEOPAGE, 3, 1);            
    crsSetDrawMode(CRS.CENTREXY);
    crsSetPen1(6);
    crsSetPen2(7);
    if bktype==1
        crsDrawMatrix(0, 0, RDBK);
    end;

    crsDrawMoveRect(CRS.HOSTPAGE, hostpages(1),[0 0],winsize1_width_height,wincenter1(newstim, :),winsize1_width_height);

    crsSetPen1(3);      % add a black patch as background for fixation spot
    crsDrawRect([0,0], [fixsize*2, fixsize*2]); 
    crsSetPen1(fixcolor(newstim));
    crsDrawOval([-sccadex,0], [fixsize, fixsize]); 
    crsDrawOval([sccadex,0], [fixsize, fixsize]); 
    crsSetPen1(2);
    crsDrawOval([-sccadex,0], [fixsize/3, fixsize/3]); 
    crsDrawOval([sccadex,0], [fixsize/3, fixsize/3]); 

    crsSetDisplayPage(3);
    fprintf('Video page # 3: saccade dots... press any key to continue.\r');   
    pause;
end
%________________________________________________ page 4: Interval page
if(testrun)
    if(isiPageType == 101) %blank
        isiPage = 1;
    elseif (isiPageType == 102) %last frame
    elseif(isiPageType == 103)
        hdPrepareOneSideDotPage(CRS.VIDEOPAGE, isiPage, wincenter1pix(1,:), 6,7,winsize1pix, dotSizeOfIntervalPage, dotDensityOfIntervalPage);          
    elseif(isiPageType > 0 && isiPageType <= Nstim)
        isiPage = 4;

        if(isiHostPage == 0)
            isiHostPage = crsPAGECreate(CRS.HOSTPAGE, [mapsize mapsize], CRS.EIGHTBITPALETTEMODE) + 1;
        end

        savedNewstim = newstim;
        newstim = isiPageType;

        map1=double(rand(ceil(mapsize/esize1(newstim)))<rddensity1(newstim));
        if esize1(newstim)~=1
            map1=imresize(map1, esize1(newstim), 'nearest');
            if filtering==1
                map1=conv2(map1, fspecial('gaussian', round(esize1(newstim)/2), floor(round(esize1(newstim)/2))), 'same');
                map1=double(map1>0.5);
            end
        end

        if elength1(newstim)>esize1(newstim)
            tempmap=map1;
            switch Cdirection1
                case {1, 5}
                    for i=esize1(newstim):esize1(newstim):elength1(newstim)
                        map1=map1+circshift(tempmap, [i,0]);
                    end
                case {2, 6}

                case {3, 7}
                    for i=esize1(newstim):esize1(newstim):elength1(newstim)
                        map1=map1+circshift(tempmap, [0,i]);
                    end           
                case {4, 8}
            end
            map1=double(map1>=0.5);
        end

        % coordinates
        xcenter1=round(mapsize/2);
        ycenter1=round(mapsize/2);
        halfdist1=round(velocity1pix(newstim)*Tcue/2);        % half of the travel distance
        angle1=(Cdirection1-1)*45*pi/180;         % dir 1 is move toward right, dir2 is move toward top right. 
        xstart1=xcenter1-round(halfdist1*cos(angle1));
        xstop1 =xcenter1+round(halfdist1*cos(angle1));
        ystart1=ycenter1+round(halfdist1*sin(angle1));  % note since Y axis in matlab is reversed, reversed to make direction change counder-clock wise
        ystop1 =ycenter1-round(halfdist1*sin(angle1));
        xx1=round(linspace(xstop1, xstart1, totalframe)); % Note: "window move direction" and "RD movd direction" are reversed
        yy1=round(linspace(ystop1, ystart1, totalframe));
        halfwinwidthpix1=round(winsize1pix(2)/2);
        halfwinheightpix1=round(winsize1pix(1)/2);    
        winleft1=round(xx1-halfwinwidthpix1);
        winright1=round(xx1+halfwinwidthpix1);
        wintop1=round(yy1-halfwinheightpix1);
        winbottom1=round(yy1+halfwinheightpix1);

        map2=double(rand(ceil(mapsize/esize2(newstim)))<rddensity2(newstim));
        if esize2(newstim)~=1
            map2=imresize(map2, esize2(newstim),'nearest');
            if filtering==1
                map2=conv2(map2, fspecial('gaussian', round(esize2(newstim)/2), floor(round(esize2(newstim)/2))), 'same');
                map2=double(map2>0.5);
            end
        end

        if elength2(newstim)>esize2(newstim)
            tempmap=map2;
            switch Cdirection2
                case {1, 5}
                    for i=esize2(newstim):esize2(newstim):elength2(newstim)
                        map2=map2+circshift(tempmap, [i,0]);
                    end
                case {2, 6}

                case {3, 7}
                    for i=esize2(newstim):esize2(newstim):elength2(newstim)
                        map2=map2+circshift(tempmap, [0,i]);
                    end           
                case {4, 8}
            end
            map2=double(map2>=0.5);
        end

        % coordinates
        xcenter2=round(mapsize/2);
        ycenter2=round(mapsize/2);
        halfdist2=round(velocity2pix(newstim)*Tcue/2);    
        angle2=(Cdirection2-1)*45*pi/180;         
        xstart2=xcenter2-round(halfdist2*cos(angle2));
        xstop2 =xcenter2+round(halfdist2*cos(angle2));
        ystart2=ycenter2+round(halfdist2*sin(angle2)); 
        ystop2 =ycenter2-round(halfdist2*sin(angle2));
        xx2=round(linspace(xstop2, xstart2, totalframe)); 
        yy2=round(linspace(ystop2, ystart2, totalframe));
        halfwinwidthpix2=round(winsize2pix(2)/2);
        halfwinheightpix2=round(winsize2pix(1)/2);    
        winleft2=round(xx2-halfwinwidthpix2);
        winright2=round(xx2+halfwinwidthpix2);
        wintop2=round(yy2-halfwinheightpix2);
        winbottom2=round(yy2+halfwinheightpix2);

        xOrigin=startxy(1,newstim)+ round(rand(1)*periodxy(1, newstim));
        yOrigin=startxy(2,newstim)+ round(rand(1)*periodxy(2, newstim));
        x0 = xOrigin;
        y0 = yOrigin;

        crsSetPen1(6);
        crsSetPen2(7);

        firstMap = zeros(winsize1pix);

        askForMask = flagBiMask(newstim) ~= 0;
        curFlagMask = askForMask   %&& (~(velocity1pix(newstim) == velocity2pix(newstim) && direction1(newstim) == direction2(newstim)));
        askForMaskLine = isiPageMaskLine && (flagMaskLine(newstim) ~= 0);

        periodx2 = 2 * periodxy(1, newstim);
        periody2 = 2 * periodxy(2, newstim);

        %draw map1 matrix to host page
        if mapPages(1) == 0
          mapPages(1) = crsPAGECreate(CRS.HOSTPAGE, [mapsize mapsize], CRS.EIGHTBITPALETTEMODE) + 1;
        end
        crsSetDrawPage(CRS.HOSTPAGE, mapPages(1), 1); 
        crsSetDrawOrigin(0,0);
        crsSetDrawMode(0);
        crsSetPen1(6);
        crsSetPen2(7);
        crsDrawMatrix(map1);

        %draw map2 matrix to host page
        if mapPages(2) == 0
          mapPages(2) = crsPAGECreate(CRS.HOSTPAGE, [mapsize mapsize], CRS.EIGHTBITPALETTEMODE) + 1;
        end
        crsSetDrawPage(CRS.HOSTPAGE, mapPages(2), 1);
        crsSetDrawOrigin(0,0);
        crsSetDrawMode(0);
        crsSetPen1(6);
        crsSetPen2(7);
        crsDrawMatrix(map2);

        if prePage == 0
          prePage = crsPAGECreate(CRS.HOSTPAGE, winsize1pix_width_height, CRS.EIGHTBITPALETTEMODE) + 1;
        end;

        debug=0;

        %It's easy to calculate location and copy range in pixel mode.
        crsSetSpatialUnits(CRS.PIXELUNIT);     % all units in pixel

        %prepare pages
        n = totalframe; % last frame
        if(askForMask || askForMaskLine)
            % 0: stable, 1: right, 2: up right, 3: up, 4: up left, 5: left, 6:left down, 7: down, 8: right down
            switch lineDirection(newstim) 
                case 0
                case 1
                    if(0 ~= periodx2)
                        x0 = xOrigin - round(n * lineVelocityPixel(newstim));
                        while(x0 <= 0)
                            x0 = x0 + periodx2;
                        end
                    end
                case 2
                    if(0 ~= periodx2)
                        x0 = xOrigin + round(n * lineVelocityPixel(newstim));
                        while(x0 > periodx2)
                            x0 = x0 - periodx2;
                        end
                    end
                case 3
                    if(0 ~= periody2)
                        y0 = yOrigin - round(n * lineVelocityPixel(newstim));
                        while(y0 <= 0)
                            y0 = y0 + periody2;
                        end
                    end
                case 4
                    if(0 ~= periody2)
                        y0 = yOrigin + round(n * lineVelocityPixel(newstim));
                        while(y0 > periody2)
                            y0 = y0 - periody2;
                        end
                    end
            end;
        end;

        if askForMask
            if curFlagMask
                crsSetDrawPage(CRS.HOSTPAGE, isiHostPage, 1);
                crsSetDrawOrigin(0,0);
                crsSetDrawMode(0);
                crsDrawMoveRect(CRS.HOSTPAGE, maskPages(1, newstim),[x0 y0],winsize1pix_width_height,[0 0],winsize1pix_width_height);

                if(debug)
                    crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
                    crsSetDrawOrigin(0,0);
                    crsSetDrawMode(0);
                    crsDrawMoveRect(CRS.HOSTPAGE, isiHostPage,[0 0],winsize1pix_width_height,[0 0],winsize1pix_width_height);
                    crsSetDisplayPage(1);
                    fprintf('Prepare : mask... press any key to continue.\r');   
                    pause;
                end;

                crsSetDrawPage(CRS.HOSTPAGE, isiHostPage, CRS.NOCLEAR);
                crsSetDrawMode(CRS.TRANSONHIGHER);
                crsDrawMoveRect(CRS.HOSTPAGE, mapPages(1),[winleft1(n) wintop1(n)],winsize1pix_width_height,[0 0],winsize1pix_width_height);

                if(debug)
                    crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
                    crsSetDrawOrigin(0,0);
                    crsSetDrawMode(0);
                    crsDrawMoveRect(CRS.HOSTPAGE, isiHostPage,[0 0],winsize1pix_width_height,[0 0],winsize1pix_width_height);
                    crsSetDisplayPage(1);
                    fprintf('Prepare : mask+map1... press any key to continue.\r');   
                    pause;
                end;

                crsSetDrawPage(CRS.HOSTPAGE, prePage, 1);
                crsSetDrawOrigin(0,0);
                crsSetDrawMode(0);
                crsDrawMoveRect(CRS.HOSTPAGE, maskPages(2, newstim),[x0 y0],winsize1pix_width_height,[0 0],winsize1pix_width_height);

                if(debug)
                    crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
                    crsSetDrawOrigin(0,0);
                    crsSetDrawMode(0);
                    crsDrawMoveRect(CRS.HOSTPAGE, prePage,[0 0],winsize1pix_width_height,[0 0],winsize1pix_width_height);
                    crsSetDisplayPage(1);
                    fprintf('Prepare : maskn... press any key to continue.\r');   
                    pause;
                end;

                crsSetDrawPage(CRS.HOSTPAGE, prePage, CRS.NOCLEAR);
                crsSetDrawMode(CRS.TRANSONHIGHER);
                crsDrawMoveRect(CRS.HOSTPAGE, mapPages(2),[winleft2(n) wintop2(n)],winsize1pix_width_height,[0 0],winsize1pix_width_height);

                if(debug)
                    crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
                    crsSetDrawOrigin(0,0);
                    crsSetDrawMode(0);
                    crsDrawMoveRect(CRS.HOSTPAGE, prePage,[0 0],winsize1pix_width_height,[0 0],winsize1pix_width_height);
                    crsSetDisplayPage(1);
                    fprintf('Prepare : maskn+map2... press any key to continue.\r');   
                    pause;
                end;

                crsSetDrawPage(CRS.HOSTPAGE, isiHostPage, CRS.NOCLEAR);
                crsSetDrawMode(CRS.TRANSONLOWER);
                crsDrawMoveRect(CRS.HOSTPAGE, prePage,[0 0],winsize1pix_width_height,[0 0],winsize1pix_width_height);

                if(debug)
                    crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
                    crsSetDrawOrigin(0,0);
                    crsSetDrawMode(0);
                    crsDrawMoveRect(CRS.HOSTPAGE, isiHostPage,[0 0],winsize1pix_width_height,[0 0],winsize1pix_width_height);
                    crsSetDisplayPage(1);
                    fprintf('Prepare : mask+map1+maskn+map2... press any key to continue.\r');   
                    pause;
                end;
            else %just display map1
                crsSetDrawPage(CRS.HOSTPAGE, isiHostPage, 1);
                crsSetDrawOrigin(0,0);
                crsSetPen1(6);
                crsSetPen2(7);
                crsSetDrawMode(0);
                crsDrawMoveRect(CRS.HOSTPAGE, mapPages(1),[winleft1(n) wintop1(n)],winsize1pix_width_height,[0 0],winsize1pix_width_height);
            end;
        else % for all 1 masks
            crsSetDrawPage(CRS.HOSTPAGE, isiHostPage, 1);
            crsSetDrawOrigin(0,0);
            crsSetPen1(6);
            crsSetPen2(7);
            crsSetDrawMode(0);
            crsDrawMoveRect(CRS.HOSTPAGE, mapPages(1),[winleft1(n) wintop1(n)],winsize1pix_width_height,[0 0],winsize1pix_width_height);
            crsSetDrawMode(CRS.TRANSONHIGHER);
            crsDrawMoveRect(CRS.HOSTPAGE, mapPages(2),[winleft2(n) wintop2(n)],winsize1pix_width_height,[0 0],winsize1pix_width_height);
        end

        if askForMaskLine
            crsSetDrawPage(CRS.HOSTPAGE, isiHostPage, CRS.NOCLEAR);
            crsSetDrawMode(CRS.TRANSONHIGHER);
            crsDrawMoveRect(CRS.HOSTPAGE, maskPages(3, newstim),[x0 y0],winsize1pix_width_height,[0 0],winsize1pix_width_height);

            if(debug)
                crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
                crsSetDrawOrigin(0,0);
                crsSetDrawMode(0);
                crsDrawMoveRect(CRS.HOSTPAGE, isiHostPage,[0 0],winsize1pix_width_height,[0 0],winsize1pix_width_height);
                crsSetDisplayPage(1);
                fprintf('Prepare : mask+map1+maskn+map2+maskline... press any key to continue.\r');   
                pause;
            end;
        end

        crsSetDrawPage(CRS.HOSTPAGE, isiHostPage, CRS.NOCLEAR);
        crsSetDrawOrigin(winsize1pix_width_height / 2); %Each host page has its own draw origin.

        crsSetDrawPage(CRS.VIDEOPAGE, isiPage, 1);
        crsSetDrawOrigin(screenSizePixel_width_height / 2);
        crsSetDrawMode(CRS.CENTREXY);

        crsSetSpatialUnits(CRS.DEGREEUNIT);     % all units in degree of angles
        crsDrawMoveRect(CRS.HOSTPAGE, isiHostPage,[0 0],winsize1_width_height,wincenter1(newstim, :),winsize1_width_height);

        newstim = savedNewstim;
    end

    crsSetDisplayPage(isiPage);
    fprintf('Video page # 4: interval page... press any key to continue.\r');   
    pause;
end
% output stim log here
