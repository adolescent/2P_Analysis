% HDRDSub04Mask.m
% Prepare stimulus mask, used for combine two patch of random dots. 
% Just a section of code, not a function. 

% note: phase 0 is dealed later for each presentation 

fprintf('Preparing masks ... ');
t0=clock;

%prepare mask matrix first.
period=round(deg2pix./SF0./2)*2;   % how many pixels per cycle, make sure this is a even number (so the width of two stripes will be the same)
period0 = period ./ 2;
masksize=round(crsGetScreenWidthPixels + 4 * max(period0));
masks=zeros(masksize, masksize, Nstim);
masks3=zeros(masksize, masksize, Nstim); % mask line
startxy=ones(2, Nstim);     % top-left corner location on the mask that will be used for masking, usually is [1,1] for some masks (e.g. 135 deg) need use right side of the big mask since it's easy to draw on the right side. 
periodxy=zeros(2, Nstim);   % period in X Y dimension, used for determin spatial phase in each trial. 
for i=1:Nstim
    xperiod = 0;
    yperiod = 0;
    
    switch orientation0(i) 
        case 0
            yperiod = period0(i); 
        case 45
            xperiod=round(period0(i)/sin(pi/4)/2)*2;
            yperiod = xperiod;
        case 90
            xperiod = period0(i);
        case 135
            xperiod=round(period0(i)/sin(pi/4)/2)*2;
            yperiod = xperiod;
        otherwise
    end
    
    if(flagBiMask(i) ~= 0 || flagMaskLine(i) ~= 0)
        periodxy(:, i)=[xperiod, yperiod];
    end;
    
    if flagBiMask(i) ~= 0 % draw mask
        %check if mask file has been pre-generated and stored.
        %Different mask should have different mask_size or different orientation or different period.
        maskFileName = sprintf('masks/mask_size%d_orientation%d_period%d.mat',masksize, orientation0(i), period0(i));
        fMaskExist = (2 == exist(maskFileName,'file'));
        
        if(fMaskExist) %load mask from file to save preparation time.
           d = load(maskFileName,'c');
           masks(:,:,i) = d.c;
        else %prepare mask and save to file.
            switch orientation0(i) 
                case 0
                    for y=1:masksize
                        if mod(y, yperiod * 2) > yperiod ||mod(y, yperiod * 2) == 0;  %% in a cycle [2*period], there are black (present 0) stripe [1 period] and white (present 1) stripe [1 period],  this 'if' code seperate the black and white stripes in a cycle.
                            masks(:,:,i)=IMDrawLine(masks(:,:,i), 1, y, masksize, y, 1);
                        end
                    end
                    c = masks(:,:,i);
                    save(maskFileName, 'c');
                case 45
                    for x=1:2 *masksize
                        if mod(x, xperiod*2)>xperiod || mod(x, xperiod*2)==0;
                            if(x > masksize)
                                fx = masksize;
                                fy = x - masksize;
                                dx = fy;
                                dy = fx;
                            else
                                fx = x;
                                fy = 1;
                                dx = fy;
                                dy = fx;
                            end
                            masks(:,:,i)=IMDrawLine(masks(:,:,i), fx, fy, dx, dy, 1);
                        end
                    end
                    c = masks(:,:,i);
                    save(maskFileName, 'c');
                case 90
                    for x=1:masksize
                        if mod(x, xperiod * 2) > xperiod || mod(x, xperiod * 2) == 0;
                            masks(:,:,i)=IMDrawLine(masks(:,:,i), x, 1, x, masksize, 1);
                        end
                    end
                    c = masks(:,:,i);
                    save(maskFileName, 'c');
                case 135
                    for x=-masksize+1:masksize
                        if mod(masksize - x + 1, xperiod*2)> xperiod || mod(masksize - x + 1, xperiod*2)==0;
                           if(x > 0)
                                fx = x;
                                fy = 1;
                                dx = masksize;
                                dy = masksize - fx + 1;
                            else
                                fx = 1;
                                fy = 1 - x;
                                dx = masksize - fy + 1;
                                dy = masksize;
                            end
                            masks(:,:,i)=IMDrawLine(masks(:,:,i), fx, fy, dx, dy, 1);
                        end
                    end
                    c = masks(:,:,i);
                    save(maskFileName, 'c');
                otherwise
            end
        end
    end
    
    if flagMaskLine(i) ~= 0 % draw mask line
        %check if maskline file has been pre-generated and stored.
        %Different maskline should have different mask_size or different
        %orientation or different period.
        mask3FileName = sprintf('masks/mask3_size%d_orientation%d_period%d.mat',masksize, orientation0(i),period0(i));
        fMask3Exist = (2 == exist(mask3FileName,'file'));
        
        if(fMask3Exist) %load from file to save preparation time.
           d = load(mask3FileName,'c');
           masks3(:,:,i) = d.c;
        else %prepare maskline and save to file
            switch orientation0(i) 
                case 0
                    for y=1:masksize
                        if 0 == mod(y, yperiod)
                            masks3(:,:,i)=IMDrawLine(masks3(:,:,i), 1, y, masksize, y, 1);
                        end
                    end
                    c = masks3(:,:,i);
                    save(mask3FileName, 'c');
                case 45
                    for x=1:2 *masksize
                        if 0 == mod(x, xperiod)
                            if(x > masksize)
                                fx = masksize;
                                fy = x - masksize;
                                dx = fy;
                                dy = fx;
                            else
                                fx = x;
                                fy = 1;
                                dx = fy;
                                dy = fx;
                            end
                            masks3(:,:,i)=IMDrawLine(masks3(:,:,i), fx, fy, dx, dy, linecolor3(i));
                        end
                    end
                    c = masks3(:,:,i);
                    save(mask3FileName, 'c');
                case 90
                    for x=1:masksize
                        if 0 == mod(x, xperiod)
                            masks3(:,:,i)=IMDrawLine(masks3(:,:,i), x, 1, x, masksize, 1);
                        end
                    end
                    c = masks3(:,:,i);
                    save(mask3FileName, 'c');
                case 135
                    for x=-masksize+1:masksize
                        if 0 == mod(masksize - x + 1, xperiod)
                           if(x > 0)
                                fx = x;
                                fy = 1;
                                dx = masksize;
                                dy = masksize - fx + 1;
                            else
                                fx = 1;
                                fy = 1 - x;
                                dx = masksize - fy + 1;
                                dy = masksize;
                            end
                            masks3(:,:,i)=IMDrawLine(masks3(:,:,i), fx, fy, dx, dy, linecolor3(i));
                        end
                    end
                    c = masks3(:,:,i);
                    save(mask3FileName, 'c');
                otherwise
            end
        end
    end
end

fprintf(' cost %3.2f sec\r', etime(clock, t0));

%draw mask matrix to host pages.
screenSize = [crsGetScreenWidthDegrees crsGetScreenHeightDegrees];
maskPages = zeros(3, Nstim);

debugMask = 0; %For debug. Use to see each maskpage prepared. Zero means ignore. 
for i=1:Nstim
    if(flagBiMask(i) ~= 0) % draw mask i
        found = 0;
        for j=1:i-1 % check pages already drawn
            isIdentical = (flagBiMask(j) ~= 0) && (orientation0(i) == orientation0(j)) && (period0(i) == period0(j));
            if(isIdentical)
                maskPages(1,i) = maskPages(1,j);
                maskPages(2,i) = maskPages(2,j);
                found = 1;
                break;
            end;
        end;
        
        if(~found)
            maskPages(1,i) = crsPAGECreate(CRS.HOSTPAGE, [masksize masksize], CRS.EIGHTBITPALETTEMODE) + 1;
            crsSetDrawPage(CRS.HOSTPAGE, maskPages(1,i), 1); 
            crsSetDrawOrigin(0, 0);
            crsSetDrawMode(0);
            crsSetPen1(6);
            crsSetPen2(7);
            crsDrawMatrix(masks(:,:,i));

            if(debugMask)
                crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
                crsSetDrawOrigin(0, 0);
                crsSetDrawMode(0);
                crsDrawMoveRect(CRS.HOSTPAGE, maskPages(1, i), [0 0],screenSize,[0 0],screenSize);
                crsSetDisplayPage(1);
                fprintf('Mask... press any key to continue.\r');   
                pause;
            end;

            maskPages(2,i) = crsPAGECreate(CRS.HOSTPAGE, [masksize masksize], CRS.EIGHTBITPALETTEMODE) + 1;
            crsSetDrawPage(CRS.HOSTPAGE, maskPages(2,i), 1); 
            crsSetDrawOrigin(0, 0);
            crsSetDrawMode(0);
            crsSetPen1(7);
            crsSetPen2(6);
            crsDrawMatrix(masks(:,:,i));

            if(debugMask)
                crsSetDrawPage(CRS.VIDEOPAGE, 1, 1);
                crsSetDrawOrigin(0, 0);
                crsSetDrawMode(0);
                crsDrawMoveRect(CRS.HOSTPAGE, maskPages(2, i), [0 0],screenSize,[0 0],screenSize);
                crsSetDisplayPage(1);
                fprintf('Maskn... press any key to continue.\r');   
                pause;
            end;
        end;
    end;
    
    if(flagMaskLine(i)~=0)
        found = 0;
        for j=1:i-1 % check pages already drawn
            isIdentical = (flagMaskLine(j) ~= 0) && (orientation0(i) == orientation0(j)) && (period0(i) == period0(j));
            if(isIdentical)
                maskPages(3,i) = maskPages(3,j);
                found = 1;
                break;
            end;
        end;
        
        if(~found)
            maskPages(3,i) = crsPAGECreate(CRS.HOSTPAGE, [masksize masksize], CRS.EIGHTBITPALETTEMODE) + 1;
            crsSetDrawPage(CRS.HOSTPAGE, maskPages(3,i), 1); 
            crsSetDrawOrigin(0, 0);
            crsSetDrawMode(0);
            crsSetPen1(6);
            crsSetPen2(7);
            crsDrawMatrix(masks3(:,:,i));

            if(debugMask)
                crsSetDrawPage(CRS.VIDEOPAGE, 2, 1);
                crsSetDrawOrigin(0, 0);
                crsSetDrawMode(0);
                crsDrawMoveRect(CRS.HOSTPAGE, maskPages(3, i), [0 0],screenSize,[0 0],screenSize);
                crsSetDisplayPage(2);
                fprintf('Mask3... press any key to continue.\r');   
                pause;
            end
        end
    end
end
% for i=1:Nstim
%     imwrite(nc(masks3(:,:,i)*255), [num2str(i), '3.bmp']);
%     imwrite(nc(masks(:,:,i)*255), [num2str(i), '.bmp']);
% end

