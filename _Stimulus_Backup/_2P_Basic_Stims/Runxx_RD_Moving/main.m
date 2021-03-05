%% This function will disp moving random dots, gray scale.
clc; clear all;
global CRS;
vsgInit;
crsIOWriteDAC(0,'Volts5');  
Scan_mode=2;

wincenterRD = [0,0];
viewdist=570;  % in mm (1400mm for awake room 2)
crsSetViewDistMM(viewdist);
wincenterRDpix = round(crsUnitToUnit(CRS.DEGREEUNIT, wincenterRD, CRS.PIXELUNIT));
winsizeRD = [crsGetScreenHeightDegrees, crsGetScreenWidthDegrees];
dotSize = 2;
dotDensity = 0.03;
screenSize_width_height= [crsGetScreenHeightDegrees, crsGetScreenWidthDegrees] ;	 % screen size in x and y in degrees
screenSizePix_width_height= [crsGetScreenWidthPixels, crsGetScreenHeightPixels] ;	
winsizePix = round(crsUnitToUnit(CRS.DEGREEUNIT, winsizeRD, CRS.PIXELUNIT));
winsizePixWidthHeight = [winsizePix(2),winsizePix(1)];
curFrameRate= crsGetSystemAttribute(CRS.FRAMERATE);
totalframe = 180; %Frame of host pages.
fprintf('\r\ttotalframe=%d\r', totalframe);
hostpagesRD=zeros(totalframe,1);
crsSetVideoMode(CRS.EIGHTBITPALETTEMODE);
crsSetDrawMode(CRS.CENTREXY);
OldUnits = crsGetSpatialUnits;
crsSetSpatialUnits(CRS.PIXELUNIT);
refreshrateRD = 6;



for n = 1:totalframe
    if refreshrateRD ~= 1 && mod(n,refreshrateRD)~= 1
        hostpagesRD(n)=hostpagesRD(n-1);
        continue;
    end
    if hostpagesRD(n) == 0 %allocate host pages used to store final images
        hostpagesRD(n) = crsPAGECreate(CRS.HOSTPAGE, screenSizePix_width_height, CRS.EIGHTBITPALETTEMODE) + 1;
    end
    crsSetDrawPage(CRS.HOSTPAGE, hostpagesRD(n), 1)
    size_pix=winsizePix;
    map = hdGenerateOneSideOneZeroRandomDotMatrix(winsizePix, dotSize, dotDensity);
    crsDrawMatrix(wincenterRDpix(1,:), map);
end
    
while(1)
    for n = 1:totalframe
        displayPages = [6 7];
        displayPage = displayPages(mod(n, 2)+1); %switch display page
        crsSetDrawPage(CRS.VIDEOPAGE, displayPage, 1);
        crsSetDrawMode(CRS.CENTREXY);
        %copy pre-generated image from host page to vedio page.
        crsDrawMoveRect(CRS.HOSTPAGE, hostpagesRD(n),[0 0],screenSizePix_width_height,[0 0],screenSizePix_width_height);
        crsSetDisplayPage(displayPage);
    end
end

