%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all;
global CRS;
vsgInit;
Scan_mode=2;
crsIOWriteDAC(0,'Volts5');  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
viewdist=570;  % in mm (1400mm for awake room 2)
crsSetViewDistMM(viewdist);
screenSize_width_height= [crsGetScreenHeightDegrees, crsGetScreenWidthDegrees] ;	 % screen size in x and y in degrees
screenSizePixel_width_height= [crsGetScreenWidthPixels, crsGetScreenHeightPixels] ;	 % screen size in x and y in pixels. 
%Stimulus 0 -- Blank page definition
blankPageType=1; % 0: blank page; 1: random dot page;
dotSizeOfBlankPage=2;        % RD element size, in pixel
dotDensityOfBlankPage = 0.03;  % RD density: the percentage of area covered by random dots.

wincenter1=[0,0];
winsize1=  screenSize_width_height;
wincenter1pix = round(crsUnitToUnit(CRS.DEGREEUNIT, wincenter1, CRS.PIXELUNIT));
winsize1pix = round(crsUnitToUnit(CRS.DEGREEUNIT, winsize1, CRS.PIXELUNIT));
crsPaletteSetPixelLevel(6, [1,1,1]);   % Pen Color
crsPaletteSetPixelLevel(7, [0,0,0]);   % Back Ground Color
hdPrepareOneSideDotPage(CRS.VIDEOPAGE, 1, wincenter1pix(1,:), 6,7,winsize1pix, dotSizeOfBlankPage, dotDensityOfBlankPage);
crsSetDisplayPage(1);

