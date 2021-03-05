% mat_to_bmp

clc; clear all;

tempfilename=struct2cell(dir([pwd,'\*.mat']));
filename=sort(tempfilename(1,:));
filenum=length(filename);

for i=1:filenum
    tifname=[pwd,'\',filename{i}];
    load (tifname);
    if length(size(picture))==2
      aa=uint8(picture*255);
      filename0=filename{i};
      imwrite(aa,[filename0(1:end-4),'.bmp'],'bmp');
    elseif length(size(picture))==3
        for j=1:size(picture,3)
          aa=uint8(squeeze(picture(:,:,j))*255);
          filename0=filename{i};
          imwrite(aa,[filename0(1:end-4),'_',num2str(j),'.bmp'],'bmp');
        end
    else continue;           
    end

end





