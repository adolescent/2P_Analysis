% load_mat

load triangle.mat;
picture=imresize(picture,cmSizep/300);
picture(picture<0.4)=0; picture(picture>1)=1;
stim_triangle=picture;

load circle_bright.mat;
picture=imresize(picture,cmSizep/300);
picture(picture<0.4)=0; picture(picture>1)=1;
stim_circle_bright=picture;

% load patch_bright.mat;
% picture=imresize(picture,cmSizep/300); 
% picture(picture<0.4)=0; picture(picture>1)=1;
% stim_patch_bright=picture;
% 




