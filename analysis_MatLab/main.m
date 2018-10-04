%% Li H, Wu X J. DenseFuse: A Fusion Approach to Infrared and Visible Images[J]. arXiv preprint arXiv:1804.08361, 2018. 
%% https://arxiv.org/abs/1804.08361

fileName_source_ir  = ["infrared image name"];
fileName_source_vis = ["visible image name"];
fileName_fused      = ["fused image name"];

source_image1 = imread(fileName_source_ir);
source_image2 = imread(fileName_source_vis);
fused_image   = imread(fileName_fused);

disp("Start");
disp('---------------------------Analysis---------------------------');
[EN,MI,Qabf,FMI_pixel,FMI_dct,FMI_w,Nabf,SCD,SSIM, MS_SSIM] = analysis_Reference(fused_image,source_image1,source_image2);
disp('Done');


