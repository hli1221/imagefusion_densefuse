function [EN,MI,Qabf,FMI_pixel,FMI_dct,FMI_w,Nabf,SCD,SSIM, MS_SSIM] = analysis_Reference(image_f,image_ir,image_vis)

[s1,s2] = size(image_ir);
imgSeq = zeros(s1, s2, 2);
imgSeq(:, :, 1) = image_ir;
imgSeq(:, :, 2) = image_vis;

image1 = im2double(image_ir);
image2 = im2double(image_vis);
image_fused = im2double(image_f);

%EN
EN = entropy(image_fused);
%MI
MI = analysis_MI(image_ir,image_vis,image_f);
%Qabf
Qabf = analysis_Qabf(image1,image2,image_fused);
%FMI
FMI_pixel = analysis_fmi(image1,image2,image_fused);
FMI_dct = analysis_fmi(image1,image2,image_fused,'dct');
FMI_w = analysis_fmi(image1,image2,image_fused,'wavelet');
%Nabf
Nabf = analysis_nabf(image_fused,image1,image2);
%SCD
SCD = analysis_SCD(image1,image2,image_fused);
% SSIM_a
SSIM1 = ssim(image_fused,image1);
SSIM2 = ssim(image_fused,image2);
SSIM = (SSIM1+SSIM2)/2;
%MS_SSIM
[MS_SSIM,t1,t2]= analysis_ms_ssim(imgSeq, image_f);

end







