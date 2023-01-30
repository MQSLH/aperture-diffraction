


clear

options.useGPU = true;
options.plotAperture = true;
options.brightnessScale = 4.5;

inFileName = 'Lecture4_Fourier.png';
outFileName = 'out.avi';

W = 2.5e-3; % slit width
D = 7e-3; % slit distance from center
slitH = 15e-3; % slit height



fclose('all');
if isfile(outFileName)
    delete(outFileName)
end
v = VideoWriter(outFileName); 
NFrames = 30*4;

Lmin = 0.000;
Lmax = 2;
L = linspace(Lmax,Lmin,NFrames); % canvas distance [m]

maxScale = zeros(size(L));
lambdas = linspace(380e-9,766e-9,30); %m

fieldSize = 25.6e-3; %m
patternSize = 20e-3;
Nx = 1920;
Ny = 1080;
x = linspace(-fieldSize/2,fieldSize/2,Nx);
y = (Ny/Nx) * linspace(-fieldSize/2,fieldSize/2,Ny);
% % create aperture function
% x = -H/2:0.5:H/2;
% y = x;

U = zeros(length(y), length(x));
 
image = imbinarize(rgb2gray(imread(inFileName)),0.5);

% [A,map,alpha] = imread('Untitled-2.png');
% imrgb = ind2rgb(A,map);
% image = imbinarize(rgb2gray(imrgb),0.5);

imageSize = size(image);


% resize image to fit 1080p frame
% if imageSize(2)/Nx> imageSize(1)/Ny %x-limited
% 
%     imscaleFactor = ((patternSize/fieldSize)*Nx)/size(image,2);
%     image = imresize(image,imscaleFactor);
% else
%     imscaleFactor = ((patternSize/fieldSize)*Ny)/size(image,1);
%     image = imresize(image,imscaleFactor);
% end

imageSize = size(image);

[xmask ymask] = find(image);

for ii = 1:size(xmask,1)
    U(floor(Ny/2)-floor(imageSize(1)/2)+xmask(ii),floor(Nx/2)-floor(imageSize(2)/2)+ymask(ii)) = 1;
end

% create slits
% U(abs(y)<=slitH/2, abs((abs(x)-D))<=W/2) = 1;
% U(abs(abs(y)-D)<=W/2, abs(x)<=slitH/2) = 1;

% Sample 'frequency'
fsx = 1/((x(2)-x(1)));     
fsy = 1/((y(2)-y(1)));  

% angular
wsx = 2*pi*fsx;
wsy = 2*pi*fsy;

% Number of samples
nx = length(x);            
ny = length(y);


kx =linspace(-pi*Nx/2/(fieldSize/2),pi*Nx/2/(fieldSize/2),Nx);

ky =linspace(-pi*Ny/2/(fieldSize/2),pi*Ny/2/(fieldSize/2),Ny);


[kxx, kyy] = meshgrid(kx,ky);

% diffraction angle = asin(fx)
thetaX = asind(kx);
thetaY = asind(ky);

scales = zeros(size(L));
open(v);
for ii = 1:length(L)
    [Y_RGB,maxScale] = getDiffractionFrame(U,kxx,kyy,lambdas,L(ii),options);

    scales(ii) = gather(maxScale);
    frame = real(gather(Y_RGB));
%     
%     
    writeVideo(v,gather(real(Y_RGB)));
    fprintf('writing frame %i, length %.2f\n',ii,L(ii));
end
close(v);



%%
fig = figure(7);

imshow(Y_RGB)
ax = gca;
ax.Visible = 'on';

%% Plot

%%
if options.plotAperture
    fig = figure(5);
    ax = gca;
    axis equal
    surf(x,y,flipud(U),'edgecolor','none'); 
    view(2)
    colormap colorcube     
    ax.FontSize = 24;
end
%%
function [Y_RGB, varargout] = getDiffractionFrame(U,kxx,kyy,lambdas,L,options)

    Y_out = complex(zeros(size(U,1),size(U,2),length(lambdas)));

    if options.useGPU

        U = gpuArray(U);
        lambdas = gpuArray(lambdas);
        kxx = gpuArray(kxx);
        kyy = gpuArray(kyy);
        L = gpuArray(L);
        Y_out = gpuArray(Y_out);
    end

    c = fftshift(fft2(U));

    for ii = 1:length(lambdas)

        kz = sqrt(((2*pi)/(lambdas(ii)))^2-(kxx).^2-(kyy).^2);

        Y = c.*exp(-1i*kz*L);

        U_L = ifft2(ifftshift(Y));
        Y_out(:,:,ii) = real(U_L.*conj(U_L));
    end

    if options.useGPU
        [Y_RGB, maxScale] = spectral2RGB_GPU(Y_out,lambdas,options);
        varargout{1} = maxScale;
    else
        Y_RGB = spectral2RGB(Y_out,lambdas);
    end

    
end

function RGBcolor = spectral2RGB(Y_out,lambdas)


    sizes = size(Y_out);

    colorData = readmatrix('colorMatchingData.txt');     

    colorinterpolantX = griddedInterpolant(colorData(:,1),colorData(:,2),'nearest');
    colorinterpolantY = griddedInterpolant(colorData(:,1),colorData(:,3),'nearest');
    colorinterpolantZ = griddedInterpolant(colorData(:,1),colorData(:,4),'nearest');
    
    
    
    RGBcolor = zeros(sizes(1:2));
    
    xyz2rgb = [3.2406 -1.5372 -0.4986;...
               -0.9689 1.8758 0.0415;...
               0.0557 -0.204 1.057];
    
        
           
    for ii = 1:size(Y_out,3)

       X = colorinterpolantX(lambdas(ii)*1e9)*Y_out(:,:,ii);  
       Y = colorinterpolantY(lambdas(ii)*1e9)*Y_out(:,:,ii);  
       Z = colorinterpolantZ(lambdas(ii)*1e9)*Y_out(:,:,ii);  
       
       XYZcolor(:,:,1) = X; 
       XYZcolor(:,:,2) = Y;
       XYZcolor(:,:,3) = Z;
        
       P = reshape(XYZcolor,sizes(1)*sizes(2),3);
       %linear RGB
       RGBcolor = RGBcolor+reshape(P*xyz2rgb',sizes(1),sizes(2),3);

    end
       % gamma correction to enhance low intensities
       RGBcolor(RGBcolor<=0.00304) = 12.92*RGBcolor(RGBcolor<=0.00304);
       RGBcolor(RGBcolor>0.00304) = 1.055*RGBcolor(RGBcolor>0.00304).^0.42-0.055;
    
       RGBcolor = RGBcolor./max(RGBcolor,[],'all');
       
end

function [RGBcolor maxScale] = spectral2RGB_GPU(Y_out,lambdas,options)


    sizes = size(Y_out);

    colorData = gpuArray(readmatrix('colorMatchingData.txt'));     

%     colorinterpolantX = griddedInterpolant(colorData(:,1),colorData(:,2),'nearest');
%     colorinterpolantY = griddedInterpolant(colorData(:,1),colorData(:,3),'nearest');
%     colorinterpolantZ = griddedInterpolant(colorData(:,1),colorData(:,4),'nearest');
%     
%     
    
    RGBcolor = gpuArray.zeros(sizes(1:2));
    
    xyz2rgb = gpuArray([3.2406 -1.5372 -0.4986;...
               -0.9689 1.8758 0.0415;...
               0.0557 -0.204 1.057]);
    
        
           
    for ii = 1:size(Y_out,3)

       X = interp1(colorData(:,1),colorData(:,2),lambdas(ii)*1e9,'linear',0)*Y_out(:,:,ii);  
       Y = interp1(colorData(:,1),colorData(:,3),lambdas(ii)*1e9,'linear',0)*Y_out(:,:,ii);  
       Z = interp1(colorData(:,1),colorData(:,4),lambdas(ii)*1e9,'linear',0)*Y_out(:,:,ii);   
       
       XYZcolor(:,:,1) = X; 
       XYZcolor(:,:,2) = Y;
       XYZcolor(:,:,3) = Z;
        
       P = reshape(XYZcolor,sizes(1)*sizes(2),3);
       %linear RGB
       RGBcolor = RGBcolor+reshape(P*xyz2rgb',sizes(1),sizes(2),3);

    end
       % gamma correction to enhance low intensities
       RGBcolor(RGBcolor<=0.00304) = 12.92*RGBcolor(RGBcolor<=0.00304);
       RGBcolor(RGBcolor>0.00304) = 1.055*RGBcolor(RGBcolor>0.00304).^0.42-0.055;
    
       if options.brightnessScale
           maxScale = options.brightnessScale;
       else
           maxScale = max(real(RGBcolor),[],'all');
       end
       RGBcolor = RGBcolor./maxScale;
       RGBcolor(RGBcolor<0) = 0;
       RGBcolor(RGBcolor>1) = 1;
       
end
