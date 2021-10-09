function h=defROI4Shape()

impossibleFiles = dir('im*.bmp');
possibleFiles = dir('new*.bmp');

nFiles=length(impossibleFiles);


for i=1:nFiles
    close all
    possibleIm = imread(possibleFiles(i).name);
    h= imshow(possibleIm);
    movegui(h, 'northwest');
    figure
    im = imread(impossibleFiles(i).name);
    bw = roipoly(im);
    imwrite(bw,sprintf('roi_%s',impossibleFiles(i).name), 'bmp');
end


end