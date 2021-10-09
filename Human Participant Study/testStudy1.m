function testStudy1(dataFilenmage, backgroundFilename)

testRemoveImages(dataFilenmage,[]);

[names, oldCohensd newCohensd] = findOutlierBackgrounds(backgroundFilename, 1);
testRemoveImages(dataFilenmage,names);
length(names)
end

function testRemoveImages(filename,imageNames)

fileNames = dir(filename + "*");

nPart = length(fileNames);

for iPart =1:nPart
    
    
    if iscell(filename)
        num = filename{1};
        txt = filename{2};
    else
        [num, txt] = xlsread(fileNames(iPart).name);
    end
    images = txt(2:end,3);
    
    h1 = imageNames;
    imageNames = string(h1) + ".bmp";
    
    correct = num(:,1);
    rt = num(:,6);
    part = num(:,end);
    label = txt(2:end,4);
    
    i = find(startsWith(images,imageNames));
    newRT = rt;
    newRT(i) = [];
    newCorrect = correct;
    newCorrect(i) = [];
    newLabel = label;
    newLabel(i) = [];
    
   
    
    i = find(contains(newLabel, 'Possible') == 1);
    accPoss(iPart) = mean(newCorrect(i));
    RTPoss(iPart) = mean(newRT(i));
    
    i = find(contains(newLabel, 'Impossible') == 1);
    accImposs(iPart) = mean(newCorrect(i));
    RTImposs(iPart) = mean(newRT(i));
    
end
poss = (accPoss+accImposs) / 2;

% outlier removal
i = find(poss < 0.6);


poss(i) = [];
accPoss(i) = [];
accImposs(i) = [];
RTPoss(i) = [];
RTImposs(i) = [];

mean(100*poss)
std(100*poss)
SEM = std(100*poss)/sqrt(length(poss));               % Standard Error
ts = tinv([0.025  0.975],length(poss)-1);      % T-Score
CI = mean(100*poss) + ts*SEM                      % Confidence Intervals

% confusion matrix

% possible stimulus
%correct possible
resPossPoss = 0.5*mean(accPoss);
% falsePositives possible
resPossImposs = 0.5*(1-mean(accPoss));

% impossile stimulus
% falsePostive
resImpossPoss = 0.5*(1-mean(accImposs));
resImpossImposs =0.5*mean(accImposs);

ta(1,1) = resPossPoss;
ta(1,2) = resPossImposs;
ta(2,1) = resImpossPoss;
ta(2,2) = resImpossImposs;

ta

% prime and bias
[dp, c] = dprime_simple(mean(accPoss),1-mean(accPoss))

[mImposs, mPoss,P, dCohen, t, df] = calcStats(RTImposs', RTPoss');
sImposs = std(RTImposs);
sPoss = std(RTPoss);
print_res(mImposs, sImposs, mPoss, sPoss, P, dCohen, t, df);

sImposs = std(accImposs);
sPoss = std(accPoss);
[mImposs, mPoss,P, dCohen, t, df] = calcStats(accImposs', accPoss');
print_res(mImposs, sImposs, mPoss, sPoss, P, dCohen, t, df);



end