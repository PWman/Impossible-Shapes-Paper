function [H,P,CI,STATS, possAcc, impossAcc, dCohen] = testRemoveImages(filename, imageNames)


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

%


poss = (accPoss+accImposs) / 2;

i = find(poss < 0.6);

RTPoss(i), RTImposs(i)

poss(i) = [];
accPoss(i) = [];
accImposs(i) = [];
RTPoss(i) = [];
RTImposs(i) = [];

possRT = mean(RTPoss)
impossRT = mean(RTImposs)
[H,P,CI,STATS] = ttest(RTPoss, RTImposs)
dCohen = (possRT - impossRT) / std([possRT impossRT]);


possAcc = mean(accPoss);
impossAcc = mean(accImposs);

hist(poss)
dCohen = (possAcc - impossAcc) / std([accPoss accImposs]);
s = std(100*[accPoss accImposs]);
m = mean(100*[accPoss accImposs]);

[H,P,CI,STATS] = ttest(accPoss, accImposs);

end