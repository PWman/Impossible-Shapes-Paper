function [names, oldCohensd newCohensd] = findRandomFigures(filename, nTobeRemoved)


NUMBER_IMAGES_CONDITION = 120;

if iscell(filename)
    num = filename{1};
    txt = filename{2};
else
    [num, txt] = xlsread(filename);
end

backgroundSize = squeeze(num(:,3));
imageNames = txt(2:end,1);

poss = backgroundSize(1:NUMBER_IMAGES_CONDITION);
imposs = backgroundSize(NUMBER_IMAGES_CONDITION+1:2*NUMBER_IMAGES_CONDITION);

oldCohensd = (mean(poss) - mean(imposs)) / std([poss; imposs]);

s = std(poss-imposs);

i = randperm(NUMBER_IMAGES_CONDITION,nTobeRemoved);

possNew = poss;
impossNew = imposs;

possNew(i)= [];
impossNew(i)= [];

newCohensd = (mean(possNew) - mean(impossNew)) / std([possNew; impossNew]);

names = [imageNames(i); imageNames(i+NUMBER_IMAGES_CONDITION)];

end