function testBackgroundChange()

[num, txt] = xlsread('Background Percentages Study 1.csv');

[poss1, imposs1] = calcComp(num,txt);

[num, txt] = xlsread('Background Percentages Study 2.csv');
[poss2, imposs2] = calcComp(num,txt);

diff1 = poss1 - imposs1;
diff2 = poss2 - imposs2;


[mDiff1, mDiff2,P, dCohen, t, df] = calcStats(diff2, diff1);

print_res(mDiff1,std(diff1), mDiff2, std(diff2), P, dCohen, t, df);

end

function [poss, imposs] = calcComp(num,txt)

background = 224*224*num(:,3);
label = txt(2:end,3);

i = find(contains(label, 'Impossible') == 1);
imposs = background(i);

i = find(contains(label, 'Possible') == 1);
poss = background(i);

[mImposs, mPoss,P, dCohen, t, df] = calcStats(imposs, poss);

print_res(mImposs, std(imposs), mPoss, std(poss),P, dCohen, t, df);

end


