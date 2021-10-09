function h = plotRes()

thresFactor = [1];

n = length(thresFactor);

p = zeros(n,1);

for iThres=1:n
    [res, p(iThres), sample] =testonRemoval('Background Proportion Behavioural.csv','ExpResults_corrected.xlsx', thresFactor(iThres));
    tMean(iThres) = mean(sample);
    tStd(iThres) = std(sample);
    tRes(iThres) = res.stats.tstat;
    pTest(iThres) = res.p;
    newd(iThres) = res.newd;
    nRes(iThres) = res.n;
end

figure
subplot(2,3,1);
plot(thresFactor, pTest, '*-');
title('p after removal');
subplot(2,3,2);
plot(thresFactor, newd, '*-');
title('effect size after removal');
subplot(2,3,4);
plot(thresFactor, p, '*-');
title('p for removal');
subplot(2,3,3);
errorbar(thresFactor, tMean, tStd, '*-'); hold
plot(thresFactor, tRes);
title('test of removal');

subplot(2,3,5);
plot(thresFactor, nRes, '*-');

end