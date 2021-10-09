function [mImposs, mPoss, P, dCohen, t, df] = calcStats(imposs, poss)

mImposs = mean(imposs);
mPoss = mean(poss);

[H,P,CI,STATS] = ttest(imposs, poss);
t = STATS.tstat;
df = STATS.df;

dCohen = (mean(imposs) - mean(poss)) / std([imposs; poss]);

end