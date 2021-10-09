function print_res(mImposs, sImposs, mPoss,sPoss, p, d, t, df)

fprintf(1, 'impossible: %.3f (%.3f) possible: %.3f (%.3f) (t(%d)=%.2f, p=%.3f, d=%.2f)\n', mImposs, sImposs, mPoss, sPoss, df, t, p, d);
end
