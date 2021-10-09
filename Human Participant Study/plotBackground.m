function plotBackground(fileName_study1, filename_study2)

set_plot_default

[num, txt] = xlsread(fileName_study1);

bg = num(:,3);

trainDiff = bg(1:32) - bg(33:64);
validDiff = bg(65:72) - bg(73:80) ;

subplot(1,2,1);
plotRes(trainDiff, validDiff);
title('Original images');


[num, txt] = xlsread(filename_study2);

bg = num(:,3);

%           Imposs     possible
trainDiff = bg(1:32) - bg(33:64);
validDiff = bg(65:72) - bg(73:80) ;

subplot(1,2,2);
plotRes(trainDiff, validDiff);
title('Corrected images');


end

function plotRes(trainDiff, validDiff)
plot(224*224*trainDiff, 'b*');
hold
plot(224*224*validDiff, 'r*');
xlabel('images');
ylabel('impossible - possible [number of pixels]');
legend('training', 'validation');
ylim([-3000 7000]);
xlim([0 33]);
text(5,5000, sprintf('mean: %.0f',224*224*mean([trainDiff; validDiff])));
text(5,4500, sprintf('std. dev.: %.0f',std(224*224*[trainDiff; validDiff])));

end
