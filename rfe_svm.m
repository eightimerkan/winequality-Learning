% Veri setini yükle
data = readtable('winequality.csv');

% Eksik verileri kontrol et ve doldur
missingData = any(ismissing(data), 2);
meanValues = mean(data{~missingData, :}, 'omitnan');
data{missingData, :} = repmat(meanValues, sum(missingData), 1);

% Veri setini normalleştir
normalizedData = normalize(table2array(data(:, 1:end-1))); % Son sütun hedef değişken olduğu için dışarıda bırakılır

% Korelasyon matrisini hesapla
correlationMatrix = corrcoef(normalizedData, 'rows', 'pairwise');

% Recursive Feature Elimination (RFE)
rfeModel = fitrlinear(normalizedData, table2array(data(:, end))); % Fit linear regression model
rfeRanking = abs(rfeModel.Beta); % Feature importance ranking based on beta coefficients

% Belirli bir eşik değeri kullanarak özellik seçimi yap
rfeThreshold = 0.1; % Eşik değeri
selectedFeatures_rfe = find(rfeRanking > rfeThreshold);

% Principal Component Analysis (PCA)
[coeff, score, ~, ~, explained] = pca(normalizedData); % Perform PCA
pcaExplainedRatio = cumsum(explained) / sum(explained); % Cumulative explained variance ratio

% Belirli bir açıklanan varyans oranı eşiğini kullanarak özellik seçimi yap
pcaThreshold = 0.95; % Açıklanan varyans oranı eşiği
selectedFeatures_pca = find(pcaExplainedRatio > pcaThreshold);

% Seçilen özelliklerin adlarını al
featureNames = data.Properties.VariableNames(1:end-1);
selectedFeatureNames_rfe = featureNames(selectedFeatures_rfe);
selectedFeatureNames_pca = featureNames(selectedFeatures_pca);

% Özellik seçimi sonuçlarını ekrana yazdır
disp("Recursive Feature Elimination (RFE) - Seçilen Özellikler:");
disp(selectedFeatureNames_rfe);

disp("Principal Component Analysis (PCA) - Seçilen Özellikler:");
disp(selectedFeatureNames_pca);

% Özellik seçimi sonuçlarını görselleştir
figure;
subplot(1, 2, 1);
heatmap(featureNames, featureNames, correlationMatrix);
title('Korelasyon Matrisi');

subplot(1, 2, 2);
bar(rfeRanking);
xlabel('Özellikler');
ylabel('Önem Sıralaması');
xticklabels(featureNames);
xtickangle(45);
title('Recursive Feature Elimination (RFE) - Özellik Önem Sıralaması');

% Normalize edilmiş veriyi seçilen özelliklerle güncelle
normalizedData_selected_rfe = normalizedData(:, selectedFeatures_rfe);
normalizedData_selected_pca = normalizedData(:, selectedFeatures_pca);

% Modelleme Aşaması

% SVM modelini eğitim verisiyle oluştur
svmModel = fitcecoc(normalizedData_selected_rfe, table2array(data(:, end)), 'Learners', 'linear');

% Veri setini eğitim ve test setlerine bölmek için oranı ayarla
trainRatio = 0.8; % Eğitim seti oranı
testRatio = 1 - trainRatio; % Test seti oranı

% Veri setini eğitim ve test setlerine böl
cv = cvpartition(size(normalizedData_selected_rfe, 1), 'Holdout', testRatio);

% Eğitim ve test setlerini oluştur
XTrain = normalizedData_selected_rfe(training(cv), :);
YTrain = table2array(data(training(cv), end));
XTest = normalizedData_selected_rfe(test(cv), :);
YTest = table2array(data(test(cv), end));

% Modeli eğit
svmModel = fitcecoc(XTrain, YTrain, 'Learners', 'linear');

% Tahmin yap
YTrainPred = predict(svmModel, XTrain);
YTestPred = predict(svmModel, XTest);

% Doğruluk oranını hesapla
trainAccuracy = sum(YTrainPred == YTrain) / numel(YTrain);
testAccuracy = sum(YTestPred == YTest) / numel(YTest);

% Sonuçları ekrana yazdır
disp("Eğitim Seti - Doğruluk Oranı: " + trainAccuracy);
disp("Test Seti - Doğruluk Oranı: " + testAccuracy);


% Model Değerlendirme Aşaması

% Tahmin yap
YTrainPred = predict(svmModel, XTrain);
YTestPred = predict(svmModel, XTest);

% Eğitim ve test seti doğruluk oranını hesapla
trainAccuracy = sum(YTrainPred == YTrain) / numel(YTrain);
testAccuracy = sum(YTestPred == YTest) / numel(YTest);

% Eğitim ve test seti hassasiyet ve duyarlılık değerlerini hesapla
trainConfusionMatrix = confusionmat(YTrain, YTrainPred);
testConfusionMatrix = confusionmat(YTest, YTestPred);
trainPrecision = diag(trainConfusionMatrix) ./ sum(trainConfusionMatrix, 2);
testPrecision = diag(testConfusionMatrix) ./ sum(testConfusionMatrix, 2);
trainRecall = diag(trainConfusionMatrix) ./ sum(trainConfusionMatrix, 1)';
testRecall = diag(testConfusionMatrix) ./ sum(testConfusionMatrix, 1)';

% Sonuçları ekrana yazdır
disp("Eğitim Seti - Doğruluk Oranı: " + trainAccuracy);
disp("Test Seti - Doğruluk Oranı: " + testAccuracy);
disp("Eğitim Seti - Hassasiyet: ");
disp(trainPrecision');
disp("Eğitim Seti - Duyarlılık: ");
disp(trainRecall');
disp("Test Seti - Hassasiyet: ");
disp(testPrecision');
disp("Test Seti - Duyarlılık: ");
disp(testRecall');