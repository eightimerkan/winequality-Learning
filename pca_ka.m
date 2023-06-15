% Veri setini yüklüyoruz.
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

%% PCA uygulayarak özellik seçimi yapma
[coeff, score, ~, ~, explained] = pca(normalizedData);
pcaExplainedRatio = cumsum(explained) / sum(explained);

% Belirli bir açıklanan varyans oranı eşiğini kullanarak özellik seçimi yapma
pcaThreshold = 0.95; % Açıklanan varyans oranı eşiği
selectedFeatures_pca = find(pcaExplainedRatio > pcaThreshold);

% Seçilen özelliklerin adlarını al
selectedFeatureNames_pca = featureNames(selectedFeatures_pca);

% Normalize edilmiş veriyi seçilen özelliklerle güncelle
normalizedData_selected_pca = normalizedData(:, selectedFeatures_pca);

% Eğitim ve test verilerini yeniden ayırma
X_train = normalizedData_selected_pca(trainIdx, :);
y_train = table2array(data(trainIdx, end));
X_test = normalizedData_selected_pca(testIdx, :);
y_test = table2array(data(testIdx, end));

% Karar ağacı modelini eğitme
treeModel = fitctree(X_train, y_train);

% Test verilerini kullanarak modeli değerlendirme
y_pred = predict(treeModel, X_test);

% Doğruluk oranını hesaplama
accuracy = sum(y_pred == y_test) / numel(y_test);

% Karmaşıklık matrisini hesaplama
confusionMatrix = confusionmat(y_test, y_pred);

% Hassasiyet ve duyarlılık değerlerini hesaplama
precision = diag(confusionMatrix) ./ sum(confusionMatrix, 2);
recall = diag(confusionMatrix) ./ sum(confusionMatrix, 1)';

% Sonuçları ekrana yazdırma
disp("PCA-Karar Ağacı Modeli - Doğruluk Oranı: " + accuracy);
disp("PCA-Karar Ağacı Modeli - Hassasiyet: ");
disp(precision');
disp("PCA-Karar Ağacı Modeli - Duyarlılık: ");
disp(recall');
disp("PCA-Karar Ağacı Modeli - Karmaşıklık Matrisi: ");
disp(confusionMatrix);

