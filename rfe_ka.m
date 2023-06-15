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

% Modelleme Aşaması (Karar Ağaçları)

% Makine öğrenmesi kütüphanesini yükle
addpath('libsvm');

% K-katlı çapraz doğrulama için parametreleri ayarla
k = 5; % K-fold değeri

% Veri setini eğitim ve test verisi olarak böl
cv = cvpartition(height(data), 'KFold', k);

% Doğruluk oranlarını depolamak için boş bir dizi oluştur
accuracies = zeros(k, 1);

% K-katlı çapraz doğrulama ile modeli değerlendir
for i = 1:k
    trainIdx = training(cv, i);
    testIdx = test(cv, i);

    % Eğitim verilerini ve hedef değişkenleri ayır
    X_train = normalizedData_selected_rfe(trainIdx, :);
    y_train = table2array(data(trainIdx, end));

    % Test verilerini ve hedef değişkenlerini ayır
    X_test = normalizedData_selected_rfe(testIdx, :);
    y_test = table2array(data(testIdx, end));

    % Karar ağacı modelini oluştur
    treeModel = fitctree(X_train, y_train);

    % Test verilerini kullanarak modeli değerlendir
    y_pred = predict(treeModel, X_test);

    % Doğruluk oranını hesapla
    accuracy = sum(y_pred == y_test) / numel(y_test);
    
    % Eğer tahminlerde sadece bir sınıf varsa doğruluk oranını 0 olarak ayarla
    if numel(unique(y_pred)) == 1
        accuracy = 0;
    end
    
    accuracies(i) = accuracy;
end

% Doğruluk oranlarının ortalamasını hesapla
meanAccuracy = mean(accuracies);

% Sonucu ekrana yazdır
disp("Karar Ağacı Modeli - Ortalama Doğruluk Oranı: " + meanAccuracy);


% Model Değerlendirme Aşaması

% Eğitim ve test verilerini yeniden ayırma
X_train = normalizedData_selected_rfe(trainIdx, :);
y_train = table2array(data(trainIdx, end));
X_test = normalizedData_selected_rfe(testIdx, :);
y_test = table2array(data(testIdx, end));

% Karar ağacı modelini tekrar eğitme
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
disp("Karar Ağacı Modeli - Doğruluk Oranı: " + accuracy);
disp("Karar Ağacı Modeli - Hassasiyet: ");
disp(precision');
disp("Karar Ağacı Modeli - Duyarlılık: ");
disp(recall');
disp("Karar Ağacı Modeli - Karmaşıklık Matrisi: ");
disp(confusionMatrix);

