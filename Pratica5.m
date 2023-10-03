% ********** Luan Gomes Magalhães Lima - 473008                      ********** 
% ********** Tópicos Especiais em Telecomunicações 1 - Prática 5     **********

% Inicializações
clear all;
close all;
clc;

%% Extração de atributos
% Definir o número de classes e a quantidade de amostras por classe
num_classes = 3;
num_amostras = 15;
total_amostras = num_classes * num_amostras;

% Base de dados
base_dados = zeros(45, 7);

% Quantidade de atributos utilizados
num_atributos = 6;

% Vetores temporários que armazena as amostras de cada classe
vetor_temp1 = zeros(num_amostras, num_atributos); % Classe 1
vetor_temp2 = zeros(num_amostras, num_atributos); % Classe 2
vetor_temp3 = zeros(num_amostras, num_atributos); % Classe 3

% Leitura dos sinais de aúdio
% OBS: foram gravados 15 amostras para cada classe

% Laço para percorrer cada classe
for classe = 1:num_classes
    % Laço para percorrer cada amostra
    for amostra = 1:num_amostras
        % Define o nome do arquivo de áudio
        if classe == 1
            nome_arquivo = sprintf('um%d.mp3', amostra);
        elseif classe == 2
            nome_arquivo = sprintf('dois%d.mp3', amostra);
        elseif classe == 3
            nome_arquivo = sprintf('tres%d.mp3', amostra);
        end
        
        % Carregar o arquivo de áudio
        [sinal, fs] = audioread(nome_arquivo);

        % Sinal de aúdio estéreo, logo seleciona-se um dos canais
        sinal = sinal(:, 1);
        
        % Laço para povoar as matrizes temporárias de cada classe
        for j = 1 : 6
            switch j
                % 1º Atributo: Média
                case 1
                    if classe == 1
                        vetor_temp1(amostra, j) = mean(sinal);
                    elseif classe == 2
                        vetor_temp2(amostra, j) = mean(sinal);
                    elseif classe == 3
                        vetor_temp3(amostra, j) = mean(sinal);
                    end
                % 2º Atributo: Kurtose
                case 2
                    if classe == 1
                        vetor_temp1(amostra, j) = kurtosis(sinal);
                    elseif classe == 2
                        vetor_temp2(amostra, j) = kurtosis(sinal);
                    elseif classe == 3
                        vetor_temp3(amostra, j) = kurtosis(sinal);
                    end
                % 3º Atributo: Assimetria
                case 3
                    if classe == 1
                        vetor_temp1(amostra, j) = skewness(sinal);
                    elseif classe == 2
                        vetor_temp2(amostra, j) = skewness(sinal);
                    elseif classe == 3
                        vetor_temp3(amostra, j) = skewness(sinal);
                    end
                % 4º Atributo: RMS
                case 4
                    if classe == 1
                        vetor_temp1(amostra, j) = rms(sinal);
                    elseif classe == 2
                        vetor_temp2(amostra, j) = rms(sinal);
                    elseif classe == 3
                        vetor_temp3(amostra, j) = rms(sinal);
                    end
                % 5º Atributo: Valor Máximo
                case 5
                    if classe == 1
                        vetor_temp1(amostra, j) = max(sinal);
                    elseif classe == 2
                        vetor_temp2(amostra, j) = max(sinal);
                    elseif classe == 3
                        vetor_temp3(amostra, j) = max(sinal);
                    end
                % 6º Atributo: Valor Mínimo
                case 6
                    if classe == 1
                        vetor_temp1(amostra, j) = min(sinal);
                    elseif classe == 2
                        vetor_temp2(amostra, j) = min(sinal);
                    elseif classe == 3
                        vetor_temp3(amostra, j) = min(sinal);
                    end
            end
        end
    end
end

% Base de dados após a extração de características
base_dados = vertcat(vetor_temp1, vetor_temp2, vetor_temp3);

% Classes da base
classes = [ones(15, 1); 2*ones(15, 1); 3*ones(15, 1)];

% Base de dados final com as classes
base_dados = [base_dados, classes];

%% Criação do Classificador
% Manipulação da base de dados
% Entrada (features)
X = base_dados(:, 1 : end-1);

% Saída (targets)
y = base_dados(:, end);

% Parâmetros a serem testados por meio do grid search
kernel_functions = {'linear', 'RBF', 'polynomial1', 'polynomial2'}; % Função kernel
box_constraints = [10^-3, 10^-2, 10^-1, 1, 10, 100]; % Constante de relaxamento C
kernel_scales = [10^-3, 10^-2, 10^-1, 1, 10, 100]; % Escala do kernel

% Inicialização das variáveis que irão selecionar a melhor acurácia e o 
% melhor modelo
melhor_acuracia = 0;
melhores_parametros = struct();

% Laço para percorrer todos os possíveis hiperparâmetros
% Laço para a função kernel
for a = 1 : length(kernel_functions)
    kernel_function_atual = kernel_functions{a};

    % Consideração para a Função Kernel do tipo Polinomial
    if strcmp(kernel_function_atual, 'polynomial1')
        ordem_polinomio = 1;
    elseif strcmp(kernel_function_atual, 'polynomial2')
        ordem_polinomio = 2;
    end

    % Laço para a constante de relaxamento
    for b = 1 : length(box_constraints)
        box_constraint_atual = box_constraints(b);
        
        % Laço para a escala do kernel
        for c = 1 : length(kernel_scales)
            kernel_scale_atual = kernel_scales(c);

            % Implementação do K-Fold
            k_fold = 3;

            % Implementação da abordagem One-vs-One
            num_classes = 3;
            num_combinacoes = nchoosek(num_classes, 2);
            ind_combinacao = 1;
            
            % Vetor que acumula as somas das acurácias
            soma_acuracia = 0;
            
            % Percorrer as três situações possíveis
            % 1: (1, 2), 2: (1, 3), 3: (2, 3)
            for i = 1 : num_classes - 1
                for j = i + 1 : num_classes
                    indices = (y == i | y == j);

                    X_binario = X(indices, :); % Base atual
                    y_binario = y(indices); % Classes correspondentes

                    % Vetor de acurácia para cada K-Fold
                    acuracia_fold = zeros(k_fold, 1);

                    % Divisão dos dados em n amostras por fold
                    num_amostras_fold = length(X_binario);

                    % Função que implementa a Validação Cruzada (K-Fold)
                    cv = cvpartition(num_amostras_fold, 'KFold', k_fold);

                    % Classificação para cada K-Fold
                    for z = 1 : k_fold
                        % Divisão dos índices em teste e treino
                        ind_teste = cv.test(z);
                        ind_treino = cv.training(z);

                        % Conjunto de treinamento
                        X_train = X_binario(ind_treino, :);
                        y_train = y_binario(ind_treino);

                        % Conjunto de teste
                        X_test = X_binario(ind_teste, :);
                        y_test = y_binario(ind_teste);
                        
                        % Implementação do SVM
                        % Treinamento do classificador para os parâmetros atuais
                        % Verificar se a Função kernel é polinomial
                        if (kernel_function_atual == "polynomial1" || kernel_function_atual == "polynomial2")
                            kernel_function_atual = "polynomial";
                            svm = fitcsvm(X_train, y_train, 'KernelFunction', kernel_function_atual, 'BoxConstraint', box_constraint_atual, 'KernelScale', kernel_scale_atual, 'PolynomialOrder', ordem_polinomio);
                        else
                            svm = fitcsvm(X_train, y_train, 'KernelFunction', kernel_function_atual, 'BoxConstraint', box_constraint_atual, 'KernelScale', kernel_scale_atual);
                        end

                        % Classificação para combinação atual
                        y_predict = predict(svm, X_test);

                        % Acurácia do K-Fold atual
                        qtd_acertos = sum(y_predict == y_test);
                        acuracia_fold_atual = qtd_acertos/length(y_test);
                        acuracia_fold(z) = acuracia_fold_atual;
                    end

                    % Acurácia para combinação atual da base
                    acuracia_kfold = mean(acuracia_fold);

                    % Atualizar a combinação das bases
                    ind_combinacao = ind_combinacao + 1;
                    soma_acuracia = soma_acuracia + acuracia_kfold; 
                end
            end

            % Calculando a acurácia média para todas as combinações
            acuracia_media = soma_acuracia / num_combinacoes;
            
            % Verificando se a acurácia é a melhor encontrada até o momento
            if acuracia_media > melhor_acuracia
                melhor_acuracia = acuracia_media;
                melhores_parametros.kernel = kernel_function_atual;
                if melhores_parametros.kernel == "polynomial"
                    melhores_parametros.ordem = ordem_polinomio;
                end
                melhores_parametros.c = box_constraint_atual;
                melhores_parametros.kernel_scale = kernel_scale_atual;
            end
        end
    end
end

% Resultados obtidos
fprintf("Melhores parâmetros encontrados:\n");
disp(melhores_parametros);
fprintf("Acurácia Média do classificador: %.2f", melhor_acuracia*100);

% OBS: O tempo de processamento do código está muito alto, observa-se que
% quanto maior for o k do K-Fold, maior será esse tempo. Logo, utiliza-se um
% k bem pequeno. Para abordagem One-vs-One verifica-se a acurácia média a 
% partir das várias combinações possíveis.A base de dados influencia diretamente
% nos valores da acurácia.