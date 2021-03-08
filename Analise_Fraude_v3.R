### Projeto 1 - Detecção de Fraudes no Tráfego de Cliques em Propagandas ###
#                             de Aplicações Mobile                                              

## Optei por utilizar somente o dataset "train_sample.csv" por ter um tamanho que viabilizou o processamento 
# em minha máquina e para possibilitar a avaliação de desempenho do modelo


## Carregando os dados 
library(data.table)

df <- fread("train_sample.csv")

View(df)
str(df)


## Verificando valores NA
library(Amelia)

missmap(df, 
        main = "Análise de Fraude - Mapa de Dados Missing", 
        col = c("yellow", "black"), 
        legend = FALSE)

# Só existem valores NA na coluna attributed_time
# Confirmando que os valores NA somente existem nos casos em que não houve download do app (is_attributed igual a 0)
library(dplyr)

df%>%
  filter(is.na(attributed_time)) %>%
  summarise(n_distinct(is_attributed))   # somente 1 valor na variável is_attributed


## Convertendo a variável target para fator
df$is_attributed <- factor(df$is_attributed, levels = c("0", "1"), labels = c("No", "Yes"))


## Análise Exploratória
require(ggplot2)


# Análise da variável target is_attributed
round(prop.table(table(df$is_attributed))*100,1)
# Será necessário balancear o dataset, há pouquíssimos casos de download de app


# Analisando variável ip
df %>% 
  ggplot(aes(x =ip), binwidth = 30) + geom_histogram() +
  ylim(c(0, 10000)) +
  stat_bin(geom='text', aes(label=..count..), vjust = -1.5)+
  ggtitle("Number of clicks per ip")
# Existe concentração de clicks em alguns ips


# Analisando variável app
df %>% 
  ggplot(aes(x =app), binwidth = 20) + geom_histogram() +
  ylim(c(0, 60000)) +
  stat_bin(geom='text', aes(label=..count..), vjust = -1.5)+
  ggtitle("Clicks per App")
# Existe concentração de clicks em 2 apps


# Analisando variável device
df %>% 
  ggplot(aes(x =device), binwidth = 20) + geom_histogram() +
  ylim(c(0, 110000)) +
  stat_bin(geom='text', aes(label=..count..), vjust = -1.5)+
  ggtitle("Clicks per device type")
# Existe uma grande concentração de clicks em 1 tipo de device


# Analisando variável os
df %>% 
  ggplot(aes(x = os), binwidth = 20) + geom_histogram() +
  ylim(c(0, 70000)) +
  stat_bin(geom='text', aes(label=..count..), vjust = -1.5)+
  ggtitle("Device OS")
# Existe uma concentração em 2 os


# Analisando variável channel
df %>% 
  ggplot(aes(x = channel), binwidth = 30) + geom_histogram() +
  ylim(c(0, 15000)) +
  stat_bin(geom='text', aes(label=..count..), vjust = -1.5)+
  ggtitle("Clicks per channel id")
# Quase 50% dos clicks vieram de 5 canais


# Analisando variável click_time
table(year(df$click_time))
table(month(df$click_time))
# Amostra possui apenas dados do ano 2017 e mês novembro, então ano e mês não serão variáveis preditoras do modelo

require(lubridate)
table(day(df$click_time))
# A amostra contém apenas 4 dias do mês, então também não será considerada como variável preditora

df$hour <- hour(df$click_time)
hist(df$hour, labels = TRUE)
# A hora pode ser uma variável preditora relevante


## Selecionando as variáveis para a criação do modelo preditivo

df_final <- df[,c(1:5, 8:9)]
View(df_final)
str(df_final)


## Analisando a correlação
require(corrplot)

numeric_var <- sapply(df_final, is.numeric)
corr_matrix <- cor(df_final[,..numeric_var])
corrplot(corr_matrix, main="\n\nGráfico de Correlação para Variáveis Numéricas", method="number")
# device e os tem uma forte correlação


## Normalizando os dados
maxs <- apply(df_final[,..numeric_var], 2, max) 
mins <- apply(df_final[,..numeric_var], 2, min)

df_norm <- as.data.frame(scale(df_final[,..numeric_var], center = mins, scale = maxs - mins))
df_norm$is_attributed <- df_final$is_attributed
View(df_norm)


## Separando dados de treino e teste
library(caTools)
amostra <- sample.split(df_norm$ip, SplitRatio = 0.7)

# Dados de treino
train = subset(df_norm, amostra == TRUE)

# Dados de teste
test = subset(df_norm, amostra == FALSE)


## Balanceamento de classes
library(ROSE)

# Balanceando os dados de treino
train_b <- ROSE(is_attributed ~ ., data = train)$data
prop.table(table(train_b$is_attributed))

# Balanceando os dados de teste
test_b <- ROSE(is_attributed ~ ., data = test)$data
prop.table(table(test_b$is_attributed))


## Feature Selection
library(caret)
formula <- "is_attributed ~ ."
formula <- as.formula(formula)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
model_selection <- train(formula, data = train_b, method = "glm", trControl = control)
importance <- varImp(model_selection, scale = FALSE)
plot(importance)
# As variáveis mais relevantes indicadas pelo modelo foram ip, app e channel


## Treinando modelo de Regressão Logística com as variáveis mais relevantes
formula_selected <- "is_attributed ~ ip + app + channel"
formula_selected <- as.formula(formula_selected)
lr_model <- glm(formula = formula_selected, data = train_b, family = "binomial")
summary(lr_model)


## Testando o modelo de Regressão Logística com os dados de teste
lr_predictions <- predict(lr_model, test_b, type = "response") 
lr_predictions <- round(lr_predictions)
lr_predictions

lr_predictions <- ifelse(lr_predictions > 0.5, "Yes", "No")


## Avaliando o modelo Regressão Logística
misClasificError <- mean(lr_predictions != test_b$is_attributed)
print(paste('Acuracia', 1-misClasificError))

roc.curve(test_b$is_attributed, lr_predictions, plotit = T, col = "red")
# Acurácia 0.6504 e AUC 0.651


## Utilizando outro modelo - Naive Bayes
library(e1071)

nb_model <- naiveBayes(formula = formula_selected, data = train_b)
nb_predictions <- predict(nb_model, test_b) 

## Avaliando o modelo Naive Bayes
library(caret)
caret::confusionMatrix(test_b$is_attributed, nb_predictions, positive = 'Yes')
roc.curve(test_b$is_attributed, nb_predictions, plotit = T, col = "red")
# Acurácia 0.6469 e AUC 0.648 (não houve melhoria com relação ao modelo de Regressão Logística)


## Outro modelo - SVM
svm_model <- svm(formula = formula_selected, 
                     data = train_b, 
                     type = 'C-classification', 
                     kernel = 'radial') 
svm_predictions <- predict(svm_model, test_b) 


## Avaliando o modelo SVM
caret::confusionMatrix(test_b$is_attributed, svm_predictions, positive = 'Yes')
roc.curve(test_b$is_attributed, svm_predictions, plotit = T, col = "red")
# Acurácia 0.7393 e AUC 0.740 (melhor modelo)


## Outro modelo - Árvore de Decisão
library(C50)
tree_model <- C5.0(formula = formula_selected, data = train_b)
tree_predictions <- predict(tree_model, test_b)

## Avaliando o modelo Árvore de Decisão
caret::confusionMatrix(test_b$is_attributed, tree_predictions, positive = 'Yes')
roc.curve(test_b$is_attributed, tree_predictions, plotit = T, col = "red")
# Acurácia 0.7417 e AUC 0.743 (performance bastante parecida com o modelo SVM)



## Conclusão: escolheria o modelo Árvore de Decisão, por ter apresentado a melhor performance e processar mais rápido
# quando comparado com o modelo SVM, apresentou um desempenho parecido.

