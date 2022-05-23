from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

#dataset
dataset = load_breast_cancer()


#divide o dataset entre um dataset pra treino e outro para teste
treino_x, teste_x, treino_y, teste_y = train_test_split(dataset.data, dataset.target, test_size=0.2)

#cria e treina o algoritmo LinearSVC
linear = LinearSVC()
linear.fit(treino_x,treino_y)

#score
print('#### score ####')
resultado = linear.score(teste_x, teste_y)
print(resultado)

#resposta correta
print('### gabarito ###')
print(teste_y[20:30])

#resposta da IA
print('### previsões ###')
prev = linear.predict(teste_x[20:30])
print(prev)




