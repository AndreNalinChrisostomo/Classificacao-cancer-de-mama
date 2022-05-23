from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

dataset = load_breast_cancer()

treino_x, teste_x, treino_y, teste_y = train_test_split(dataset.data, dataset.target, test_size=0.2)

linear = LinearSVC()
linear.fit(treino_x,treino_y)

#score
print('#### score ####')
resultado = linear.score(teste_x, teste_y)
print(resultado)

print('### gabarito ###')
print(teste_y[20:30])


print('### previsões ###')
prev = linear.predict(teste_x[20:30])
print(prev)




