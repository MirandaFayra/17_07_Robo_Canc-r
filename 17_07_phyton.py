#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


camera = cv2.VideoCapture (0)

kernel= np.array ([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

while True:

# cada imagem que a camera captura
    _,frame = camera.read()
# para sair a imagem a imagem com o filtro
    saida = cv2.filter2D(frame, -1, kernel)
#  abrir janela com camera
# manter a ordem e não esquecer de chamar a saída 
    cv2.imshow ('Camera', saida)  
# para guardar a imagem da camera retorno da tecla apertada
    pressedKey = cv2.waitKey (1)
#     0xFF codigo hexadecimal 'q' fechar janela
    if pressedKey & 0xFF == ord ('q'):
        break
camera.release()
cv2.destroyAllWindows()


# In[14]:


# KNN 
# importação das bibliotecas
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
#IMPORT SKLEARN BIBLIOTECA DE AI MCL
from sklearn.datasets import load_breast_cancer


# In[15]:


dados = load_breast_cancer()
print (dados['DESCR'])


# ORGANIZANDO DADOS

# In[16]:


x = dados.data # variáveis preditoras
y = dados.target #variáveis de resposta


# In[20]:


print(y)


# In[18]:


print(x)


# In[34]:


# TREINAR O ALGORITIMO PARA TREINAR E TESTAR

#DIVIDIR A BASE PARA TER VARIAVÉIS PARA TESTAR AS RESPOSTAS
from sklearn.model_selection import train_test_split
# CRIAR AS VARIAVEIS DE TREINO E TESTE E RECORTAR - retorna uma lista, dividida entre as variáveis 
# (75% treino aprender e 25% teste)
xTreino, xTeste, yTreino, yTeste = train_test_split(x,y)


# In[36]:


xTreino.shape

CRIANDO UM MODELO
# In[37]:


from sklearn.neighbors import KNeighborsClassifier


# In[38]:


# MODELO PARA CLASSIFICAR OS VIZINHOS E CLASSIFICAR O TIPO DE CANCER
classf= KNeighborsClassifier(n_neighbors= 5)
#TREINANDO MODELO  COM OS DADOS DE TREINO - FIT -> TREINAR MODELO
classf.fit(xTreino,yTreino)


# In[39]:


# FAZER PREDIÇÃO A PARTIR DOS DADOS E COMPARAR COM A BASE
preditos= classf.predict(xTeste)


# In[40]:


print(preditos)


# In[43]:


# TESTAR A EFETIVIDADE DO ALGORITIMO -> ACURACIA -> TABELA CONFUSÃO -> sklearn.metrics

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[44]:


accuracy_score(yTeste, preditos)


# In[45]:


# verdadeiros positivos 54 verdadeiros negativos 80 -> y = respostas do gabarito
confusion_matrix(yTeste, preditos, labels=[0,1])


# In[47]:


# aumentando os vizinho e checar a acuracia
classf= KNeighborsClassifier(n_neighbors=10)
classf.fit(xTreino,yTreino)
preditos=classf.predict(xTeste)
print('Acurácia:', accuracy_score(yTeste,preditos))


# ACHANDO O MELHOR K -> NÚMERO DE VIZINHOS

# In[52]:


listaAc= []
listak = []

# REPETIÇÃO LISTA POR LISTA PARA AS REPETIÇÕES E VERIFICAÇÕES QUAL ESTÁ MAIS CERTO

for k in range (1,50):
    classf = KNeighborsClassifier(n_neighbors=k)
    classf.fit(xTreino,yTreino)
    preditos=classf.predict(xTeste)
    acuracia=accuracy_score (yTeste, preditos)
# soma numeros a lista de acuracia
    listaAc.append(acuracia)
    listak.append(k)
    


# In[53]:


listaAc= np.array(listaAc)


# In[54]:


# max -> maior valor na lista
print('Valor Máximo: ', listaAc.max())

# argmax -> indice da posição de onde está guardado o maior valor

print('Número K do valor máximo é ', listaAc.argmax()+1)


# In[56]:


plt.plot(listak, listaAc)


# EXERCÍCIO ; CRIAR UM MODELO DE CLASSIFICAÇÃO DE FLORES DO GÊNERO IRIS (SCIKIT-LEARN). 
# https://scikit-learn.org/stable/datasets/index.html, USANDO KNN
# 

# In[ ]:




