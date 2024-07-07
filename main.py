import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os
import time
import matplotlib.pyplot as plt
from openai import OpenAI
import numpy as np
import random

# Configure sua chave de API da OpenAI
client = OpenAI(api_key='seu api key')

# Caminhos dos arquivos
MODEL_FILE = 'model.pkl'
DATA_FILE = 'data.csv'
CATEGORY_COUNT_FILE = 'category_count.csv'

# Função para carregar o dataset
def load_data():
    if os.path.exists(DATA_FILE):
        print("Carregando dados existentes")
        return pd.read_csv(DATA_FILE, delimiter=';')
    else:
        # Dataset inicial com um exemplo categorizado
        print("Carregando dados escritos")
        data = {
            'text': [
                'O time ganhou a partida de futebol',  # Esportes
                'O time ganhou a partida de futebol',  # Esportes
                'O time ganhou a partida de futebol',  # Esportes
                'O time ganhou a partida de futebol',  # Esportes
                'O time ganhou a partida de futebol',  # Esportes
                
            ],
            'category': ['Esportes', 'Esportes', 'Esportes', 'Esportes', 'Esportes']
        }
        df = pd.DataFrame(data)
        df.to_csv(DATA_FILE, index=False, sep=';')
        return df


# Função para carregar ou inicializar o contador de categorias
def load_category_count():
    if os.path.exists(CATEGORY_COUNT_FILE):
        category_count = pd.read_csv(CATEGORY_COUNT_FILE)
        if 'category' not in category_count.columns or 'count' not in category_count.columns:
            category_count = pd.DataFrame(columns=['category', 'count'])
    else:
        category_count = pd.DataFrame(columns=['category', 'count'])
    return category_count


# Função para salvar o contador de categorias
def save_category_count(category_count):
    category_count.to_csv(CATEGORY_COUNT_FILE, index=False)

# Função para incrementar o contador de categorias
def increment_category_count(category):
    global category_count
    if category in category_count['category'].values:
        category_count.loc[category_count['category'] == category, 'count'] += 1
    else:
        new_row = pd.DataFrame({'category': [category], 'count': [1]})
        category_count = pd.concat([category_count, new_row], ignore_index=True)
    save_category_count(category_count)

# Função para encontrar a categoria com maior quantidade de itens
def get_most_common_category():
    if category_count.empty:
        return None
    return category_count.loc[category_count['count'].idxmax(), 'category']

# Função para encontrar a categoria com menor quantidade de itens
def get_least_common_category():
    if category_count.empty:
        return None
    min_count = category_count['count'].min()
    least_common_categories = category_count[category_count['count'] == min_count]['category'].tolist()
    return random.choice(least_common_categories)

# Função para carregar o modelo
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        return make_pipeline(TfidfVectorizer(), MultinomialNB())

# Carregar o dataset e o modelo
df = load_data()
category_count = load_category_count()
model = load_model()

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.3, random_state=42)

# Treinar o modelo
model.fit(X_train, y_train)

# Salvar o modelo treinado
joblib.dump(model, MODEL_FILE)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo com diferentes métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)

print("===================== DADOS INICIAIS =====================\n")
print(f'Acurácia: {accuracy * 100:.2f}%')
print(f'Precisão: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')
print('Matriz de Confusão:')
print(conf_matrix)
print('Relatório de Classificação:')
print(classification_report(y_test, y_pred, zero_division=0))
print("\n===================== DADOS INICIAIS =====================")


# Validação cruzada
cv_scores = cross_val_score(model, df['text'], df['category'], cv=3)
print(f'Acurácia média na validação cruzada: {cv_scores.mean() * 100:.2f}%')


# Função para adicionar novos dados e re-treinar o modelo
def add_data(text, category):
    global df, model
    # Adicionar a nova entrada ao DataFrame
    new_data = pd.DataFrame({'text': [text], 'category': [category]})
    df = pd.concat([df, new_data], ignore_index=True)
    # Salvar o DataFrame atualizado no arquivo CSV
    df.to_csv(DATA_FILE, index=False, sep=';')
    # Incrementar o contador de categorias
    increment_category_count(category)
    # Re-dividir o conjunto de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.3, random_state=42)
    # Re-treinar o modelo com o conjunto de dados atualizado
    model.fit(X_train, y_train)
    # Salvar o modelo treinado
    joblib.dump(model, MODEL_FILE)
    # Fazer novas previsões no conjunto de teste
    y_pred = model.predict(X_test)
    # Reavaliar o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    
    # Imprimir as novas métricas
    print(f'Acurácia: {accuracy * 100:.2f}%')
    return accuracy * 100, report, conf_matrix

# Função para fazer perguntas ao modelo
def ask_question(question):
    prediction = model.predict([question])[0]
    return prediction

# Função para corrigir a resposta do modelo
def correct_answer(question, predicted_category, correct_category):
    if predicted_category != correct_category:
        #print(f'Corrigindo: {question} -> {correct_category}')
        add_data(question, correct_category)
    else:
        #print(f'Registrando: {question} -> {correct_category}')
        add_data(question, correct_category)


def get_chatgpt_response(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Você é um assistente de IA."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip().strip('"')


def verify_classification(question, predicted_category):
    verification_prompt = f"A pergunta '{question}' foi classificada como '{predicted_category}'. Isso está correto? Se não, qual seria a classificação correta? Diga somente o nome da categoria que se enquadra, sem explicações."
    response = get_chatgpt_response(verification_prompt)
    return response

def print_counters(df):
    num_categories = df['category'].nunique()
    total_items = len(df)
    print("\n=====> ML: Contadores")
    print(f"=====> ML: Categorias: {num_categories}")
    print(f"=====> ML: Itens registrados: {total_items}\n")

# Loop para gerar perguntas e obter respostas do ChatGPT automaticamente
def generate_and_classify_automatically():
    accuracies = []
    iterations = []
    plt.ion()  # Modo interativo ligado
    fig, ax = plt.subplots(figsize=(10, 6))  # Ajustar a altura da janela
    ax.set_xlabel('Iteração')
    ax.set_ylabel('Acurácia (%)')
    ax.set_title('Acurácia de Aprendizado ao Longo do Tempo')
    iteration = 0
    
    print("\n\n\n ========= INICIO INTERAÇÃO  ========= \n")
    while True:
        least_common_category = get_least_common_category()
        
        if least_common_category:
            prompt = f"Gere um texto sobre o tema {least_common_category} até 150 caracteres, não coloque o texto entre aspas duplas."
        else:
            prompt = "Gere um texto sobre qualquer tema até 150 caracteres, não coloque o texto entre aspas duplas."
        
        print(prompt)
        
        #prompt = "Gere um texto sobre qualquer tema até 150 caracteres, não coloque o texto entre aspas duplas."
        text = get_chatgpt_response(prompt)
        predicted_category = ask_question(text)

        prompt = f"O texto '{text}' foi classificado como '{predicted_category}'. Está correto? Digite apenas (s - para sim /n - para não). Caso a classificação esteja errada, não explique, apenas diga n"
        correct_response = get_chatgpt_response(prompt)

        if correct_response.lower() == 'n':
            print("## ERROU!")
            correct_category_prompt = f"Qual é a categoria correta para o texto '{text}'? escreva apenas a categoria, todas as letras maiusculas, sem aspas duplas nem ponto final, somente a palavra"
            correct_category = get_chatgpt_response(correct_category_prompt.strip().strip("."))
            print("Tema: "+correct_category)
            accuracy, report, conf_matrix = add_data(text, correct_category)
        else:
            print("## ACERTOU!")
            print("Tema: "+predicted_category)
            accuracy, report, conf_matrix = add_data(text, predicted_category)

        # Adicionando a acurácia à lista e atualizando o gráfico
        iteration += 1
        iterations.append(iteration)
        accuracies.append(accuracy)

        # Calculando a média acumulada das acurácias
        mean_accuracies = np.cumsum(accuracies) / np.arange(1, len(accuracies) + 1)

        ax.clear()
        ax.set_xlabel('Iteração')
        ax.set_ylabel('Acurácia (%)')
        ax.set_title('Acurácia de Aprendizado ao Longo do Tempo')
        ax.plot(iterations, accuracies)  # Remover label
        ax.plot(iterations, mean_accuracies, label='Média de Acurácia', linestyle='--')

        # Adicionar anotações apenas na última média de acurácias
        if len(mean_accuracies) > 0:
            ax.annotate(f'{mean_accuracies[-1]:.2f}%', (iterations[-1], mean_accuracies[-1]), textcoords="offset points", xytext=(0,10), ha='center')

        ax.legend()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        
        print_counters(df)
        print("\n ========= FIM INTERAÇÃO  =========")

        # Adicionando um timer de 30 segundos no fim do loop
        #print("=====> ML: Aguardando 30 segundos antes da próxima iteração...")
        #time.sleep(30)



# Iniciar o processo de geração e classificação automática
generate_and_classify_automatically()
