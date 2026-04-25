import os
import numpy as np
import tensorflow as tf
import plotly.express as px
from sklearn.metrics import confusion_matrix
from src.processamento.prepararDados import PrepararDados

def gerar_matriz_confusao():
    print("[INFO] Iniciando avaliação detalhada do modelo...")

    caminho_teste = "dados/raw/test"

    if not os.path.exists(caminho_teste):
        print(f"[ERRO] Pasta de teste não encontrada em {caminho_teste}.")
        print("Dica: Se você não tem a pasta 'test', mude o caminho para 'dados/raw/train' apenas para visualizar a matriz.")
        return
    
    preparador = PrepararDados()
    x_teste, y_teste = preparador.carregar_dataset(caminho_teste)

    if len(x_teste) == 0:
        return
    
    caminho_modelo = "dados/processados/melhor_modelo_emocoes.keras"
    if not os.path.exists(caminho_modelo):
        print("[ERRO] Modelo não encontrado. Treine a IA primeiro.")
        return
    
    print("[INFO] Carregando a Rede Neural...")
    modelo = tf.keras.models.load_model(caminho_modelo)

    print("[INFO] A IA está fazendo a prova final. Aguarde...")
    predicoes_brutas = modelo.predict(x_teste, verbose=1)

    # Transforma as probabilidades (ex: [0.1, 0.8, 0.1...]) no número final da emoção (ex: 1)
    y_pred = np.argmax(predicoes_brutas, axis=1)

    # 4. Construindo a Matriz de Confusão
    # O dicionário de emoções na ordem correta (0 a 6)
    labels_emocoes = ['Raiva', 'Nojo', 'Medo', 'Alegria', 'Neutro', 'Tristeza', 'Surpresa']
    matriz = confusion_matrix(y_teste, y_pred)

    fig = px.imshow(
        matriz,
        text_auto=True, 
        labels=dict(x="O que a IA respondeu", y="A Emoção Real", color="Nº de Imagens"),
        x=labels_emocoes,
        y=labels_emocoes,
        title="Matriz de Confusão: Avaliação de Acertos e Erros",
        color_continuous_scale="Blues"
    )

    fig.update_layout(xaxis_title_font=dict(size=14, weight='bold'), 
                      yaxis_title_font=dict(size=14, weight='bold'))
    
    caminho_html = "dados/processados/matriz_confusao.html"
    fig.write_html(caminho_html)

    print(f"\n[SUCESSO] Matriz gerada! Abra no seu navegador: {caminho_html}")

if __name__ == "__main__":
    gerar_matriz_confusao()
