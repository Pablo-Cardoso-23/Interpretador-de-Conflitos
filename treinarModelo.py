import os
import tensorflow as tf
import numpy as np
from src.processamento.prepararDados import PrepararDados
from src.modelo.analisarEmocoes import CriadorModeloEmocoes

def iniciar_treinamento():
    print("[INFO] Iniciando o pipeline de Treinamento da IA...")
    
    preparador = PrepararDados()
    caminho_treino = "dados/raw/train" 
    
    if not os.path.exists(caminho_treino):
        print(f"[ERRO] A pasta {caminho_treino} não foi encontrada. Verifique se extraiu o ZIP corretamente.")
        return

    x_treino, y_treino = preparador.carregar_dataset(caminho_treino)
    
    # Construindo a Arquitetura da Rede Neural
    criador = CriadorModeloEmocoes()
    modelo = criador.construir_modelo()
    
    # Mostra um resumo do modelo no terminal
    modelo.summary()
    
    # Callbacks de Segurança (Boas práticas)
    if not os.path.exists("dados/processados"):
        os.makedirs("dados/processados")
        
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="dados/processados/melhor_modelo_emocoes.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    print("[INFO] Calculando balanceamento de pesos para emoções raras (ex: Nojo)")
    total_amostras = len(y_treino)
    num_classes = 7
    pesos_classes = {}

    for i in range(num_classes):
        # Conta quantas imagens existem desta emoção específica
        contagem = np.sum(y_treino == i)
        peso = total_amostras / (num_classes * (contagem + 1))
        pesos_classes[i] = peso
    
    print(f"[INFO] Pesos aplicados: {pesos_classes}")
    
    # Iniciar o Treinamento
    print("\n[INFO] O treinamento vai começar. Isso pode levar alguns minutos...")
    historico = modelo.fit(
        x_treino, y_treino,
        batch_size=64, 
        epochs=30,     
        validation_split=0.2, 
        callbacks=[checkpoint]
    )
    print("\n[SUCESSO] Treinamento concluído. Modelo salvo em 'dados/processados/melhor_modelo_emocoes.keras'!")

if __name__ == "__main__":
    iniciar_treinamento()