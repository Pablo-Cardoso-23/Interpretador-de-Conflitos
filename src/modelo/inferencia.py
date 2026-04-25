import os
import cv2
import numpy as np
import tensorflow as tf

class AnalisadorEmocoes:
    """
    Módulo responsável por carregar o modelo treinado e realizar
    predições em tempo real sobre imagens de rostos recortados.
    """
    def __init__(self, caminho_modelo="dados/processados/melhor_modelo_emocoes.keras"):
        if not os.path.exists(caminho_modelo):
            raise FileNotFoundError(f"[ERRO] Modelo não encontrado em {caminho_modelo}. Treine o modelo primeiro.")
        
        print("[INFO] Carregando a Rede Neural... Isso pode levar alguns segundos.")
        self.modelo = tf.keras.models.load_model(caminho_modelo)
        
        # Mapeamento reverso: transforma a saída numérica da IA (0 a 6) em texto legível
        self.dicionario_emocoes = {
            0: 'Raiva', 
            1: 'Nojo', 
            2: 'Medo', 
            3: 'Alegria', 
            4: 'Neutro', 
            5: 'Tristeza', 
            6: 'Surpresa'
        }
        
    def prever_emocao(self, rosto_recortado):
        """
        Recebe um recorte de rosto em tons de cinza, formata para a IA e retorna a emoção.
        """
        # 1. Padronização: Só entende imagens 48x48
        rosto_redimensionado = cv2.resize(rosto_recortado, (48, 48))
        
        # 2. Normalização e Formatação
        rosto_normalizado = rosto_redimensionado / 255.0
        
        # Adiciona as dimensões de "Lote" e "Canal" exigidas pelo Keras (1, 48, 48, 1)
        rosto_pronto = np.expand_dims(rosto_normalizado, axis=0)
        rosto_pronto = np.expand_dims(rosto_pronto, axis=-1)
        
        # 3. Predição: A rede devolve uma lista de probabilidades para as 7 emoções
        probabilidades = self.modelo.predict(rosto_pronto, verbose=0) # verbose=0 esconde os logs repetitivos
        
        # Pega o índice numérico (0 a 6) com a maior probabilidade
        indice_maior = np.argmax(probabilidades[0])
        
        # Traduz o número para a palavra em português
        emocao_detectada = self.dicionario_emocoes[indice_maior]
        
        return emocao_detectada