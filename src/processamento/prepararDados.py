import os
import cv2
import numpy as np

class PrepararDados:
    """
    Classe responsável por carregar, processar e normalizar imagens 
    para o treinamento da Rede Neural de Emoções.
    """
    def __init__(self, tamanho_imagem=(48, 48)):
        self.tamanho_imagem = tamanho_imagem
        self.mapa_emocoes = {
            'angry': 0, 'disgust': 1, 'fear': 2, 
            'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6
        }
    
    # Padronizado para o singular
    def carregar_dataset(self, caminho_arquivo):
        """
        Lê todas as imagens da pasta, redimensiona e converte para matrizes NumPy.
        """
        imagens = []
        rotulos = []

        print(f"[INFO] Carregando dados: {caminho_arquivo}...")

        for nome_emocao in os.listdir(caminho_arquivo):
            caminho_emocao = os.path.join(caminho_arquivo, nome_emocao)

            # CORREÇÃO 1: Validar se 'caminho_emocao' é uma pasta
            if not os.path.isdir(caminho_emocao):
                continue

            emocao_chave = nome_emocao.lower().split('_')[-1]
            if emocao_chave not in self.mapa_emocoes:
                continue

            label = self.mapa_emocoes[emocao_chave]

            # CORREÇÃO 2: Listar arquivos dentro da pasta específica da emoção
            for nome_imagem in os.listdir(caminho_emocao):
                caminho_imagem = os.path.join(caminho_emocao, nome_imagem)

                imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

                if imagem is not None:
                    # Para colocar o tamanho 48x48
                    imagem_redimensionada = cv2.resize(imagem, self.tamanho_imagem)
                    imagens.append(imagem_redimensionada)
                    rotulos.append(label)

        x_dados = np.array(imagens, dtype="float32")
        y_rotulos = np.array(rotulos)

        # Proteção: Só normaliza se realmente encontrou imagens
        # ... (código anterior da normalização) ...
        if len(imagens) > 0:
            x_dados = x_dados / 255.0
            x_dados = x_dados.reshape(x_dados.shape[0], self.tamanho_imagem[0], self.tamanho_imagem[1], 1)
            
            # --- NOVA IMPLEMENTAÇÃO: EMBARALHAMENTO (SHUFFLE) ---
            # Gera uma lista de índices (0, 1, 2, 3...) do tamanho do nosso dataset
            indices = np.arange(x_dados.shape[0])
            # Embaralha esses números aleatoriamente
            np.random.shuffle(indices)
            
            # Reorganiza as imagens e os rótulos usando a mesma ordem embaralhada
            x_dados = x_dados[indices]
            y_rotulos = y_rotulos[indices]
            # ----------------------------------------------------

        else:
            print("[ALERTA CRÍTICO] Nenhuma imagem foi encontrada para carregar.")

        print(f"[CONCLUÍDO] {len(imagens)} imagens carregadas, normalizadas e embaralhadas.")

        return x_dados, y_rotulos