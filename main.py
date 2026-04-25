import cv2
from src.captura.cameraManager import CameraManager
from src.processamento.faceTracker import RastreamentoFacial
from src.processamento.dinamicaGrupo import AnalisadorDinamica
from src.modelo.inferencia import AnalisadorEmocoes
from src.processamento.registradorDados import RegistradorDados

def iniciar_aplicacao():
    print("[INFO] Iniciando o sistema de Inferência em Tempo Real...")
    
    rastreador = RastreamentoFacial()
    analisador = AnalisadorEmocoes()
    dinamica = AnalisadorDinamica()
    registrador = RegistradorDados()

    cores_emocoes = {
        'Alegria': (0, 255, 0), # Verde
        'Raiva': (0, 0, 255), # Vermelho
        'Tristeza': (255, 0, 0), # Azul
        'Medo': (0, 165, 255), # Laranja
        'Nojo': (128, 0, 128), # Roxo
        'Surpresa': (255, 255, 0), # Ciano
        'Neutro': (255, 255, 255) # Branco
    }

    with CameraManager(camera_index=0) as camera:
        while True:
            sucesso, frame = camera.read()

            if not sucesso:
                print("[AVISO] Falha ao capturar o frame. Encerrando...")
                break

            # 1. Preparação da Imagem
            frame_espelhado = cv2.flip(frame, 1)
            frame_cinza = cv2.cvtColor(frame_espelhado, cv2.COLOR_BGR2GRAY)

            # 2. Visão Computacional (Detectar onde os rostos estão)
            rostos = rastreador.processar_frame(frame_espelhado)

            # Lista para armazenar todas as emoções alisadas
            emocao_grupo = []

            # 3. Integração com a IA de Emoções
            for (x, y, w, h) in rostos:
                # Proteção contra recortes fora da tela (evita crash se o rosto sair pela borda)
                if x < 0 or y < 0 or w == 0 or h == 0:
                    continue
                    
                # Extrai apenas o quadrado onde o rosto está (Imagem em tons de cinza)
                rosto_recortado = frame_cinza[y:y+h, x:x+w]
                
                # Pede para o cérebro analisar este recorte
                emocao = analisador.prever_emocao(rosto_recortado)
                emocao_grupo.append(emocao)
                
                # Desenha o resultado na tela (Substitui o método antigo do faceTracker)
                cor = cores_emocoes.get(emocao, (0, 255, 0))

                cv2.rectangle(frame_espelhado, (x, y), (x+w, y+h), cor, 2)
                cv2.putText(frame_espelhado, emocao, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
                
            # Análise do Ambiente
            texto_status, cor_status = dinamica.analisar_grupo(emocao_grupo)
            registrador.registrar_frame(emocao_grupo, texto_status)

            # Criar um HUD na tela
            altura_barra = 40
            cv2.rectangle(frame_espelhado, (0, 0), (frame_espelhado.shape[1], altura_barra), (0, 0, 0), -1)
            cv2.putText(frame_espelhado, f"STATUS DA SALA: {texto_status}", (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, cor_status, 2)

            # 4. Exibição
            cv2.imshow("Dinamica Social - Analise em Tempo Real", frame_espelhado)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    print("\n[INFO] Encerrando o sistema de vídeo...")
    #registrador.exportar_csv()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_aplicacao()