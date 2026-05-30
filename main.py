import cv2
import numpy as np
from src.captura.cameraManager import CameraManager
from src.processamento.faceTracker import RastreamentoFacial
from src.processamento.dinamicaGrupo import AnalisadorDinamica
from src.modelo.inferencia import AnalisadorEmocoes
from src.processamento.registradorDados import RegistradorDados
from src.processamento.gerarDashboard import GeradorDashboard

def processar_ia_no_frame(frame, rastreador, analisador, cores_emocoes):
    """
    Função auxiliar para aplicar a IA de rastreamento e emoção em qualquer câmera.
    Retorna a lista de emoções encontradas naquele quadro específico.
    """
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostos = rastreador.processar_frame(frame)
    emocoes_frame = []
    
    for (x, y, w, h) in rostos:
        # Proteção contra recortes fora da tela
        if x < 0 or y < 0 or w == 0 or h == 0: 
            continue
            
        rosto_recortado = frame_cinza[y:y+h, x:x+w]
        emocao = analisador.prever_emocao(rosto_recortado)
        emocoes_frame.append(emocao)
        
        # Desenha as marcações no rosto
        cor = cores_emocoes.get(emocao, (0, 255, 0))
        cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
        cv2.putText(frame, emocao, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
        
    return emocoes_frame

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

    usar_multi_camera = False
    #url_celular = "http://192.168.1.11:8080/video"
    camera_index_2 = 1

    print("[INFO] Abrindo câmera principal...")
    with CameraManager(camera_index=0) as cam_notebook:
        
        # Se o usuário optou por câmera única, criamos um fluxo vazio para a externa
        if not usar_multi_camera:
            print("[INFO] Executando em modo de Câmera Única.")
            while True:
                sucesso_notebook, frame_notebook = cam_notebook.read()
                if not sucesso_notebook:
                    print("[AVISO] Falha ao capturar o frame da câmera interna. Encerrando...")
                    break

                frame_notebook = cv2.flip(frame_notebook, 1)
                emocoes_notebook = processar_ia_no_frame(frame_notebook, rastreador, analisador, cores_emocoes)
                
                texto_status, cor_status = dinamica.analisar_grupo(emocoes_notebook)
                registrador.registrar_frame(emocoes_notebook, texto_status)

                # Redimensiona para manter o padrão visual do painel
                frame_not_redim = cv2.resize(frame_notebook, (640, 480))
                cv2.putText(frame_not_redim, "CAM 1: INTERNAL", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # HUD Superior
                altura_barra = 40
                cv2.rectangle(frame_not_redim, (0, 0), (frame_not_redim.shape[1], altura_barra), (0, 0, 0), -1)
                cv2.putText(frame_not_redim, f"STATUS: {texto_status}", (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, cor_status, 2)

                cv2.imshow("Dinamica Social - Analise em Tempo Real", frame_not_redim)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Se configurado para multi-câmera, abre o segundo gerenciador de contexto
        else:
            print(f"[INFO] Conectando à câmera secundária (Índice/URL: {camera_index_2})...")
            with CameraManager(camera_index=camera_index_2) as cam_externa:
                while True:
                    sucesso_notebook, frame_notebook = cam_notebook.read()
                    sucesso_externa, frame_externa = cam_externa.read()

                    if not sucesso_notebook or not sucesso_externa:
                        print("[AVISO] Falha ao capturar frames de uma das câmeras. Encerrando...")
                        break

                    frame_notebook = cv2.flip(frame_notebook, 1)

                    emocoes_notebook = processar_ia_no_frame(frame_notebook, rastreador, analisador, cores_emocoes)
                    emocoes_externa = processar_ia_no_frame(frame_externa, rastreador, analisador, cores_emocoes)

                    emocao_grupo_total = emocoes_notebook + emocoes_externa
                    texto_status, cor_status = dinamica.analisar_grupo(emocao_grupo_total)
                    registrador.registrar_frame(emocao_grupo_total, texto_status)

                    frame_not_redim = cv2.resize(frame_notebook, (640, 480))
                    frame_ext_redim = cv2.resize(frame_externa, (640, 480))

                    cv2.putText(frame_not_redim, "CAM 1", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame_ext_redim, "CAM 2", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    painel_cftv = np.hstack((frame_not_redim, frame_ext_redim))

                    altura_barra = 40
                    cv2.rectangle(painel_cftv, (0, 0), (painel_cftv.shape[1], altura_barra), (0, 0, 0), -1)
                    cv2.putText(painel_cftv, f"STATUS GLOBAL DA SALA: {texto_status}", (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, cor_status, 2)

                    cv2.imshow("Dinamica Social - Analise em Tempo Real", painel_cftv)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    
    print("\n[INFO] Encerrando o sistema de vídeo...")
    registrador.exportar_csv()

    try:
        gerador = GeradorDashboard()
        gerador.gerar_relatorio_html()
        print("[INFO] Dashboard HTML gerado com sucesso!")
    except Exception as e:
        print(f"[ERRO] Falha ao gerar o Dashboard: {e}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_aplicacao()