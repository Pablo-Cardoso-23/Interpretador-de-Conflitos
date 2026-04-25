import os
import pandas as pd
from datetime import datetime

class RegistradorDados:
    """
    Módulo responsável por coletar o histórico de emoções e dinâmica
    em tempo real e exportar um relatório analítico estruturado.
    """
    def __init__(self):
        self.historico = []
        self.pata_destino = "dados/processados"

        if not os.path.exists(self.pata_destino):
            os.makedirs(self.pata_destino)

    def registrar_frame(self, emocoes_grupo, status_sala):
        """
        Salva o momento atual na memória RAM. É super rápido e não trava o vídeo.
        """
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_pessoas = len(emocoes_grupo)

        registro = {
            "Timestamp": agora,
            "Total_Pessoas": total_pessoas,
            "Emocoes_Detectadas": ", ".join(emocoes_grupo) if total_pessoas > 0 else "Nenhuma",
            "Status_Sala": status_sala
        }

        self.historico.append(registro)

    def exportar_csv(self):
        """
        Converte todo o histórico guardado em um DataFrame do Pandas e salva em disco.
        Este método só deve ser chamado quando a aplicação for encerrada.
        """
        if len(self.historico) == 0:
            print("[AVISO] Nenhum dado capturado para exportar.")
            return
        
        df = pd.DataFrame(self.historico)

        nome_arquivo = f"relatorio_dinamica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        caminho_completo = os.path.join(self.pata_destino, nome_arquivo)

        df.to_csv(caminho_completo, index=False, encoding='utf-8')
        print(f"\n[SUCESSO] Relatório da sessão salvo com sucesso em: {caminho_completo}")
        print(f"Total de registros (frames) analisados: {len(df)}")
