import os
import glob
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class GeradorDashboard:
    """
    Módulo responsável por ler os relatórios de Dinâmica Social (CSV)
    e gerar Dashboards interativos em HTML utilizando Plotly.
    """
    def __init__(self, pasta_dados="dados/processados"):
        self.pasta_dados = pasta_dados

    def obter_relatorio_recente(self):
        """
        Busca o relatório .csv mais recente na pasta de dados.
        """
        arquivo_csv = glob.glob(os.path.join(self.pasta_dados, "*.csv"))
        if not arquivo_csv:
            return None
        
        # Retorna o arquivo com a data de modificação mais recente
        return max(arquivo_csv, key=os.path.getctime)
    
    def gerar_relatorio_html(self):
        arquivo_alvo = self.obter_relatorio_recente()

        if not arquivo_alvo:
            print("[ERRO] Nenhum relatório CSV encontrado. Rode o (main.py) primeiro.")
            return
        
        print(f"[INFO] Gerando Dashboard com base no arquivo: {arquivo_alvo}")
        
        df = pd.read_csv(arquivo_alvo)

        df_filtrado = df[df['Status_Sala'] != "Aguardando Pessoas..."]

        if df_filtrado.empty:
            print("[AVISO] O relatório só contém frames vazios (sem pessoas). Gráficos não gerados.")
            return
        
        # GRÁFICO 1: Distribuição do Clima - Tempo em que a sala ficou em cada clima
        contagem_status = df_filtrado['Status_Sala'].value_counts().reset_index()
        contagem_status.columns = ['Status_Sala', 'Contagem']

        fig_pizza = px.pie(
            contagem_status,
            values='Contagem',
            names='Status_Sala',
            title='Distribuição do Clima da Reunião/Sala',
            hole=0.4
        )

        # GRÁFICO 2: Linha do Tempo
        df_filtrado['Tempo_Segundos'] = range(len(df_filtrado))

        fig_linha = px.scatter(
            df_filtrado,
            x='Tempo_Segundos',
            y='Status_Sala',
            color='Status_Sala',
            title='Linha do Tempo: Oscilação da Dinâmica Social',
            labels={'Tempo_Segundos': 'Passagem do Tempo (Frames)', 'Status_Sala': 'Clima do Grupo'}
        )

        fig_linha.update_traces(mode='lines+markers')

        caminho_pizza = os.path.join(self.pasta_dados, "dashboard_pizza.html")
        caminho_linha = os.path.join(self.pasta_dados, "dashboard_linha_tempo.html")

        fig_pizza.write_html(caminho_pizza)
        fig_linha.write_html(caminho_linha)

        print(f"[SUCESSO] Dashboards gerados com sucesso!")
        print(f"-> Abra no seu navegador: {caminho_pizza}")
        print(f"-> Abra no seu navegador: {caminho_linha}")

if __name__ == "__main__":
    gerador = GeradorDashboard()
    gerador.gerar_relatorio_html()

