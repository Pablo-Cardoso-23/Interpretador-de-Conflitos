class AnalisadorDinamica:
    """
    Módulo responsável por ler as emoções individuais de um grupo
    e inferir o clima/dinâmica social do ambiente em tempo real.
    """

    def analisar_grupo(self, emocoes_detectadas):
        total_pessoas = len(emocoes_detectadas)

        # Cenário 1: A sala está vazia
        if total_pessoas == 0:
            return "Aguardando Pessoas...", (150, 150, 150)
        
        # Cenário 2: Análise Individual (Fallback)
        if total_pessoas == 1:
            return f"Individuo Insolado ({emocoes_detectadas[0]})", (255, 255, 255)
        
        emocoes_unicas = set(emocoes_detectadas)

        # Alerta 1: Conflito/Tensão
        if "Raiva" in emocoes_unicas:
            return "ALERTA: Tensao/Conflito Detectado", (0, 0, 255)
        
        # Alerta 2: Alerta de Desconforto
        if "Medo" in emocoes_unicas or "Nojo" in emocoes_unicas:
            return "ALERTA: Desconforto no Ambiente", (0, 165, 255)
        
        # Todos na mesma sintonia
        if len(emocoes_unicas) == 1:
            emocao = emocoes_detectadas[0]
            if emocao == "Alegria":
                return "Sinergia Alta: Ambiente Positivo", (0, 255, 0)
            elif emocao == "Neutro":
                return "Coesao Alta: Foco/Atencao Coletiva", (255, 200, 0)
            elif emocao == "Tristeza":
                return "Coesao: Melancolia/Empatia Compartilhada", (255, 0, 0)
        
        contagem = {emo: emocoes_detectadas.count(emo) for emo in emocoes_unicas}
        emocao_predominante = max(contagem, key=contagem.get)

        return f"Dinamica Mista (Maioria: {emocao_predominante})", (0, 255, 255)
