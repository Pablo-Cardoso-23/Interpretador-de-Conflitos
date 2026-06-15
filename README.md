
---

# Interpretador de Dinâmica Social via Visão Computacional

## Propósito do Projeto

O Interpretador de Dinâmica Social é um sistema autónomo de monitorização e análise de interações humanas em tempo real. Operando estritamente sob o paradigma de Edge Computing (processamento local na borda, sem envio de imagens para a nuvem), o projeto garante Privacy by Design enquanto utiliza técnicas avançadas de Visão Computacional e Redes Neurais Convolucionais (CNNs) para identificar microexpressões faciais.

O principal objetivo é processar emoções individuais de forma agregada para inferir o clima social de um ambiente. A ferramenta fornece dados objetivos sobre o envolvimento, a coesão e o nível de tensão em grupos, permitindo intervenções baseadas em dados em salas de aula, ambientes corporativos e espaços públicos.

## Funcionalidades Principais

* **Pipeline Multi-Câmara Sincronizado:** Suporte para processamento simultâneo de múltiplos dispositivos de vídeo, permitindo a fusão de matrizes de câmaras físicas (USB/Embutidas) e fluxos de rede (Câmaras IP/Telemóveis via HTTP/RTSP).
* **Rastreamento Estocástico de Baixa Latência:** Deteção facial multi-agente otimizada utilizando Haar Cascades via OpenCV, garantindo alta taxa de quadros (FPS) e eficiência de CPU.
* **Inteligência Artificial Anti-Overfitting:** Rede Neural Convolucional construída de raiz (TensorFlow/Keras) com camadas de Data Augmentation, Batch Normalization e Dropout, mitigando vieses e penalizando o desbalanceamento de classes através de Class Weights.
* **Motor Heurístico de Dinâmica Social:** Algoritmo que funde o estado emocional de todos os agentes detetados para diagnosticar instantaneamente a atmosfera do ambiente (ex: Sinergia, Tensão, Foco, Conflito).
* **Heads-Up Display (HUD) Global:** Interface vetorial sobreposta ao vídeo em tempo real, gerando um painel estilo CFTV profissional.
* **Auditoria Visual (Snapshots):** Sistema de gravação automática com temporizador de cooldown (prevenindo estrangulamentos de I/O), que armazena os frames validados com bounding boxes para auditoria posterior.
* **Geração Autónoma de Analytics:** Criação automática de registos estruturados em .csv e compilação de Dashboards HTML interativos via Plotly no encerramento seguro da sessão.

## Importância e Aplicabilidade

A avaliação manual de dinâmica de grupo é falha, subjetiva e propensa a vieses cognitivos. Este projeto oferece uma camada de inteligência e telemetria vital para:

* **Educação:** Mapear quebras de atenção, envolvimento ou frustração coletiva durante as aulas.
* **Gestão Corporativa (RH):** Avaliar a recetividade de equipas a novas propostas e o clima de stress em reuniões.
* **Segurança Institucional:** Deteção precoce de anomalias comportamentais e focos de conflito intenso em locais fechados.

## Arquitetura Técnica

A aplicação foi desenhada respeitando a Clean Architecture e o princípio da Responsabilidade Única (SRP). O acoplamento entre hardware, matemática e interface foi isolado:

* `/src/captura`: Abstração de hardware e rotinas de leitura de buffer de vídeo.
* `/src/processamento`: Módulos de rastreamento geométrico, motor de dinâmica de grupo e estruturação de Dashboards.
* `/src/modelo`: Encapsulamento dos tensores da IA e scripts de inferência probabilística.
* `/dados`: Diretório protegido (.gitignored) para alocação de datasets originais, registos tabulares temporais, snapshots de auditoria e relatórios finais.

## Tecnologias e Dependências

* **Linguagem Base:** Python 3.12
* **Visão Computacional e Matrizes:** OpenCV (cv2) e NumPy
* **Deep Learning:** TensorFlow e Keras
* **Engenharia de Dados:** Pandas
* **Visualização Web:** Plotly

## Como Executar

**1. Preparação do Ambiente**
Certifique-se de que está a correr o sistema dentro de um ambiente virtual (`.venv`) com as dependências instaladas:

```bash
pip install -r requirements.txt

```

**2. Execução da Inferência em Tempo Real**
O sistema pode ser testado com câmara única ou em modo CFTV (Multi-câmara). Abra o ficheiro `main.py` e configure as variáveis na inicialização:

* `usar_multi_camera = False` (Testes locais apenas com a webcam do computador).
* `usar_multi_camera = True` (Ativa a procura pela câmara IP ou USB secundária).

Inicie o orquestrador:

```bash
python main.py

```

**3. Encerramento Seguro e Geração de Relatórios**

* Com a janela de vídeo em foco, **pressione a tecla `q**`.
* O sistema fará o teardown gracioso, desligando o hardware, guardando os registos de sessão no disco e compilando imediatamente o ficheiro de Dashboard HTML interativo. As imagens de auditoria estarão disponíveis na pasta `dados/capturas_emocoes/`.

---

*Desenvolvido como projeto aplicado de Ciência da Computação, Inteligência Artificial, focando na intersecção entre redes neurais convolucionais e análise comportamental.*