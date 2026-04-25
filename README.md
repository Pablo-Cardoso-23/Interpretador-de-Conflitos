# Interpretador de Dinâmica Social via Visão Computacional

## Propósito do Projeto

O Interpretador de Dinâmica Social é um sistema de monitoramento e análise de interações humanas em tempo real. Utilizando técnicas avançadas de Visão Computacional e Aprendizagem Profunda (Deep Learning), o projeto visa identificar estados emocionais individuais e processar essas informações de forma agregada para inferir o clima social de um ambiente.

O principal objetivo é fornecer dados objetivos sobre o engajamento, a coesão e o nível de tensão em grupos, permitindo intervenções mais precisas em ambientes educacionais, reuniões corporativas ou análises de segurança e bem-estar.

## Funcionalidades Principais

* Rastreamento Facial Multi-Agente: Detecção simultânea de múltiplos rostos utilizando algoritmos de Haar Cascades via OpenCV.
* Classificação de Emoções em Tempo Real: Rede Neural Convolucional (CNN) treinada para identificar sete estados emocionais: Raiva, Nojo, Medo, Alegria, Neutro, Tristeza e Surpresa.
* Motor de Regras de Dinâmica Social: Algoritmo que interpreta a combinação de emoções do grupo para diagnosticar estados como Sinergia, Tensão, Conflito ou Atenção Coletiva.
* Logging Analítico: Registro automatizado de métricas em formato tabular (CSV) para auditoria e análise histórica.
* Dashboards Interativos: Geração de relatórios visuais utilizando Plotly para visualização de tendências e distribuição de clima social.

## Importância e Aplicabilidade

A análise manual de dinâmica de grupo é subjetiva e propensa a vieses humanos. Este projeto oferece uma camada de inteligência de dados que pode ser vital em diversos setores:

1.  Educação: Auxiliar professores a identificar queda de atenção ou frustração coletiva durante uma aula.
2.  Recursos Humanos: Avaliar o clima de reuniões e a recepção de novas propostas por parte da equipe.
3.  Segurança e Saúde: Detectar precocemente sinais de conflito intenso ou desconforto em locais públicos ou instituições de cuidado.

## Arquitetura Técnica

O projeto foi construído seguindo os princípios de Responsabilidade Única (SRP) e Código Limpo (Clean Code), dividido nos seguintes módulos:

* Captura: Gerenciamento de dispositivos de vídeo e fluxos de câmera.
* Processamento: Rastreamento facial e normalização de dados brutos.
* Modelo: Inferência via TensorFlow/Keras utilizando modelos salvos em formato HDF5/Keras.
* Analítico: Motores de regras sociais, registro de logs e geração de dashboards.

### Tecnologias Utilizadas

* Linguagem: Python 3.12
* Visão Computacional: OpenCV
* Inteligência Artificial: TensorFlow, Keras e Scikit-Learn
* Manipulação de Dados: Pandas e NumPy
* Visualização: Plotly

## Estrutura do Repositório

* src/captura: Módulos de interface com hardware de vídeo.
* src/processamento: Lógica de rastreamento facial e preparação de dados.
* src/modelo: Definição da arquitetura da rede neural e scripts de inferência.
* dados/raw: Diretório para armazenamento dos datasets originais (ex: FER2013).
* dados/processados: Local de saída para modelos treinados, logs e relatórios interativos.

## Como Executar

### Pré-requisitos
* Ambiente Virtual (venv) configurado com Python 3.12.
* Instalação das dependências listadas no arquivo de requisitos.

### Execução do Sistema
1.  Para treinar o modelo: Execute o script treinarModelo.py garantindo a presença do dataset no diretório correto.
2.  Para iniciar o monitoramento: Execute o script main.py.
3.  Para gerar relatórios: Após encerrar a sessão de monitoramento, execute o script gerarDashboard.py.

## Desenvolvimento e Autoria

Este projeto foi desenvolvido como parte de um estudo avançado em Inteligência Artificial e Visão Computacional, focando na aplicação prática de redes neurais para a resolução de problemas de interpretação de sinais e dinâmicas de grupo.
