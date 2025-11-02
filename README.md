# MobileNetV2-WebAPP
Aplicação Web para criação de dataset, treinamento e reconhecimento de imagens de objetos diversos nas categorias CONFORME e NÃO CONFORME

# Sistema de Reconhecimento por Imagem com Flask e MobileNetV2

## Descrição do Projeto

O sistema tem como objetivo realizar **reconhecimento de objetos** utilizando **aprendizado de máquina** com redes neurais baseadas na arquitetura **MobileNetV2**.  
Seu funcionamento está dividido em **três etapas principais**: Captura, Treinamento e Classificação.

### 1. Captura
O usuário define o objeto a ser reconhecido e fornece imagens de duas categorias:
- **Conforme:** exemplos corretos do objeto esperado.  
- **Não Conforme:** exemplos com defeitos ou imperfeições.  

As imagens são capturadas por uma câmera conectada ao computador que executa o sistema.

### 2. Treinamento
As imagens capturadas passam por **pré-processamento** (normalização e remoção de ruídos) e são utilizadas para treinar uma **rede neural artificial**.  
O modelo é avaliado com métricas de **acurácia**, **erro (loss)** e **F1-score**.

### 3. Classificação
Após o treinamento, o modelo é utilizado para **classificar novas imagens** capturadas pela câmera, determinando se pertencem à categoria **Conforme** ou **Não Conforme**.

A aplicação possui **interface web** desenvolvida com **Flask**, composta por quatro páginas:
- **Página Inicial:** ponto de navegação entre as demais páginas.  
- **Página de Captura:** exibe o feed da câmera e permite capturar imagens para cada categoria.  
- **Página de Treinamento:** realiza o treinamento do modelo e exibe métricas de desempenho.  
- **Página de Classificação:** utiliza o modelo treinado para classificar novas imagens em tempo real.

---

## Instalação e Execução

### 1. Clone o repositório
```bash
git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd SEU_REPOSITORIO
```

### 2. Crie e ative um ambiente virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
.env\Scriptsctivate        # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Execute a aplicação
```bash
flask run
```

A aplicação estará disponível em:  
**http://127.0.0.1:5000**

---

## Tecnologias Utilizadas

- **Python 3.x**  
- **Flask** – microframework web  
- **TensorFlow / Keras** – treinamento do modelo MobileNetV2  
- **OpenCV** – captura e processamento de imagens  
- **HTML / CSS / JavaScript** – interface web  

---

## Estrutura Simplificada do Projeto
├───app
│   ├───controllers
│   ├───models
│   ├───static
│   │   ├───css
│   │   ├───dataset
│   │   │   ├───Conforme
│   │   │   └───Naoconforme
│   │   └───images
│   ├───templates
├───Datasets Desenvolvidos
│   ├───Dataset Maçãs (10 50 e 100)
│   │   ├───Conforme10
│   │   ├───Conforme100
│   │   ├───Conforme50
│   │   ├───Naoconforme10
│   │   ├───Naoconforme100
│   │   └───Naoconforme50
│   ├───Dataset Maçãs (Resoluções)
│   │   ├───1080p
│   │   │   ├───Conforme
│   │   │   └───Naoconforme
│   │   ├───480p
│   │   │   ├───Conforme
│   │   │   └───Naoconforme
│   │   └───720p
│   │       ├───Conforme
│   │       └───Naoconforme
│   ├───Dataset Pregadores (Resoluções)
│   │   ├───1080p
│   │   │   ├───Conforme
│   │   │   └───Naoconforme
│   │   ├───480p
│   │   │   ├───Conforme
│   │   │   └───Naoconforme
│   │   └───720p
│   │       ├───Conforme
│   │       └───Naoconforme
│   └───Datasets Pregadores (10 50 e 100)
│       ├───Conforme10
│       ├───Conforme100
│       ├───Conforme50
│       ├───Naoconforme10
│       ├───Naoconforme100
│       └───Naoconforme50
```

---

## Autor

Desenvolvido por **Fernando Figueredo**  
E-mail: [feluiz.figueredo@gmail.com]  
GitHub: [https://github.com/fernando-figueredo](https://github.com/fernando-figueredo)
