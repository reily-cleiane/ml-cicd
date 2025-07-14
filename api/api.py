print('Preparando ambiente')
print('-- carregando as bibliotecas')

from fastapi import FastAPI, Query
import cloudpickle
import os
import pandas as pd
import wandb

# AFAZER: ver se precisa dessas linhas abaixo. Elas servem pra usar a pasta do wandb
#         no mesmo lugar que o pipeline usa. Se não colocar, vai criar a pasta wandb e a artifacts
#         no diretório que o script de inicialização da API for executado
import os
os.environ["WANDB_DIR"] = "/pipeline-machine-learning"
os.environ["WANDB_CACHE_DIR"] = "/pipeline-machine-learning/artifacts"

wandb.init(
    project="intencao-dialogar",
    job_type="selecao-modelo",
)

# AFAZER: considere remover esse condicional. Precisei colocar para poder rodar o modelo que
#         que treinei na minha máquina usando a versão do Python diferente da sua
def carregar_classificador(usar_wandb=True):
    if usar_wandb:
        print('---- baixando dados do Wandb')
        artefato = wandb.use_artifact('cleiane-projetos/intencao-dialogar/modelo:latest', type='modelo')
        diretorio_download = artefato.download()
        caminho_arquivo = os.path.join(diretorio_download, 'modelo_completo.pkl')
        with open(caminho_arquivo, 'rb') as arq:
            classificador = cloudpickle.load(arq)
    else:
        # AFAZER: considerar apagar esse arquivo também. Ele foi gerado usando Python 3.12.8
        with open('pipeline-machine-learning/modelo_completo.pkl', 'rb') as arq:
            classificador = cloudpickle.load(arq)
    return classificador

print('-- carregando classificador')
classificador = carregar_classificador()

print('Inicializando API')
controller = FastAPI()

print('-- definindo rotas')

@controller.get('/health/')
async def health():
    return {"status": "Executando"}

@controller.get('/help/')
async def health():
    return {"rotas_disponiveis": {
        "health":{
            "parametros": {},
            "descricao": "Retorna um objeto com o status da API, em caso de execução, no formato {'status': 'Executando'}"
        },
        "help":{
            "parametros": {},
            "descricao": "Retorna um objeto com a lista de rotas disponíveis, bem como os parâmetros esperados e suas descrições"
        },
        "intencao":{
            "parametros": {"texto": {"tipo": "string", "descricao": "Texto com o input do usuário a ter sua intenção classificada."}},
            "descricao": "Retorna um objetono formato {'intencao': 'valor'} com o resultado da classificação de intenção do input od usuário. Os valores podem ser: 'ping', 'out', 'inadeq', 'consulta' e 'doc'."
        }
    }}

@controller.get('/intencao/')
async def classificar_intencao(texto: str = Query(..., description="Texto para classificar")):
    try:
        resultado = classificador.predict(pd.Series([texto]))[0]
        return {"intencao": resultado}
    except Exception as e:
        return {"erro": str(e)}
    
print('Inicialização completa!')