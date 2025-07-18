from fastapi import FastAPI, Query
import cloudpickle
import os
import pandas as pd
import wandb
from pydantic import BaseModel

os.environ["WANDB_CACHE_DIR"] = "./.wandb_cache"
wandb.init(
    project="intencao-dialogar",
    job_type="execucao-api",
)
# classificador = None

# def classificador():
#     global classificador
#     if classificador is None:
#         classificador = carregar_classificador()
#     return classificador

def carregar_classificador():
    artefato = wandb.use_artifact('cleiane-projetos/intencao-dialogar/classificador:latest', type='modelo')
    caminho = artefato.get_path("modelo_completo.pkl").download()
    with open(caminho, 'rb') as arq:
        classificador = cloudpickle.load(arq)
    return classificador
    
print('-- carregando classificador')
classificador = carregar_classificador()

print('Inicializando API')
controller = FastAPI(title="API de Classificação de Intenção para o DIALOGAR da ALRN", version="1.0")

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

class TextoInput(BaseModel):
    texto: str
class IntencaoOutput(BaseModel):
    intencao: str

@controller.post('/intencao/', response_model=IntencaoOutput, summary="Classificar intenção", description="Classifica a intenção do texto enviado.")
async def classificar_intencao(payload: TextoInput):
    try:
        resultado = classificador.predict(pd.Series([payload.texto]))[0]
        print(f"Dado: {payload.texto} . Classificação: {resultado}")
        return {"intencao": resultado}
    except Exception as e:
        return {"erro": str(e)}
    
print('Inicialização completa!')