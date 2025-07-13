import argparse
import wandb
import pathlib

def disponibilizar_dataset_treinamento(dataset: pathlib.Path, run: "wandb.run"):
    """Disponibiliza o dataset no repositório de artefatos

    Args:
        dataset (pathlib.Path): caminho do dataset
        run (wandb.run): The wandb run to log the artifact to.
    """

    artefato = wandb.Artifact(name="dataset_treinamento", type="dataset", description="Dataset de treinamento", metadata={"origem": str(dataset)})
    artefato.add_file(str(dataset))  # adiciona o arquivo original
    run.log_artifact(artefato)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_treinamento",
        type=pathlib.Path,
        default="./dataset-treinamento/dataset-treinamento.csv",
        help="Caminho para o dataset de treinamento",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    run = wandb.init(project='intencao-dialogar', job_type="preparacao-dataset-treinamento")
    disponibilizar_dataset_treinamento(args.dataset_treinamento, run)
    run.finish()


if __name__ == "__main__":
    main()