from loguru import logger

from app.data_loader import SparkDataLoader
from app.preprocessing import SparkPreprocessor


def main():
    logger.info("=" * 60)
    logger.info("BankGraphAI")
    logger.info("Inicialização concluída.")
    logger.info("=" * 60)

    # -------------------------
    # 1. LOAD SPARK
    # -------------------------
    logger.info("1. Leitura Spark")

    loader = SparkDataLoader(
        app_name="BankGraphAI",
        master="local[*]",
        shuffle_partitions=8,
    )

    df = loader.read("data/").df

    # -------------------------
    # 2. PREPROCESSAMENTO + CLUSTERS
    # -------------------------
    logger.info("2. Pré-processamento")

    preprocessor = SparkPreprocessor()
    df_processed = preprocessor.transform(df)

    logger.info("Pré-processamento concluído.")

    # -------------------------
    # 3. FINALIZA PIPELINE (SEM GRAFO)
    # -------------------------
    logger.info("Pipeline finalizado até estágio intermediário.")
    logger.info("Grafo será gerado sob demanda (Streamlit / botão).")

    return df_processed


if __name__ == "__main__":
    main()