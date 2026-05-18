import asyncio
import os
from typing import List

import nest_asyncio
import pandas as pd
from browser_use import Agent, ChatGoogle
from dotenv import load_dotenv
from pydantic import BaseModel

from prompts.instagram_prompts import (
    get_instagram_scrape_prompt,
    get_instagram_scrape_prompt_with_previous_posts,
)

# Configuración para entornos interactivos (Notebooks)
nest_asyncio.apply()


# 1. Modelo de datos estructurado (según requerimientos del usuario)
class InstagramPost(BaseModel):
    user_name: str
    post_hashtags: List[str]
    likes_count: int | None
    post_link: str
    post_caption: str
    image_description: str
    comments_count: int | None
    repost_count: int | None


class InstagramScrapeResults(BaseModel):
    results: List[InstagramPost]


class InstagramScraperV2:
    def __init__(self, model_name="gemini-flash-latest"):
        load_dotenv()
        self.user = os.getenv("USER_EMAIL")
        self.password = os.getenv("USER_PASSWORD")
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name

        if not all([self.user, self.password, self.api_key]):
            raise ValueError(
                "Faltan credenciales en el archivo .env (USER_EMAIL, USER_PASSWORD, GOOGLE_API_KEY)"
            )

    async def scrape_hashtag(self, hashtag: str, n_results: int = 10):
        """
        Ejecuta el agente para scrapear un hashtag específico.
        """
        # Comprobar si existen resultados previos para este hashtag
        data_path = os.path.join("data", f"scrape_results_{hashtag}.csv")
        previous_links = []

        if os.path.exists(data_path):
            try:
                df_prev = pd.read_csv(data_path)
                if "post_link" in df_prev.columns:
                    previous_links = df_prev["post_link"].unique().tolist()
            except Exception as e:
                print(f"Error al cargar resultados previos: {e}")

        if previous_links:
            print(
                f"Se encontraron {len(previous_links)} posts previos. Usando prompt con historial."
            )
            task_prompt = get_instagram_scrape_prompt_with_previous_posts(
                hashtag=hashtag,
                n_results=n_results,
                user_email=self.user,
                user_password=self.password,
                previous_posts=previous_links,
            )
        else:
            task_prompt = get_instagram_scrape_prompt(
                hashtag=hashtag,
                n_results=n_results,
                user_email=self.user,
                user_password=self.password,
            )

        agent = Agent(
            task=task_prompt,
            llm=ChatGoogle(model=self.model_name, api_key=self.api_key),
            use_thinking=True,
            output_model_schema=InstagramScrapeResults,
            use_vision=True,
        )

        print(f"Iniciando extracción de #{hashtag} ({n_results} posts)...")
        history = await agent.run()
        result_json = history.final_result()

        if result_json:
            parsed = InstagramScrapeResults.model_validate_json(result_json)
            df = pd.DataFrame([post.model_dump() for post in parsed.results])

            # Añadimos la columna de búsqueda para trazabilidad
            if not df.empty:
                df.insert(0, "searched_for", f"#{hashtag}")

            return df, history
        else:
            print("No se obtuvieron resultados finales.")
            return pd.DataFrame(), history


async def run_standalone_scrape(hashtag="ia", n=5):
    scraper = InstagramScraperV2()
    df, history = await scraper.scrape_hashtag(hashtag=hashtag, n_results=n)

    if not df.empty:
        # Carpeta de salida y ruta del archivo
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        output_file = os.path.join(data_dir, f"scrape_results_{hashtag}.csv")

        # Si ya existen resultados previos, los cargamos y concatenamos los nuevos
        if os.path.exists(output_file):
            print(f"Cargando resultados anteriores de {output_file}...")
            existing_df = pd.read_csv(output_file)
            df = pd.concat([existing_df, df], ignore_index=True)
            print(f"Nuevos resultados agregados. Total: {len(df)} posts.")

        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\nExtracción completada. Datos guardados en {output_file}")
        print(df.tail())
    else:
        print("\nLa extracción no generó datos.")

    # Mostrar métricas básicas de uso
    usage = history.usage
    print(f"\nTokens totales: {usage.total_tokens}")
    print(f"Pasos del agente: {len(history.history)}")


if __name__ == "__main__":
    # Soporte básico para argumentos por línea de comandos
    import sys

    target_hashtag = sys.argv[1] if len(sys.argv) > 1 else "tecnologia"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    asyncio.run(run_standalone_scrape(hashtag=target_hashtag, n=count))
