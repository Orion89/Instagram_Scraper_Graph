import asyncio
import os
from typing import List

import nest_asyncio
import pandas as pd
from browser_use import Agent, ChatGoogle
from dotenv import load_dotenv
from pydantic import BaseModel

from instagram_prompts import get_instagram_scrape_prompt

# Configuración para entornos interactivos (Notebooks)
nest_asyncio.apply()


# 1. Modelo de datos estructurado (según requerimientos del usuario)
class InstagramPost(BaseModel):
    user_name: str
    post_hashtags: List[str]
    likes_count: int
    post_link: str
    post_caption: str
    image_description: str


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
        task_prompt = get_instagram_scrape_prompt(
            hashtag=hashtag,
            n_results=n_results,
            user_email=self.user,
            user_password=self.password,
        )

        agent = Agent(
            task=task_prompt,
            llm=ChatGoogle(model=self.model_name, api_key=self.api_key),
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
        output_file = f"scrape_results_{hashtag}.csv"
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\nExtracción completada. {len(df)} posts guardados en {output_file}")
        print(df.head())
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
