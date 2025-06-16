import os
import asyncio
import json
import pandas as pd
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, LLMExtractionStrategy

# Modelo Pydantic para la oferta
class JobOffer(BaseModel):
    title: str = Field(..., description="Título de la oferta")
    company: str = Field(..., description="Empresa")
    description: str = Field(..., description="Descripción del puesto")
    location: str = Field(..., description="Ubicación")
    requirements: str = Field(..., description="Requisitos")
    technologies: str = Field(..., description="Tecnologías mencionadas")
    salary: str = Field(..., description="Salario")
    modality: str = Field(..., description="Modalidad (remoto, presencial, híbrido)")


# PASO 1: Obtener URLs de ofertas desde una página de búsqueda
async def get_job_links(search_url: str, limit: int = 10) -> list[str]:
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=search_url,
            extraction_strategy=JsonCssExtractionStrategy(
                selector="a",  # Extraemos todos los links
                attributes=["href"],
                schema={"href": "str"}  # <-- Añadido schema requerido
            ),
            config=CrawlerRunConfig(
               exclude_external_images=True,

            )
        )
        raw_links = json.loads(result.extracted_content)
        urls = [link["href"] for link in raw_links if "/of-" in link["href"]]
        # Filtra duplicados, limita cantidad, completa enlaces relativos
        urls = list(set(urls))
        full_urls = ["https://www.infojobs.net" + url if url.startswith("/") else url for url in urls]
        return full_urls[:limit]


# PASO 2: Extraer datos de cada oferta con LLM
async def scrape_offer(url: str):
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=url,
                extraction_strategy=LLMExtractionStrategy(
                    provider="openai/gpt-4o",
                    api_token=os.getenv("AZURE_OPENAI_API_KEY"),
                    schema=JobOffer.model_json_schema(),
                    extraction_type="schema",
                    instruction="Extrae los campos definidos de esta oferta de empleo: título, empresa, descripción, ubicación, requisitos, tecnologías, salario y modalidad.",
                ),
                config=CrawlerRunConfig(
                    exclude_selectors=["script", "style", "svg", "iframe"],
                    # max_depth=0,
                )
            )
            return json.loads(result.extracted_content)
    except Exception as e:
        print(f"[ERROR] Fallo en {url}: {e}")
        return None


# PASO 3: Controlador principal
async def main():
    # URL de búsqueda (puedes cambiar la keyword)
    search_url = "https://www.infojobs.net/jobsearch/search-results/list.xhtml?keyword=cientifico+de+datos"

    print("📥 Obteniendo enlaces de ofertas...")
    offer_links = await get_job_links(search_url, limit=10)
    print(f"🔗 {len(offer_links)} enlaces encontrados")

    print("🤖 Haciendo scraping de cada oferta con LLM...")
    offers = await asyncio.gather(*(scrape_offer(url) for url in offer_links))
    offers = [o for o in offers if o]

    print("💾 Guardando resultados en CSV...")
    df = pd.DataFrame(offers)
    df.to_csv("ofertas_scrapeadas.csv", index=False)
    print("✅ Listo: 'ofertas_scrapeadas.csv'")

if __name__ == "__main__":
    asyncio.run(main())
