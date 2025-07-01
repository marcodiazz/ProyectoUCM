import asyncio
import json
import csv
import os
from dotenv import load_dotenv
from typing import List

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlResult, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from pydantic import BaseModel
from crawl4ai import LLMExtractionStrategy, LLMConfig

# Cargamos la API Key de OpenAI
load_dotenv()

# ==========
# Esquemas CSS para la extracción
# ==========

# Enlaces de ofertas en el listado
SCHEMA_LISTADO = {
    "baseSelector": "a.ij-OfferCardContent-description-title-link",
    "fields": [
        {"name": "href", "type": "attribute", "attribute": "href"},
        {"name": "titulo", "type": "text"}
    ]
}

# ==========
# Modelo de oferta para extracción LLM
# ==========

class JobOffer(BaseModel):
    title: str
    company: str
    description: str
    location: str
    requirements: List[str]
    technologies: List[str]
    salary: str
    modality: str
    languages: List[str]
    contract_type: str

# ==========
# Función 1: Extraer enlaces del listado
# ==========

async def get_offer_links(keyword="ingeniero software", location="madrid", pages=1):
    base_url = f"https://www.infojobs.net/jobsearch/search-results/list.xhtml?keyword={keyword}+{location}&page={{}}"
    links = []

    async with AsyncWebCrawler() as crawler:
        for page in range(1, pages + 1):
            url = base_url.format(page)
            print(f"🔎 Página {page}: {url}")
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                    extraction_strategy=JsonCssExtractionStrategy(schema=SCHEMA_LISTADO),
                    cache_mode=CacheMode.BYPASS,
                    scan_full_page=True
                )
            )
            items = json.loads(result.extracted_content or "[]")
            for item in items:
                href = item.get("href", "")
                full_url = "https:" + href if href.startswith("//") else href
                links.append({"titulo": item.get("titulo", "").strip(), "link": full_url})

    return links
# ==========
# Función 2: Extraer detalles con LLM usando el texto de la página
# ==========

async def extract_offer_details(url: str):
    llm_config = LLMConfig(
        provider="openai/gpt-4o-mini",
        api_token=os.getenv("OPENAI_API_KEY")
    )

    # No pasamos schema ni selectores, solo el texto completo de la página
    strategy = LLMExtractionStrategy(
        llm_config=llm_config,
        schema=JobOffer.model_json_schema(),
        extraction_type="schema",
        instruction="""
            Extrae los siguientes campos de la oferta de empleo a partir del texto completo de la página:
            {
                title (string),
                company (string),
                description (string),
                location (string),
                requirements (list[string]),
                technologies (list[string]),
                salary (string),
                modality (string),
                languages (list[string]),
                contract_type (string)
            }
            Para el salario, devuelve solo un valor numérico sin el símbolo €, si no está especificado, dejalo vacío. Si es un rango, devuelve el valor medio.
            El campo "modality" elige y pon solo una de estas opciones: "remoto", "presencial" o "híbrido". Si no está especificado, devuelve el campo vacío.
            Para el campo "languages", devuelve una lista de lenguajes mencionados, en español, (Ej: español, inglés) o devuelve el campo vacío si no se menciona ninguno.
            Si alguno campo no está presente, dejalo vacío.
            El resultado debe ser un JSON válido con estos campos.
        """,
        input_format="markdown", 
        apply_chunking=False,
        chunk_token_threshold=2000,
        overlap_rate=0.1,
        extra_args={"temperature": 0.1, "max_tokens": 2000}
    )

    config = CrawlerRunConfig(
        extraction_strategy=strategy,
        cache_mode=CacheMode.BYPASS,
        scan_full_page=True,
        wait_for="css:body",
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)
        # Imprimir el texto fuente de la página
        # Imprimir el HTML bruto recibido
        print("\n--- HTML BRUTO DE LA PÁGINA ---\n")
        if hasattr(result, 'html') and result.html:
            print(result.html[:10000])  # Solo los primeros 10000 caracteres
        elif hasattr(result, 'raw_html') and result.raw_html:
            print(result.raw_html[:10000])
        elif hasattr(result, 'content') and result.content:
            print(result.content[:10000])
        else:
            print('[No html/raw_html/content disponible]')
        print("\n--- FIN HTML BRUTO ---\n")
        if result.success:
            data = json.loads(result.extracted_content or "{}")
            if isinstance(data, list):
                return data[0] if data else {}
            return data
        else:
            return {}

# ==========
# Función 3: Flujo completo
# ==========

async def scrape_infojobs_to_csv(pages=3, output_file="100_ofertas_infojobs.csv"):
    print("🚀 Extrayendo enlaces de InfoJobs...")
    links = await get_offer_links(pages=pages)

    print(f"✅ {len(links)} enlaces extraídos.")
    ofertas = []

    for i, link in enumerate(links, 1):
        print(f"[{i}/{len(links)}] Procesando: {link['link']}")
        detalles = await extract_offer_details(link["link"])
        detalles["link"] = link["link"]
        ofertas.append(detalles)

    if not ofertas:
        print("❌ No se extrajo ninguna oferta.")
        return

    # Guardar en CSV
    # Orden de columnas deseado
    desired_order = [
        "title", "company", "description", "location", "requirements", "technologies", "salary", "modality", "languages", "contract_type", "error", "link"
    ]
    # Unir todas las claves presentes en todas las ofertas
    all_keys = {k for row in ofertas for k in row.keys()}
    # Añadir las que falten al final
    fieldnames = desired_order + [k for k in all_keys if k not in desired_order]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ofertas:
            writer.writerow({k: ", ".join(v) if isinstance(v, list) else v for k, v in row.items() if k in fieldnames})

    print(f"📁 Archivo guardado: {output_file}")

# ==========
# Ejecutar
# ==========

if __name__ == "__main__":
    asyncio.run(scrape_infojobs_to_csv(pages=6))
