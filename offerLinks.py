import asyncio
import json
import os
from typing import List
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlResult, CrawlerRunConfig, LLMConfig, LLMExtractionStrategy
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from pydantic import BaseModel
import csv

# Esquema para la página de listado de ofertas
schema = {
    "baseSelector": "a.ij-OfferCardContent-description-title-link",
    "fields": [
        {
            "name": "href",
            "type": "attribute",
            "attribute": "href"
        },
        {
            "name": "titulo",
            "type": "text"
        }
    ]
}

# Esquema para la página de detalle de la oferta
schema_detalle = {
    "baseSelector": "main.ij-OfferDetailPage-main",
    "fields": [
        {"name": "titulo",
         "type": "text",
         "selector": "h1.ij-Heading-title1",
         },
        {"name": "empresa", 
         "type": "text", 
         "selector": "a.ij-Heading-headline2"
        },
        {"name": "ubicacion", 
         "type": "text", 
         "selector": "a.ij-Text-body1",
         },
        {"name": "modalidad", 
         "type": "text", 
         "selector": "p.ij-Text-body1",
         },
        {"name": "descripcion", 
         "type": "text", 
         "selector": "article.ij-Box",
         },
        {"name": "tecnologias", 
         "type": "text", 
         "selector": "a.sui-AtomTag-actionable",
         },

    ]
}

async def extract_infojobs_offer_links(num_pages=5):
    # base_url = "https://www.infojobs.net/jobsearch/search-results/list.xhtml?keyword=cientifico+de+datos+madrid&page={}" # 32 ofertas
    base_url = "https://www.infojobs.net/jobsearch/search-results/list.xhtml?keyword=ingeniero+software+madrid&page={}"  # Cambia la keyword según necesites
    all_rows = []
    seen_links = set()

    async with AsyncWebCrawler() as crawler:
        for page in range(1, num_pages + 1):
            url = base_url.format(page)
            print(f"Scrapeando página {page}: {url}")
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                   extraction_strategy=JsonCssExtractionStrategy(
                    schema=schema,
                    ),
                    cache_mode=CacheMode.BYPASS,
                    scan_full_page=True,
                    scroll_delay=0.5,
                )
            )
            if not result.extracted_content:
                print(f"❌ No se extrajo contenido en la página {page}.")
                continue
            extracted = json.loads(result.extracted_content)
            for item in extracted:
                href = item.get("href")
                titulo = item.get("titulo", "")
                if href:
                    url = "https:" + href if href.startswith("//") else href
                    if url not in seen_links:
                        all_rows.append({"titulo": titulo.strip(), "link": url})
                        seen_links.add(url)

    # Mostrar
    print("🔗 Enlaces encontrados:")
    for row in all_rows:
        print(f"{row['titulo']} -> {row['link']}")

    # Guardar los links y títulos básicos
    import csv
    with open("offer_links.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["titulo", "link"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✅ {len(all_rows)} enlaces guardados en offer_links.csv")

    # Ahora scrapea cada link para extraer los detalles
    print("\n⏳ Extrayendo detalles de cada oferta...")
    detailed_rows = []
    async with AsyncWebCrawler() as crawler:
        for i, row in enumerate(all_rows, 1):
            link = row["link"]
            print(f"[{i}/{len(all_rows)}] Scrapeando detalles: {link}")
            result = await crawler.arun(
                url=link,
                config=CrawlerRunConfig(
                    extraction_strategy=JsonCssExtractionStrategy(schema=schema_detalle),
                    cache_mode=CacheMode.BYPASS,
                    scan_full_page=True,
                    scroll_delay=0.5,
                )
            )
            if not result.extracted_content:
                print(f"❌ No se extrajo detalle para: {link}")
                continue
            detalle = json.loads(result.extracted_content)
            # Debug: mostrar el resultado bruto extraído
            print(f"Resultado extraído para {link}: {detalle}")
            if isinstance(detalle, list):
                if detalle:
                    detalle = detalle[0]
                else:
                    detalle = {}
            for k in detalle:
                if isinstance(detalle[k], list):
                    detalle[k] = "; ".join([str(x) for x in detalle[k] if x])
            detalle["link"] = link
            detailed_rows.append(detalle)

    # Guardar dataset completo
    if detailed_rows:
        fieldnames = list(detailed_rows[0].keys())
        with open("offer_links_full.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_rows)
        print(f"\n✅ Dataset completo guardado en offer_links_full.csv con {len(detailed_rows)} filas.")
    else:
        print("❌ No se extrajeron detalles de ninguna oferta.")

class JobOffer(BaseModel):
    title: str
    company: str
    description: str 
    location: str
    requirements: str
    technologies: list[str]
    salary: str
    modality: str
    languages: list[str]
    contract_type: str 


async def extract_offer_info(target_url: str):
    """Extrae los detalles de una oferta de InfoJobs."""
    print(f"Scrapeando oferta: {url}")

    # 1. Define the LLM extraction strategy
    llm_strategy = LLMExtractionStrategy(
        llm_config = LLMConfig(
            provider="openai/gpt-4o-mini",
            api_token=os.getenv("OPENAI_API_KEY"),
            
        ),
        schema=JobOffer.model_json_schema(),
        extraction_type="schema",
        instruction=
        """Extract from this job offer the following fields:
        {
            title (string),
            company (string),
            description (string),
            location (string),
            requirements (list[string]),
            technologies (list[string]),
            salary (string),
            modality (string),
            languages (list[string])
            contract_type (string).
        }

        The output should be a valid JSON object with these fields.
        If any field is not present, it should be set to an empty string or an empty list as appropriate.
        """,
        chunk_token_threshold=1200,
        overlap_rate=0.1,
        apply_chunking=True,
        input_format="markdown",
        extra_args={"temperature": 0.1, "max_tokens": 1000},
        verbose=True,
    )

    # 2. Build the crawler config
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
        scan_full_page=True,
        scroll_delay=1  # Adjust this if needed
    )

    async with AsyncWebCrawler() as crawler:
        # 4. Let's say we want to crawl a single page
        result = await crawler.arun(
            url=target_url,
            config=crawl_config
        )

        if result.success:
            # 5. The extracted content is presumably JSON
            data = json.loads(result.extracted_content)
            print("Extracted items:", data)

            # 6. Show usage stats
            llm_strategy.show_usage()  # prints token usage
        else:
            print("Error:", result.error_message)


async def extract_all_offers_to_csv(num_pages=5, csv_filename="ofertas_infojobs_llm.csv"):
    """Extrae todos los links y la información detallada de cada oferta y la guarda en un CSV."""
    base_url = "https://www.infojobs.net/jobsearch/search-results/list.xhtml?keyword=ingeniero+software+madrid&page={}"
    all_rows = []
    seen_links = set()

    async with AsyncWebCrawler() as crawler:
        for page in range(1, num_pages + 1):
            url = base_url.format(page)
            print(f"Scrapeando página {page}: {url}")
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                   extraction_strategy=JsonCssExtractionStrategy(
                    schema=schema,
                    ),
                    cache_mode=CacheMode.BYPASS,
                    scan_full_page=True,
                    scroll_delay=0.5,
                )
            )
            if not result.extracted_content:
                print(f"❌ No se extrajo contenido en la página {page}.")
                continue
            extracted = json.loads(result.extracted_content)
            for item in extracted:
                href = item.get("href")
                titulo = item.get("titulo", "")
                if href:
                    url = "https:" + href if href.startswith("//") else href
                    if url not in seen_links:
                        all_rows.append({"titulo": titulo.strip(), "link": url})
                        seen_links.add(url)

    print(f"\n✅ {len(all_rows)} enlaces extraídos.")

    # 2. Extraer detalles de cada oferta usando LLM
    ofertas_detalladas = []
    for i, row in enumerate(all_rows, 1):
        link = row["link"]
        print(f"[{i}/{len(all_rows)}] Extrayendo detalles con LLM: {link}")
        try:
            detalle = await extract_offer_info_return_dict(link)
            print(f"✅ Detalles extraídos para {link}: {detalle[-1]}")
            if detalle:
                # Si es lista, tomar el primer elemento
                if isinstance(detalle, list):
                    if detalle:
                        detalle = detalle[-1]
                    else:
                        detalle = {}
                detalle["link"] = link
                ofertas_detalladas.append(detalle)
        except Exception as e:
            print(f"❌ Error extrayendo detalles para {link}: {e}")

    if not ofertas_detalladas:
        print("❌ No se extrajeron detalles de ninguna oferta.")
        return

    # 3. Guardar en CSV
    fieldnames = list(ofertas_detalladas[0].keys())
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for oferta in ofertas_detalladas:
            row = {col: oferta.get(col, "") if not isinstance(oferta.get(col, ""), list) else ", ".join(oferta.get(col, [])) for col in fieldnames}
            writer.writerow(row)
    print(f"\n✅ Archivo CSV guardado como {csv_filename} con {len(ofertas_detalladas)} ofertas.")

# Helper para que extract_offer_info devuelva dict
async def extract_offer_info_return_dict(target_url: str):
    """Extrae los detalles de una oferta de InfoJobs y devuelve un dict."""
    llm_strategy = LLMExtractionStrategy(
        llm_config = LLMConfig(
            provider="openai/gpt-4o-mini",
            api_token=os.getenv("OPENAI_API_KEY"),
        ),
        schema=JobOffer.model_json_schema(),
        extraction_type="schema",
        instruction="""Extract from this job offer the following fields: { title (string), company (string), description (string), location (string), requirements (list[string]), technologies (list[string]), salary (string), modality (string), languages (list[string]), contract_type (string) }. The output should be a valid JSON object with these fields. If any field is not present, it should be set to an empty string or an empty list as appropriate.""",
        chunk_token_threshold=1200,
        overlap_rate=0.1,
        apply_chunking=True,
        input_format="markdown",
        extra_args={"temperature": 0.1, "max_tokens": 1000},
        verbose=False,
    )
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
        scan_full_page=True,
        scroll_delay=1
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=target_url,
            config=crawl_config
        )
        if result.success:
            data = json.loads(result.extracted_content)
            return data
        else:
            print(f"Error: {result.error_message}")
            return None

if __name__ == "__main__":
    # asyncio.run(extract_infojobs_offer_links(num_pages=1))
    url = "https://www.infojobs.net/madrid/ingeniero-software/of-i64a87a0eb948dd8bc2f4858b574610?applicationOrigin=search-new&page=1&sortBy=RELEVANCE"
    # asyncio.run(extract_offer_info(url))
    # Para lanzar el flujo completo y guardar en CSV:
    asyncio.run(extract_all_offers_to_csv(num_pages=1, csv_filename="ofertas_infojobs_llm.csv"))