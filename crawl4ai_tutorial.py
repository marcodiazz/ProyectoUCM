import asyncio
import json
import os
from crawl4ai import AsyncWebCrawler, BM25ContentFilter, BrowserConfig, CacheMode, CrawlerRunConfig, DefaultMarkdownGenerator, JsonCssExtractionStrategy, LLMConfig, LLMExtractionStrategy, PruningContentFilter
from dotenv import load_dotenv

load_dotenv()

async def main():
    
    court_html_schema = """
        <div data-court="AA" class="schedule-court"><div class="courtName" data="1300" id="AA" tabindex="0">Centre Court<br>1:30pm</div><div class="schedule-content"><div data-match="2201" class="match" data-players="wta320760,null,wta320983,null"><div class="row"><div class="match-info header" colspan="3"><span class="event">Ladies' Singles</span> - <span class="round">Second Round</span></div><div class="scores header"></div></div><div class="row teams"><div></div><div class="schedule-team content"><span><span><div class="schedule-player player1 " data-player="wta320760" id="wta320760"><a class="name" aria-label="A. Sabalenka player profile" alt="A. Sabalenka player profile" href="/en_GB/players/overview/wta320760.html">A. Sabalenka</a><span class="nation">  </span><span class="seed">1</span></div></span></span></div><div class="versus content">v</div><div class="schedule-team content"><span><span><div class="schedule-player player1 " data-player="wta320983" id="wta320983"><a class="name" aria-label="M. Bouzkova player profile" alt="M. Bouzkova player profile" href="/en_GB/players/overview/wta320983.html">M. Bouzkova</a><span class="nation"> (CZE) </span><span class="seed"></span></div></span></span></div><div class="status content"></div></div><div class="row mobile"><div class="scores"></div><div class="status"></div></div></div></div></div>
    """
    
    
    llm_config = LLMConfig(
            provider="openai/gpt-4o-mini",
            api_token=os.getenv("OPENAI_API_KEY")
    )
    
    schema = JsonCssExtractionStrategy.generate_schema(
        html=court_html_schema,
        llm_config=llm_config,
        query="""From https://www.wimbledon.com/en_GB/scores/schedule/index.html, extract the match schedule from the Wimbledon page. 
        For each match, provide the following fields in the schema: date, time, court, event_type, match, result (result of the match), and status (status of the match). 
        If there are doubles, include all player names in the list.
        """,
    )
    
    print("Schema generado:", json.dumps(schema, indent=2, ensure_ascii=False))
    
    schema_extraction_strategy = JsonCssExtractionStrategy(schema=schema)
    
    browser_config = BrowserConfig(
        headless=True,  # Ejecutar en modo headless (sin interfaz gráfica)
    )
      
    # llm_extraction_strategy = LLMExtractionStrategy(
      
    #     instruction="""Extract the match schedule from the Wimbledon page. The schedule is in a table format with columns for date, time, court, and match details. Provide the schedule in a structured format.""",  
    #     extraction_type="schema",
    #     schema={
    #         "day": "string",
    #         "time": "string",
    #         "court": "string",
    #         "match": "string"
    #     },
    #     extra_args={
    #         "temperature": 0.0,  # Temperatura baja para respuestas más precisas
    #         "max_tokens": 4096,  # Límite de tokens para la respuesta
    #     },
    #     verbose=True
    # )
    
    #     # Configuración del crawler
    
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,  # No usar caché
        # markdown_generator = DefaultMarkdownGenerator(
        #     content_filter = PruningContentFilter(
        #         threshold=0.5,  # Umbral de filtrado del contenido
        #     ),  # Filtro de contenido BM25   
        # ),  # Generador de Markdown por defecto
        extraction_strategy=schema_extraction_strategy,
    )
    

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun(
           url="https://www.wimbledon.com/en_GB/scores/schedule/index.html",
           config=run_config
        )
      
        # for result in results:
        result = results[0]
        print("Contenido sin filtrar", result.markdown.raw_markdown)
        print("Length del contenido filtrado", len(result.markdown.fit_markdown))   
        data = json.loads(result.extracted_content)
        print("Datos extraídos:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    
if __name__ == "__main__":
    asyncio.run(main())