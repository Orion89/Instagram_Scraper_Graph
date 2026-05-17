# Instagram Scraper Graph

A Python-based toolkit for scraping Instagram data and performing network analysis on hashtags to identify communities and relationships.

## Project Overview

This project provides a pipeline to:
1.  **Scrape**: Use Selenium and BeautifulSoup to collect Instagram posts associated with specific hashtags.
2.  **Analyze**: Process the text data, extract features, and build a network graph where nodes are hashtags and edges represent co-occurrence.
3.  **Visualize**: Generate interactive plots (using Plotly) and static visualizations (using Seaborn) to explore the hashtag network and identified communities.

## Core Components

### 1. InstagramScraper (`InstagramScraper.py` / `.ipynb`)
Handles data collection.
- `log_in()`: Authenticates with Instagram using Selenium.
- `get_links()`: Retrieves post URLs for a given hashtag.
- `get_data()`: Multi-threaded scraping of post details (caption, user, tags, location, etc.).

### 2. InstagramGraph (`InstagramGraph.py` / `.ipynb`)
Handles data processing and network analysis.
- `get_features()`: NLP-based feature extraction (supports translation).
- `select_data()`: Filtering and cleaning (e.g., removing bots/verified accounts).
- `build_graph()`: Constructs a NetworkX graph from hashtag co-occurrences.
- `plot_graph()` / `plot_community()`: Interactive visualizations of the hashtag network.

## Technologies
- **Scraping**: `selenium`, `beautifulsoup4`, `requests`
- **Data Analysis**: `pandas`, `numpy`, `scikit-learn`
- **NLP**: `spacy`, `langdetect`, `emoji`, `regex`
- **Network Analysis**: `networkx`, `more-itertools`
- **Visualization**: `plotly`, `seaborn`, `matplotlib`
- **Environment**: Designed for `Jupyter Notebook` (`ipython`, `tqdm`)

## Setup and Requirements

### Dependencies
Install the required libraries:
```bash
pip install pandas numpy selenium beautifulsoup4 requests tqdm spacy langdetect networkx plotly seaborn matplotlib scikit-learn emoji more-itertools regex
```

### WebDriver
Since this project uses Selenium, you must have a WebDriver (e.g., ChromeDriver) installed and accessible in your PATH.

### NLP Models
You may need to download a Spacy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage Workflow

1.  **Scraping**:
    ```python
    from InstagramScraper import InstagramScraper
    scraper = InstagramScraper()
    scraper.log_in()
    scraper.get_links(hashtag='nature', n=100)
    df = scraper.get_data()
    # Save to CSV
    df.to_csv('nature_posts.csv', index=False)
    ```

2.  **Analysis**:
    ```python
    from InstagramGraph import InstagramGraph
    graph = InstagramGraph(csv='nature_posts.csv')
    graph.get_features()
    graph.select_data()
    graph.build_graph()
    graph.plot_graph()
    ```

## Development Conventions
- The project is primarily designed for interactive use in Jupyter Notebooks.
- `InstagramScraper` requires valid Instagram credentials.
- Use the `.ipynb` files for step-by-step execution and visualization.

## General guidelines

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.
