# Fastbreak-AI-Developer-Challenge
Grace Smith

## Contents
- [Setup Instructions](#setup-instructions)
- [Architecture Decisions](#architecture-decisions)
- [Search Implementation Explanation](#search-implementation-explanation)
- [Brief Documentation](#brief-documentation)


## Setup Instructions
### Prerequisites
- Python (version >= 3.7, <3.13)
   - Installation of Python 3.12.10 can be found here: [https://www.python.org/downloads/release/python-31210/](https://www.python.org/downloads/release/python-31210/)
- Git
- Packages used can be found in requirements.txt

### Installation and Usage
1. **Clone the repository**
   ``` bash
   git clone https://github.com/gracevsmith/Fastbreak-AI-Developer-Challenge.git
   cd Fastbreak-AI-Developer-Challenge
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
2. **Running the code**
   When inside the .\Fastbreak-AI-Developer-Challenge\ filepath,
   ```bash
   python -3.12 main.py
   ```
   User will be prompted to type in password in order to use API keys.

## Architecture Decisions
1. **Data Integration**
   - Multi-Source Data Aggregation: Pulling data from APIs (sportsdata.io), knowledge bases (DBpedia), and handmade examples, then unifying them in Algolia
   - Embedding Caching Design: Pre-computing and caching entity embeddings to avoid expensive OpenAI API calls when processing the same dataset repeatedly
2. **Search and NLP Architecture**
   - Mutli-Modal Extraction: Combining classical NLP techniques (rule-based patterns, spaCy Named Entity Recognition) with semantic search using OpenAI embeddings
   - Hybrid Search Pipeline: Implementing multi-stage appraoch that uses both keyword matching and semantic similarity for template identification and parameter extraction
3. **Configuration and Deployment**
   - Conditional Data Loading: Initialization checks if data already exists in Algolia to avoid redudant uploads and API calls
   - Semantic Search Optimization: Configuring Algolia index with searchable attributes tailored for sports constraint parsing
4. **Performance Considerations**
   - Batch Processing: Entity embedding in batches of 200 for optimal API usage
   - Selective Embedding: Filter out invalid entities before expensive API calls
   - Memory Management: Pickle-based caching strategy for large embedding datasets
5. **API Rate Limit Management**
   - OpenAI API: Multiple caching layers to minimize embedding API calls
   - SportsData.io API: Batching of team data requests across multiple sports leagues
6. **Data Flow Architecture**
```
Data Flow Architecture:
───────────────────────────────────────────────────────────
User Input → Text Processing → Semantic Search → 
Template Classification → Parameter Extraction → 
Formatted Output → (Optional Feedback → New Data)
───────────────────────────────────────────────────────────
```
   - Feedback only incorporates if OpenAI embeddings are recalculated on modified Aloglia index (expensive; did not fully incorporate this into current code because of my API key's call limitations)


## Search Implementation Explanation
Hybrid Search Pipeline: Algolia with OpenAI Embeddings
1. Data Acquisition and Storage
   - Data Collection
      - Sports teams from Sportsdata.io API (NBA, NFL, MLB, etc.)
      - Venues from DBpedia SPARQL endpoint
      - Hendcrafted patterns for parameters to extract
      - Handcrafted template examples for classification task
   - Data Storage
      - Store all data to Algolia as structured search index
      - Create OpenAI word embeddings for all Algolia entities
      - Local caching of embeddings to minimize API costs
      - Algolia serves as primary data backup
2. Input Processing
   - NER: use spaCy NER to help identify teams, venues, networks, etc
   - N-gram generation: (1-gram) or (1-gram AND 2-grams) phrases extracted from user's prompt
        - n-gram number is user's choice
   - Context preservation: Include full query as context phrase
3. Semantic Parameter Extraction
   - Embedding generation: create OpenAI embeddings for all search phrases
   - Similarity matching: use cosine similarity of search phrase embedding against cached entity embeddings; return cached entity with largest cosine similarity
   - Confidence based results:
      - 0.8: Automatic acceptance
      - 0.6-0.8: Present top choice but show alternatives
      - < 0.6: Require user disambiguation
   - Parameter mapping: entities mapped to constraint parameters
4. Template Classification and Population
   - Embedding comparison: compare user prompt against template example embeddings
   - Best match selection: highest cosine similarity template selected
   - Confidence scores: template match confidence returned
   - Populate template: use extracted parameters to fill template slots
5. user Feedback
   - Interactive feedback: optional user correction system
   - Continuous Learning: feedback incorporated into Algolia index
        - Feedback only incorporates if OpenAI embeddings are recalculated on modified Aloglia index (expensive; did not fully incorporate this into current code because of my API key's call limitations)

## Brief Documentation
### Why I Chose Algolia/ Open AI
I chose Algolia to store my data because it is specifically designed with fast search results in mind, which is ideal for entity matching. It is less expense to lookup information because it utilizes neural hashing in its storage of data. It scales well with large datasets and is easy to use in Python.

I opted to use the OpenAI embeddings mainly for performance advantages. OpenAi's embeddings have 1563 dimensions, much larger than its competitors. It is skilled at capturing semantic meaning, and is trained on a less specilized dataset than Algolia, which is mostly used for e-commerce product searches rather than sports scheduling. OpenAI embeddings allowed me to create my own NLP pipeline using the embeddings, rather than using a company's blackbox method. Additionally, the way I've utilized OpenAI's embeddings in my code allows me to swap OpenAI's embeddings for another method's in the future without needing to restructure the code.

### Challenged Faced and How I Solved Them
The main challenge that I faced was a lack of data. I made up for this by creating handmade data and scraping the internet for data relevate to sports scheduling. I also added a feedback section to the code, which would essentially create a new entity from the user's prompt and the corrected filled-out template; in this way, the model creates new data the more it is used. Note that due to the way data is called in the version of the code I've uploaded to GitHub (called from files rather than making repeated API calls, which are expensive for the free version), the new data is not integrated. However, the code to integrate the data is functioning, and if one were to load the data through Algolia rather than the premade .pkl datafile, the new data would be incorporated.

Another challenge that I faced was knowing which words in an NLP prompt were associated with which labels/ parameters. This was a little difficult to figure out because there weren't any examples of a natural language prompt with extracted parameter outputs. I made educated guesses on the parameters that didn't seem very distinct. In the future, having some labeled data and further access to/ understanding of the core optimization model would vastly improve the extracted parameter portion of the code.

### Trade-offs I Considered
The two main trade-offs I considered were accuracy vs speed, accuracy vs expense (w.r.t API calls), and speed vs memory usage.

The n-grams case exemplifies of the accuracy-speed tradeoff. The more n-grams that are considered, the more likely the model is able to identify entities that should be extracted parameters. I limited the maximum amount of n-grams to 2, as most phrases we would need to consider can be captured in two words (e.g. venue: "Staples Center"). This has a small tradeoff in accuracy, as three word phrases like "Christmas Day games" are missed by the model but this is justified by significant runtime improvement.

When deciding to load the data from .pkl files instead of directly calling the Algolia and OpenAI APIs, all three tradeoffs were considered. Loading the data has less accuracy, because we are not incorporating the data from the user's feedback, and require more memory, but we decrease runtime and expensive API calls.
