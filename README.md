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
   - ``` bash
     User input --> Text Processing --> Semantic Search --> Template Classification --> Parameter Extraction --> Constraint Validation --> Formatted Output (--> Optional Feedback which becomes New Data)
   ```
        - Feedback only incorporated if OpenAI embeddings are recalculated on modified Aloglia index (expensive; did not fully incorporate this into current code because of my API key's call limitations)


## Search Implementation Explanation
Integrating Database features of Algolia with OpenAI Embeddings
PIPELINE Incorporating Algolia and OpenAI Embeddings:
1. Finding Data and Data Storage
   - Finding Data that will help identify parameters to extract
      - Scrape data from sports.io api for sports teams, DBpedia for venues
      - Handmake some NLP rule-based data to help extract all other parameters
      - Handmake some prompt that follow each template
   - Data Storage
      - Store all data to Algolia as structures database
      - Create OpenAI word embeddings using Algolia data
      - Cache Algolia data and word embeddings locally (mostly to avoid expense of calling APIs over and over again)
      - Algolia is backup with all data
2. Process Input; given a user prompt:
   - Use spaCy's built-in NER to tag entities (names, venues, networks, etc)
       - !! Could try to add a way to filter out n-grams that definitely aren't params (how to do it not rule based?)
   - Generate n-grams (1,2,3) to capture partial phrases of user's prompt
   - Include full query as context phrase
3. Semantic Search
   - Create embeddings for n-grams and full query
   - Do semantic search against cached embeddings (from step 1), return top 2 params & confidence score
   - If top 2 params differ and param_2 is not None:
      - If top_param Confidence > 0.8: Automatic acceptance
      - If top_param Confidence 0.6-0.8: Present top choice but show alternatives
      - If top_param Confidence < 0.6: Require user disambiguation
   - Populate "parameters" with parms and associated confidence scores
4. Template Classification
   - Compute embedding of full use query
   - Compare against template example embeddings, return best matching template and confidence score
5. Template Population with Extracted Paramterers
   - Use extracted parameters to populate the template fields
   - Check if extracted parameters make sense for the chosen template
   - Ask for feedback from user, incoportate feedback in form of a dictionary into Algolia index

## Brief Documentation
### My Approach to Semantic Seach

### Why I Chose Algolia/ Open AI

### Challenged Faced and How I Solved Them

### Trade-offs I Considered




Sports Scheduling Constraint Parser

Creating the GitHub page for developer challenge

Explanation for choice of Search Implementations
| Method | Pros | Cons |
|--------| -----| -----|
| OpeanAI Embeddings + Vector Search | <ul><li>Considered very accurate wrt capturing semantic meaning (due to high-dimensional vector embeddings) </li><li> Compatible with Python </li></ul>| <ul><li>Pretrained on unknown data (introduces biases that will be hard to account for) </li><li> Generally computationally expensive </li></ul>|
| Algolia with AI Search | <ul><li> Semantic search built in (their method seems conceptually similar to vector embeddings) </li><li> Less expensive than most vector embedding methods (due to neural hashing that compresses vectors)</li><li>"Hybrid search" combines semantic and keyword search </li><li> Can supplement pretrained model with user's own data </li><li>Typo resistant</li></ul> | <ul><li> Pricing for more advanced versions </li><li> Unclear what data the pretrained model was trained on (possible biases) </li></ul>|
| Pinecone | <ul><li>Vector database (stores vector embeddings) so should capture semantic meaning well </li><li> Compatible with Python </li></ul> | <ul><li> Vector database doesn't compress data the way neural hashing does </li><li> Cost increase for larger projects </li></ul>  |
| Supabase pgvector | <ul><li> Stores vector embeddings so should capture semantic meaning well </li><li> Seems more computationally efficient than Pinecone/ Vector Database methods (likely because it's an extension of the Supabase SQL database, rather than a new/ separate database) </li></ul> | <ul><li> May be less accurate than Vector Database methods </li><li> Real-time updates with large amounts of data require reindexing that can be expensive/ timely (more true for IVFFlat, debately still true for HNSW) </li></ul>  |
| Alt1: pretrained LLM (thinking RoBERTa)  | <ul><li> Can apply "context" better </li><li> Input immediately results in output</li><li> Can finetune with scheduling dataset easily </li></ul> | <ul><li> Need training data </li><li>  Cost/ efficiency scales with usage </li><li> Accuracy not guarenteed/ consistent (same prompt may get different outputs) </li></ul>  |
| Alt2: Traditional NLP Techniques (Name Entity Recognition, dependency parsing, rules, etc)  | <ul><li> Great for structured NLP inputs </li></ul> | <ul><li> Rule-based approaches can perform poorly and are very case-dependent </li><li> No actual semantic understanding </li><li>Would scale very poorly </li></ul>  |
















