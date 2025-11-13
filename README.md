# Fastbreak-AI-Developer-Challenge
Sports Scheduling Constraint Parser

Creating the GitHub page for developer challenge

Explanation for choice of Search Implementations
| Method | Pros | Cons |
|--------| -----| -----|
| OpeanAI Embeddings + Vector Search | <ul><li>Considered very accurate wrt capturing semantic meaning (due to high-dimensional vector embeddings) </li><li> Compatible with Python </li></ul>| <ul><li>Pretrained on unknown data (introduces biases that will be hard to account for) </li><li> Generally computationally expensive </li></ul>|
| Algolia with AI Search | <ul><li> Semantic search built in (their method seems conceptually similar to vector embeddings) </li><li> Less expensive than most vector embedding methods (due to neural hashing that compresses vectors)</li><li>"Hybrid search" combines semantic and keyword search </li><li> Can supplement pretrained model with user's own data </li><li>Typo resistant</li></ul> | <ul><li> Pricing for more advanced versions </li><li> Unclear what data the pretrained model was trained on (possible biases) </li></ul>|
| Pinecone | <ul><li>Vector database (stores vector embeddings) so should capture semantic meaning well </li><li> Compatible with Python </li></ul> | <ul><li> Vector database doesn't compress data the way neural hashing does </li><li> Cost increase for larger projects </li></ul>  |
| Supabase pgvector | <ul><li> Stores vector embeddings so should capture semantic meaning well </li><li> Seems more computationally efficient than Pinecone/ Vector Database methods (likely because it's an extension of the Supabase SQL database, rather than a new/ separate database) </li></ul> | <ul><li> May be less accurate than Vector Database methods </li><li> Real-time updates with large amounts of data require reindexing that can be expensive/ timely (more true for IVFFlat, debately still true for HNSW) </li></ul>  |
| Alt1: LLM  | <ul><li> Can apply "context" better </li><li> Input immediately results in output</li><li> Can finetune with scheduling dataset easily </li></ul> | <ul><li> Cost/ efficiency scales with usage </li><li> Accuracy not guarenteed/ consistent (same prompt may get different outputs) </li></ul>  |
| Alt2: Traditional NLP Techniques (Name Entity Recognition, dependency parsing, rules, etc)  | <ul><li> </li><li>  </li></ul> | <ul><li> Rule-based approaches can perform poorly and are very case-dependent </li><li> No actual semantic understanding </li><li>Would scale very poorly </li></ul>  |
