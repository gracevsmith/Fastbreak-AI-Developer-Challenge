## import packages needed
import spacy
import re
import time
import os
import pickle
from dataclasses import fields
from typing import Dict, List
from algoliasearch.search.client import SearchClientSync
from openai import OpenAI
import numpy as np
import Parameters_to_Extract



class PreMadeSportsExtractor:
    def __init__(self, algolia_app_id, algolia_api_key, openai_api_key,
                 openai_model_for_embeddings="text-embedding-3-small", cache_path="entity_embeddings",
                 n_gram = False):
        
        self.extracted_Parameters = Parameters_to_Extract.Extracted_Parameters()
        self.client = SearchClientSync(app_id=algolia_app_id, api_key=algolia_api_key)
        self.index_name = "premade_sports_knowledge"

        ## load spacy english dict for NER
        self.nlp = spacy.load("en_core_web_sm")

        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = openai_model_for_embeddings
        self.cache_path = cache_path
        self.n_gram = n_gram
        

        ## Map Algolia types to Extracted_Parameters fields
        self.type_mapping = {
            "team": "teams",
            "venue": "venues",
            "network": "networks",
            "game_type": "games",
            "round": "rounds",
            "round_range": "rounds",
            "status": "home_away_bye_active",
            "location": "home_away_bye_active",
            "constraint": "min_val",
            "quantifier": "each_or_all",
            "pattern": "each_or_all",
            "matchup": "matchups",
            "bye": "byes"
        }

        ## Map template placeholder to Extracted_Parameters
        self.field_mapping = {
            "<min>": "min_val",
            "<max>": "max_val", 
            "<k>": "k",
            "<m>": "m",
            "<each_or_all>": "each_or_all",
            "<teams>": "teams",
            "<games>": "games",
            "<venues>": "venues", 
            "<networks>": "networks",
            "<rounds>": "rounds",
            "<round1>": "round1",
            "<round2>": "round2",
            "<home_away_bye_active>": "home_away_bye_active"
        }

        self.template_display_names = {
            "game_scheduling_constraints": "Template 1: Game Scheduling Constraints",
            "sequence_constraints": "Template 2: Sequence Constraints", 
            "team_schedule_pattern_constraints": "Template 3: Team Schedule Pattern Constraints"
        }

        self.template_examples = {
            "game_scheduling_constraints": [
                "Ensure that at least <min> and at most <max> games from <games or matchups or byes> are scheduled across <rounds> and played in any venue from <venues> and assigned to any of <networks>.",
                "Schedule all rivalry games on a weekend on ESPN",
                "Don't schedule high profile games on a weekday",
                "At least 2 of UTN@VU, ALA@AU, MSU@UM should all be scheduled on the final 2 dates of the season and on either CBS or ESPN",
                "Make sure UTN, UK, USC, LSU do not have any weekday byes",
                "Schedule at least two, but no more than four, of the top-10 ranked team matchups in the first five weeks. These must be held in Eastern Conference stadiums and broadcast on either ABC, NBC, or FOX.",
                "We need to avoid overloading the schedule with divisional byes. Please ensure only one or two Pacific Division teams have a bye in either week 10 or 11.",
                "For the holiday slots on Christmas and New Year's, feature at least three of these marquee matchups: Lakers at Celtics, Knicks at Heat, Warriors at Bulls. All should be on their home courts and televised by either ESPN or TNT.",
                "Do not schedule any team for a West-Coast-to-East-Coast game on a weekday (Monday through Thursday).",
                "The season opener in Week 1 must be hosted by a previous Super Bowl city and broadcast exclusively on NBC.",
                "In the final three weeks of the season, prioritize our classic rivalries. At least four of our designated 'protected rivalry' games should be scheduled during this period.",
                "Ensure that neither of the two potential playoff preview games (e.g., #1 vs #2 seeded teams from the previous year) are scheduled for the same weekend."
            ],

            "sequence_constraints": [
                "Ensure at least <min> and at most <max> cases where there is a sequence <games or matchups or byes>, <games or matchups or byes>, ... across rounds <round1>, <round2>.",
                "Make sure Oregon, Washington, UCLA, USC do not play at home on either side of their bye week",
                "Make sure Penn State plays at UCLA and at USC in back-to-back weeks",
                "For Oregon, Washington, UCLA, and USC, avoid scheduling a home game immediately before or immediately after their bye week.",
                "Penn State must have a back-to-back road trip to play at UCLA and at USC during the second half of the season.",
                "No team should ever have a sequence of three consecutive away games.",
                "After the All-Star break, make sure no NBA team has a brutal stretch of five away games within a seven-night period.",
                "We want to create a compelling narrative for the new expansion team. Please schedule their first three games as a sequence of home, away, home.",
                "For teams traveling from the West Coast to the East Coast, avoid scheduling an early (1 PM ET) game the week after a Monday Night Football game.",
                "In the final four weeks, ensure there is at least one instance where two top-tier teams play each other in consecutive weeks."
            ],

            "team_schedule_pattern_constraints": [
                "Ensure that <each of/all> teams in <teams> have at least <min> and at most <max> instances where they play at least <k> and at most <m> <home/away/bye/active> games across <rounds> where the game is assigned to any of <networks> and played in any venue from <venues>.",
                "No cases of 3 games in 3 nights for any NBA team",
                "No cases of 5 away games in 7 nights after the all star break",
                "At most 2 cases of 3 away games in 4 rounds for Western Conference teams",
                "No team in the league should ever be scheduled for three games in three nights.",
                "After the All-Star break, no Western Conference team should have more than two instances where they play three away games in a four-game stretch.",
                "Each team in the Southeast Division must have at least one, but no more than two, homestands of at least three consecutive games during the first half of the season.",
                "Make sure all of the California-based NBA teams (Lakers, Clippers, Warriors, Kings) have at least one instance of a three-game home stand broadcast on either TNT or ESPN.",
                "For any team that has a five-game road trip, do not schedule a Monday night game immediately upon their return.",
                "Every team must have a bye week scheduled between weeks 5 and 12.",
                "Limit the number of times any team has back-to-back games against opponents who were playoff teams from the previous season to a maximum of three instances."            ]
        }

        self.template_embeddings = {
            name: [self._get_openai_embedding(text) for text in examples]
            for name, examples in self.template_examples.items()
        }

        self.entities_w_embeddings = self.load_or_build_entity_embeddings()
        

    def load_or_build_entity_embeddings(self, batch_size=200):
        """
        Load embeddings from cache or build them using OpenAI API.
        Uses batching, caching, and retries on rate-limit errors.
        """

        ## If the files exist in the git (which they should rn),
            ## load and merge files
        part_aa = f"{self.cache_path}.pkl.part_aa"
        part_ab = f"{self.cache_path}.pkl.part_ab"
        combined = "entity_embeddings_combined.pkl"
        
        with open(combined, 'wb') as outfile:
            for part in [part_aa, part_ab]:
                with open(part, 'rb') as infile:
                    outfile.write(infile.read())
    
        # Now try loading the combined file
        with open(combined, 'rb') as f:
            return pickle.load(f)


        print("No cache found. Getting all entities from Algolia…")

        ## -----------------------
        ## THIS LOADS DATA IF IT DOESN'T EXIST
        ## BUT I'M CLOSE TO MY QUOTA SO DON'T ALLOW FOR NOW
        ## -----------------------

        ## If we don't have values saved, calc them

        all_entities = []
    
        def agg(batch):
            all_entities.extend(batch.hits)


        try:
            self.client.browse_objects(
                index_name=self.index_name, 
                aggregator=agg,
                browse_params={"hitsPerPage": 1000}  ## max per Algolia batch
            )

        except Exception as e:
            print(f"Error getting entities from Algolia: {e}")
            return []


        print(f"Got {len(all_entities)} total entities")

        all_entities_dicts = [dict(e) for e in all_entities]

        ## Generate embeddings with OpenAI
        print("Embedding entities...")
        ## Extract names for embedding
        
        # names = [e.get("name") for e in all_entities_dicts]

        texts = []
        valid_entries = []

        for entity in all_entities_dicts:
            text = self._get_embeddings_text(entity)
            if text and isinstance(text, str) and text.strip():      # last condition makes it so we ignore any text that's just whitespace
                texts.append(text.strip())
                valid_entries.append(entity)

            else:
                print(f"Warning: Skipping entity with no embeddable text: {entity.get('objectID', 'unknown')}")



        print(f"Computing embeddings for {len(texts)} valid entities (filtered out {len(all_entities_dicts) - len(valid_entries)} invalid)")
        print(f"Computing embeddings in batches of {batch_size}")

        all_entities_dicts = valid_entries

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            # DEBUG: Check batch content
            print(f"Batch {i//batch_size + 1} types: {[type(x) for x in batch]}")
            print(f"Batch {i//batch_size + 1} sample: {batch[:3]}")
            
            # Check for None, empty strings, or non-string values
            problematic = [item for item in batch if not isinstance(item, str) or not item.strip()]
            if problematic:
                print(f"Problematic items: {problematic}")
                # Skip or clean this batch
                batch = [item for item in batch if isinstance(item, str) and item.strip()]

            if not batch:  # Skip if batch is empty after cleaning
                print(f"Batch {i//batch_size + 1} empty after cleaning, skipping...")
                continue

            success = False
            while not success:
                try:
                    response = self.openai_client.embeddings.create(
                        model = self.embedding_model,
                        input = batch
                    )

                    batch_embeddings = [np.array(item.embedding) for item in response.data]

                    # Attach embeddings to entities
                    for j, emb in enumerate(batch_embeddings):
                        all_entities_dicts[i+j]["embedding"] = emb

                    print(f"Batch {i//batch_size + 1} embedded successfully ({len(batch)} items)")
                    success = True

                except Exception as e:
                    print(f"Embedding batch failed {e}. Retrying in 10 seconds")
                    time.sleep(10)
            

        with open(self.cache_path, "wb") as f:
            pickle.dump(all_entities_dicts, f)
            print(f"Saved embeddings to {self.cache_path}")

        return all_entities_dicts

    def _get_embeddings_text(self, entity):
        """
        Get the appropriate text to embed for each entity type
        """

        entity_type = entity.get("type")
        
        if entity_type == "team":
            return entity.get("name")  # "Los Angeles Lakers"
        elif entity_type == "venue":
            return entity.get("name")  # "Madison Square Garden"
        elif entity_type == "quantifier":
            return entity.get("pattern") or entity.get("name")  # "at least <min>"
        elif entity_type in ["event_category", "temporal_range", "network", "game_state"]:
            return entity.get("name")  # "rivalry games", "ESPN", "home games"
        elif entity_type == "temporal_pattern":
            return entity.get("name")  # "back-to-back weeks"
        else:
            return entity.get("name") or entity.get("pattern") or f"{entity_type} {entity.get('category', '')}".strip()


    def _get_openai_embedding(self, text: str) -> np.ndarray:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)


    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


    def _add_value(self, params, field: str, value: str):
        """
        Adds a value to params (helps w multiples i.e. 2 teams mentioned)
        """
        current = getattr(params, field)
        if current is None:
            setattr(params, field, [value])
        else:
            if value not in current:
                current.append(value)



    def _semantic_search_multi(self, phrase: str, top_k: int = 2):
        """
        Semantic search returning top K parameter matches with confidence scores
        """
        phrase_vec = self._get_openai_embedding(phrase)
        matches = []

        ## Compare embeddings against cached entities
        for entity in self.entities_w_embeddings:
            if "embedding" not in entity:
                continue

            similarity = self._cosine_similarity(phrase_vec, entity["embedding"])
            
            matches.append({
                "entity": entity,
                "similarity": similarity,
                "phrase": phrase,
                "mapped_field": self.type_mapping.get(entity.get("type"))
            })


        matches.sort(key = lambda x: x["similarity"], reverse = True)
        top_matches = matches[:top_k]

        if top_matches[0]["similarity"] > 0.8:
            return {
                "entity": top_matches[0]["entity"], 
                "phrase": top_matches[0]["phrase"],
                "mapped_field": top_matches[0]["mapped_field"],
                "confidence": top_matches[0]["similarity"]
            }


        elif top_matches[0]["similarity"] >= 0.6 and top_matches[1] is not None:
            return [{
                "entity": top_matches[0]["entity"], 
                "phrase": top_matches[0]["phrase"],
                "mapped_field": top_matches[0]["mapped_field"],
                "confidence": top_matches[0]["similarity"]},
                {
                "entity": top_matches[1]["entity"], 
                "phrase": top_matches[1]["phrase"],
                "mapped_field": top_matches[1]["mapped_field"],
                "confidence": top_matches[1]["similarity"]},
            ]


        return 

    def _extract_numeric_constraints(self, user_prompt: str) -> Dict[str, any]:
        """
        Extract numeric values that follow constraint words (rule-based)
        """
        constraints = {
            "min_val": None,
            "max_val": None, 
            "k": None,
            "m": None
        }
        
        # Patterns for constraint extraction
        patterns = [
            (r"at least (\d+)", "min_val"),
            (r"at most (\d+)", "max_val"), 
            (r"no more than (\d+)", "max_val"),
            (r"between (\d+) and (\d+)", ["min_val", "max_val"]),
            (r"(\d+) or more", "min_val"),
            (r"(\d+) or fewer", "max_val"),
            (r"exactly (\d+)", ["min_val", "max_val"]),  # min = max
            (r"(\d+) games in (\d+) nights", ["k", "m"]),  # 3 games in 5 nights
            (r"(\d+) games in (\d+) rounds", ["k", "m"]),
        ]
        
        for pattern, field in patterns:
            matches = re.finditer(pattern, user_prompt.lower())
            for match in matches:
                if isinstance(field, list):
                    # Multiple captures (e.g., "between X and Y")
                    for i, f in enumerate(field):
                        if i < len(match.groups()):
                            constraints[f] = int(match.group(i + 1))
                else:
                    # Single capture
                    constraints[field] = int(match.group(1))
        
        return constraints

    def _process_semantic_matches(self, all_matches):
        """
        Processes semantic matches and populates parameters with possible entries
        Handles single match and multiple matches (if low confidence)
        """

        params = self.extracted_Parameters
        params_confidence = {}
        alt_interpretations = {}

        ## Grouping matches by parameter field
        field_matches = {}

        for matches in all_matches:
            
            ## if one output
            if isinstance(matches, dict):
                matches_to_process = [matches]
            ## if two outputs (low confidence)
            elif isinstance(matches, list):   
                matches_to_process = matches
            else:
                continue

            for single_match in matches_to_process:
                field = single_match.get("mapped_field")
                if field and hasattr(params, field):
                    if field not in field_matches:
                        field_matches[field] = []
                    field_matches[field].append(single_match)

            
        for field, matches in field_matches.items():
            if not matches:
                continue

            # Sort by confidence for this field
            matches.sort(key=lambda x: x["confidence"], reverse=True)
            best_match = matches[0]
            
            # Get the value from the entity
            entity = best_match["entity"]
            value = entity.get("name") or entity.get("pattern") or entity.get("description", "")
            
            # Add to parameters
            self._add_value(params, field, value)
            params_confidence[field] = best_match["confidence"]
            
            # If there are multiple matches for this field, store alternatives
            if len(matches) > 1:
                alt_interpretations[field] = [
                    {
                        "value": alt_match["entity"].get("name") or alt_match["entity"].get("pattern"),
                        "confidence": alt_match["confidence"],
                        "source_phrase": alt_match["phrase"]
                    }
                    for alt_match in matches[1:]  # All alternatives
                ]  
        
        return {
            "parameters": params,
            "confidences": params_confidence,
            "alternatives": alt_interpretations,
            "raw_matches": all_matches
        }



    def extract(self, user_prompt: str):
        """
        1. Process input
        2. Semantic search
        3. Assign word to parameter
        """

        ## STEP 1: Process Input
        search_phrases = self.process_input(user_prompt)

        ## STEP 2: Semantic Search
        all_matches = []

        for phrase in search_phrases:
            matches = self._semantic_search_multi(phrase, top_k = 2)
            all_matches.append(matches)

        # Process matches
        result = self._process_semantic_matches(all_matches)
        numeric_constraints = self._extract_numeric_constraints(user_prompt)

        for field, value in numeric_constraints.items():
            if value is not None and hasattr(result["parameters"], field):
                setattr(result["parameters"], field, value)
                # Update confidence for numeric constraints
                result["confidences"][field] = 0.95

        # Template classification
        template_result = self.classify_template(user_prompt)
        
        # Format the output in the desired JSON structure
        template_display_names = self.template_display_names
        
        # Get the filled template
        template_population = self.populate_template(template_result["best_template"])
        
        # Convert to the desired output format
        formatted_output = {
            "template": template_display_names.get(template_result["best_template"], template_result["best_template"]),
            "confidence": float(template_result["confidence"]),  # Convert numpy to native Python float
            "parsedConstraint": template_population["filled_template"],
            "parameters": self._format_parameters(),
            "parameterConfidences": self._format_parameter_confidences_from_matches(result["confidences"], result["raw_matches"]),
            "alternatives": self._format_alternatives(result["alternatives"])
        }
        
        return formatted_output

    def _format_alternatives(self, alternatives: Dict) -> List:
        """
        Convert alternatives to the desired list format
        """

        formatted_alternatives = []
        
        for field, alt_list in alternatives.items():
            for alt in alt_list:
                if alt["confidence"]>0.7:
                    formatted_alternatives.append({
                        "field": field,
                        "value": alt["value"],
                        "confidence": float(alt["confidence"]),  # Convert numpy to native Python float
                        "sourcePhrase": alt["source_phrase"]
                    })
        
        return formatted_alternatives

    def _format_parameter_confidences_from_matches(self, confidences: Dict, raw_matches: List):
        """
        Create confidence scores for every parameter using actual semantic search outputs
        """
        params = self.extracted_Parameters

        parameter_confidences = {}
        
        # Define all parameter fields
        all_fields = [
            'min', 'max', 'games', 'matchups', 'byes', 'rounds', 
            'venues', 'networks', 'teams', 'round1', 'round2', 'k', 'm', 
            'home_away_bye_active', 'each_or_all'
        ]
        
        # First, use the confidences from the processed matches (these are the best matches)
        for field in all_fields:
            if field in confidences:
                # Use the confidence from semantic matching
                parameter_confidences[field] = float(confidences[field])
            else:
                # Check if this parameter has any value but no confidence score
                value = getattr(params, field, None)
                if value is not None and (isinstance(value, list) and value or (not isinstance(value, list) and value)):
                    # Parameter has value but no confidence - check raw matches
                    confidence_from_raw = self._find_confidence_from_raw_matches(field, raw_matches)
                    parameter_confidences[field] = confidence_from_raw
        
        return parameter_confidences


    def _format_parameters(self) -> Dict:
        """
        Convert extracted_Parameters object to the desired parameters dictionary format
        """

        params = self.extracted_Parameters
        parameters_nonempty_dict = {}
        
        fields =  {
            "min": getattr(params, 'min_val', None),
            "max": getattr(params, 'max_val', None),
            "games": getattr(params, 'games', []),
            "matchups": getattr(params, 'matchups', []),
            "byes": getattr(params, 'byes', []),
            "rounds": getattr(params, 'rounds', []),
            "venues": getattr(params, 'venues', []),
            "networks": getattr(params, 'networks', []),
            "teams": getattr(params, 'teams', []),
            "round1": getattr(params, 'round1', None),
            "round2": getattr(params, 'round2', None),
            "k": getattr(params, 'k', None),
            "m": getattr(params, 'm', None),
            "home_away_bye_active": getattr(params, 'home_away_bye_active', []),
            "each_or_all": getattr(params, 'each_or_all', None)
        }

        for field, value in fields.items():
            if value is not None:
                parameters_nonempty_dict[field] = value
        
        return parameters_nonempty_dict


    def process_input(self, user_prompt: str, min_n = 1, max_n = 2):
        """ 
        Input: user_prompt
        Output: n-grams and NER tagged words
        """
        if self.n_gram == False:
            max_n = min_n

        candidate_phrases = set()

        ## STEP 1: spaCy NER tagging
        doc = self.nlp(user_prompt)
        
        ## Extract named entities from spaCy document
        ner_phrases = []
        
        for ent in doc.ents:
            ## Keep the entity text
            ner_phrases.append(ent.text)
            
            ## Also add any sub-phrases for compound entities
            if len(ent.text.split()) > 1:
                ## Add individual words from multi-word entities
                words = ent.text.split()
                ner_phrases.extend(words)

        candidate_phrases.update(ner_phrases)


        ## STEP 2: Generate n-grams (1,2,3 words)
        words = user_prompt.split()
        ngrams = []
        
        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i + n])
                ngrams.append(ngram)

        candidate_phrases.update(ngrams)


        ## STEP 3: Include full query for context
        candidate_phrases.add(user_prompt)


        ## Convert to a list, remove any entries that are empty
        return [phrase for phrase in candidate_phrases if phrase.strip()]



    def classify_template(self, user_prompt):
        """
        Input: user's promt
        Output: dict w best template, confidence scores, alternatives
        """

        query_vec = self._get_openai_embedding(user_prompt)
        best_template = None
        best_score = -1
        all_scores = []

        ## Compare query embedding against each template embedding
        for template_name, embeddings in self.template_embeddings.items():
            for temp_vec in embeddings:
                score = self._cosine_similarity(query_vec, temp_vec)
                all_scores.append({
                    "template": template_name,
                    "score": score,
                    "examples": self.template_examples[template_name]
                })
                
                if score > best_score:
                    best_score = score
                    best_template = template_name


        # Sort all scores for alternatives
        all_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "best_template": best_template,
            "confidence": best_score,
            "all_matches": all_scores[:3]  # Top 3 alternatives
        }


    def populate_template(self, template_name):
        """
        Populate template fields with extracted parameters
        Returns: dict with filled template and validation results
        """

        parameters = self.extracted_Parameters

        # Get template structure
        template_info = {
            "game_scheduling_constraints": {
                "optional": ["games", "rounds","venues", "networks", "min_val", "max_val"],
                "template": "Ensure that at least <min> and at most <max> games from <games> are scheduled across <rounds> and played in any venue from <venues> and assigned to any of <networks>."
            },
            "sequence_constraints": {
                "optional": ["games", "matchups", "byes", "min_val", "max_val","round1", "round2"],
                "template": "Ensure at least <min> and at most <max> cases where there is a sequence <games> across rounds <round1>, <round2>."
            },
            "team_schedule_pattern_constraints": {
                "optional": ["min_val", "max_val", "k", "m", "networks", "venues", "each_or_all","teams", "home_away_bye_active", "rounds"],
                "template": "Ensure that <each_or_all> teams in <teams> have at least <min> and at most <max> instances where they play at least <k> and at most <m> <home_away_bye_active> games across <rounds> where the game is assigned to any of <networks> and played in any venue from <venues>."
            }
        }
        
        template_data = template_info.get(template_name, {})
        filled_template = template_data.get("template", "")
        validation_results = self._validate_parameters()
        
        # Define field mapping from template placeholders to parameter attributes
        field_mapping = self.field_mapping
    
        # Fill in the template with actual values
        for template_field, param_field in field_mapping.items():
            value = getattr(parameters, param_field, None)
            if value is not None:
                if isinstance(value, list):
                    filled_value = ", ".join(str(v) for v in value) if value else "all"
                else:
                    filled_value = str(value)
                filled_template = filled_template.replace(template_field, f"<{filled_value}>")

        return {
            "filled_template": filled_template,
            "validation": validation_results,
            "template_structure": template_data
        }




    def _validate_parameters(self):
        ## Check parameter consistency for numeric values
        ## Maybe think about adding ways to check for word parameters

        parameters = self.extracted_Parameters

        validation_results = {
            "warnings": []
        }

        if hasattr(parameters, 'min_val') and hasattr(parameters, 'max_val'):
            if parameters.min_val and parameters.max_val and parameters.min_val > parameters.max_val:
                validation_results["warnings"].append("min_val cannot be greater than max_val")
        
        if hasattr(parameters, 'k') and hasattr(parameters, 'm'):
            if parameters.k and parameters.m and parameters.k > parameters.m:
                validation_results["warnings"].append("k cannot be greater than m")


    def get_user_feedback_and_update(self, user_query: str, extraction_result: Dict, user_corrections: Dict) -> None:
        """
        Incorporate user feedback into Algolia index for continuous learning
        Feedback only works if we recalculate embeddings every time we run this i.e. don't load (outdated) data files
            (which is fine, but for now we skip bc of api usage constraints and for speed)
        """
        try:
            # Create a new entity from user corrections
            feedback_entity = {
                "objectID": f"feedback_{int(time.time())}_{hash(user_query) % 10000}",
                "type": "user_correction",
                "original_query": user_query,
                "corrected_parameters": user_corrections,
                "extraction_result": extraction_result,
                "timestamp": time.time(),
                "source": "user_feedback"
            }
            
            # Add the name/pattern for embedding generation
            if "parameters" in user_corrections:
                feedback_entity["name"] = user_corrections["parameters"]
                feedback_entity["template"] = user_corrections["template"]
            
            # Upload to Algolia
            self.client.save_object(self.index_name, feedback_entity)
            print("User feedback incorporated into Algolia index")

        except Exception as e:
            print(f"Error saving feedback: {e}")


    def _nice_display(self, result):
        print(f"\n=== EXTRACTION RESULTS ===")
        print(f"Template: {result["template"]} "
            f"(confidence: {result["confidence"]:.2f})")
        print(f"Filled template: {result["parsedConstraint"]}")
        
        
        # Show parameter confidences
        for param_name, param_value in result["parameters"].items():
            confidence = result["confidence"]
            print(f"  - {param_name}: {param_value} (confidence: {confidence:.2f})")

        

    # Example usage with feedback loop:
    def interactive_extraction(self, user_prompt: str):
        """
        Interactive version that asks for user feedback when needed
        """
        result = self.extract(user_prompt)
        
        self._nice_display(result)

        # Ask for feedback if needed
        corrections = self._collect_user_corrections(result)
        self.get_user_feedback_and_update(user_prompt, result, corrections)

        return corrections


    def _collect_user_corrections(self, result: Dict) -> Dict:
        """
        Collect corrections from user input
        """

        special_mappings = {"min_val":"min", "max_val":"max"}
        
        corrections = {}
        print("\n=== PROVIDE CORRECTIONS ===")

        ## Allow for correction of template
        print("\nCorrect the template: (press enter to skip)")
        print("\nAvailable templates:")
        for template_name in self.template_display_names.values():
            print(f"  {template_name}")

        current_template = result["template"]
        new_temp = input(f"\n Template (Current Template: {current_template}) \n Enter a value (1/2/3): ")
        if new_temp:
            corrections["template"] = list(self.template_display_names.values())[int(new_temp)]


        
        ## Allow corrections for any parameter
        print("\nCorrect any parameter (press enter to skip):")
        for field in fields(self.extracted_Parameters):
            param_name = field.name
            
            result_key = special_mappings.get(param_name)

            if result_key in result["parameters"]:
                current = result["parameters"][result_key]
                new_value = input(f"  {param_name} (current: {current}): ")
                if new_value.strip():
                    corrections[param_name] = new_value

            elif param_name in result["parameters"]:
                current = result["parameters"][param_name]
                new_value = input(f"  {param_name} (current: {current}): ")
                if new_value.strip():
                    corrections[param_name] = new_value


            else:
                new_value = input(f"  {param_name}: ")
                if new_value.strip():
                    corrections[param_name] = new_value

        return corrections



