## import packages needed
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import Dict, List
from algoliasearch.search.client import SearchClientSync
import requests



class CompleteSportsDataUploader_Algolia:

    def __init__ (self, algolia_app_id, algolia_api_key):
        self.client = SearchClientSync(app_id=algolia_app_id, api_key=algolia_api_key)
        self.index_name = "premade_sports_knowledge"

    def upload_all_data(self):
        """
        Check if data already exists in Algolia
        If not, upload it
        """

        if self._data_already_exists():
            return


        ## Data doesn't exist, so we must upload it

        all_entities = []


        ## 1. get sports teams from API
        all_entities.extend(self._get_nba_teams_from_api())
        all_entities.extend(self._get_nfl_teams_from_api())
        all_entities.extend(self._get_mlb_teams_from_api())
        all_entities.extend(self._get_nhl_teams_from_api())
        all_entities.extend(self._get_wnba_teams_from_api())
        all_entities.extend(self._get_cwbb_teams_from_api())
        all_entities.extend(self._get_cbb_teams_from_api())
        all_entities.extend(self._get_cfb_teams_from_api())


        ## 2. use NLP handmade constraint/ parameter dictionaries
        all_entities.extend(self._get_all_constraint_data())


        ## 3. Configure Index
        self._configure_index()


        ## 4. Upload to Algolia
        self._upload_to_aloglia(all_entities)

        


    def _data_already_exists(self) -> bool:
        """
        Checking if data already exists in algolia to avoid unnecessary uploads 
        """

        try:
            # Try to search for a few representative records from each category
            test_filters = [
                {"filter": "type:team", "expected_min": 50},  # Should have way more than 50 teams
                {"filter": "type:quantifier", "expected_min": 5},  # Should have constraint patterns
                {"filter": "type:event_category", "expected_min": 7},  # Should have event types
            ]
            
            all_exist = True
            for test in test_filters:
                results = self.client.search_single_index(
                    self.index_name, 
                    {"filters": test["filter"], "hitsPerPage": 1}
                )
                if results.nb_hits < test["expected_min"]:
                    print(f"Not enough {test["filter"]} records found: {results.nb_hits}")
                    all_exist = False
                else:
                    print(f"Found {results.nb_hits} {test["filter"]} records")
            
            return all_exist

        except Exception as e:
            print(f"Error checking existing data: {e}")
            return False


    def _get_all_constraint_data(self):
        """
        Place to dump all handmade constraints and DBpedia venues
        """
        
        constraint_entities = []

        ## Initialize data sources
        constraint_entities.extend(self._init_quantitative_patterns())
        constraint_entities.extend(self._init_event_categories())
        constraint_entities.extend(self._init_temporal_terms())
        constraint_entities.extend(self._init_broadcast_networks())
        constraint_entities.extend(self._init_quantifiers())
        constraint_entities.extend(self._init_game_states())
        constraint_entities.extend(self._init_venue_database())


        ## Add objectIDs to constraint entities for algolia's benefit
        for i, entity in enumerate(constraint_entities):
            entity["objectID"] = f"constraint_{i}"


        return constraint_entities


    def _init_quantitative_patterns(self):
        """
        Function defines quantative relationship to help identify min, max, range
        """

        return [
            {"pattern": r"at least (\d+)", "type": "quantifier", "capture": "min"},
            {"pattern": r"at most (\d+)", "type": "quantifier", "capture": "max"},
            {"pattern": r"between (\d+) and (\d+)", "type": "quantifier", "capture": "range"},
            {"pattern": r"exactly (\d+)", "type": "quantifier", "capture": "exact"},
            {"pattern": r"no more than (\d+)", "type": "quantifier", "capture": "max"},
            {"pattern": r"(\d+) or more", "type": "quantifier", "capture": "min"},
            {"pattern": r"minimum of (\d+)", "type": "quantifier", "capture": "min"},
            {"pattern": r"maximum of (\d+)", "type": "quantifier", "capture": "max"},
            {"pattern": r"up to (\d+)", "type": "quantifier", "capture": "max"},
            {"pattern": r"less than (\d+)", "type": "quantifier", "capture": "max", "adjustment": -1},
            {"pattern": r"more than (\d+)", "type": "quantifier", "capture": "min", "adjustment": 1}
        ]
        
    def _init_event_categories(self):
        """
        Function defines event categories to capture game types
        """
        return [
            {"name": "rivalry games", "type": "event_category", "synonyms": ["rivalries", "derby games", "rivalry matchups"]},
            {"name": "bye weeks", "type": "event_category", "synonyms": ["byes", "off weeks", "rest weeks"]},
            {"name": "high profile games", "type": "event_category", "synonyms": ["marquee matchups", "premier games", "featured games"]},
            {"name": "weekend games", "type": "temporal_category", "synonyms": ["Saturday games", "Sunday games", "weekend matchups"]},
            {"name": "weekday games", "type": "temporal_category", "synonyms": ["weeknight games", "Monday games", "Thursday games"]},
            {"name": "primetime games", "type": "broadcast_category", "synonyms": ["night games", "prime time", "evening games"]},
            {"name": "playoff games", "type": "event_category", "synonyms": ["postseason games", "playoff matchups"]},
            {"name": "conference games", "type": "event_category", "synonyms": ["conference matchups", "intraconference games"]},
            {"name": "division games", "type": "event_category", "synonyms": ["division matchups", "divisional games"]},
            {"name": "home games", "type": "event_category", "synonyms": ["home matchups", "home contests"]},
            {"name": "away games", "type": "event_category", "synonyms": ["away matchups", "road games"]}
        ]


    def _init_temporal_terms(self):
        """
        Function define temporal ranges (help w rounds, sequences)
        """

        return[
            {"name": "final 2 dates", "type": "temporal_range", "position": "end", "count": 2},
            {"name": "playoff rounds", "type": "temporal_range", "phase": "postseason"},
            {"name": "regular season", "type": "temporal_range", "phase": "regular"},
            {"name": "early season", "type": "temporal_range", "segment": "early"},
            {"name": "mid-season", "type": "temporal_range", "segment": "mid"},
            {"name": "late season", "type": "temporal_range", "segment": "late"},
            {"name": "back-to-back weeks", "type": "temporal_pattern", "pattern": "consecutive"},
            {"name": "consecutive rounds", "type": "temporal_pattern", "pattern": "consecutive"},
            {"name": "final month", "type": "temporal_range", "position": "end", "unit": "month"},
            {"name": "opening week", "type": "temporal_range", "position": "start", "unit": "week"}
        ]

    def _init_broadcast_networks(self):
        """
        Finite amount of networks, so just list them all
        Might be missing online streaming networks, hopefully semantic search will get the idea though
        """
        
        return [
            {"name": "ESPN", "type": "network", "aliases": ["ESPN", "ESPN2", "ESPN+", "ESPN Plus"]},
            {"name": "CBS", "type": "network", "aliases": ["CBS", "CBS Sports"]},
            {"name": "Fox", "type": "network", "aliases": ["Fox", "Fox Sports", "FS1", "Fox Sports 1"]},
            {"name": "NBC", "type": "network", "aliases": ["NBC", "NBC Sports"]},
            {"name": "ABC", "type": "network", "aliases": ["ABC"]},
            {"name": "TNT", "type": "network", "aliases": ["TNT"]},
            {"name": "NBA TV", "type": "network", "aliases": ["NBA TV"]},
            {"name": "NFL Network", "type": "network", "aliases": ["NFL Network"]},
            {"name": "MLB Network", "type": "network", "aliases": ["MLB Network"]},
            {"name": "NHL Network", "type": "network", "aliases": ["NHL Network"]}
        ]

    def _init_quantifiers(self):
        """
        Function helps with <each/all> constraint
        """

        return [
            {"name": "each of", "type": "quantifier", "scope": "universal"},
            {"name": "all", "type": "quantifier", "scope": "universal"},
            {"name": "any", "type": "quantifier", "scope": "existential"},
            {"name": "no", "type": "quantifier", "scope": "negative"},
            {"name": "every", "type": "quantifier", "scope": "universal"},
            {"name": "none of", "type": "quantifier", "scope": "negative"}
        ]

    def _init_game_states(self):
        """
        Function helps identify <home/away/bye/active>
        """

        return [
            {"name": "home games", "type": "game_state", "location": "home"},
            {"name": "away games", "type": "game_state", "location": "away"},
            {"name": "bye games", "type": "game_state", "status": "inactive"},
            {"name": "active games", "type": "game_state", "status": "active"}
        ]


    def _init_venue_database(self):
        """
        Function identifies possible venues from DBpedia
        Will just load from Algolia if already pushed there
        """
        try:
            venues = self._load_venues_from_algolia()
            if venues:
                print(f"Successfully loaded {len(venues)} venues from Algolia")
                return venues
            else:
                print("Algolia index empty, fetching from DBpedia...")
        except Exception as e:
            print(f"Error loading from Algolia: {e}. Fetching from DBpedia...")

        ## If data not already in Algolia, get it from DBpedia and push to algolia
        venues = self._fetch_venues_from_dbpedia()
        return venues


    def _load_venues_from_algolia(self):

        venues = []

        try:
            ## Search specifically for venue objects
            results = self.client.search_single_index(
                self.index_name,
                {"query": "", "filters": "type:venue", "hitsPerPage": 1000}
            )
            
            for hit in results.hits:
                venue = {
                    "name": hit.get("name"),
                    "type": "venue"
                }
                venues.append(venue)
            return venues

        except Exception as e:
            print(f"Error searching Algolia index: {e}")
            return []


    def _fetch_venues_from_dbpedia(self) -> List[Dict]:
        """
        Query DBpedia to discover venues - keep same logic, no Algolia push
        """

        try:
            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
            query = """
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT DISTINCT ?venueName
            WHERE {
                ?venue a dbo:Venue ;
                    rdfs:label ?venueName .
                FILTER(LANG(?venueName) = "en")
                FILTER(STRLEN(?venueName) > 3)
            }
            LIMIT 200
            """
            
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            
            venues = []
            for result in results["results"]["bindings"]:
                venue = {
                    "name": result["venueName"]["value"],
                    "type": "venue"
                }
                venues.append(venue)
            
            print(f"Successfully fetched {len(venues)} venues from DBpedia")
            return venues
        except Exception as e:
            print(f"Error fetching DBpedia data: {e}")


    def _get_nba_teams_from_api(self) -> List[Dict]:
        """
        Get real NBA teams from Ball Don't Lie API
        """

        try:
            headers = {
                'Ocp-Apim-Subscription-Key': 'f4e2eb13c2344e3d9f8cf0ed67020293'
            }
            response = requests.get('https://api.sportsdata.io/v3/nba/scores/json/Teams',
                                  headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            nba_teams = []
            for team in data:
                full_name = f"{team.get('City', '')} {team.get('Name', '')}".strip()
                nba_teams.append({
                    "objectID": f"nba_team_{team['TeamID']}",
                    "name": full_name,
                    "type": "team",
                    "sport": "basketball",
                    "league": "NBA",
                    "city": team['City'],
                    "conference": team['Conference'],
                    "division": team['Division'],
                    "abbreviation": team['Key'],
                    "category": "team",
                    "source": "sportsdata_api"
                })
            return nba_teams
        except Exception as e:
            print(f"Error fetching NBA teams: {e}")


    def _get_nfl_teams_from_api(self) -> List[Dict]:
        """
        Get NFL teams from SportsData.io API (free tier)
        """

        try:
            headers = {
                'Ocp-Apim-Subscription-Key': 'f4e2eb13c2344e3d9f8cf0ed67020293'
            }
            response = requests.get('https://api.sportsdata.io/v3/nfl/scores/json/Teams',
                                  headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            nfl_teams = []
            for team in data:
                full_name = f"{team.get('City', '')} {team.get('Name', '')}".strip()
                nfl_teams.append({
                    "objectID": f"nfl_team_{team['TeamID']}",
                    "name": full_name,
                    "type": "team",
                    "sport": "football",
                    "league": "NFL",
                    "city": team['City'],
                    "conference": team['Conference'],
                    "division": team['Division'],
                    "abbreviation": team['Key'],
                    "category": "team",
                    "source": "sportsdata_api"
                })

            return nfl_teams
        except Exception as e:
            print(f"Error fetching NBA teams: {e}")


    def _get_mlb_teams_from_api(self) -> List[Dict]:
        """
        Get NFL teams from SportsData.io API (free tier)
        """

        try:
            headers = {
                'Ocp-Apim-Subscription-Key': 'f4e2eb13c2344e3d9f8cf0ed67020293'
            }
            response = requests.get('https://api.sportsdata.io/v3/mlb/scores/json/Teams',
                                  headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            mlb_teams = []
            for team in data:
                full_name = f"{team.get('City', '')} {team.get('Name', '')}".strip()
                mlb_teams.append({
                    "objectID": f"mlb_team_{team['TeamID']}",
                    "name": full_name,
                    "type": "team",
                    "league": "MLB",
                    "city": team['City'],
                    "division": team['Division'],
                    "abbreviation": team['Key'],
                    "category": "team",
                    "source": "sportsdata_api"
                })

            return mlb_teams
        except Exception as e:
            print(f"Error fetching MLB teams: {e}")



    def _get_nhl_teams_from_api(self) -> List[Dict]:
        """
        Get NHL teams from SportsData.io API (free tier)
        """

        try:
            headers = {
                'Ocp-Apim-Subscription-Key': 'f4e2eb13c2344e3d9f8cf0ed67020293'
            }
            response = requests.get('https://api.sportsdata.io/v3/nhl/scores/json/Teams',
                                  headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            nhl_teams = []
            for team in data:
                full_name = f"{team.get('City', '')} {team.get('Name', '')}".strip()
                nhl_teams.append({
                    "objectID": f"nhl_team_{team['TeamID']}",
                    "name": full_name,
                    "type": "team",
                    "league": "NHL",
                    "city": team['City'],
                    "division": team['Division'],
                    "abbreviation": team['Key'],
                    "category": "team",
                    "source": "sportsdata_api"
                })

            return nhl_teams
        except Exception as e:
            print(f"Error fetching NHL teams: {e}")


    def _get_wnba_teams_from_api(self) -> List[Dict]:
        """
        Get wnba teams from SportsData.io API (free tier)
        """

        try:
            headers = {
                'Ocp-Apim-Subscription-Key': 'f4e2eb13c2344e3d9f8cf0ed67020293'
            }
            response = requests.get('https://api.sportsdata.io/v3/wnba/scores/json/Teams',
                                  headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            wnba_teams = []
            for team in data:
                full_name = f"{team.get('City', '')} {team.get('Name', '')}".strip()
                wnba_teams.append({
                    "objectID": f"soccer_team_{team['TeamID']}",
                    "name": full_name,
                    "type": "team",
                    "city": team['City'],
                    "league": "WNBA",
                    "abbreviation": team['Key'],
                    "category": "team",
                    "source": "sportsdata_api"
                })

            return wnba_teams
        except Exception as e:
            print(f"Error fetching wnba teams: {e}")

    def _get_cwbb_teams_from_api(self) -> List[Dict]:
        """
        Get college womens basketballl teams from SportsData.io API (free tier)
        """

        try:
            headers = {
                'Ocp-Apim-Subscription-Key': 'f4e2eb13c2344e3d9f8cf0ed67020293'
            }
            response = requests.get('https://api.sportsdata.io/v3/cwbb/scores/json/Teams',
                                  headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            cwbb_teams = []
            for team in data:
                full_name = f"{team.get('City', '')} {team.get('Name', '')}".strip()
                cwbb_teams.append({
                    "objectID": f"cwbb_team_{team['TeamID']}",
                    "name": full_name,
                    "type": "team",
                    "league": "cwbb",
                    "abbreviation": team['Key'],
                    "category": "team",
                    "source": "sportsdata_api"
                })

            return cwbb_teams
        except Exception as e:
            print(f"Error fetching cwbb teams: {e}")

    def _get_cbb_teams_from_api(self) -> List[Dict]:
        """
        Get cbb teams from SportsData.io API (free tier)
        """

        try:
            headers = {
                'Ocp-Apim-Subscription-Key': 'f4e2eb13c2344e3d9f8cf0ed67020293'
            }
            response = requests.get('https://api.sportsdata.io/v3/cbb/scores/json/Teams',
                                  headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            cbb_teams = []
            for team in data:
                full_name = f"{team.get('City', '')} {team.get('Name', '')}".strip()
                cbb_teams.append({
                    "objectID": f"cbb_team_{team['TeamID']}",
                    "name": full_name,
                    "type": "team",
                    "league": "CBB",
                    "abbreviation": team['Key'],
                    "category": "team",
                    "source": "sportsdata_api"
                })

            return cbb_teams
        except Exception as e:
            print(f"Error fetching CBB teams: {e}")

    def _get_cfb_teams_from_api(self) -> List[Dict]:
        """
        Get cfb teams from SportsData.io API (free tier)
        """

        try:
            headers = {
                'Ocp-Apim-Subscription-Key': 'f4e2eb13c2344e3d9f8cf0ed67020293'
            }
            response = requests.get('https://api.sportsdata.io/v3/cfb/scores/json/Teams',
                                  headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            cfb_teams = []
            for team in data:
                full_name = f"{team.get('City', '')} {team.get('Name', '')}".strip()
                cfb_teams.append({
                    "objectID": f"cfb_team_{team['TeamID']}",
                    "name": full_name,
                    "type": "team",
                    "league": "CFB",
                    "abbreviation": team['Key'],
                    "category": "team",
                    "source": "sportsdata_api"
                })

            return cfb_teams
        except Exception as e:
            print(f"Error fetching CFB teams: {e}")



    def _upload_to_aloglia(self, entities):
        """
        Upload entities to algolia
        """

        if not entities:
            print("No entities")
            return

        try:
            batch_requests = [
                {"action": "addObject", "body": obj}
                for obj in entities
            ]

            chunk_size = 1000
            for i in range(0, len(batch_requests), chunk_size):
                chunk = batch_requests[i:i + chunk_size]
                print(f"Uploading chunk {i//chunk_size + 1} with {len(chunk)} records...")
                response = self.client.batch(self.index_name, {"requests": chunk})
                print(f"Chunk {i//chunk_size + 1} uploaded successfully")

            print(f"All {len(entities)} entities uploaded to Algolia")

        except Exception as e:
            print(f"Upload failed: {e}")
            import traceback
            traceback.print_exc()


    def _configure_index(self):
        """
        Configure the index for optimal semantic search based on your data structure
        """
        settings = {
            # SEARCHABLE FIELDS - in order of importance for semantic search
            'searchableAttributes': [
                'name',              # Most important - team names, venue names, constraint names
                'type',              # team, quantifier, event_category, etc.
                'sport',             # basketball, football, hockey, etc.
                'league',            # NBA, NFL, MLB, etc.
                'category',          # team, venue, constraint, etc.
                'city',              # Location context for teams and venues
                'conference',        # Eastern, Western, etc.
                'division',          # Atlantic, Pacific, etc.
                'abbreviation',      # Team codes like LAL, BOS, etc.
                'pattern',           # Constraint patterns like "at least <min>"
                'synonyms',          # Alternative names for constraints
                'aliases',           # Network aliases
                'capture',           # What the quantifier captures (min, max, range)
                'scope',             # Universal, existential quantifiers
                'location',          # Home, away, etc.
                'status',            # Active, inactive
                'position',          # Temporal position (start, end)
                'phase',             # postseason, regular season
                'segment',           # early, mid, late season
                'description'        # Venue descriptions
            ],
            
            # FACETING FIELDS - for filtering and categorization
            'attributesForFaceting': [
                'type',              # Filter by: team, quantifier, event_category, etc.
                'sport',             # Filter by sport type
                'league',            # Filter by league
                'category',          # Filter by category
                'conference',        # Filter by conference
                'division',          # Filter by division
                'source',            # Filter by data source
                'searchable(city)',  # Search within city filters
                'capture',           # Filter by what quantifiers capture
                'scope',             # Filter by quantifier scope
                'location',          # Filter by location type
                'status',            # Filter by status
                'phase',             # Filter by season phase
                'searchable(synonyms)' # Search within synonym filters
            ],
            
            ## More advanced semantic parsing settings
            'queryType': 'prefixLast',  
            'advancedSyntax': True,      
            
            
            ## Typo tolerance
            'typoTolerance': True,
            'minWordSizefor1Typo': 2,
            'minWordSizefor2Typos': 5,
            
            # Results shown
            'hitsPerPage': 20
        }
        
        try:
            self.client.set_settings(
                index_name=self.index_name,
                index_settings=settings,
                forward_to_replicas=True
            )
            print("Index configured successfully for semantic search")

        except Exception as e:
            print(f"Error configuring index: {e}")


