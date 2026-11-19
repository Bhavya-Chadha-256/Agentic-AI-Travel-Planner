# Dubai AI Travel Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit prototype that turns a natural-language Dubai trip request into a validated, budgeted, day-by-day itinerary, produced by five specialized agents and runnable in two modes: LangGraph and CrewAI.

**Architecture:** All planning logic lives in a deterministic, framework-agnostic `core/` package (parse → research → logistics → budget → review → revise), backed by a curated JSON dataset with optional live APIs that fall back to it. LangGraph and CrewAI are thin orchestration layers that wrap the same core functions and both return the same `Itinerary` Pydantic model, so their runs can be compared side by side. An `EventLog` records every agent step so the UI can show which agent is running and how the plan changes across revisions.

**Tech Stack:** Python 3.11, uv, Pydantic v2, LangGraph, LangChain (`init_chat_model`), CrewAI, Streamlit, httpx, python-dotenv, pytest, respx.

**Spec:** [Problem Statement.md](Problem%20Statement.md)

## Global Constraints

- Python **3.11** managed by `uv` (the system Python is 3.9.6, and CrewAI needs >=3.10).
- Scope: **Dubai only**, trips of **1–7 days**, all money in **AED**, budget **excludes flights**.
- Out of scope: live hotel bookings, ticket purchases, guaranteed real-time prices.
- Every cost and travel time carries a `Provenance`: `verified` (curated, with source URL), `api` (live API result) or `estimated` (computed or assumed).
- Both modes return the same `Itinerary` schema; `Itinerary.framework` is `"langgraph"` or `"crewai"`.
- The review loop is capped at `MAX_REVISIONS = 3`.
- LLM provider and model come from env `LLM_PROVIDER` / `LLM_MODEL` (default `anthropic` / `claude-sonnet-5-5`).
- Live APIs only run when `USE_LIVE_APIS=true` and the needed key is set. Any API failure falls back to the local dataset without raising.
- The test suite runs **offline with no API keys**. Tests that need network access or an LLM are marked `@pytest.mark.live` and skipped by default.
- Secrets live in `.env`, which is gitignored. `.env.example` lists every variable.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Unsupported request** ("10-day trip", "trip to Abu Dhabi"): the user sees a clear "Dubai trips of 1–7 days only" message in the UI, not a stack trace. Tested in Task 5 and Task 13.
2. **Missing budget or day count**: the plan uses defaults (3 days; no budget means the budget check is skipped) and shows them as visible assumptions. It never invents an `OVER_BUDGET` violation. Tested in Task 5 and Task 9.
3. **Budget too low to meet** (AED 300 for 5 days): the loop stops after `MAX_REVISIONS` and returns a best-effort itinerary with `review.passed == False`, the unresolved `OVER_BUDGET` violation and non-empty alternatives. It never loops forever. Tested in Task 11.
4. **Live API enabled but failing** (timeout, 500 error, missing key): planning still completes, using fallback values marked `estimated`. Tested in Task 3 and Task 4.
5. **Running out of places** (7 days + avoid crowds + one niche interest): the result has all 7 days, no repeated attraction, and a "free time" note on short days. It never crashes or repeats attractions. Tested in Task 7.

---

## File Structure

```
pyproject.toml              # deps, pytest config (markers: live)
.env.example  .gitignore  README.md
app.py                      # Streamlit UI
src/dubai_planner/
  config.py                 # constants + Settings from env
  schemas.py                # all Pydantic models (single source of truth)
  events.py                 # EventLog + agent name constants
  llm.py                    # configurable LLM factories (LangChain + CrewAI)
  data/attractions.json restaurants.json hotels.json transport.json
  data_store.py             # loads + validates curated data
  tools/geo.py              # haversine
  tools/routes.py           # Router: legs, durations, fares (Google Routes optional)
  tools/places.py           # Overpass restaurant discovery (optional)
  tools/weather.py          # Open-Meteo forecast (optional)
  core/parser.py            # NL request -> TripRequest
  core/research.py          # Destination Research agent logic
  core/logistics.py         # hotel choice + schedule building
  core/budget.py            # cost estimate + alternatives
  core/review.py            # rule-based validation
  core/revise.py            # violations -> RevisionDirectives
  core/summary.py           # optional LLM narrative
  modes/langgraph_mode.py   # Mode A
  modes/crewai_mode.py      # Mode B
  compare.py                # run both, collect metrics
tests/                      # mirrors src layout; conftest.py holds shared fixtures
```

---

### Task 1: Project scaffold and schemas

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `src/dubai_planner/__init__.py`, `src/dubai_planner/config.py`, `src/dubai_planner/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces (`config.py`): constants `MAX_REVISIONS=3`, `DAY_START="09:30"`, `DAY_END="22:00"`, `LUNCH_AT="13:00"`, `DINNER_AT="19:30"`, `MAX_ATTRACTIONS_PER_DAY=3`, `CROWD_THRESHOLD=4`, `BREAKFAST_AED=25.0`, `WALK_MAX_KM=1.0`, `METRO_MIN_KM=3.0`, `ROAD_FACTOR=1.3`, `SPEED_KMH={"walk":4.5,"metro":35.0,"taxi":30.0}`, `ACCESS_MIN={"walk":0,"metro":12,"taxi":5}`, `USD_TO_AED=3.6725`, `HTTP_TIMEOUT_S=5.0`, `MIN_DAYS=1`, `MAX_DAYS=7`; `@dataclass Settings(llm_provider: str, llm_model: str, use_live_apis: bool, google_maps_api_key: str | None)`; `get_settings() -> Settings` (reads env via python-dotenv).
- Produces (`schemas.py`):
  - `Provenance(str, Enum)`: `VERIFIED`, `API`, `ESTIMATED`
  - `Interest(str, Enum)`: `culture, food, beach, shopping, adventure, nature, nightlife, family`
  - `HotelTier = Literal["budget","mid","luxury"]`; `Mode = Literal["walk","metro","taxi"]`
  - `TripRequest(raw_text: str, days: int [1..7], budget_aed: float | None [>0], interests: list[Interest], hotel_tier: HotelTier = "mid", avoid_crowds: bool = False, travelers: int = 1 [>=1], start_date: date | None = None, assumptions: list[str] = [])`
  - `Place(id, name, kind: Literal["attraction","restaurant"], area, lat, lon, tags: list[Interest], cost_aed: float, duration_min: int, crowd_level: int [1..5], rating: float [0..5], opens: time, closes: time, near_metro: bool, indoor: bool, provenance: Provenance, source_url: str | None)`
  - `Hotel(id, name, area, lat, lon, tier: HotelTier, nightly_aed: float, near_metro: bool, good_for: list[Interest], provenance, source_url: str | None)`
  - `Leg(from_id, to_id, mode: Mode, distance_km: float, duration_min: int, cost_aed: float, provenance)`
  - `ScheduledItem(place: Place, start: time, end: time, leg_in: Leg | None)`
  - `DayPlan(day: int, area_focus: str, items: list[ScheduledItem], notes: list[str] = [])`
  - `BudgetLine(category: Literal["accommodation","food","transport","activities"], amount_aed: float, provenance, detail: str)`
  - `BudgetBreakdown(lines: list[BudgetLine], budget_aed: float | None, alternatives: list[str] = [])` with computed properties `total_aed` and `within_budget` (True when `budget_aed is None`)
  - `ViolationCode = Literal["OVER_BUDGET","SCHEDULE_OVERRUN","DUPLICATE_ACTIVITY","PREFERENCE_MISMATCH","CROWDED_PLACE","MISSING_INFO"]`; `Violation(code, message, day: int | None = None, item_id: str | None = None)`
  - `ReviewResult(passed: bool, violations: list[Violation], revision: int)`
  - `RevisionDirectives(exclude_ids: set[str] = set(), prefer_free: bool = False, max_hotel_nightly_aed: float | None = None, max_meal_aed: float | None = None, max_attractions_per_day: int = 3, applied: list[str] = [])`
  - `PlanSnapshot(revision: int, hotel_name: str, total_aed: float, violations: list[Violation], applied: list[str])`
  - `AgentEvent(ts: datetime, agent: str, kind: Literal["start","output","revision","error"], summary: str, data: dict = {})`
  - `Itinerary(framework: Literal["langgraph","crewai"], request: TripRequest, hotel: Hotel, days: list[DayPlan], budget: BudgetBreakdown, review: ReviewResult, snapshots: list[PlanSnapshot], events: list[AgentEvent], narrative: str | None = None)`

- [ ] **Step 1: Set up tooling.** Run `git init`, `uv init --python 3.11 --package --name dubai-planner`, then `uv add pydantic langgraph langchain langchain-anthropic crewai streamlit httpx python-dotenv` and `uv add --dev pytest respx`. In `pyproject.toml`, set `[tool.pytest.ini_options] markers = ["live: needs network or LLM"]` and `addopts = "-m 'not live'"`. `.gitignore` covers `.env`, `.venv`, `__pycache__`. `.env.example` lists `LLM_PROVIDER`, `LLM_MODEL`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_MAPS_API_KEY` and `USE_LIVE_APIS=false`.

- [ ] **Step 2: Write the failing tests**

```python
def test_trip_request_rejects_days_outside_range():
    with pytest.raises(ValidationError):
        TripRequest(raw_text="x", days=8, budget_aed=1000, interests=[])
    with pytest.raises(ValidationError):
        TripRequest(raw_text="x", days=0, budget_aed=1000, interests=[])

def test_budget_breakdown_totals_and_flags():
    b = BudgetBreakdown(budget_aed=1000, lines=[
        BudgetLine(category="accommodation", amount_aed=600, provenance="verified", detail=""),
        BudgetLine(category="food", amount_aed=500, provenance="estimated", detail="")])
    assert b.total_aed == 1100
    assert b.within_budget is False

def test_no_budget_counts_as_within_budget():
    assert BudgetBreakdown(budget_aed=None, lines=[]).within_budget is True

def test_settings_defaults(monkeypatch):
    for k in ("LLM_PROVIDER", "LLM_MODEL", "USE_LIVE_APIS"):
        monkeypatch.delenv(k, raising=False)
    s = get_settings()
    assert (s.llm_provider, s.llm_model, s.use_live_apis) == ("anthropic", "claude-sonnet-5-5", False)
```

- [ ] **Step 3: Run** `uv run pytest tests/test_schemas.py -v`. Expected: FAIL (ImportError).
- [ ] **Step 4: Implement** `config.py` and `schemas.py` from the Interfaces block. Use `@computed_field` for `total_aed` (rounded to 2 decimals) and `within_budget`.
- [ ] **Step 5: Run** `uv run pytest tests/test_schemas.py -v`. Expected: 4 passed.
- [ ] **Step 6: Commit:** `git add -A && git commit -m "feat: project scaffold, config and shared schemas"`

---

### Task 2: Curated Dubai dataset and DataStore

**Files:**
- Create: `src/dubai_planner/data/{attractions,restaurants,hotels,transport}.json`, `src/dubai_planner/data_store.py`
- Test: `tests/test_data_store.py`

**Interfaces:**
- Consumes: `Place`, `Hotel`, `Provenance`, `Interest` (Task 1).
- Produces: `FareTable(metro_zone_fares_aed: dict[int, float], taxi_base_aed: float, taxi_per_km_aed: float, taxi_min_aed: float, airport_taxi_base_aed: float, source_url: str, as_of: date)`; `DataStore(attractions: list[Place], restaurants: list[Place], hotels: list[Hotel], fares: FareTable, airport: Place)`, with `get_place(id: str) -> Place` (raises `KeyError`) and `area_centroids() -> dict[str, tuple[float, float]]`; `load_store(data_dir: Path | None = None) -> DataStore` (cached with `lru_cache`).

**Dataset requirements** (curated by hand from Visit Dubai and published prices; cite `source_url` and use provenance `verified` for any official price, otherwise `estimated`):
- ≥ 30 attractions across ≥ 7 areas: Al Fahidi/Bur Dubai, Deira, Al Seef, Downtown, Jumeirah, Dubai Marina/JBR, Al Barsha, Business Bay. Include ≥ 8 free ones (e.g. public beaches, Al Fahidi district, abra-side souks) and ≥ 6 with `crowd_level >= 4` (e.g. Burj Khalifa, Dubai Mall).
- ≥ 25 restaurants tagged `food`, with costs from AED 15 to 400 per person and ≥ 2 per area.
- ≥ 12 hotels: ≥ 4 per tier, spread across ≥ 6 areas, with `budget` tier priced ≤ AED 300/night.
- `transport.json`: RTA Nol Silver fares `{1: 3.0, 2: 5.0, 3: 7.5}`, taxi base 5.0, per km 1.97, minimum 12.0, airport base 25.0, `source_url` set to rta.ae, `as_of` date. **Check these against rta.ae before the demo.**
- `airport`: DXB entry with id `"DXB"`, `kind="attraction"`, `cost_aed=0`, `near_metro=true`.

- [ ] **Step 1: Write the failing tests**

```python
DUBAI_BBOX = (24.75, 25.40, 54.90, 55.65)  # lat_min, lat_max, lon_min, lon_max

def test_dataset_meets_minimums():
    s = load_store()
    assert len(s.attractions) >= 30 and len(s.restaurants) >= 25 and len(s.hotels) >= 12
    assert sum(p.cost_aed == 0 for p in s.attractions) >= 8
    assert {h.tier for h in s.hotels} == {"budget", "mid", "luxury"}

def test_all_coordinates_in_dubai_and_ids_unique():
    s = load_store()
    every = s.attractions + s.restaurants + s.hotels
    assert len({x.id for x in every}) == len(every)
    for x in every:
        assert DUBAI_BBOX[0] <= x.lat <= DUBAI_BBOX[1] and DUBAI_BBOX[2] <= x.lon <= DUBAI_BBOX[3], x.id

def test_verified_entries_cite_a_source():
    s = load_store()
    for x in s.attractions + s.restaurants + s.hotels:
        if x.provenance == Provenance.VERIFIED:
            assert x.source_url, x.id

def test_get_place_unknown_raises():
    with pytest.raises(KeyError):
        load_store().get_place("nope")
```

- [ ] **Step 2: Run** `uv run pytest tests/test_data_store.py -v`. Expected: FAIL.
- [ ] **Step 3: Write the four JSON files** to the requirements above, then **implement `load_store`**, validating each record with the Pydantic models. Ship the JSON with the package (hatch `include` or `importlib.resources`).
- [ ] **Step 4: Run** `uv run pytest tests/test_data_store.py -v`. Expected: 4 passed.
- [ ] **Step 5: Add a `tests/conftest.py`** with fixtures `store` (calls `load_store()`), `example_request` (the spec example: 4 days, AED 5000, `[culture, food, beach]`, `budget` tier, `avoid_crowds=True`) and `tight_request` (3 days, AED 1200, `[culture, beach]`, `mid` tier).
- [ ] **Step 6: Commit:** `git commit -am "feat: curated Dubai dataset and DataStore"` (run `git add` on new files first)

---

### Task 3: Routing, travel times and fares

**Files:**
- Create: `src/dubai_planner/tools/geo.py`, `src/dubai_planner/tools/routes.py`
- Test: `tests/tools/test_routes.py`

**Interfaces:**
- Consumes: `DataStore.fares`, `Settings`, `Leg`, config speed constants.
- Produces: `haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float`; `class Located(Protocol): id: str; lat: float; lon: float; near_metro: bool`; `class Router(store: DataStore, settings: Settings, client: httpx.Client | None = None)` with `leg(origin: Located, dest: Located) -> Leg`.

**Mode and cost rules** (fallback path, `provenance=estimated`):
- If straight-line km ≤ `WALK_MAX_KM`, walk at a cost of 0.
- Else if both ends are `near_metro` and km ≥ `METRO_MIN_KM`, take the metro. Zones: ≤ 5 km is 1, ≤ 15 km is 2, otherwise 3. Fare comes from `metro_zone_fares_aed`.
- Otherwise take a taxi: road km = km × `ROAD_FACTOR`; cost = `max(taxi_min, base + per_km × road_km)`. Use `airport_taxi_base_aed` as the base when either end is `"DXB"`.
- Duration = `ceil(road_km / SPEED_KMH[mode] × 60) + ACCESS_MIN[mode]`. For walk and metro, use the straight-line km × `ROAD_FACTOR`.
- Live path: when `settings.use_live_apis` and a Google key are set, POST to Google Routes `computeRoutes` (`DRIVE` for taxi, `TRANSIT` for metro) with timeout `HTTP_TIMEOUT_S`, and use its distance and duration with `provenance=api`. The fare is still computed from the RTA table. On any `httpx.HTTPError`, non-200 response or bad body, return the fallback leg.

- [ ] **Step 1: Write the failing tests**

```python
def test_haversine_known_distance():
    # Burj Khalifa -> Al Fahidi ~ 9.5 km straight line
    assert 9.0 < haversine_km((25.1972, 55.2744), (25.2637, 55.2972)) < 10.0

def test_short_hop_is_free_walk(router, store): ...  # two places < 1 km apart -> mode "walk", cost 0
def test_far_metro_pair_uses_zone_fare(router, store): ...  # both near_metro, ~12 km -> "metro", cost 5.0
def test_taxi_respects_minimum_fare(router, store): ...  # 1.2 km, not near metro -> "taxi", cost 12.0
def test_airport_taxi_uses_airport_base(router, store): ...  # DXB -> hotel by taxi, cost >= 25.0

@respx.mock
def test_live_api_failure_falls_back(store):
    respx.post(url__startswith="https://routes.googleapis.com").mock(side_effect=httpx.ConnectTimeout("t"))
    r = Router(store, Settings("anthropic", "m", use_live_apis=True, google_maps_api_key="k"))
    leg = r.leg(a, b)  # two far apart places
    assert leg.provenance == Provenance.ESTIMATED and leg.duration_min > 0
```

Build the `router` fixture with `use_live_apis=False`. Pick concrete place pairs from your dataset so each test triggers the mode it is named for, and assert exact costs.

- [ ] **Step 2: Run** `uv run pytest tests/tools/test_routes.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `geo.py` and `routes.py` following the rules above.
- [ ] **Step 4: Run** `uv run pytest tests/tools/test_routes.py -v`. Expected: 6 passed.
- [ ] **Step 5: Commit:** `git commit -m "feat: router with RTA fares and Google Routes fallback"`

---

### Task 4: Optional live data tools (places, weather)

**Files:**
- Create: `src/dubai_planner/tools/places.py`, `src/dubai_planner/tools/weather.py`
- Test: `tests/tools/test_places.py`, `tests/tools/test_weather.py`

**Interfaces:**
- Produces: `discover_restaurants_near(lat: float, lon: float, settings: Settings, radius_m: int = 1500, client: httpx.Client | None = None) -> list[Place]`. This queries Overpass (`amenity=restaurant`) and returns `Place(kind="restaurant", tags=[food], cost_aed=60.0, duration_min=60, provenance=api)`, with defaults `crowd_level=3`, `rating=3.5`, opening hours 11:00–23:00 and `near_metro=False`. It returns `[]` when live APIs are off or on any error.
- Produces: `DayWeather(date: date, max_c: float, precip_mm: float)`; `forecast(start: date | None, days: int, settings: Settings, client=None) -> list[DayWeather] | None`, which calls Open-Meteo daily `temperature_2m_max,precipitation_sum` for lat 25.2048 / lon 55.2708. It returns `None` when `start` is None, the start is more than 16 days out, live APIs are off, or the call fails. Also `advisories(weather: list[DayWeather] | None) -> dict[int, list[str]]`: for a day number, `max_c >= 38` adds `"Very hot (NN°C): favour indoor sights 11:00–16:00"` and `precip_mm >= 2` adds `"Rain expected: keep indoor options handy"`.

- [ ] **Step 1: Write the failing tests:** `test_places_returns_empty_when_live_disabled`, `test_places_parses_overpass_payload` (respx returns 2 elements, so expect 2 Places with `provenance == api`), `test_places_returns_empty_on_http_500`, `test_forecast_none_without_start_date`, `test_forecast_none_beyond_16_days`, `test_advisories_flags_hot_and_rainy_days` (input `[DayWeather(d, 41, 0), DayWeather(d2, 30, 5)]`, expected keys `{1, 2}`, with day 1's message containing `"41°C"`).
- [ ] **Step 2: Run** `uv run pytest tests/tools -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** both modules. All exceptions are caught inside, and failures are logged with `logging.warning`.
- [ ] **Step 4: Run** `uv run pytest tests/tools -v`. Expected: all pass.
- [ ] **Step 5: Commit:** `git commit -m "feat: optional Overpass and Open-Meteo tools with fallbacks"`

---

### Task 5: LLM configuration and request parser (Orchestrator: understand request)

**Files:**
- Create: `src/dubai_planner/llm.py`, `src/dubai_planner/core/parser.py`
- Test: `tests/core/test_parser.py`

**Interfaces:**
- Produces (`llm.py`): `LLMNotConfiguredError(RuntimeError)`; `get_chat_model(settings: Settings) -> BaseChatModel` (`init_chat_model(f"{provider}:{model}")`; raises `LLMNotConfiguredError` when the provider's API key env var is missing); `get_crewai_llm(settings: Settings) -> crewai.LLM` (`LLM(model=f"{provider}/{model}")`, same error rule); `has_llm(settings) -> bool`.
- Produces (`parser.py`): `UnsupportedRequestError(ValueError)`; `parse_request(text: str, llm: BaseChatModel | None = None) -> TripRequest`.

**Parsing rules:** With an `llm`, call `llm.with_structured_output(TripRequest)`. On any exception or validation error, fall back to the rules below. Always run the scope checks on the result.
- Days: `(\d+|one|two|…|seven)[\s-]*(day|night)`. Nights count as N+1 days. If no day count is found, use 3 and add the assumption `"Trip length not given; assumed 3 days"`.
- Budget: `AED|dirhams?|Dhs` next to a number (commas allowed). `$` or `USD` amounts are multiplied by `USD_TO_AED`, with the assumption `"Converted USD to AED at 3.6725"`. If no budget is found, use `None` and add the assumption `"No budget given; budget check skipped"`.
- Interests use a keyword map, e.g. culture: cultur/heritage/museum/history; food: food/cuisine/eat/street food; beach: beach/sea/swim.
- Tier: `affordable|cheap|budget|low-cost` maps to `budget`, `luxury|5-star` to `luxury`, otherwise `mid`.
- `avoid_crowds`: matches `avoid crowd|crowded|quiet|less busy`.
- Scope: a number of days outside 1–7 raises `UnsupportedRequestError("Dubai trips of 1–7 days only")`. Naming another emirate or city (Abu Dhabi, Sharjah, Paris, …) without "Dubai" raises `UnsupportedRequestError("Only Dubai is supported")`.

- [ ] **Step 1: Write the failing tests**

```python
SPEC_EXAMPLE = ("Plan a 4-day trip to Dubai with a budget of AED 5,000, excluding flights. I enjoy "
                "cultural attractions, local food, and beaches. I prefer affordable hotels and want "
                "to avoid crowded places.")

def test_parses_spec_example_offline():
    r = parse_request(SPEC_EXAMPLE)
    assert r.days == 4 and r.budget_aed == 5000
    assert set(r.interests) == {Interest.culture, Interest.food, Interest.beach}
    assert r.hotel_tier == "budget" and r.avoid_crowds is True

def test_word_numbers_and_usd():
    r = parse_request("A three-day Dubai trip for $1,000, love museums")
    assert r.days == 3 and r.budget_aed == pytest.approx(3672.5)
    assert any("3.6725" in a for a in r.assumptions)

def test_missing_fields_become_assumptions():
    r = parse_request("Dubai trip, I like beaches")
    assert r.days == 3 and r.budget_aed is None and len(r.assumptions) == 2

@pytest.mark.parametrize("text", ["Plan a 10-day trip to Dubai", "Plan 3 days in Abu Dhabi"])
def test_out_of_scope_raises(text):
    with pytest.raises(UnsupportedRequestError):
        parse_request(text)

def test_llm_failure_falls_back_to_rules():
    class Broken:
        def with_structured_output(self, _): raise RuntimeError("boom")
    assert parse_request(SPEC_EXAMPLE, llm=Broken()).days == 4

def test_missing_key_raises_llm_not_configured(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMNotConfiguredError):
        get_chat_model(Settings("anthropic", "claude-sonnet-5-5", False, None))
```

- [ ] **Step 2: Run** `uv run pytest tests/core/test_parser.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `llm.py` and `parser.py` following the rules above.
- [ ] **Step 4: Run** `uv run pytest tests/core/test_parser.py -v`. Expected: 7 passed.
- [ ] **Step 5: Commit:** `git commit -m "feat: configurable LLM and request parser with rule fallback"`

---

### Task 6: Destination Research agent logic

**Files:**
- Create: `src/dubai_planner/core/research.py`
- Test: `tests/core/test_research.py`

**Interfaces:**
- Consumes: `TripRequest`, `DataStore`, `RevisionDirectives`.
- Produces: `ScoredPlace(place: Place, score: float, reasons: list[str])`; `ResearchResult(attractions: list[ScoredPlace], restaurants: list[ScoredPlace])`; `research_places(request: TripRequest, store: DataStore, directives: RevisionDirectives, extra_restaurants: list[Place] = []) -> ResearchResult`.

**Scoring:**
- Drop any place in `directives.exclude_ids`.
- If `avoid_crowds`, drop places with `crowd_level >= CROWD_THRESHOLD`.
- Score = 3 × (number of matching interest tags) + `rating`.
- If `prefer_free`, subtract `cost_aed / 50`.
- Keep attractions with at least one matching tag (or all attractions if `interests` is empty). Sort by score descending, then id.
- For restaurants, drop those with cost above `directives.max_meal_aed` when it is set. A `budget` tier also caps meals at AED 80 on the first pass.
- `reasons` holds a readable string per scoring factor, shown in the UI.

- [ ] **Step 1: Write the failing tests:** `test_ranks_interest_matches_first` (top 5 attractions for `example_request` each share a tag with the request), `test_avoid_crowds_excludes_busy_places` (no result has `crowd_level >= 4`), `test_excluded_ids_are_dropped`, `test_prefer_free_moves_free_places_up` (mean cost of the top 6 with `prefer_free=True` < mean cost without it), `test_budget_tier_caps_meal_cost` (all restaurants ≤ 80 for a budget-tier request), `test_reasons_are_populated`.
- [ ] **Step 2: Run** `uv run pytest tests/core/test_research.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `research_places`.
- [ ] **Step 4: Run** `uv run pytest tests/core/test_research.py -v`. Expected: 6 passed.
- [ ] **Step 5: Commit:** `git commit -m "feat: destination research scoring"`

---

### Task 7: Logistics agent logic (hotel and schedule)

**Files:**
- Create: `src/dubai_planner/core/logistics.py`
- Test: `tests/core/test_logistics.py`

**Interfaces:**
- Consumes: `Router.leg` (Task 3), `ResearchResult` (Task 6), `advisories` (Task 4).
- Produces: `choose_hotel(request: TripRequest, store: DataStore, directives: RevisionDirectives) -> Hotel`; `build_schedule(request: TripRequest, research: ResearchResult, hotel: Hotel, router: Router, weather_notes: dict[int, list[str]] | None = None, max_per_day: int = MAX_ATTRACTIONS_PER_DAY) -> list[DayPlan]`.

**`choose_hotel`:** Pick hotels of `request.hotel_tier` with `nightly_aed <= directives.max_hotel_nightly_aed` (when set). Score each as 10 × (matches between `good_for` and interests) − `nightly_aed / 100`, and return the best. If none pass the filters, return the cheapest hotel overall. This step runs in parallel with research, so it must not depend on `ResearchResult`.

**`build_schedule` algorithm** (the signature leaves this open, so follow it):
1. Take attractions in research order and assign each to a day. Group by `area`, then fill days area by area (areas sorted by distance from the hotel) with up to `max_per_day` attractions. No attraction appears twice.
2. Within a day, order attractions by nearest neighbour starting from the hotel.
3. Walk the day from `DAY_START`, adding `leg_in` durations. If you arrive before `opens`, wait. If the visit would end after `closes`, move the attraction to a later day with space, or drop it.
4. Insert lunch at the first gap at or after `LUNCH_AT` and dinner at or after `DINNER_AT`. Pick the highest-scored unused restaurant within 3 km of the previous stop, otherwise the nearest one. Restaurants repeat only once all have been used.
5. Day 1's first leg starts from `store.airport` (the arrival transfer). Every other day's first leg starts from the hotel.
6. A day with fewer than `max_per_day` attractions gets the note `"Free time: relax near {hotel.area}"`. Merge `weather_notes[day]` into notes. `area_focus` is the area that appears most among the day's attractions.

- [ ] **Step 1: Write the failing tests:**
  - `test_choose_hotel_respects_tier_and_cap` (budget tier, cap 250, so `nightly_aed <= 250`)
  - `test_choose_hotel_falls_back_to_cheapest` (cap 1, so the cheapest hotel)
  - `test_schedule_has_one_dayplan_per_day_with_meals` (4 days, each with exactly one lunch and one dinner item)
  - `test_items_are_chronological_and_end_by_day_end` (`end <= time(22, 0)` and starts increasing)
  - `test_no_duplicate_attractions`
  - `test_day_one_starts_from_airport` (`days[0].items[0].leg_in.from_id == "DXB"`)
  - **Review Focus 5:** `test_seven_day_niche_trip_does_not_crash_or_repeat` (7 days, `interests=[Interest.beach]`, `avoid_crowds=True`: 7 DayPlans, unique attraction ids, and at least one day with a `"Free time"` note)
- [ ] **Step 2: Run** `uv run pytest tests/core/test_logistics.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `choose_hotel` and `build_schedule` following the algorithm above.
- [ ] **Step 4: Run** `uv run pytest tests/core/test_logistics.py -v`. Expected: 7 passed.
- [ ] **Step 5: Commit:** `git commit -m "feat: logistics hotel selection and geographic scheduling"`

---

### Task 8: Budget agent logic

**Files:**
- Create: `src/dubai_planner/core/budget.py`
- Test: `tests/core/test_budget.py`

**Interfaces:**
- Produces: `estimate_budget(request: TripRequest, hotel: Hotel, days: list[DayPlan], router: Router, store: DataStore) -> BudgetBreakdown`; `suggest_alternatives(request, hotel, days, budget: BudgetBreakdown, store) -> list[str]`.

**Rules:**
- Rooms = `ceil(travelers / 2)`; nights = `max(days − 1, 1)`. Accommodation = `nightly_aed × nights × rooms`, with the hotel's provenance.
- Food = (sum of restaurant `cost_aed` + `BREAKFAST_AED` × days) × travelers, provenance `estimated`.
- Activities = sum of attraction `cost_aed` × travelers, provenance `verified` only if every priced item is verified, otherwise `estimated`.
- Transport = (sum of all `leg_in.cost_aed` + the return transfer `router.leg(hotel, store.airport).cost_aed`) × travelers.
- When `within_budget` is False, `alternatives = suggest_alternatives(...)`. Each entry names a concrete change and its saving in AED:
  - the cheapest hotel in a lower tier
  - the 2 most expensive attractions, each paired with the best free attraction that shares a tag
  - "Use metro for legs X–Y" when any taxi leg is between near-metro ends

- [ ] **Step 1: Write the failing tests:**
  - `test_four_categories_present`
  - `test_accommodation_uses_nights_not_days` (4 days at AED 200/night = 600)
  - `test_travelers_scale_food_and_activities` (2 travelers ≈ 2× food of 1)
  - `test_alternatives_only_when_over_budget` (empty when `budget_aed=100000`; non-empty with AED amounts when `budget_aed=300`)
  - `test_no_budget_means_no_alternatives`
- [ ] **Step 2: Run** `uv run pytest tests/core/test_budget.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** following the rules above.
- [ ] **Step 4: Run** `uv run pytest tests/core/test_budget.py -v`. Expected: 5 passed.
- [ ] **Step 5: Commit:** `git commit -m "feat: budget estimation and alternatives"`

---

### Task 9: Review agent and revision directives

**Files:**
- Create: `src/dubai_planner/core/review.py`, `src/dubai_planner/core/revise.py`
- Test: `tests/core/test_review.py`, `tests/core/test_revise.py`

**Interfaces:**
- Produces: `review_itinerary(request: TripRequest, hotel: Hotel, days: list[DayPlan], budget: BudgetBreakdown, revision: int) -> ReviewResult`; `directives_from_review(review: ReviewResult, current: RevisionDirectives, hotel: Hotel, store: DataStore) -> RevisionDirectives` (returns a new object; never mutates `current`).

**Review checks** (one `Violation` per finding; `passed = not violations`):
- `OVER_BUDGET` when `budget.within_budget` is False; the message includes the total and the overage. This check is skipped when `budget_aed is None`.
- `SCHEDULE_OVERRUN` when any item ends after `DAY_END`, or two items overlap.
- `DUPLICATE_ACTIVITY` when an attraction id appears more than once across the trip.
- `CROWDED_PLACE` when `avoid_crowds` and an item has `crowd_level >= CROWD_THRESHOLD`.
- `PREFERENCE_MISMATCH` when a day has no attraction matching any interest (only if `interests` is non-empty).
- `MISSING_INFO` when a day lacks lunch or dinner, or a day has zero items.

**Revision escalation** (append a readable entry to `applied` for each lever used):
- `OVER_BUDGET` applies the **next unused** lever, in this order:
  1. `max_hotel_nightly_aed` = the price of the next-cheaper hotel below the current one
  2. `prefer_free=True`
  3. `max_meal_aed=60`
  4. `max_attractions_per_day -= 1` (minimum 1)
- `CROWDED_PLACE` and `DUPLICATE_ACTIVITY` add `item_id` to `exclude_ids`.
- `SCHEDULE_OVERRUN` applies `max_attractions_per_day -= 1` (minimum 1).
- `PREFERENCE_MISMATCH` and `MISSING_INFO` have no lever. They are recorded in `applied` as `"unresolved: …"`.

- [ ] **Step 1: Write the failing tests:** For review, build each violation with small hand-made `DayPlan`s: one test per code (6 tests), plus `test_clean_plan_passes` and **Review Focus 2:** `test_no_budget_never_flags_over_budget`. For revise:
  - `test_over_budget_first_lowers_hotel_cap`
  - `test_over_budget_escalates_in_order` (apply 4 times, then check `applied` order matches the list above)
  - `test_crowded_item_is_excluded`
  - `test_does_not_mutate_input`
- [ ] **Step 2: Run** `uv run pytest tests/core/test_review.py tests/core/test_revise.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules following the checks and escalation order above.
- [ ] **Step 4: Run** the same command. Expected: 12 passed.
- [ ] **Step 5: Commit:** `git commit -m "feat: rule-based review agent and revision directives"`

---

### Task 10: Event log and optional narrative

**Files:**
- Create: `src/dubai_planner/events.py`, `src/dubai_planner/core/summary.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Produces: agent name constants `ORCHESTRATOR="Orchestrator"`, `RESEARCH="Destination Research"`, `LOGISTICS="Logistics"`, `BUDGET="Budget"`, `REVIEW="Review"`; `class EventLog(listener: Callable[[AgentEvent], None] | None = None)` with `emit(agent: str, kind: str, summary: str, **data) -> AgentEvent` and `events: list[AgentEvent]`. The listener is called on every emit, and any exception it raises is caught and logged so the UI never breaks a run. Also `summarize(itinerary: Itinerary, llm: BaseChatModel | None) -> str | None`, which returns None without an LLM or on error. The prompt asks for 4–6 friendly sentences covering the highlights, the budget status and any unresolved issues.
- **Every stage helper in Tasks 11 and 12 emits a `start` and an `output` event**. `output.data` holds a compact summary (counts, chosen hotel, total AED, violation codes).

- [ ] **Step 1: Write the failing tests:** `test_emit_records_and_notifies_listener`, `test_listener_exception_does_not_propagate`, `test_summarize_without_llm_returns_none`.
- [ ] **Step 2: Run** `uv run pytest tests/test_events.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** `uv run pytest tests/test_events.py -v`. Expected: 3 passed.
- [ ] **Step 5: Commit:** `git commit -m "feat: agent event log and optional narrative"`

---

### Task 11: Mode A — LangGraph

**Files:**
- Create: `src/dubai_planner/modes/langgraph_mode.py`
- Test: `tests/modes/test_langgraph_mode.py`

**Interfaces:**
- Consumes: all of `core/*`, `Router`, `forecast`/`advisories`, `discover_restaurants_near`, `EventLog`.
- Produces: `PlanState(TypedDict, total=False)` with `request_text, request, directives, research, hotel, days, budget, review, revision: int, snapshots: Annotated[list[PlanSnapshot], operator.add]`; `build_graph(store: DataStore, router: Router, log: EventLog, llm: BaseChatModel | None = None) -> CompiledStateGraph`; `run_langgraph(text: str, *, llm=None, listener=None, store=None, router=None) -> Itinerary`.

**Graph:**
- Nodes:
  - `parse` (Orchestrator): `parse_request`, initial `RevisionDirectives()`, `revision=0`
  - `research` (Destination Research)
  - `hotel` (Logistics: `choose_hotel`)
  - `schedule` (Logistics: `build_schedule` with `max_per_day=directives.max_attractions_per_day`)
  - `budget`
  - `review` (appends a `PlanSnapshot`)
  - `replan` (Orchestrator: `directives_from_review`, `revision += 1`, emits `kind="revision"`)
  - `finalize` (Orchestrator: `summarize`)
- Edges:
  - `START→parse`
  - `parse→research` and `parse→hotel` (**parallel fan-out**)
  - `research→schedule` and `hotel→schedule` (**join**)
  - `schedule→budget→review`
  - `review` goes to a conditional `route_after_review(state) -> Literal["replan","finalize"]`: `"finalize"` if `review.passed` or `revision >= MAX_REVISIONS`, else `"replan"`
  - `replan→research` and `replan→hotel` (**feedback loop**)
  - `finalize→END`
- `run_langgraph` catches `UnsupportedRequestError` from parse and re-raises it unchanged, so the UI can show the message.

- [ ] **Step 1: Write the failing tests**

```python
def test_graph_topology_shows_parallel_and_loop(store, router):
    g = build_graph(store, router, EventLog()).get_graph()
    edges = {(e.source, e.target) for e in g.edges}
    assert {("parse", "research"), ("parse", "hotel"), ("research", "schedule"),
            ("hotel", "schedule"), ("replan", "research"), ("replan", "hotel")} <= edges

def test_spec_example_runs_offline_and_passes():
    it = run_langgraph(SPEC_EXAMPLE)
    assert it.framework == "langgraph" and len(it.days) == 4 and it.review.passed

def test_tight_budget_triggers_revision_then_passes():
    it = run_langgraph("Plan a 3-day trip to Dubai with a budget of AED 1,200. I love culture and beaches.")
    assert len(it.snapshots) >= 2
    assert any(v.code == "OVER_BUDGET" for v in it.snapshots[0].violations)
    assert it.review.passed and it.budget.within_budget

def test_impossible_budget_stops_after_max_revisions():  # Review Focus 3
    it = run_langgraph("Plan a 5-day trip to Dubai with a budget of AED 300. I like culture.")
    assert it.review.passed is False and it.review.revision == MAX_REVISIONS
    assert any(v.code == "OVER_BUDGET" for v in it.review.violations) and it.budget.alternatives

def test_events_cover_all_five_agents():
    agents = {e.agent for e in run_langgraph(SPEC_EXAMPLE).events}
    assert agents == {ORCHESTRATOR, RESEARCH, LOGISTICS, BUDGET, REVIEW}
```

- [ ] **Step 2: Run** `uv run pytest tests/modes/test_langgraph_mode.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the graph as specified above. Node functions return partial state dicts, emit events through the shared `EventLog` passed in by closure, and `run_langgraph` builds the `Itinerary` from the final state plus `log.events`. If the tight-budget test can't pass with the Task 2 data, adjust the **data** (hotel prices) rather than the test, and keep the AED 1,200 / 3-day scenario reachable with a budget-tier hotel.
- [ ] **Step 4: Run** `uv run pytest tests/modes/test_langgraph_mode.py -v`. Expected: 5 passed.
- [ ] **Step 5: Commit:** `git commit -m "feat: LangGraph mode with parallel fan-out and review loop"`

---

### Task 12: Mode B — CrewAI

**Files:**
- Create: `src/dubai_planner/modes/crewai_mode.py`
- Test: `tests/modes/test_crewai_mode.py`

**Interfaces:**
- Consumes: the same core functions as Task 11; `get_crewai_llm`.
- Produces: `@dataclass PlanningSession(log: EventLog, request: TripRequest | None = None, directives: RevisionDirectives = RevisionDirectives(), research=None, hotel=None, days=None, budget=None, review=None, snapshots: list = [])`; `make_tools(session, store, router) -> list[BaseTool]`; `build_crew(session, store, router, llm: crewai.LLM) -> Crew`; `run_crewai(text: str, *, llm=None, listener=None, store=None, router=None) -> Itinerary`.

**Design:**
- **Tools** (built with `@tool`, bound to the session by closure): `parse_request_tool(text)`, `research_places_tool()`, `choose_hotel_tool()`, `build_schedule_tool()`, `estimate_budget_tool()`, `review_itinerary_tool()`. Each tool calls the core function, **stores the result on the session**, emits events, and returns a compact JSON summary string for the LLM. A tool called before its inputs exist returns `"ERROR: run <prerequisite tool> first"` and does not raise.
- **Agents:** five agents with role, goal and backstory taken from the spec's agent descriptions. Each agent only gets the tools for its own job. The Review agent has `allow_delegation=True` (it can ask Budget or Logistics to rework); the others have `False`.
- **Tasks** (`Process.sequential`):
  1. `parse` (Orchestrator)
  2. `research` (Research, `async_execution=True`)
  3. `hotel` (Logistics, `async_execution=True`)
  4. `schedule` (Logistics, `context=[research, hotel]`)
  5. `budget` (`context=[schedule]`)
  6. `review` (`context=[budget]`)
  7. `synthesize` (Orchestrator, `context=[all prior]`), which writes the narrative

  Task descriptions take the `{request_text}` and `{feedback}` inputs.
- **`run_crewai` loop:**
  - Raises `LLMNotConfiguredError` when no LLM is available. For `revision` in `0..MAX_REVISIONS`: kick off with `feedback` (empty on the first pass, otherwise the directives' `applied` list and violation messages), with `Crew(step_callback=..., task_callback=...)` forwarding to the `EventLog`.
  - After kickoff, **fill gaps deterministically**: if the LLM skipped any tool, run that core function directly and emit `kind="error"` with `"<tool> not called by agent; filled deterministically"`.
  - Then run `review_itinerary` itself. Stop if it passes or the cap is reached; otherwise apply `directives_from_review` and loop.
  - The `Itinerary` comes from the session plus the narrative from the crew's final output.

- [ ] **Step 1: Write the failing tests** (offline; set a fake `ANTHROPIC_API_KEY` with monkeypatch so `crewai.LLM` can be built without calling out):
  - `test_tools_store_results_on_session` (call the tools in order directly and check that `session.budget` is set)
  - `test_tool_out_of_order_returns_error_string`
  - `test_crew_has_five_agents_and_parallel_tasks` (`len(crew.agents) == 5`; research and hotel tasks are async; schedule's context includes both)
  - `test_only_review_agent_can_delegate`
  - `test_run_without_llm_raises` (no key, so `LLMNotConfiguredError`)
  - `@pytest.mark.live test_live_crewai_spec_example` (`framework == "crewai"`, 4 days, all five agents appear in events)
- [ ] **Step 2: Run** `uv run pytest tests/modes/test_crewai_mode.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the module as specified above.
- [ ] **Step 4: Run** `uv run pytest tests/modes/test_crewai_mode.py -v`. Expected: 5 passed, 1 deselected. With a real key, also run `uv run pytest -m live tests/modes/test_crewai_mode.py -v`. Expected: 1 passed.
- [ ] **Step 5: Commit:** `git commit -m "feat: CrewAI mode with role agents, delegation and deterministic validation"`

---

### Task 13: Comparison runner and Streamlit app

**Files:**
- Create: `src/dubai_planner/compare.py`, `app.py`, `README.md`
- Test: `tests/test_compare.py`, `tests/test_app.py`

**Interfaces:**
- Produces: `RunMetrics(framework: str, seconds: float, revisions: int, total_aed: float, passed: bool, agent_steps: int, itinerary: Itinerary | None, error: str | None)`; `compare(text: str, *, llm_settings: Settings | None = None) -> tuple[RunMetrics, RunMetrics]`. It runs LangGraph and then CrewAI. A failing mode yields `RunMetrics` with `error` set and does not raise.

**UI (`app.py`):**
- Sidebar:
  - mode radio: `LangGraph`, `CrewAI`, `Compare both`
  - preset select: `Spec example` (the spec request), `Tight budget` (AED 1,200 / 3 days), `Impossible budget` (AED 300 / 5 days)
  - a status line showing the LLM provider and model, and whether live APIs are on or off
- Main area: a text area and a **Plan trip** button. During a run, an `st.status` block streams each `AgentEvent` as `"{agent}: {summary}"` through the `EventLog` listener, so the user can see which agent is running.
- Result tabs:
  - **Itinerary**: per-day table with time, place, area, transport mode and duration, and cost, plus hotel, assumptions, weather notes and narrative
  - **Budget**: per-category `st.bar_chart`, total against budget, alternatives, and a provenance badge per line
  - **Review & Revisions**: a table of snapshots (revision, hotel, total AED, violation codes, levers applied), so the user can see how the plan changed
  - **Agent Trace**: the full event list
  - **Compare** (compare mode only): side-by-side `RunMetrics` with a short note on the orchestration differences (graph with explicit state and edges vs role-based task delegation)
- Errors: `UnsupportedRequestError` → `st.warning(message)`; `LLMNotConfiguredError` → `st.info("CrewAI mode needs an LLM key; see .env.example")`.

- [ ] **Step 1: Write the failing tests:**
  - `test_compare_reports_crewai_error_without_key` (LangGraph metrics are `passed`; the CrewAI one has `error` set)
  - `test_app_runs_spec_example_in_langgraph_mode` (`AppTest.from_file("app.py")`, choose LangGraph, click the button, no exceptions, a metric label contains `"AED"`)
  - **Review Focus 1:** `test_app_shows_warning_for_unsupported_request` (input `"Plan a 10-day trip to Dubai"`, so `at.warning[0].value` contains `"1–7 days"`)
- [ ] **Step 2: Run** `uv run pytest tests/test_compare.py tests/test_app.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `compare.py` and `app.py` as specified above. In `README.md`, cover setup (`uv sync`, copy `.env.example`), `uv run streamlit run app.py`, the two modes, the data provenance legend, and the reminder to check RTA fares and curated prices.
- [ ] **Step 4: Run** `uv run pytest tests/test_compare.py tests/test_app.py -v`. Expected: 3 passed.
- [ ] **Step 5: Run the full suite** with `uv run pytest -v`. Expected: all pass, with live tests deselected.
- [ ] **Step 6: Manual demo check:** run `uv run streamlit run app.py` and try each preset in LangGraph mode. Expected: the spec example passes on the first review, Tight budget shows at least 2 snapshots ending in a pass, and Impossible budget shows 4 snapshots (revisions 0–3) with alternatives. With an LLM key, run Compare both on the spec example; both columns should fill in.
- [ ] **Step 7: Commit:** `git commit -m "feat: Streamlit UI with agent trace and framework comparison"`

---

## Success Criteria Traceability

| Spec success criterion | Proven by |
|---|---|
| Natural-language request to structured Dubai itinerary | Task 5 parser tests; Task 11 `test_spec_example_runs_offline_and_passes` |
| Meaningful collaboration among specialized agents | Task 11 `test_events_cover_all_five_agents`; Task 12 crew structure and delegation tests |
| Realistic schedules and estimated budgets | Task 7 schedule tests; Task 8 budget tests; Task 2 dataset checks |
| Detect and revise at least one constraint violation | Task 11 `test_tight_budget_triggers_revision_then_passes` |
| Visualize agent execution and compare frameworks | Task 13 app tests and manual demo check |
