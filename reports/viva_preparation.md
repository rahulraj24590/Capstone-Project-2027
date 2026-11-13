# LandslideGuard — Complete Capstone Viva Examination Guide (100 Questions & Answers)

**Project Title:** LandslideGuard: AI-Based Dynamic Landslide Risk Prediction and Early Warning System  
**Academic Degree:** Bachelor of Technology (B.Tech) in Computer Science & Engineering  
**Version:** Integrated Release (Phases 1–9)  
**Total Questions:** Exactly 100 Technical, ML, GIS, Architecture, and Limitations Questions & Answers  

---

## Table of Contents
1. [Category 1: General Technical Viva Questions (Q1 – Q40)](#category-1-general-technical-viva-questions-q1--q40)
2. [Category 2: Machine Learning & Statistical Modeling Questions (Q41 – Q60)](#category-2-machine-learning--statistical-modeling-questions-q41--q60)
3. [Category 3: Geospatial, GIS & Remote Sensing Questions (Q61 – Q75)](#category-3-geospatial-gis--remote-sensing-questions-q61--q75)
4. [Category 4: System Architecture, Database & Security Questions (Q76 – Q90)](#category-4-system-architecture-database--security-questions-q76--q90)
5. [Category 5: Limitations, Scientific Ethics & Future Scope Questions (Q91 – Q100)](#category-5-limitations-scientific-ethics--future-scope-questions-q91--q100)

---

## Category 1: General Technical Viva Questions (Q1 – Q40)

#### Q1: What is the primary objective of LandslideGuard?
**A:** LandslideGuard is an academic capstone platform designed to demonstrate how static terrain characteristics (slope, elevation, aspect) can be dynamically synthesized with real-time meteorological indicators (hourly rainfall, antecedent accumulations, soil moisture) to produce explainable machine learning risk estimates, GIS visualizations, and stateful alert lifecycles.

#### Q2: What programming language and runtime environment does the project use?
**A:** Python 3.14.4 on Windows 64-bit, leveraging standard scientific and web libraries including Streamlit, FastAPI, scikit-learn, ReportLab, Folium, and SQLite 3.

#### Q3: How is the user interface implemented?
**A:** Using Streamlit 1.40+, structured into a multi-page interactive dashboard comprising 15 specialized views (Overview, Live Monitoring, Risk Analysis, Geospatial Map, Weather Analysis, India Regional Monitoring, Historical Events, Model Performance, AI Explainability, Alert Management, AI Assistant & Vision, Disaster Response & Community, Admin Panel, Data Management, and Settings).

#### Q4: How is the backend REST API implemented?
**A:** Using FastAPI (`api.py`), exposing 47 registered routes (42 application endpoints) with automatic OpenAPI/Swagger documentation (`/docs`), Pydantic request/response schema validation, and Bearer/Header token authorization.

#### Q5: What database engine is used and why?
**A:** SQLite 3 (`data/landslide_guard.db`), selected for its serverless, zero-configuration architecture, ACID transaction guarantees, fast in-process query execution, and portability across development and evaluation environments.

#### Q6: How does the system handle Python 3.14 and NumPy 2.x compatibility?
**A:** By using modern standard library typing (`str | None`), removing deprecated package calls, avoiding C-extension incompatibilities, and configuring targeted filters in `pytest.ini` for upstream NumPy 2.x shape-assignment deprecation notices in Joblib.

#### Q7: What is the purpose of the `src/` directory structure?
**A:** It enforces modular separation of concerns: `alerts/` (alert evaluation and notification), `assistant/` (RAG engine), `data_processing/` (cleaning and feature engineering), `data_providers/` (weather and GIS API clients), `database/` (models and connection management), `geospatial/` (mapping, terrain analysis, routing), `models/` (training, prediction, SHAP XAI), `monitoring/` (live stations, analytics, drift), `risk/` (dynamic risk engine), and `utils/` (auth, validation, PDF generation).

#### Q8: How is configuration managed across the project?
**A:** Centrally via `config/settings.yaml` (defining default cooldowns, risk thresholds, API timeouts) and environment variables loaded safely via `python-dotenv` in `src/utils/settings.py`.

#### Q9: What happens if an API key is missing?
**A:** The system operates under a graceful degradation policy: core weather data is fetched from Open-Meteo (which requires no API key); if optional keys (e.g., IMD API key) are absent, the UI cleanly displays an informational badge ("Not configured") without raising uncaught exceptions.

#### Q10: How are coordinates validated throughout the system?
**A:** In `src/utils/validation.py`, `validate_coordinates()` checks that latitude falls strictly within $[-90.0, +90.0]$ and longitude within $[-180.0, +180.0]$, ensuring inputs conform to the WGS84 geographic coordinate reference system (EPSG:4326).

#### Q11: What is the role of `LiveMonitor` in Phase 1?
**A:** It coordinates multi-station surveillance by querying active monitoring stations from SQLite, fetching current weather feeds, computing 1h/24h/72h rainfall accumulations, invoking the risk engine, evaluating alert conditions, and persisting observations atomically.

#### Q12: How are station health and telemetry freshness tracked?
**A:** Each weather observation records an `observed_at` timestamp and a provenance badge (`LIVE` vs. `CACHED`). If an external API call fails, cached data is used and labeled explicitly with its age.

#### Q13: What is the purpose of the India Regional Monitoring module?
**A:** It provides localized monitoring coverage for all 36 Indian States and Union Territories and 735 Districts, featuring spatial bounding lookups, representative district coordinates, SACHET CAP alert integration, and IMD status indicators.

#### Q14: How does the system prevent UI rendering crashes when data is unavailable?
**A:** All KPI metric cards, dataframes, and charts use safe dictionary lookups (`.get()`) and explicit unavailable-state fallbacks (rendering `"Unavailable"` or `"N/A"`), preventing `KeyError`, `IndexError`, or `NoneType` attribute errors.

#### Q15: How does the PDF Report Generator work?
**A:** In `src/utils/pdf_generator.py`, ReportLab dynamically constructs a publication-ready `%PDF-` document containing station metadata, environmental telemetry tables, ML risk scores, nearby historical events, nearest emergency facilities, and academic research disclaimers.

#### Q16: What is the role of the Admin Panel?
**A:** It provides authorized administrators (`role='admin'`) with capabilities to inspect and moderate crowd-sourced community hazard reports, approve or reject submissions, activate or pause monitoring stations, and review alert audit logs.

#### Q17: What credentials are used for administrator authentication?
**A:** Default evaluation credentials: Username `admin`, Password `landslideguard2026`, and REST API Secret `landslideguard-admin-secret-2026`.

#### Q18: How is password comparison implemented securely?
**A:** Using `hmac.compare_digest()` to compare SHA-256 hashes in constant time, preventing timing-based side-channel attacks.

#### Q19: What is the Community Incident Reporting module in Phase 9?
**A:** It allows local citizens and field observers to submit crowd-sourced reports of slope instability, tension cracks, or active mudflows, including descriptions, GPS coordinates, timestamps, and optional photographic evidence.

#### Q20: What is the Strict Data Segregation Invariant?
**A:** Community reports are stored strictly in the `community_reports` database table. Under no circumstances are unverified crowd-sourced reports merged into the scientific historical landslide observation catalog (`observations`), preventing data corruption and training bias.

#### Q21: What stages exist in the community report review lifecycle?
**A:** `submitted` $\rightarrow$ `under_review` $\rightarrow$ `verified` (or `rejected`), with timestamps and reviewer notes recorded in the database.

#### Q22: How does the emergency facility locator function?
**A:** In `src/geospatial/facilities.py`, it queries the OpenStreetMap Overpass API for nearby emergency amenities (hospitals, police stations, fire stations, disaster shelters) within a configurable radius, backed by an authentic offline regional fallback catalog.

#### Q23: How does the routing module calculate evacuation paths?
**A:** In `src/geospatial/routing.py`, it queries the Open Source Routing Machine (OSRM) driving API to obtain actual road geometries, distances in kilometers, and estimated travel times, falling back to great-circle geodesic vectors if the external network is unreachable.

#### Q24: What disclaimer is attached to evacuation routes?
**A:** All suggested routes are explicitly labeled as informational estimates. The system issues an explicit non-operational disclaimer warning that suggested roads may be blocked, washed out, or hazardous during an active landslide event.

#### Q25: What is the Historical Event Replay module?
**A:** It allows researchers to select an authentic past disaster (e.g., Wayanad 2024, Malin 2014) and replay historical rainfall and terrain conditions to evaluate model response retrospectively.

#### Q26: What safeguard prevents historical replays from generating false warnings?
**A:** All replay simulations are tagged `is_replay = True` and write exclusively to simulation audit tables; they can never trigger live alerts or dispatch notifications.

#### Q27: How does the LandslideGuard AI Assistant operate?
**A:** It uses a Retrieval-Augmented Generation (RAG) knowledge engine (`src/assistant/knowledge_engine.py`) that indexes project architecture, live SQLite station states, trained model metadata, and verified NDMA/GSI disaster guidelines to answer user questions factually.

#### Q28: How does the AI Assistant prevent factual hallucinations?
**A:** The knowledge engine enforces strict query constraints: if requested information (e.g., unmonitored district, unavailable weather station) does not exist in the database, it explicitly replies that the data is unavailable rather than fabricating a plausible number.

#### Q29: What languages are supported for emergency safety guidance?
**A:** English, Hindi (हिन्दी), and Punjabi (ਪੰਜਾਬੀ), providing authentic NDMA-standard "Before, During, and After a Landslide" safety protocols.

#### Q30: How does the Computer Vision terrain analyzer work?
**A:** In `src/geospatial/terrain_cv.py`, uploaded slope images are processed using OpenCV filters (Sobel/Canny edge detection for tension cracks, HSV color thresholding for bare soil/scar exposure, and Laplacian texture gradients for hummocky ground deformation).

#### Q31: What is the non-operational disclaimer for the Computer Vision module?
**A:** *"An image classification result is NOT proof of an active landslide and must NEVER independently trigger an official warning."* It serves strictly as an experimental decision-support aid.

#### Q32: What is the file upload size limit and how is it enforced?
**A:** A strict 10 MB limit (`10 * 1024 * 1024` bytes) enforced in `app.py` before decoding or processing uploaded GeoJSON or image files.

#### Q33: How does the offline synchronization cache work?
**A:** In `data/cache/offline_sync/`, recent weather feeds and station metadata are serialized to disk with SHA-256 integrity hashes, allowing the system to run in disconnected mountain environments.

#### Q34: What is the role of `pytest.ini` in the test suite?
**A:** It configures warning filters that suppress harmless third-party upstream deprecation notices (such as Joblib's NumPy 2.x array shape mutation and Starlette TestClient warnings), ensuring clean and interpretable test runs.

#### Q35: How many automated tests exist in the project, and what is their status?
**A:** 238 automated tests across 16 test modules; 100% pass with 0 failures and 0 warnings.

#### Q36: How is the application packaged for containerized deployment?
**A:** Via a multi-stage `Dockerfile` based on `python:3.14-slim`, configuring headless Streamlit, initializing SQLite tables, generating baseline data, and exposing port 8501.

#### Q37: What is the purpose of `.dockerignore` and `.gitignore`?
**A:** To ensure that credentials (`.env`), virtual environments (`.venv/`), temporary caches (`.cache/`, `__pycache__/`), local database instances (`*.db`), and large binaries are never committed to version control or baked into image layers.

#### Q38: How does the system handle HTTP 429 rate limiting from external APIs?
**A:** The base provider client (`src/data_providers/base.py`) implements exponential backoff retry logic and checks HTTP status codes; upon encountering a 429 response, it pauses, falls back to disk cache, and logs an informational warning.

#### Q39: What is the purpose of `src/utils/logging_config.py`?
**A:** It provides structured, consistent application logging across all modules, formatting log records with ISO timestamps, module names, log levels, and masking sensitive tokens.

#### Q40: What command starts the complete interactive dashboard?
**A:** `streamlit run app.py`

---

## Category 2: Machine Learning & Statistical Modeling Questions (Q41 – Q60)

#### Q41: What is the exact machine learning task defined in LandslideGuard?
**A:** Binary classification: predicting the probability $P(Y=1 | X)$ that a specific terrain point under given meteorological conditions will experience slope failure ($Y=1$) versus remaining stable ($Y=0$).

#### Q42: What 7 core features constitute the model input vector?
**A:** `slope_degrees`, `elevation_m`, `rainfall_24h_mm`, `rainfall_72h_mm`, `rainfall_1h_mm`, `vegetation_index`, and `soil_moisture`.

#### Q43: Which machine learning algorithms were trained and benchmarked?
**A:** Logistic Regression (with StandardScaler), Random Forest, scikit-learn Gradient Boosting, and XGBoost Classifier.

#### Q44: Which model was selected as `best_model.pkl` and why?
**A:** Logistic Regression, selected because it achieved the highest validation PR-AUC (1.0000 on synthetic benchmark), the lowest Brier calibration score (0.0080 on test split), fast deterministic inference, and native compatibility with SHAP `LinearExplainer`.

#### Q45: How was the dataset split between training, validation, and testing?
**A:** Using a chronological split: 68% training (1,360 rows), 12% validation (240 rows), and 20% held-out testing (400 rows).

#### Q46: Why is chronological splitting preferred over random splitting in hazard modeling?
**A:** Chronological splitting prevents temporal data leakage by ensuring that the model is evaluated exclusively on future events that occurred after the training window, mimicking true operational deployment.

#### Q47: What is class imbalance and how severe is it in natural landslide datasets?
**A:** In real-world terrain, landslides are rare events representing $< 5\%$ of observations. In our synthetic development dataset, the class distribution is 25.0% positive (500 events) and 75.0% negative (1,500 events).

#### Q48: How did you mitigate class imbalance during model training?
**A:** By configuring `class_weight='balanced'` in all estimators, which weights positive instances inversely proportional to their class frequency during loss minimization.

#### Q49: Why is PR-AUC (Precision-Recall Area Under Curve) chosen as the primary evaluation metric over ROC-AUC?
**A:** ROC-AUC includes False Positive Rate ($FPR = \frac{FP}{FP + TN}$), which is artificially suppressed when True Negatives ($TN$) are overwhelmingly large, giving a false impression of high performance. PR-AUC focuses exclusively on Precision ($\frac{TP}{TP + FP}$) and Recall ($\frac{TP}{TP + FN}$), accurately reflecting performance on rare positive events.

#### Q50: What were the measured test metrics for the selected model?
**A:** On the held-out test split ($N=400$): Precision = 0.9450, Recall = 1.0000, F1-Score = 0.9717, ROC-AUC = 1.0000, PR-AUC = 1.0000, and Brier Score = 0.0080.

#### Q51: What were the confusion matrix values on the held-out test set?
**A:** Out of 400 test samples (103 positives, 297 negatives): True Negatives = 291, False Positives = 6, False Negatives = 0, and True Positives = 103.

#### Q52: What is the Brier Score and what does it measure?
**A:** The Brier Score is the mean squared error of probabilistic predictions:
$$BS = \frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$$
It measures probabilistic calibration. A lower score ($BS \rightarrow 0$) indicates that predicted probabilities closely match true empirical outcomes.

#### Q53: What is spatial data leakage and how does it affect landslide models?
**A:** Spatial data leakage arises from spatial autocorrelation (Tobler's First Law). If landslide points from the same mountain slope or geological formation are randomly partitioned into train and test sets, the model memorizes local geographic signatures rather than learning generalizable geotechnical relationships, causing overoptimistic test metrics.

#### Q54: What cross-validation strategy should be used to combat spatial leakage in future operational versions?
**A:** Spatial Block Cross-Validation or Leave-One-Region-Out (LORO) cross-validation, where entire geographical zones (e.g., Western Ghats vs. Himalayas) are held out for testing.

#### Q55: How are missing values handled during preprocessing?
**A:** Static terrain features use training-set medians stored in `models/training_stats.json`. Missing rainfall values are **never** filled with zero; they are preserved as `NaN` to avoid masking genuine hazards, triggering a fallback scorer with explicit missing-data warnings.

#### Q56: What is SHAP (SHapley Additive exPlanations)?
**A:** A game-theoretic approach to model explainability based on Shapley values. It allocates each feature's marginal contribution to the difference between the model's actual prediction and the average base value:
$$f(x) = \phi_0 + \sum_{j=1}^M \phi_j$$

#### Q57: How is SHAP implemented in LandslideGuard?
**A:** In `src/models/explainability.py`, using `shap.LinearExplainer` for Logistic Regression and `shap.TreeExplainer` for tree models, accelerated by a 128-entry LRU cache to deliver real-time attributions.

#### Q58: What is the difference between SHAP attribution and physical causality?
**A:** SHAP explains the mathematical behavior of the model's decision function ("Feature X increased the model's score by 0.35"). It does NOT prove physical causation ("Rainfall caused the rock slope to shear"). LandslideGuard displays explicit disclaimers emphasizing that statistical correlation does not imply physical causation.

#### Q59: How does the system detect environmental anomalies in Phase 6?
**A:** In `src/monitoring/environmental_analytics.py`, by comparing current rainfall against a 30+ day ERA5 historical baseline using statistical Z-scores ($Z = \frac{x - \mu}{\sigma}$), interquartile range (IQR) thresholds, and an Isolation Forest when $\ge 60$ days of baseline data exist.

#### Q60: How is model drift monitored in Phase 6?
**A:** In `src/monitoring/drift_monitor.py`, by computing the Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS) tests between inference feature distributions and the baseline training distributions.

---

## Category 3: Geospatial, GIS & Remote Sensing Questions (Q61 – Q75)

#### Q61: What Coordinate Reference System (CRS) is standardized across LandslideGuard?
**A:** WGS84 geographic coordinate system, standard EPSG:4326 (latitude and longitude in decimal degrees).

#### Q62: How are interactive geospatial maps rendered in the dashboard?
**A:** Using Folium (Leaflet.js Python wrapper), incorporating `MarkerCluster` for historical landslide catalogs, custom circular risk markers with interactive HTML popups, and kernel density `HeatMap` overlays.

#### Q63: How are terrain slope and aspect calculated from DEM data?
**A:** Using finite-difference gradient approximations on 30 m elevation grids. If local DEM tiles are unavailable, point elevation is queried from Open-Elevation / SRTM APIs, and slope is derived from local elevation deltas:
$$\text{Slope} = \arctan\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right)$$

#### Q64: What is the Terrain Roughness Index (TRI)?
**A:** Following Riley et al. (1999), TRI measures the root-mean-square elevation difference between a focal DEM cell and its 8 neighboring cells:
$$TRI = \sqrt{\sum_{i=1}^8 (z_i - z_0)^2}$$
High TRI values indicate rugged, fractured terrain susceptible to rockfalls.

#### Q65: What is the Topographic Wetness Index (TWI)?
**A:** Formulated by Beven & Kirkby (1979), TWI describes the tendency of water to accumulate at any point in a catchment:
$$TWI = \ln\left(\frac{a}{\tan \beta}\right)$$
where $a$ is the specific upslope contributing area and $\beta$ is the local slope gradient in radians.

#### Q66: What is the "Zero Hallucination" TWI invariant in LandslideGuard?
**A:** Because computing specific catchment area $a$ strictly requires D8 flow-direction routing over a continuous hydrologically conditioned DEM, LandslideGuard **refuses** to calculate TWI for isolated point coordinates, explicitly displaying *"TWI Refused / Unavailable: Requires continuous catchment flow grid."*

#### Q67: How are spatial landslide hotspots identified in Phase 7?
**A:** Using DBSCAN (Density-Based Spatial Clustering of Applications with Noise) with the Haversine metric on radian coordinates (`eps_km = 25.0`, `min_samples = 3`), clustering historical occurrences into regional hazard zones while filtering noise outliers.

#### Q68: What is Kernel Density Estimation (KDE) and how is it used?
**A:** 2D Gaussian Kernel Density Estimation computes a continuous surface of historical event concentration over the Indian landmass, visualizing relative landslide density per square kilometer without claiming future hazard timing.

#### Q69: What disclaimer accompanies historical spatial clustering?
**A:** *"Do not call an area a future landslide hotspot based only on historical clustering."* Historical clusters reflect past occurrences and reporting bias, not guaranteed future failure sites.

#### Q70: How is vegetation health monitored using satellite remote sensing?
**A:** Using the Normalized Difference Vegetation Index (NDVI) derived from Sentinel-2 Level-2A surface reflectance:
$$NDVI = \frac{NIR - Red}{NIR + Red} = \frac{B08 - B04}{B08 + B04}$$

#### Q71: What quality control is applied to satellite NDVI calculations?
**A:** Strict cloud screening using the Scene Classification Layer (SCL). If cloud or shadow coverage exceeds 20%, the observation is flagged as cloud-obscured and filtered out to prevent false disturbance alerts.

#### Q72: How does multi-temporal satellite change detection work?
**A:** By calculating the pixel-by-pixel difference in NDVI between two cloud-screened acquisition dates ($t_1 \rightarrow t_2$):
$$\Delta NDVI = NDVI_{t_2} - NDVI_{t_1}$$
A negative shift ($\Delta NDVI < -0.25$) over non-agricultural mountain terrain highlights candidate slope disturbances or vegetative stripping.

#### Q73: What disclaimer accompanies satellite vegetation change detection?
**A:** *"Do not interpret vegetation changes alone as proof of landslide activity."* Vegetation loss can be caused by timber harvesting, forest fires, seasonal leaf drop, or farming activities.

#### Q74: How are 3D terrain surfaces generated in the application?
**A:** Using Plotly's `go.Surface` component, rendering elevation rasters in three-dimensional perspective with customizable texture drape modes (Elevation colormap, Slope gradient, or Roughness shading).

#### Q75: How does the system validate uploaded GeoJSON boundary files?
**A:** In `src/geospatial/spatial_analysis.py`, `validate_geojson()` checks RFC 7946 compliance, geometry types (Polygon/MultiPolygon), coordinate winding order, and rejects files exceeding 10 MB.

---

## Category 4: System Architecture, Database & Security Questions (Q76 – Q90)

#### Q76: Explain the 10-step canonical data flow in LandslideGuard.
**A:**
1. Region selection $\rightarrow$ 2. Geographic metadata loading $\rightarrow$ 3. Weather telemetry retrieval $\rightarrow$ 4. Input validation $\rightarrow$ 5. Risk model inference $\rightarrow$ 6. SHAP explainability $\rightarrow$ 7. Alert condition evaluation $\rightarrow$ 8. Dashboard view assembly $\rightarrow$ 9. Atomic SQLite persistence $\rightarrow$ 10. PDF assessment report generation.

#### Q77: How are SQLite database migrations managed?
**A:** Idempotently in `src/database/db.py` (`initialize_database()`), executing `CREATE TABLE IF NOT EXISTS` statements and safe `ALTER TABLE ADD COLUMN` queries wrapped in exception handlers before creating composite indexes.

#### Q78: Name 5 key database tables in the schema and their purposes.
**A:**
1. `locations`: Monitored station metadata (coordinates, elevation, slope, state).
2. `environmental_observations`: Time-series weather telemetry (1h/24h/72h rainfall, soil moisture, temperature).
3. `prediction_records`: Historical model inferences (probabilities, risk categories, SHAP inputs, warnings).
4. `alerts`: Triggered experimental alerts with lifecycle status (`new`, `acknowledged`, `resolved`).
5. `community_reports`: Segregated citizen hazard reports and photo references.

#### Q79: How are foreign keys enforced in SQLite?
**A:** Every connection executes `PRAGMA foreign_keys = ON` immediately upon opening (`src/database/db.py`), ensuring that orphaned prediction or alert records cannot be inserted without a valid linked `location_id`.

#### Q80: How does the system prevent SQL injection?
**A:** 100% of SQL queries across the application use parameterized query binding with `?` placeholders. No dynamic SQL strings are constructed via concatenation or f-strings.

#### Q81: How is the alert cooldown mechanism implemented?
**A:** In `src/alerts/alert_manager.py`, when a risk threshold is breached, the manager queries SQLite for prior alerts at that location within a configurable cooldown window (default: 120 minutes). If an active alert of equal or lower severity exists, duplicate notification is suppressed.

#### Q82: What is the upward severity escalation override?
**A:** If environmental conditions worsen during an active cooldown window (e.g., risk escalates from `High` to `Very High`), the cooldown suppression is bypassed, creating a new escalated alert tagged `is_escalation = True`.

#### Q83: What alert severity tiers are supported?
**A:** Four progressive tiers: `Low` ($P < 0.35$), `Moderate` ($0.35 \le P < 0.55$), `High` ($0.55 \le P < 0.75$), and `Very High` ($P \ge 0.75$).

#### Q84: How are external notification channels (Email/SMS) governed?
**A:** In `src/alerts/notifications.py`, notification dispatchers are implemented but configured to `enabled: false` by default in `config/settings.yaml`, ensuring that experimental alerts are never broadcast to the public without explicit administrator configuration.

#### Q85: What security measures protect file uploads in the application?
**A:**
1. Strict 10 MB file size cutoff.
2. File extension verification (`.geojson`, `.json`, `.jpg`, `.jpeg`, `.png`, `.webp`).
3. Binary magic bytes header inspection (verifying JPEG `FF D8 FF`, PNG `89 50 4E 47`, WEBP `RIFF`).
4. Filename sanitization to prevent path traversal attacks.

#### Q86: How does the application implement Role-Based Access Control (RBAC)?
**A:** In `src/utils/auth.py`:
- `admin`: Granted full write privileges (station creation, community report moderation, alert resolution).
- `viewer`: Read-only access to monitoring dashboards, maps, and analytics.

#### Q87: How are REST API routes secured in `api.py`?
**A:** Administrative endpoints (e.g., `/api/v1/admin/*`, `/api/v1/reports/{id}/review`) require an `X-Admin-Key` header or `Authorization: Bearer <token>` verified via `verify_admin_key()`.

#### Q88: How are external API credentials managed?
**A:** Using environment variables loaded through `python-dotenv`. An `.env.example` file is provided with placeholder tokens (`<your_key_here>`), and `.gitignore` prevents real `.env` files from ever entering source control.

#### Q89: How is memory managed during heavy SHAP and geospatial calculations?
**A:**
1. SHAP calculations are cached via `functools.lru_cache(maxsize=128)`.
2. Folium map markers use clustering (`MarkerCluster`) rather than rendering individual DOM nodes.
3. Satellite rasters and 3D surface meshes are subsampled (`max_samples=200`, `downsample_factor=2`) before rendering.

#### Q90: How does the system ensure zero unhandled exceptions across all pages?
**A:** By wrapping external API calls, model inferences, and database queries in comprehensive `try/except` blocks that log structured warnings and return explicit fallback states (e.g., `ValidationResult(is_valid=False)` or `"Unavailable"` UI badges).

---

## Category 5: Limitations, Scientific Ethics & Future Scope Questions (Q91 – Q100)

#### Q91: What is the most critical scientific limitation of the current LandslideGuard implementation?
**A:** All machine learning models were trained and evaluated on a **synthetic development dataset** (`data/processed/dataset.parquet`, `is_synthetic = 1`). While the software pipelines, algorithms, and visualizations are real and fully functional, the trained model weights **do not possess validated operational predictive skill in the real world**.

#### Q92: Why was synthetic data used instead of real landslide inventories for initial model training?
**A:** To build, test, and verify the end-to-end multi-phase computational architecture without being blocked by restricted geospatial data licenses, complex NASA Earthdata authentications, or unverified positive-only spatial event catalogs.

#### Q93: Can LandslideGuard be used as an official operational early warning system today?
**A:** **No.** It is an academic B.Tech research prototype. Under Indian disaster management law, official warnings may only be issued by designated statutory authorities: the Geological Survey of India (GSI), India Meteorological Department (IMD), and National Disaster Management Authority (NDMA).

#### Q94: What is the "Positive-Only Reporting Bias" in global landslide catalogs (like NASA GLC)?
**A:** Catalogs like NASA GLC only record landslides that occurred near human settlements, roads, or infrastructure where people noticed and reported them. Landslides in remote wilderness are omitted, and stable slopes (negatives) are not recorded, creating severe spatial sampling bias.

#### Q95: How should non-landslide (negative) samples be collected in real-world landslide susceptibility studies?
**A:** Using stratified random spatial sampling or buffer-based pseudo-absence sampling across geologically stable slopes with slope angles, lithology, and rainfall distributions representative of the region.

#### Q96: What is the spatial resolution limitation of open weather data?
**A:** Open-Meteo and ERA5 reanalysis grids have horizontal resolutions of ~5 km to 25 km. Severe mountain landslide events are often triggered by localized cloudbursts or convective cells spanning $< 2\text{ km}$, which coarse weather grids can underestimate.

#### Q97: Why is geological lithology and soil shear strength important, and how is it limited here?
**A:** Slope stability depends fundamentally on internal geotechnical properties: soil cohesion ($c$), internal angle of friction ($\phi$), and pore-water pressure ($u$) governed by the Factor of Safety ($FS$):
$$FS = \frac{c' + (\sigma - u)\tan\phi'}{\tau}$$
Because high-resolution borehole and geotechnical soil surveys are not openly available as continuous digital grids, LandslideGuard uses elevation and slope as accessible proxies.

#### Q98: What is the future scope for integrating IoT geotechnical sensors?
**A:** Deploying wireless in-situ sensor networks (tiltmeters, vibrating-wire piezometers measuring pore-water pressure, and time-domain reflectometry soil moisture probes) linked to LandslideGuard's station ingestion API for real-time slope deformation tracking.

#### Q99: What is the future scope for satellite radar remote sensing?
**A:** Integrating Sentinel-1 Synthetic Aperture Radar (SAR) Interferometry (InSAR) to detect millimeter-scale pre-failure surface subsidence and creep months before catastrophic slope collapse occurs.

#### Q100: How should LandslideGuard be validated in a real-world field study?
**A:** By establishing a multi-year pilot study in a high-risk district (e.g., Idukki or Wayanad in Kerala), partnering with state disaster authorities, calibrating regional empirical rainfall-intensity thresholds ($I\text{-}D$ curves), and executing blind prospective validation across monsoon seasons.
