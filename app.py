"""
CrossFit Games Open Leaderboard API

FastAPI backend for the CrossFit leaderboard scraper.
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

from crossfit_leaderboard_scraper import (
    CrossFitLeaderboardScraper,
    PercentileCalculator,
)

# Path to the CSV data file
CSV_DATA_PATH = os.path.join(os.path.dirname(__file__), "results_2025_men_worlwide_Rx.csv")

# Pre-load the CSV data at startup
_csv_data: Optional[pd.DataFrame] = None

app = FastAPI(
    title="CrossFit Open Leaderboard API",
    description="API for scraping CrossFit Games Open leaderboard and calculating percentiles",
    version="1.0.0",
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for scraped data to avoid repeated API calls
_cache: dict = {}


def get_cache_key(year: int, division: str, region: str, scaled: str, sort: str) -> str:
    """Generate a cache key for the given parameters."""
    return f"{year}:{division}:{region}:{scaled}:{sort}"


# --- Pydantic Models ---


class LeaderboardParams(BaseModel):
    """Parameters for leaderboard queries."""
    year: int = Field(default=2025, description="Competition year")
    division: str = Field(default="Men Individual", description="Division name")
    region: str = Field(default="Worldwide", description="Region name")
    scaled: str = Field(default="Rx'd", description="Workout type")
    sort: str = Field(default="Overall", description="Sort order")
    max_pages: Optional[int] = Field(default=None, description="Max pages to scrape (None for all)")


class PercentileRequest(BaseModel):
    """Request body for percentile calculation."""
    workout_num: int = Field(..., ge=1, le=10, description="Workout number (1-10)")
    result: str = Field(..., description="Result string (e.g., '300 reps', '4:01', '130 lbs')")


class OverallPercentileRequest(BaseModel):
    """Request body for overall percentile calculation."""
    results: dict[int, str] = Field(
        ...,
        description="Dictionary mapping workout number to result string",
        example={1: "300 reps", 2: "4:30", 3: "10:00"}
    )


class PercentileResponse(BaseModel):
    """Response for percentile calculation."""
    workout_num: int
    result: str
    percentile: float
    rank: int
    total_athletes: int


class OverallPercentileResponse(BaseModel):
    """Response for overall percentile calculation."""
    percentile: float
    rank: int
    total_athletes: int
    score: int
    individual_ranks: list[dict]


class AthleteData(BaseModel):
    """Athlete data from the leaderboard."""
    rank: int
    name: str
    country: str
    workouts: dict


class LeaderboardResponse(BaseModel):
    """Response for leaderboard data."""
    year: int
    division: str
    region: str
    scaled: str
    sort: str
    total_athletes: int
    num_workouts: int
    athletes: list[dict]


class OptionsResponse(BaseModel):
    """Available options for API parameters."""
    divisions: list[str]
    regions: list[str]
    scaled: list[str]
    sort: list[str]


class AvailableDataResponse(BaseModel):
    """Available data in the CSV file."""
    years: list[int]
    divisions: list[str]
    regions: list[str]
    scaled: list[str]
    num_workouts: int
    total_athletes: int


class DistributionPoint(BaseModel):
    """A single point in the distribution histogram."""
    percentile_range: str
    athlete_count: int
    is_user_bucket: bool


class WorkoutResultResponse(BaseModel):
    """Response for workout result calculation."""
    percentile: float
    rank: int
    total_athletes: int
    num_workouts: int
    distribution: list[DistributionPoint]


# --- Helper Functions ---


def load_csv_data() -> pd.DataFrame:
    """Load and return the CSV data, caching it globally."""
    global _csv_data
    if _csv_data is None:
        if not os.path.exists(CSV_DATA_PATH):
            raise HTTPException(status_code=500, detail=f"CSV data file not found: {CSV_DATA_PATH}")
        _csv_data = pd.read_csv(CSV_DATA_PATH)
    return _csv_data


def get_or_create_scraper(year: int, division: str, region: str, scaled: str, sort: str = "Overall") -> CrossFitLeaderboardScraper:
    """Get a scraper instance from cache or create one from CSV data."""
    cache_key = get_cache_key(year, division, region, scaled, sort)
    
    if cache_key in _cache:
        return _cache[cache_key]
    
    # Load from CSV
    df = load_csv_data()
    
    # Filter data
    filtered_df = df[
        (df["year"] == year) &
        (df["division"] == division) &
        (df["region"] == region) &
        (df["scaled"] == scaled)
    ]
    
    if filtered_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for year={year}, division={division}, region={region}, scaled={scaled}"
        )
    
    # Create scraper instance
    scraper = CrossFitLeaderboardScraper(
        year=year,
        division=division,
        region=region,
        scaled=scaled,
        sort=sort,
    )
    scraper.df = filtered_df.copy()
    scraper.num_workouts = scraper._detect_num_workouts()
    
    # Cache it
    _cache[cache_key] = scraper
    
    return scraper


def calculate_distribution(df: pd.DataFrame, workout_num: int, user_percentile: float) -> list[DistributionPoint]:
    """Calculate the distribution of athletes across percentile buckets."""
    display_col = f"workout_{workout_num}_display"
    
    if display_col not in df.columns:
        return []
    
    # Filter out null values
    valid_count = df[display_col].notna().sum()
    
    if valid_count == 0:
        return []
    
    # Create distribution buckets (0-5%, 5-10%, ..., 95-100%)
    distribution = []
    bucket_size = 5
    
    for i in range(0, 100, bucket_size):
        bucket_start = i
        bucket_end = i + bucket_size
        
        # Calculate athlete count for this bucket (approximate based on uniform distribution)
        # In a real scenario, we'd calculate this based on actual rank distribution
        athlete_count = int(valid_count * bucket_size / 100)
        
        # Check if user is in this bucket
        is_user_bucket = bucket_start <= user_percentile < bucket_end
        
        distribution.append(DistributionPoint(
            percentile_range=f"{bucket_start}-{bucket_end}%",
            athlete_count=athlete_count,
            is_user_bucket=is_user_bucket
        ))
    
    return distribution


# --- API Endpoints ---


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "message": "CrossFit Open Leaderboard API is running"}


@app.get("/available-data", response_model=AvailableDataResponse, tags=["Data"])
async def get_available_data():
    """
    Get the available data options from the CSV file.
    
    Returns the unique years, divisions, regions, and scaled types available,
    plus the total number of athletes and workouts.
    """
    try:
        df = load_csv_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Get unique values
    years = sorted(df["year"].unique().tolist(), reverse=True)
    divisions = sorted(df["division"].unique().tolist())
    regions = sorted(df["region"].unique().tolist())
    scaled = sorted(df["scaled"].unique().tolist())
    
    # Count workouts by checking columns
    num_workouts = 0
    for i in range(1, 20):
        if f"workout_{i}_display" in df.columns:
            num_workouts = i
        else:
            break
    
    return AvailableDataResponse(
        years=years,
        divisions=divisions,
        regions=regions,
        scaled=scaled,
        num_workouts=num_workouts,
        total_athletes=len(df)
    )


@app.get("/load-data", tags=["Data"])
async def load_data(
    year: int = Query(default=2025, description="Competition year"),
    division: str = Query(default="Men Individual", description="Division name"),
    region: str = Query(default="Worldwide", description="Region name"),
    scaled: str = Query(default="Rx'd", description="Workout type"),
):
    """
    Load and cache data for the specified parameters.
    
    This endpoint loads data from the CSV file and caches it for subsequent
    percentile calculations.
    """
    try:
        scraper = get_or_create_scraper(year, division, region, scaled)
        df = scraper.get_dataframe()
        
        return {
            "success": True,
            "message": "Data loaded successfully",
            "total_athletes": len(df) if df is not None else 0,
            "num_workouts": scraper.get_num_workouts()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workout-result", response_model=WorkoutResultResponse, tags=["Results"])
async def get_workout_result(
    workout_num: int = Query(..., ge=1, le=10, description="Workout number (1-10)"),
    result: str = Query(..., description="Result string (e.g., '300 reps', '4:01', '130 lbs')"),
    year: int = Query(default=2025, description="Competition year"),
    division: str = Query(default="Men Individual", description="Division name"),
    region: str = Query(default="Worldwide", description="Region name"),
    scaled: str = Query(default="Rx'd", description="Workout type"),
):
    """
    Calculate percentile and get distribution for a workout result.
    
    This is the main endpoint for the frontend to get all the data needed
    to display the results dashboard.
    """
    try:
        scraper = get_or_create_scraper(year, division, region, scaled)
        df = scraper.get_dataframe()
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        if workout_num > scraper.get_num_workouts():
            raise HTTPException(
                status_code=400,
                detail=f"Workout {workout_num} not available. Max workouts: {scraper.get_num_workouts()}"
            )
        
        calculator = PercentileCalculator(df)
        percentile, rank, total = calculator.calculate_percentile_and_rank(workout_num, result)
        
        # Calculate distribution
        distribution = calculate_distribution(df, workout_num, percentile)
        
        return WorkoutResultResponse(
            percentile=percentile,
            rank=rank,
            total_athletes=total,
            num_workouts=scraper.get_num_workouts(),
            distribution=distribution
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/options", response_model=OptionsResponse, tags=["Options"])
async def get_options():
    """Get available options for divisions, regions, scaled types, and sort orders."""
    return OptionsResponse(
        divisions=list(CrossFitLeaderboardScraper.DIVISIONS.values()),
        regions=list(CrossFitLeaderboardScraper.REGIONS.values()),
        scaled=list(CrossFitLeaderboardScraper.SCALED.values()),
        sort=list(CrossFitLeaderboardScraper.SORT.values()),
    )


@app.get("/leaderboard", response_model=LeaderboardResponse, tags=["Leaderboard"])
async def get_leaderboard(
    year: int = Query(default=2025, description="Competition year"),
    division: str = Query(default="Men Individual", description="Division name"),
    region: str = Query(default="Worldwide", description="Region name"),
    scaled: str = Query(default="Rx'd", description="Workout type"),
    sort: str = Query(default="Overall", description="Sort order"),
    max_pages: Optional[int] = Query(default=None, description="Max pages to scrape"),
    limit: int = Query(default=100, ge=1, le=1000, description="Number of athletes to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    use_cache: bool = Query(default=True, description="Use cached data if available"),
):
    """
    Fetch leaderboard data from CrossFit Games API.
    
    This endpoint scrapes the official CrossFit Games leaderboard and returns
    athlete rankings and workout results.
    """
    cache_key = get_cache_key(year, division, region, scaled, sort)
    
    # Check cache
    if use_cache and cache_key in _cache:
        scraper = _cache[cache_key]
    else:
        # Validate parameters
        if division not in CrossFitLeaderboardScraper.DIVISIONS_REVERSE:
            raise HTTPException(status_code=400, detail=f"Invalid division: {division}")
        if region not in CrossFitLeaderboardScraper.REGIONS_REVERSE:
            raise HTTPException(status_code=400, detail=f"Invalid region: {region}")
        if scaled not in CrossFitLeaderboardScraper.SCALED_REVERSE:
            raise HTTPException(status_code=400, detail=f"Invalid scaled type: {scaled}")
        if sort not in CrossFitLeaderboardScraper.SORT_REVERSE:
            raise HTTPException(status_code=400, detail=f"Invalid sort order: {sort}")
        
        try:
            scraper = CrossFitLeaderboardScraper(
                year=year,
                division=division,
                region=region,
                scaled=scaled,
                sort=sort,
            )
            scraper.scrape(max_pages=max_pages, verbose=False)
            _cache[cache_key] = scraper
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error scraping data: {str(e)}")
    
    df = scraper.get_dataframe()
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    
    # Apply pagination
    paginated_df = df.iloc[offset:offset + limit]
    
    # Convert to list of dicts
    athletes = paginated_df.to_dict(orient="records")
    
    return LeaderboardResponse(
        year=year,
        division=division,
        region=region,
        scaled=scaled,
        sort=sort,
        total_athletes=len(df),
        num_workouts=scraper.get_num_workouts(),
        athletes=athletes,
    )


@app.post("/percentile", response_model=PercentileResponse, tags=["Percentile"])
async def calculate_percentile(
    request: PercentileRequest,
    year: int = Query(default=2025, description="Competition year"),
    division: str = Query(default="Men Individual", description="Division name"),
    region: str = Query(default="Worldwide", description="Region name"),
    scaled: str = Query(default="Rx'd", description="Workout type"),
    sort: str = Query(default="Overall", description="Sort order"),
):
    """
    Calculate percentile for a single workout result.
    
    Returns the percentile ranking (0-100, higher is better) and position
    in the leaderboard for the given workout result.
    """
    try:
        scraper = get_or_create_scraper(year, division, region, scaled, sort)
        df = scraper.get_dataframe()
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        if request.workout_num > scraper.get_num_workouts():
            raise HTTPException(
                status_code=400,
                detail=f"Workout {request.workout_num} not available. Max workouts: {scraper.get_num_workouts()}"
            )
        
        calculator = PercentileCalculator(df)
        percentile, rank, total = calculator.calculate_percentile_and_rank(
            request.workout_num, request.result
        )
        
        return PercentileResponse(
            workout_num=request.workout_num,
            result=request.result,
            percentile=percentile,
            rank=rank,
            total_athletes=total,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/percentile/overall", response_model=OverallPercentileResponse, tags=["Percentile"])
async def calculate_overall_percentile(
    request: OverallPercentileRequest,
    year: int = Query(default=2025, description="Competition year"),
    division: str = Query(default="Men Individual", description="Division name"),
    region: str = Query(default="Worldwide", description="Region name"),
    scaled: str = Query(default="Rx'd", description="Workout type"),
    sort: str = Query(default="Overall", description="Sort order"),
):
    """
    Calculate overall percentile based on combined workout rankings.
    
    Requires results for all workouts. Returns the overall percentile,
    rank, and breakdown of individual workout rankings.
    """
    try:
        scraper = get_or_create_scraper(year, division, region, scaled, sort)
        df = scraper.get_dataframe()
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Convert string keys to int if needed (JSON serialization converts int keys to strings)
        results = {int(k): v for k, v in request.results.items()}
        
        calculator = PercentileCalculator(df)
        percentile, rank, total, score, individual_ranks = calculator.calculate_overall_percentile_and_rank(results)
        
        return OverallPercentileResponse(
            percentile=percentile,
            rank=rank,
            total_athletes=total,
            score=score,
            individual_ranks=[
                {"workout": w, "rank": r} for w, r in individual_ranks
            ],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache", tags=["Cache"])
async def clear_cache():
    """Clear the cached leaderboard data."""
    global _cache
    _cache = {}
    return {"message": "Cache cleared"}


@app.get("/cache/status", tags=["Cache"])
async def cache_status():
    """Get the status of cached data."""
    return {
        "cached_keys": list(_cache.keys()),
        "count": len(_cache),
    }
