"""
CrossFit Games Open Leaderboard API

FastAPI backend for the CrossFit leaderboard scraper.
Uses Supabase PostgreSQL for data storage.
"""

import os
from contextlib import contextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import yaml

from crossfit_leaderboard_scraper import (
    CrossFitLeaderboardScraper,
    PercentileCalculator,
)

# Path to Supabase connection config
SUPABASE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "supabase_conn.yaml")

# Connection pool (initialized on startup)
_pool: Optional[ThreadedConnectionPool] = None


def get_db_config() -> dict:
    """Load database configuration from YAML file."""
    with open(SUPABASE_CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def init_connection_pool():
    """Initialize the database connection pool."""
    global _pool
    if _pool is None:
        config = get_db_config()
        _pool = ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=config['HOST'],
            port=config['PORT'],
            dbname=config['DBNAME'],
            user=config['USER'],
            password=config['PASSWORD']
        )


@contextmanager
def get_db_connection():
    """Get a database connection from the pool."""
    if _pool is None:
        init_connection_pool()
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        _pool.putconn(conn)

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


@app.on_event("startup")
async def startup_event():
    """Initialize database connection pool on startup."""
    init_connection_pool()


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection pool on shutdown."""
    global _pool
    if _pool is not None:
        _pool.closeall()


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


class WorkoutInfo(BaseModel):
    """Information about a single workout."""
    workout_num: int
    result_type: str  # 'time', 'reps', or 'weight'


class AvailableDataResponse(BaseModel):
    """Available data in the CSV file."""
    years: list[int]
    divisions: list[str]
    regions: list[str]
    scaled: list[str]
    num_workouts: int
    total_athletes: int
    workout_types: list[WorkoutInfo]


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


# --- Database Helper Functions ---


def get_filter_conditions(year: int, division: str, region: str, scaled: str) -> tuple[str, tuple]:
    """Build SQL WHERE conditions for filtering."""
    conditions = "year = %s AND division = %s AND region = %s AND scaled = %s"
    params = (year, division, region, scaled)
    return conditions, params


def get_num_workouts_from_db(cursor, year: int, division: str, region: str, scaled: str) -> int:
    """Get the number of workouts with data for the given filters."""
    conditions, params = get_filter_conditions(year, division, region, scaled)
    
    # Check which workout columns have non-null values
    num_workouts = 0
    for i in range(1, 8):
        cursor.execute(f"""
            SELECT COUNT(*) FROM crossfit_open_results 
            WHERE {conditions} AND workout_{i}_display IS NOT NULL
            LIMIT 1
        """, params)
        count = cursor.fetchone()[0]
        if count > 0:
            num_workouts = i
        else:
            break
    return num_workouts


def detect_workout_type_from_db(cursor, year: int, division: str, region: str, scaled: str, workout_num: int) -> str:
    """Detect the result type for a workout based on sample display values."""
    conditions, params = get_filter_conditions(year, division, region, scaled)
    
    cursor.execute(f"""
        SELECT workout_{workout_num}_display 
        FROM crossfit_open_results 
        WHERE {conditions} AND workout_{workout_num}_display IS NOT NULL
        LIMIT 100
    """, params)
    
    rows = cursor.fetchall()
    if not rows:
        return "reps"
    
    time_count = 0
    reps_count = 0
    weight_count = 0
    
    import re
    for (val,) in rows:
        val_str = str(val).strip().lower()
        if ':' in val_str and 'lbs' not in val_str and 'kg' not in val_str:
            time_count += 1
        elif 'lbs' in val_str or 'kg' in val_str or 'lb' in val_str:
            weight_count += 1
        elif re.match(r'^\d+\s*(reps?)?$', val_str):
            reps_count += 1
    
    if time_count >= reps_count and time_count >= weight_count:
        return "time"
    elif weight_count >= reps_count:
        return "weight"
    else:
        return "reps"


def calculate_percentile_from_db(cursor, year: int, division: str, region: str, scaled: str, 
                                   workout_num: int, result: str) -> tuple[float, int, int]:
    """
    Calculate percentile and rank for a workout result using database queries.
    
    Returns: (percentile, rank, total_athletes)
    """
    conditions, params = get_filter_conditions(year, division, region, scaled)
    
    # Parse the user's result
    user_type, user_value = PercentileCalculator.parse_result(result)
    
    # Get total athletes with valid results for this workout
    cursor.execute(f"""
        SELECT COUNT(*) FROM crossfit_open_results 
        WHERE {conditions} AND workout_{workout_num}_display IS NOT NULL
    """, params)
    total_athletes = cursor.fetchone()[0]
    
    if total_athletes == 0:
        raise ValueError(f"No valid scores available for workout {workout_num}")
    
    # Get all results and calculate percentile
    cursor.execute(f"""
        SELECT workout_{workout_num}_display FROM crossfit_open_results 
        WHERE {conditions} AND workout_{workout_num}_display IS NOT NULL
    """, params)
    
    athletes_beaten = 0
    athletes_better = 0
    
    for (display_val,) in cursor.fetchall():
        try:
            other_type, other_value = PercentileCalculator.parse_result(str(display_val))
            
            if user_type == "time":
                if other_type == "reps":
                    athletes_beaten += 1
                elif other_type == "time":
                    if other_value > user_value:
                        athletes_beaten += 1
                    elif other_value < user_value:
                        athletes_better += 1
            elif user_type == "weight":
                if other_type == "weight":
                    if other_value < user_value:
                        athletes_beaten += 1
                    elif other_value > user_value:
                        athletes_better += 1
            else:  # reps
                if other_type == "time":
                    athletes_better += 1
                elif other_type == "reps":
                    if other_value < user_value:
                        athletes_beaten += 1
                    elif other_value > user_value:
                        athletes_better += 1
        except ValueError:
            continue
    
    rank = athletes_better + 1
    # Percentile = percentage of athletes you performed better than or equal to
    # If you're ranked R out of N, you beat (N - R) athletes
    percentile = ((total_athletes - rank) / total_athletes) * 100
    
    return round(percentile, 2), rank, total_athletes


def calculate_distribution_from_db(cursor, year: int, division: str, region: str, scaled: str,
                                     workout_num: int, user_percentile: float) -> list[DistributionPoint]:
    """Calculate the distribution of athletes across percentile buckets."""
    conditions, params = get_filter_conditions(year, division, region, scaled)
    
    # Get total valid athletes
    cursor.execute(f"""
        SELECT COUNT(*) FROM crossfit_open_results 
        WHERE {conditions} 
        AND workout_{workout_num}_display IS NOT NULL 
        AND workout_{workout_num}_rank IS NOT NULL
    """, params)
    total_valid = cursor.fetchone()[0]
    
    if total_valid == 0:
        return []
    
    # Use ROW_NUMBER to get sequential positions (handles ties correctly)
    # Then count how many athletes fall into each percentile bucket
    # Position 1 = best performer, Position N = worst performer
    # Percentile for position P out of N = ((N - P) / N) * 100
    # So for bucket X-Y%, we want positions where: X <= ((N-P)/N)*100 < Y
    # Solving: position > N * (100 - Y) / 100 AND position <= N * (100 - X) / 100
    
    distribution = []
    bucket_size = 5
    
    for i in range(0, 100, bucket_size):
        bucket_start = i
        bucket_end = i + bucket_size
        
        # Calculate position range for this percentile bucket
        # Higher percentile = lower position (better rank)
        pos_start = int(total_valid * (100 - bucket_end) / 100)  # exclusive lower bound
        pos_end = int(total_valid * (100 - bucket_start) / 100)  # inclusive upper bound
        
        # Use ROW_NUMBER to assign sequential positions based on workout rank
        cursor.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT ROW_NUMBER() OVER (ORDER BY workout_{workout_num}_rank ASC) as pos
                FROM crossfit_open_results 
                WHERE {conditions} 
                AND workout_{workout_num}_display IS NOT NULL 
                AND workout_{workout_num}_rank IS NOT NULL
            ) AS ranked
            WHERE pos > %s AND pos <= %s
        """, params + (pos_start, pos_end))
        
        athlete_count = cursor.fetchone()[0]
        is_user_bucket = bucket_start <= user_percentile < bucket_end
        
        distribution.append(DistributionPoint(
            percentile_range=f"{bucket_start}-{bucket_end}%",
            athlete_count=athlete_count,
            is_user_bucket=is_user_bucket
        ))
    
    return distribution


def calculate_overall_percentile_from_db(cursor, year: int, division: str, region: str, scaled: str,
                                          results: dict[int, str]) -> tuple[float, int, int, int, list]:
    """
    Calculate overall percentile based on combined workout rankings.
    
    Returns: (percentile, rank, total_athletes, score, individual_ranks)
    """
    conditions, params = get_filter_conditions(year, division, region, scaled)
    
    # Find individual ranks for user's results
    user_ranks = []
    for workout_num, result in results.items():
        user_type, user_value = PercentileCalculator.parse_result(result)
        
        # Find matching results to get the rank
        cursor.execute(f"""
            SELECT workout_{workout_num}_rank FROM crossfit_open_results 
            WHERE {conditions} 
            AND workout_{workout_num}_display IS NOT NULL 
            AND workout_{workout_num}_rank IS NOT NULL
        """, params)
        
        matching_ranks = []
        for (rank_val,) in cursor.fetchall():
            # We need to check against display to find matching results
            cursor.execute(f"""
                SELECT workout_{workout_num}_display, workout_{workout_num}_rank 
                FROM crossfit_open_results 
                WHERE {conditions} 
                AND workout_{workout_num}_display IS NOT NULL
            """, params)
            break  # Only need to query once
        
        # Re-query to find matches
        cursor.execute(f"""
            SELECT workout_{workout_num}_display, workout_{workout_num}_rank 
            FROM crossfit_open_results 
            WHERE {conditions} 
            AND workout_{workout_num}_display IS NOT NULL 
            AND workout_{workout_num}_rank IS NOT NULL
        """, params)
        
        for display_val, rank_val in cursor.fetchall():
            try:
                other_type, other_value = PercentileCalculator.parse_result(str(display_val))
                if other_type == user_type and other_value == user_value:
                    matching_ranks.append(int(rank_val))
            except ValueError:
                continue
        
        if matching_ranks:
            avg_rank = round(sum(matching_ranks) / len(matching_ranks))
            user_ranks.append((workout_num, avg_rank))
        else:
            _, calc_rank, _ = calculate_percentile_from_db(
                cursor, year, division, region, scaled, workout_num, result
            )
            user_ranks.append((workout_num, calc_rank))
    
    user_score = sum(rank for _, rank in user_ranks)
    
    # Build dynamic SQL for sum of ranks
    workout_nums = list(results.keys())
    rank_cols = [f"workout_{w}_rank" for w in workout_nums]
    rank_sum_sql = " + ".join(rank_cols)
    null_checks = " AND ".join([f"{col} IS NOT NULL" for col in rank_cols])
    
    # Get total athletes with all workouts completed
    cursor.execute(f"""
        SELECT COUNT(*) FROM crossfit_open_results 
        WHERE {conditions} AND {null_checks}
    """, params)
    total_athletes = cursor.fetchone()[0]
    
    if total_athletes == 0:
        raise ValueError("No valid overall scores available")
    
    # Count athletes with better (lower) and worse (higher) scores
    cursor.execute(f"""
        SELECT COUNT(*) FROM crossfit_open_results 
        WHERE {conditions} AND {null_checks} AND ({rank_sum_sql}) > %s
    """, params + (user_score,))
    athletes_beaten = cursor.fetchone()[0]
    
    cursor.execute(f"""
        SELECT COUNT(*) FROM crossfit_open_results 
        WHERE {conditions} AND {null_checks} AND ({rank_sum_sql}) < %s
    """, params + (user_score,))
    athletes_with_better_score = cursor.fetchone()[0]
    
    percentile = (athletes_beaten / total_athletes) * 100
    rank = athletes_with_better_score + 1
    
    return round(percentile, 2), int(rank), total_athletes, user_score, user_ranks


# --- API Endpoints ---


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "message": "CrossFit Open Leaderboard API is running"}


@app.get("/available-data", response_model=AvailableDataResponse, tags=["Data"])
async def get_available_data():
    """
    Get the available data options from the database.
    
    Returns the unique years, divisions, regions, and scaled types available,
    plus the total number of athletes and workouts.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get unique values
            cursor.execute("SELECT DISTINCT year FROM crossfit_open_results ORDER BY year DESC")
            years = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("SELECT DISTINCT division FROM crossfit_open_results ORDER BY division")
            divisions = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("SELECT DISTINCT region FROM crossfit_open_results ORDER BY region")
            regions = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("SELECT DISTINCT scaled FROM crossfit_open_results ORDER BY scaled")
            scaled = [row[0] for row in cursor.fetchall()]
            
            # Get total athletes
            cursor.execute("SELECT COUNT(*) FROM crossfit_open_results")
            total_athletes = cursor.fetchone()[0]
            
            # Get number of workouts (check which columns have data)
            # Use the first available combination to detect workouts
            if years and divisions and regions and scaled:
                num_workouts = get_num_workouts_from_db(
                    cursor, years[0], divisions[0], regions[0], scaled[0]
                )
                
                # Detect workout types
                workout_types = []
                for i in range(1, num_workouts + 1):
                    result_type = detect_workout_type_from_db(
                        cursor, years[0], divisions[0], regions[0], scaled[0], i
                    )
                    workout_types.append(WorkoutInfo(workout_num=i, result_type=result_type))
            else:
                num_workouts = 0
                workout_types = []
            
            cursor.close()
            
            return AvailableDataResponse(
                years=years,
                divisions=divisions,
                regions=regions,
                scaled=scaled,
                num_workouts=num_workouts,
                total_athletes=total_athletes,
                workout_types=workout_types
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/load-data", tags=["Data"])
async def load_data(
    year: int = Query(default=2025, description="Competition year"),
    division: str = Query(default="Men Individual", description="Division name"),
    region: str = Query(default="Worldwide", description="Region name"),
    scaled: str = Query(default="Rx'd", description="Workout type"),
):
    """
    Check data availability for the specified parameters.
    
    This endpoint checks if data exists in the database for the given filters.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            conditions, params = get_filter_conditions(year, division, region, scaled)
            
            cursor.execute(f"""
                SELECT COUNT(*) FROM crossfit_open_results WHERE {conditions}
            """, params)
            total_athletes = cursor.fetchone()[0]
            
            if total_athletes == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for year={year}, division={division}, region={region}, scaled={scaled}"
                )
            
            num_workouts = get_num_workouts_from_db(cursor, year, division, region, scaled)
            cursor.close()
            
            return {
                "success": True,
                "message": "Data available",
                "total_athletes": total_athletes,
                "num_workouts": num_workouts
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
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if data exists
            conditions, params = get_filter_conditions(year, division, region, scaled)
            cursor.execute(f"SELECT COUNT(*) FROM crossfit_open_results WHERE {conditions}", params)
            if cursor.fetchone()[0] == 0:
                raise HTTPException(status_code=404, detail="No data available")
            
            # Get number of workouts
            num_workouts = get_num_workouts_from_db(cursor, year, division, region, scaled)
            
            if workout_num > num_workouts:
                raise HTTPException(
                    status_code=400,
                    detail=f"Workout {workout_num} not available. Max workouts: {num_workouts}"
                )
            
            # Calculate percentile
            percentile, rank, total = calculate_percentile_from_db(
                cursor, year, division, region, scaled, workout_num, result
            )
            
            # Calculate distribution
            distribution = calculate_distribution_from_db(
                cursor, year, division, region, scaled, workout_num, percentile
            )
            
            cursor.close()
            
            return WorkoutResultResponse(
                percentile=percentile,
                rank=rank,
                total_athletes=total,
                num_workouts=num_workouts,
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
    max_pages: Optional[int] = Query(default=None, description="Max pages to scrape (ignored, kept for compatibility)"),
    limit: int = Query(default=100, ge=1, le=1000, description="Number of athletes to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    use_cache: bool = Query(default=True, description="Ignored, kept for compatibility"),
):
    """
    Fetch leaderboard data from database.
    
    Returns athlete rankings and workout results with pagination.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            conditions, params = get_filter_conditions(year, division, region, scaled)
            
            # Get total count
            cursor.execute(f"SELECT COUNT(*) FROM crossfit_open_results WHERE {conditions}", params)
            total_athletes = cursor.fetchone()[0]
            
            if total_athletes == 0:
                raise HTTPException(status_code=404, detail="No data found")
            
            # Get number of workouts
            num_workouts = get_num_workouts_from_db(cursor, year, division, region, scaled)
            
            # Build column list for workout data
            workout_cols = []
            for i in range(1, num_workouts + 1):
                workout_cols.extend([
                    f"workout_{i}_display",
                    f"workout_{i}_rank",
                    f"workout_{i}_score"
                ])
            workout_cols_str = ", ".join(workout_cols) if workout_cols else ""
            
            # Fetch paginated data
            base_cols = "id, year, division, region, scaled, sort, rank, name, country"
            all_cols = f"{base_cols}, {workout_cols_str}" if workout_cols_str else base_cols
            
            cursor.execute(f"""
                SELECT {all_cols} FROM crossfit_open_results 
                WHERE {conditions} 
                ORDER BY rank 
                LIMIT %s OFFSET %s
            """, params + (limit, offset))
            
            # Get column names
            col_names = [desc[0] for desc in cursor.description]
            
            # Convert to list of dicts
            athletes = []
            for row in cursor.fetchall():
                athlete = dict(zip(col_names, row))
                athletes.append(athlete)
            
            cursor.close()
            
            return LeaderboardResponse(
                year=year,
                division=division,
                region=region,
                scaled=scaled,
                sort=sort,
                total_athletes=total_athletes,
                num_workouts=num_workouts,
                athletes=athletes,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")


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
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if data exists
            conditions, params = get_filter_conditions(year, division, region, scaled)
            cursor.execute(f"SELECT COUNT(*) FROM crossfit_open_results WHERE {conditions}", params)
            if cursor.fetchone()[0] == 0:
                raise HTTPException(status_code=404, detail="No data available")
            
            # Get number of workouts
            num_workouts = get_num_workouts_from_db(cursor, year, division, region, scaled)
            
            if request.workout_num > num_workouts:
                raise HTTPException(
                    status_code=400,
                    detail=f"Workout {request.workout_num} not available. Max workouts: {num_workouts}"
                )
            
            # Calculate percentile
            percentile, rank, total = calculate_percentile_from_db(
                cursor, year, division, region, scaled, request.workout_num, request.result
            )
            
            cursor.close()
            
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
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if data exists
            conditions, params = get_filter_conditions(year, division, region, scaled)
            cursor.execute(f"SELECT COUNT(*) FROM crossfit_open_results WHERE {conditions}", params)
            if cursor.fetchone()[0] == 0:
                raise HTTPException(status_code=404, detail="No data available")
            
            # Convert string keys to int if needed
            results = {int(k): v for k, v in request.results.items()}
            
            # Calculate overall percentile
            percentile, rank, total, score, individual_ranks = calculate_overall_percentile_from_db(
                cursor, year, division, region, scaled, results
            )
            
            cursor.close()
            
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


@app.get("/db/status", tags=["Database"])
async def db_status():
    """Check database connection status."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM crossfit_open_results")
            count = cursor.fetchone()[0]
            cursor.close()
            return {
                "status": "connected",
                "total_records": count
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
