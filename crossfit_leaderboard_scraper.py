#!/usr/bin/env python3
"""
CrossFit Games Open Leaderboard Scraper

This script scrapes the CrossFit Games Open leaderboard and provides:
1. A dataframe with ranking, athlete name, country, and workout results
2. An interactive console to calculate percentiles for given workout results

Usage:
    python crossfit_leaderboard_scraper.py --year 2025 --division "Men Individual" --region "Worldwide" --scaled "Rx'd" --sort "Overall"
    python crossfit_leaderboard_scraper.py --input data.csv --division "Men Individual"
"""

import argparse
import re
import sys
import time as time_module
from datetime import datetime, timezone
from typing import Optional, Tuple

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import requests
import yaml


class CrossFitLeaderboardScraper:
    """Scrapes CrossFit Games Open leaderboard data."""

    BASE_URL = "https://c3po.crossfit.com/api/leaderboards/v2/competitions/open/{year}/leaderboards"

    # Division mappings (ID -> Name)
    DIVISIONS = {
        1: "Men Individual",
        2: "Women Individual",
        3: "Men 35-39",
        4: "Women 35-39",
        5: "Men 40-44",
        6: "Women 40-44",
        7: "Men 45-49",
        8: "Women 45-49",
        9: "Men 50-54",
        10: "Women 50-54",
        11: "Men 55-59",
        12: "Women 55-59",
        13: "Men 60-64",
        14: "Women 60-64",
        15: "Men 65+",
        16: "Women 65+",
        17: "Teen Boys 14-15",
        18: "Teen Girls 14-15",
        19: "Teen Boys 16-17",
        20: "Teen Girls 16-17",
    }

    # Reverse mapping (Name -> ID)
    DIVISIONS_REVERSE = {v: k for k, v in DIVISIONS.items()}

    # Region mappings
    REGIONS = {
        0: "Worldwide",
        28: "Asia",
        29: "Europe",
        30: "Africa",
        31: "Central America",
        32: "Oceania",
        33: "South America",
        34: "North America West",
        35: "North America East",
    }
    REGIONS_REVERSE = {v: k for k, v in REGIONS.items()}

    # Scaled mappings
    SCALED = {
        0: "Rx'd",
        1: "Scaled",
        2: "Foundations",
    }
    SCALED_REVERSE = {v: k for k, v in SCALED.items()}

    # Sort mappings
    SORT = {
        0: "Overall",
        1: "Workout 1",
        2: "Workout 2",
        3: "Workout 3",
    }
    SORT_REVERSE = {v: k for k, v in SORT.items()}

    def __init__(
        self,
        year: int = 2025,
        division: str = "Men Individual",
        region: str = "Worldwide",
        scaled: str = "Rx'd",
        sort: str = "Overall",
    ):
        """
        Initialize the scraper with the given parameters.

        Args:
            year: Competition year (e.g., 2025)
            division: Division name (e.g., "Men Individual", "Women Individual")
            region: Region name (e.g., "Worldwide", "Europe")
            scaled: Workout type ("Rx'd", "Scaled", "Foundations")
            sort: Sort order ("Overall", "Workout 1", "Workout 2", "Workout 3")
        """
        self.year = year
        # Store human-readable names
        self.division_name = division
        self.region_name = region
        self.scaled_name = scaled
        self.sort_name = sort
        # Convert to IDs for API calls
        self.division_id = self.DIVISIONS_REVERSE.get(division, 1)
        self.region_id = self.REGIONS_REVERSE.get(region, 0)
        self.scaled_id = self.SCALED_REVERSE.get(scaled, 0)
        self.sort_id = self.SORT_REVERSE.get(sort, 0)
        self.df: Optional[pd.DataFrame] = None
        self.total_pages = 0
        self.total_competitors = 0
        self.elapsed_time = 0.0
        self.num_workouts = 0  # Will be detected from data

    def _build_url(self, page: int) -> str:
        """Build the API URL for the given page."""
        base = self.BASE_URL.format(year=self.year)
        return (
            f"{base}?view=0&division={self.division_id}&region={self.region_id}"
            f"&scaled={self.scaled_id}&sort={self.sort_id}&page={page}"
        )

    def _fetch_page(self, page: int) -> dict:
        """Fetch a single page of leaderboard data."""
        url = self._build_url(page)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def _parse_athlete_data(self, row: dict) -> dict:
        """Parse a single athlete row from the API response."""
        entrant = row.get("entrant", {})
        scores = row.get("scores", [])

        # Extract basic info with input parameters
        data = {
            "year": self.year,
            "division": self.division_name,
            "region": self.region_name,
            "scaled": self.scaled_name,
            "sort": self.sort_name,
            "rank": int(row.get("overallRank", 0)),
            "name": entrant.get("competitorName", ""),
            "country": entrant.get("countryOfOriginName", ""),
        }

        # Extract workout scores dynamically (based on available workouts)
        # Determine max ordinal from scores
        max_ordinal = max((s.get("ordinal", 0) for s in scores), default=0)
        for i in range(1, max_ordinal + 1):
            score_data = next(
                (s for s in scores if s.get("ordinal") == i), {}
            )
            data[f"workout_{i}_display"] = score_data.get("scoreDisplay", "")
            data[f"workout_{i}_rank"] = (
                int(score_data.get("rank", 0)) if score_data.get("rank") else None
            )
            # Store raw score for percentile calculations
            data[f"workout_{i}_score"] = (
                int(score_data.get("score", 0)) if score_data.get("score") else None
            )

        return data

    def scrape(self, max_pages: Optional[int] = None, verbose: bool = True) -> pd.DataFrame:
        """
        Scrape all pages of the leaderboard.

        Args:
            max_pages: Maximum number of pages to scrape (None for all)
            verbose: Print progress information

        Returns:
            DataFrame with all athlete data
        """
        start_time = time_module.perf_counter()
        athletes = []

        # Fetch first page to get total pages
        if verbose:
            print(f"Fetching leaderboard for {self.year} Open...")
            print(f"Division: {self.division_name}")
            print(f"Region: {self.region_name}")
            print(f"Workout Type: {self.scaled_name}")
            print(f"Sort: {self.sort_name}")

        first_page_data = self._fetch_page(1)
        pagination = first_page_data.get("pagination", {})
        self.total_pages = pagination.get("totalPages", 0)
        self.total_competitors = pagination.get("totalCompetitors", 0)

        if verbose:
            print(f"Total competitors: {self.total_competitors:,}")
            print(f"Total pages: {self.total_pages:,}")

        pages_to_fetch = min(max_pages, self.total_pages) if max_pages else self.total_pages

        # Parse first page
        for row in first_page_data.get("leaderboardRows", []):
            athletes.append(self._parse_athlete_data(row))

        if verbose:
            print(f"Page 1/{pages_to_fetch} fetched ({len(athletes)} athletes)")

        # Fetch remaining pages
        for page in range(2, pages_to_fetch + 1):
            try:
                page_data = self._fetch_page(page)
                for row in page_data.get("leaderboardRows", []):
                    athletes.append(self._parse_athlete_data(row))

                if verbose:
                    print(f"Page {page}/{pages_to_fetch} fetched ({len(athletes)} athletes)")

                # Small delay to be respectful to the API
                time_module.sleep(0.1)

            except requests.RequestException as e:
                print(f"Error fetching page {page}: {e}")
                continue

        self.df = pd.DataFrame(athletes)
        self.elapsed_time = time_module.perf_counter() - start_time

        # Detect number of workouts from columns
        self.num_workouts = self._detect_num_workouts()

        if verbose:
            print(f"\nTotal athletes scraped: {len(self.df):,}")
            print(f"Number of workouts detected: {self.num_workouts}")
            print(f"Time elapsed: {self.elapsed_time:.2f} seconds")

        return self.df

    def _detect_num_workouts(self) -> int:
        """Detect the number of workouts from the DataFrame columns."""
        if self.df is None or self.df.empty:
            return 0
        workout_count = 0
        for i in range(1, 20):  # Check up to 20 workouts (more than enough)
            if f"workout_{i}_display" in self.df.columns:
                workout_count = i
            else:
                break
        return workout_count

    def get_num_workouts(self) -> int:
        """Get the number of workouts in the dataset."""
        return self.num_workouts

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the scraped dataframe."""
        return self.df

    def save_to_csv(self, filename: str) -> None:
        """Save the dataframe to a CSV file."""
        if self.df is not None:
            self.df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        else:
            print("No data to save. Run scrape() first.")

    def save_to_supabase(self, config_path: str, verbose: bool = True, batch_size: int = 1000) -> None:
        """
        Save the dataframe to a Supabase PostgreSQL table using batch inserts.

        Args:
            config_path: Path to the YAML file containing Supabase connection details
            verbose: Print progress information
            batch_size: Number of rows to insert per batch (default: 1000)
        """
        if self.df is None or self.df.empty:
            print("No data to save. Run scrape() first.")
            return

        # Load connection config from YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Connect to Supabase PostgreSQL
        conn = psycopg2.connect(
            host=config['HOST'],
            port=config['PORT'],
            dbname=config['DBNAME'],
            user=config['USER'],
            password=config['PASSWORD']
        )
        cursor = conn.cursor()

        if verbose:
            print(f"Connected to Supabase database")
            print(f"Inserting {len(self.df):,} rows into crossfit_open_results (batch size: {batch_size})...")

        # Prepare the insert statement with all workout columns (1-7)
        insert_sql = """
            INSERT INTO crossfit_open_results (
                year, division, region, scaled, sort, rank, name, country,
                workout_1_display, workout_1_rank, workout_1_score,
                workout_2_display, workout_2_rank, workout_2_score,
                workout_3_display, workout_3_rank, workout_3_score,
                workout_4_display, workout_4_rank, workout_4_score,
                workout_5_display, workout_5_rank, workout_5_score,
                workout_6_display, workout_6_rank, workout_6_score,
                workout_7_display, workout_7_rank, workout_7_score,
                inserted_at
            ) VALUES %s
        """

        inserted_at = datetime.now(timezone.utc)

        # Helper to safely get column value or None
        def get_val(row, col):
            if col in row.index and pd.notna(row[col]):
                return row[col]
            return None

        # Prepare all rows as tuples
        all_values = []
        for _, row in self.df.iterrows():
            values = (
                get_val(row, 'year'),
                get_val(row, 'division'),
                get_val(row, 'region'),
                get_val(row, 'scaled'),
                get_val(row, 'sort'),
                get_val(row, 'rank'),
                get_val(row, 'name'),
                get_val(row, 'country'),
                get_val(row, 'workout_1_display'),
                get_val(row, 'workout_1_rank'),
                get_val(row, 'workout_1_score'),
                get_val(row, 'workout_2_display'),
                get_val(row, 'workout_2_rank'),
                get_val(row, 'workout_2_score'),
                get_val(row, 'workout_3_display'),
                get_val(row, 'workout_3_rank'),
                get_val(row, 'workout_3_score'),
                get_val(row, 'workout_4_display'),
                get_val(row, 'workout_4_rank'),
                get_val(row, 'workout_4_score'),
                get_val(row, 'workout_5_display'),
                get_val(row, 'workout_5_rank'),
                get_val(row, 'workout_5_score'),
                get_val(row, 'workout_6_display'),
                get_val(row, 'workout_6_rank'),
                get_val(row, 'workout_6_score'),
                get_val(row, 'workout_7_display'),
                get_val(row, 'workout_7_rank'),
                get_val(row, 'workout_7_score'),
                inserted_at,
            )
            all_values.append(values)

        # Insert in batches using execute_values (much faster than individual inserts)
        total_inserted = 0
        for i in range(0, len(all_values), batch_size):
            batch = all_values[i:i + batch_size]
            execute_values(cursor, insert_sql, batch, page_size=batch_size)
            total_inserted += len(batch)
            if verbose:
                print(f"Inserted {total_inserted:,} / {len(all_values):,} rows...")

        conn.commit()
        cursor.close()
        conn.close()

        if verbose:
            print(f"Successfully inserted {total_inserted:,} rows into crossfit_open_results")

    @classmethod
    def load_from_csv(
        cls,
        filename: str,
        year: Optional[int] = None,
        division: Optional[str] = None,
        region: Optional[str] = None,
        scaled: Optional[str] = None,
        sort: Optional[str] = None,
        verbose: bool = True,
    ) -> Tuple["CrossFitLeaderboardScraper", pd.DataFrame]:
        """
        Load data from a CSV file and optionally filter by parameters.

        Args:
            filename: Path to the CSV file
            year: Filter by year (optional)
            division: Filter by division name (optional)
            region: Filter by region name (optional)
            scaled: Filter by scaled type (optional)
            sort: Filter by sort order (optional)
            verbose: Print progress information

        Returns:
            Tuple of (scraper instance, filtered DataFrame)
        """
        start_time = time_module.perf_counter()

        if verbose:
            print(f"Loading data from {filename}...")

        df = pd.read_csv(filename)

        if verbose:
            print(f"Loaded {len(df):,} rows from file")

        # Apply filters
        if year is not None:
            df = df[df["year"] == year]
            if verbose:
                print(f"Filtered by year={year}: {len(df):,} rows")
        if division is not None:
            df = df[df["division"] == division]
            if verbose:
                print(f"Filtered by division='{division}': {len(df):,} rows")
        if region is not None:
            df = df[df["region"] == region]
            if verbose:
                print(f"Filtered by region='{region}': {len(df):,} rows")
        if scaled is not None:
            df = df[df["scaled"] == scaled]
            if verbose:
                print(f"Filtered by scaled='{scaled}': {len(df):,} rows")
        if sort is not None:
            df = df[df["sort"] == sort]
            if verbose:
                print(f"Filtered by sort='{sort}': {len(df):,} rows")

        elapsed_time = time_module.perf_counter() - start_time

        # Create a scraper instance with the parameters
        scraper = cls(
            year=year or (df["year"].iloc[0] if not df.empty and "year" in df.columns else 2025),
            division=division or (df["division"].iloc[0] if not df.empty and "division" in df.columns else "Men Individual"),
            region=region or (df["region"].iloc[0] if not df.empty and "region" in df.columns else "Worldwide"),
            scaled=scaled or (df["scaled"].iloc[0] if not df.empty and "scaled" in df.columns else "Rx'd"),
            sort=sort or (df["sort"].iloc[0] if not df.empty and "sort" in df.columns else "Overall"),
        )
        scraper.df = df
        scraper.elapsed_time = elapsed_time
        scraper.num_workouts = scraper._detect_num_workouts()

        if verbose:
            print(f"\nTotal athletes after filtering: {len(df):,}")
            print(f"Number of workouts detected: {scraper.num_workouts}")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")

        return scraper, df


class PercentileCalculator:
    """Calculate percentiles for workout results."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the calculator with the scraped dataframe.

        Args:
            df: DataFrame with workout scores
        """
        self.df = df

    @staticmethod
    def parse_result(result: str) -> Tuple[str, float]:
        """
        Parse a workout result string into a normalized value.

        Args:
            result: Result string (e.g., "300 reps", "4:01", "300", "130 lbs")

        Returns:
            Tuple of (result_type, normalized_value)
            - result_type: "reps", "time", or "weight"
            - normalized_value: reps/weight as positive number, time as total seconds
        """
        result = result.strip().lower()

        # Check for time format (MM:SS or H:MM:SS)
        time_pattern = r"^(\d+):(\d{2})(?::(\d{2}))?$"
        time_match = re.match(time_pattern, result)
        if time_match:
            groups = time_match.groups()
            if groups[2] is not None:  # H:MM:SS format
                hours = int(groups[0])
                minutes = int(groups[1])
                seconds = int(groups[2])
                total_seconds = hours * 3600 + minutes * 60 + seconds
            else:  # MM:SS format
                minutes = int(groups[0])
                seconds = int(groups[1])
                total_seconds = minutes * 60 + seconds
            return ("time", total_seconds)

        # Check for weight format (e.g., "130 lbs", "130 lb", "60 kg")
        weight_pattern = r"^(\d+(?:\.\d+)?)\s*(?:lbs?|kg|kilos?)$"
        weight_match = re.match(weight_pattern, result)
        if weight_match:
            weight = float(weight_match.group(1))
            return ("weight", weight)

        # Check for reps format
        reps_pattern = r"^(\d+)\s*(?:reps?)?$"
        reps_match = re.match(reps_pattern, result)
        if reps_match:
            reps = int(reps_match.group(1))
            return ("reps", reps)

        raise ValueError(f"Cannot parse result: '{result}'")

    def _parse_all_results(self, workout_num: int) -> list:
        """
        Parse all results for a workout into a list of (type, value) tuples.

        Args:
            workout_num: Workout number (1, 2, or 3)

        Returns:
            List of (result_type, value) tuples
        """
        display_col = f"workout_{workout_num}_display"
        results = []
        for val in self.df[display_col].dropna():
            try:
                parsed = self.parse_result(str(val))
                results.append(parsed)
            except ValueError:
                continue
        return results

    def calculate_percentile_and_rank(self, workout_num: int, result: str) -> Tuple[float, int, int]:
        """
        Calculate the percentile and ranking for a given workout result.

        For time-capped workouts, results can be either:
        - A time (MM:SS): athlete finished the workout within the time cap
        - A rep count: athlete did not finish, this is how many reps they completed

        Scoring logic:
        - Any time result beats any rep result (finishing is better than not finishing)
        - Among times: lower time is better
        - Among reps/weight: higher value is better

        Args:
            workout_num: Workout number (1, 2, or 3)
            result: Result string (e.g., "300 reps", "4:01", "130 lbs")

        Returns:
            Tuple of (percentile, rank, total_athletes)
            - percentile: 0-100, higher is better
            - rank: position in leaderboard (1 = best)
            - total_athletes: total number of athletes
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data available for percentile calculation")

        # Parse the user's result
        user_type, user_value = self.parse_result(result)

        # Get all results for this workout
        all_results = self._parse_all_results(workout_num)
        if not all_results:
            raise ValueError(f"No valid scores available for workout {workout_num}")

        total_athletes = len(all_results)
        athletes_beaten = 0  # For percentile
        athletes_better = 0  # For rank

        for other_type, other_value in all_results:
            if user_type == "time":
                # User finished with a time
                if other_type == "reps":
                    # User beats anyone who didn't finish (has reps instead of time)
                    athletes_beaten += 1
                elif other_type == "time":
                    # Both have times: lower time is better
                    if other_value > user_value:
                        athletes_beaten += 1
                    elif other_value < user_value:
                        athletes_better += 1
                elif other_type == "weight":
                    # Time doesn't directly compare to weight (different workout types)
                    pass
            elif user_type == "weight":
                # Weight-based workout: higher weight is better
                if other_type == "weight":
                    if other_value < user_value:
                        athletes_beaten += 1
                    elif other_value > user_value:
                        athletes_better += 1
                # Weight doesn't compare to time/reps (different workout types)
            else:
                # User has reps (didn't finish)
                if other_type == "time":
                    # Anyone with a time beats the user (they finished)
                    athletes_better += 1
                elif other_type == "reps":
                    # Both have reps: higher reps is better
                    if other_value < user_value:
                        athletes_beaten += 1
                    elif other_value > user_value:
                        athletes_better += 1

        percentile = (athletes_beaten / total_athletes) * 100
        rank = athletes_better + 1  # Rank = number of athletes with strictly better results + 1

        return round(percentile, 2), rank, total_athletes

    def calculate_percentile(self, workout_num: int, result: str) -> float:
        """
        Calculate the percentile for a given workout result.

        Args:
            workout_num: Workout number (1, 2, or 3)
            result: Result string (e.g., "300 reps", "4:01", "130 lbs")

        Returns:
            Percentile (0-100, higher is better)
        """
        percentile, _, _ = self.calculate_percentile_and_rank(workout_num, result)
        return percentile

    def calculate_distribution(self, workout_num: int, user_percentile: float) -> list[dict]:
        """
        Calculate the distribution of athletes across percentile buckets.

        Args:
            workout_num: Workout number (1, 2, or 3)
            user_percentile: The user's percentile for highlighting their bucket

        Returns:
            List of dicts with percentile_range, athlete_count, is_user_bucket
        """
        if self.df is None or self.df.empty:
            return []

        rank_col = f"workout_{workout_num}_rank"
        display_col = f"workout_{workout_num}_display"

        if rank_col not in self.df.columns or display_col not in self.df.columns:
            return []

        # Filter out null values - only count athletes with valid results
        valid_df = self.df[self.df[display_col].notna() & self.df[rank_col].notna()]
        total_valid = len(valid_df)

        if total_valid == 0:
            return []

        # Create distribution buckets (0-5%, 5-10%, ..., 95-100%)
        distribution = []
        bucket_size = 5

        for i in range(0, 100, bucket_size):
            bucket_start = i
            bucket_end = i + bucket_size

            # Calculate the rank range for this percentile bucket
            # Percentile X means X% of athletes are worse (have higher rank)
            # So percentile 0-5% means ranks from 95% to 100% of total
            rank_start = int(total_valid * (100 - bucket_end) / 100) + 1
            rank_end = int(total_valid * (100 - bucket_start) / 100)

            # Count athletes in this rank range
            athlete_count = len(valid_df[(valid_df[rank_col] >= rank_start) & (valid_df[rank_col] <= rank_end)])

            # Check if user is in this bucket
            is_user_bucket = bucket_start <= user_percentile < bucket_end

            distribution.append({
                "percentile_range": f"{bucket_start}-{bucket_end}%",
                "athlete_count": athlete_count,
                "is_user_bucket": is_user_bucket
            })

        return distribution

    def calculate_overall_percentile_and_rank(
        self, results: dict
    ) -> Tuple[float, int, int, int, list]:
        """
        Calculate the overall percentile based on combined workout rankings.

        Args:
            results: Dictionary mapping workout number to result string
                     e.g., {1: "300 reps", 2: "4:30", 3: "10:00"}

        Returns:
            Tuple of (percentile, rank, total_athletes, score, individual_ranks)
            - percentile: 0-100, higher is better
            - rank: overall position
            - total_athletes: total number of athletes
            - score: sum of individual workout ranks (lower is better)
            - individual_ranks: list of (workout_num, rank) tuples
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data available for percentile calculation")

        # Find individual ranks for user's results by looking up matching results in the data
        # This uses the official ranks from the dataset for accuracy
        user_ranks = []
        for workout_num, result in results.items():
            display_col = f"workout_{workout_num}_display"
            rank_col = f"workout_{workout_num}_rank"

            # Find all athletes with the exact same result and use the average rank
            # CrossFit assigns sequential ranks to ties, so average gives the middle position
            user_type, user_value = self.parse_result(result)

            # Vectorized approach: filter DataFrame for matching results
            matching_ranks = []
            valid_mask = self.df[display_col].notna() & self.df[rank_col].notna()
            valid_df = self.df[valid_mask]

            for display_val, rank_val in zip(valid_df[display_col], valid_df[rank_col]):
                try:
                    other_type, other_value = self.parse_result(str(display_val))
                    if other_type == user_type and other_value == user_value:
                        matching_ranks.append(int(rank_val))
                except ValueError:
                    continue

            if matching_ranks:
                # Use average rank among ties (expected rank for this result)
                avg_rank = round(sum(matching_ranks) / len(matching_ranks))
                user_ranks.append((workout_num, avg_rank))
            else:
                # If no exact match found, calculate the rank
                _, calc_rank, _ = self.calculate_percentile_and_rank(workout_num, result)
                user_ranks.append((workout_num, calc_rank))

        user_score = sum(rank for _, rank in user_ranks)

        # Calculate scores for all athletes using vectorized pandas operations
        # Score = sum of ranks for each workout
        rank_cols = [f"workout_{w}_rank" for w in results.keys()]

        # Check that all rank columns exist
        missing_cols = [c for c in rank_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Create a mask for rows where all workout ranks are valid (not NaN)
        valid_mask = self.df[rank_cols].notna().all(axis=1)
        valid_df = self.df.loc[valid_mask, rank_cols]

        if valid_df.empty:
            raise ValueError("No valid overall scores available")

        # Sum ranks across all workouts (vectorized)
        athlete_scores = valid_df.sum(axis=1)

        total_athletes = len(athlete_scores)
        # Count how many athletes have a higher (worse) score
        athletes_beaten = (athlete_scores > user_score).sum()
        percentile = (athletes_beaten / total_athletes) * 100
        rank = (athlete_scores < user_score).sum() + 1

        return round(percentile, 2), int(rank), total_athletes, user_score, user_ranks


def interactive_console(calculator: PercentileCalculator, num_workouts: int = 3) -> None:
    """
    Run an interactive console for percentile calculations.

    Args:
        calculator: PercentileCalculator instance
        num_workouts: Number of workouts available
    """
    print("\n" + "=" * 60)
    print("CROSSFIT OPEN PERCENTILE CALCULATOR")
    print("=" * 60)
    print("\nEnter your workout results to see your percentile ranking.")
    print("Results can be entered as:")
    print("  - Reps: '300', '300 reps'")
    print("  - Time: '4:01' (minutes:seconds)")
    print("  - Weight: '130 lbs', '60 kg'")
    print("\nType '*' to calculate overall percentile for all workouts.")
    print("Type 'quit' or 'exit' to leave the calculator.")
    print("Type 'help' for more information.")
    print("-" * 60)

    # Store user's workout results for overall calculation
    user_results: dict = {}

    while True:
        try:
            # Ask for workout number
            workout_input = input(f"\nEnter workout number (1-{num_workouts}), '*' for overall, or 'quit': ").strip().lower()

            if workout_input in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break

            if workout_input == "help":
                print("\nHelp:")
                print(f"  - Enter a workout number (1 to {num_workouts})")
                print("  - Then enter your result for that workout")
                print("  - The calculator will show your percentile and ranking")
                print("  - A percentile of 95 means you beat 95% of athletes")
                print("  - Enter '*' to see your overall percentile across all workouts")
                print("  - Results can be: reps (300), time (4:01), weight (130 lbs)")
                if user_results:
                    print(f"\n  Your saved results: {user_results}")
                continue

            if workout_input == "*":
                # Calculate overall percentile
                missing_workouts = [w for w in range(1, num_workouts + 1) if w not in user_results]
                if missing_workouts:
                    print(f"\nMissing results for workout(s): {missing_workouts}")
                    print("Please enter results for all workouts first.")
                    for w in missing_workouts:
                        result = input(f"Enter your result for Workout {w}: ").strip()
                        if result.lower() in ("quit", "exit", "q"):
                            break
                        if result:
                            try:
                                # Validate the result
                                calculator.parse_result(result)
                                user_results[w] = result
                            except ValueError as e:
                                print(f"Error: {e}")
                                break
                    # Check again after collecting
                    missing_workouts = [w for w in range(1, num_workouts + 1) if w not in user_results]
                    if missing_workouts:
                        print("Cannot calculate overall percentile without all workout results.")
                        continue

                # Calculate overall
                try:
                    percentile, rank, total, score, individual_ranks = calculator.calculate_overall_percentile_and_rank(user_results)
                    print(f"\n{'=' * 50}")
                    print("  OVERALL RESULTS")
                    print(f"{'=' * 50}")
                    print("  Individual Workouts:")
                    for w_num, w_rank in individual_ranks:
                        print(f"    Workout {w_num}: {user_results[w_num]} - ranked {w_rank:,} out of {total:,}")
                    print(f"{'=' * 50}")
                    print(f"  Total Score: {score:,} points (sum of ranks)")
                    print(f"  Overall Ranking: ranked {rank:,} out of {total:,} athletes")
                    print(f"  Overall Percentile: {percentile:.2f}%")
                    print(f"  You beat {percentile:.1f}% of athletes!")
                    print(f"{'=' * 50}")
                except ValueError as e:
                    print(f"Error: {e}")
                continue

            try:
                workout_num = int(workout_input)
                if workout_num < 1 or workout_num > num_workouts:
                    print(f"Please enter a number between 1 and {num_workouts}")
                    continue
            except ValueError:
                print("Please enter a valid workout number")
                continue

            # Ask for result
            result = input(f"Enter your result for Workout {workout_num}: ").strip()

            if result.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break

            if not result:
                print("Please enter a result")
                continue

            # Calculate percentile and rank
            try:
                percentile, rank, total = calculator.calculate_percentile_and_rank(workout_num, result)
                # Store the result for overall calculation
                user_results[workout_num] = result
                print(f"\n{'=' * 50}")
                print(f"  Workout {workout_num} Result: {result}")
                print(f"  Ranking: ranked {rank:,} out of {total:,} athletes")
                print(f"  Percentile: {percentile:.2f}%")
                print(f"  You beat {percentile:.1f}% of athletes!")
                print(f"{'=' * 50}")
                
                # Calculate and display distribution
                distribution = calculator.calculate_distribution(workout_num, percentile)
                if distribution:
                    print(f"\n  DISTRIBUTION (Workout {workout_num})")
                    print(f"  {'-' * 46}")
                    
                    # Find max count for scaling the bar chart
                    max_count = max(d["athlete_count"] for d in distribution)
                    bar_width = 25
                    
                    for bucket in distribution:
                        pct_range = bucket["percentile_range"]
                        count = bucket["athlete_count"]
                        is_user = bucket["is_user_bucket"]
                        
                        # Scale bar length
                        bar_len = int((count / max_count) * bar_width) if max_count > 0 else 0
                        bar = "█" * bar_len + "░" * (bar_width - bar_len)
                        
                        # Highlight user's bucket
                        marker = " ◄── YOU" if is_user else ""
                        print(f"  {pct_range:>10} |{bar}| {count:>6,}{marker}")
                    
                    print(f"  {'-' * 46}")
            except ValueError as e:
                print(f"Error: {e}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def main():
    """Main entry point for the script."""
    # Get available options for help text
    division_choices = list(CrossFitLeaderboardScraper.DIVISIONS.values())
    region_choices = list(CrossFitLeaderboardScraper.REGIONS.values())
    scaled_choices = list(CrossFitLeaderboardScraper.SCALED.values())
    sort_choices = list(CrossFitLeaderboardScraper.SORT.values())

    parser = argparse.ArgumentParser(
        description="Scrape CrossFit Games Open leaderboard and calculate percentiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  Scrape 2025 Open Men's division:
    python crossfit_leaderboard_scraper.py --year 2025 --division "Men Individual"

  Load from CSV and filter:
    python crossfit_leaderboard_scraper.py --input data.csv --division "Women Individual"

  Scrape limited pages for testing:
    python crossfit_leaderboard_scraper.py --max-pages 10 --output test.csv

Available divisions: {', '.join(division_choices)}
Available regions: {', '.join(region_choices)}
Available scaled options: {', '.join(scaled_choices)}
Available sort options: {', '.join(sort_choices)}
"""
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Competition year (default: 2025)",
    )
    parser.add_argument(
        "--division",
        type=str,
        default="Men Individual",
        help="Division name (default: 'Men Individual')",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="Worldwide",
        help="Region name (default: 'Worldwide')",
    )
    parser.add_argument(
        "--scaled",
        type=str,
        default="Rx'd",
        help="Workout type (default: 'Rx'd')",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="Overall",
        help="Sort order (default: 'Overall')",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to scrape (default: all)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV file path to load instead of scraping",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (optional, used when --output-type is 'csv')",
    )
    parser.add_argument(
        "--output-type",
        type=str,
        choices=["csv", "supabase"],
        default="csv",
        help="Output destination type: 'csv' (default) or 'supabase'",
    )
    parser.add_argument(
        "--supabase-config",
        type=str,
        default="supabase_conn.yaml",
        help="Path to Supabase connection YAML config file (default: supabase_conn.yaml)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive percentile calculator",
    )

    args = parser.parse_args()

    # Either load from CSV or scrape
    if args.input:
        # Load from CSV file
        scraper, df = CrossFitLeaderboardScraper.load_from_csv(
            filename=args.input,
            year=args.year if args.year != 2025 else None,  # Only filter if explicitly set
            division=args.division if args.division != "Men Individual" else None,
            region=args.region if args.region != "Worldwide" else None,
            scaled=args.scaled if args.scaled != "Rx'd" else None,
            sort=args.sort if args.sort != "Overall" else None,
        )
    else:
        # Create scraper and fetch data
        scraper = CrossFitLeaderboardScraper(
            year=args.year,
            division=args.division,
            region=args.region,
            scaled=args.scaled,
            sort=args.sort,
        )
        df = scraper.scrape(max_pages=args.max_pages)

    # Display sample of data
    if not df.empty:
        print("\n" + "=" * 60)
        print("SAMPLE DATA (Top 10)")
        print("=" * 60)
        # Build display columns dynamically based on number of workouts
        display_cols = ["rank", "name", "country"]
        for i in range(1, scraper.num_workouts + 1):
            display_cols.append(f"workout_{i}_display")
        available_cols = [c for c in display_cols if c in df.columns]
        print(df[available_cols].head(10).to_string(index=False))

        # Save output if requested (and not just loading from CSV)
        if not args.input:
            if args.output_type == "csv" and args.output:
                scraper.save_to_csv(args.output)
            elif args.output_type == "supabase":
                scraper.save_to_supabase(args.supabase_config)

        # Run interactive calculator
        if not args.no_interactive:
            calculator = PercentileCalculator(df)
            interactive_console(calculator, num_workouts=scraper.num_workouts)
    else:
        print("No data was loaded or scraped.")
        sys.exit(1)


if __name__ == "__main__":
    main()
