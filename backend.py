"""
Media Collection Manager - Backend API
A FastAPI application for managing movies and TV shows with TMDB integration
"""

import os
import httpx
import hashlib
import secrets
import psycopg2
import threading

from enum     import Enum
from typing   import Generator, Optional, List, Dict, Any
from dotenv   import load_dotenv
from datetime import datetime

from fastapi                 import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware

from pydantic        import BaseModel, Field, ConfigDict, field_validator
from contextlib      import asynccontextmanager
from psycopg2.extras import RealDictCursor, Json

#============================================================
load_dotenv()

TMDB_API_KEY    = os.getenv("TMDB_API_KEY", "your_api_key_here")
TMDB_BASE_URL   = "https://api.themoviedb.org/3"
DATABASE_URL    = os.getenv("DATABASE_URL")  # PostgreSQL connection string from Supabase

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

_thread_local   = threading.local()
active_sessions = {}

#============================================================
def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt"""
    salt     = secrets.token_hex(16)
    pwd_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}${pwd_hash}"

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a hash"""
    try:
        salt, pwd_hash = hashed.split('$')
        return hashlib.sha256((password + salt).encode()).hexdigest() == pwd_hash
    except:
        return False

#============================================================
def create_session_token() -> str:
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)

#============================================================
class MediaType(str, Enum):
    MOVIE           = "movie"
    TV_SERIES       = "tv_series"

#============================================================
class Quality(str, Enum):
    SD              = "SD"
    HD              = "HD"
    FHD             = "FHD"
    UHD_4K          = "4K"
    UHD_8K          = "8K"

#============================================================
class BackupStatus(str, Enum):
    NOT_BACKED_UP   = "not_backed_up"
    BACKED_UP       = "backed_up"
    PENDING         = "pending"

#============================================================
class MediaBase(BaseModel):
    title           : str
    media_type      : MediaType
    tmdb_id         : Optional[int] = None
    overview        : Optional[str] = None
    release_date    : Optional[str] = None
    runtime         : Optional[int] = None
    genres          : Optional[List[str]] = []
    cast_list       : Optional[List[Dict[str, Any]]] = []
    crew            : Optional[List[Dict[str, Any]]] = []
    poster_path     : Optional[str] = None
    backdrop_path   : Optional[str] = None
    tmdb_rating     : Optional[float] = None
    tmdb_vote_count : Optional[int]  = None
    
    # Additional fields
    director        : Optional[str] = None
    cast_names      : Optional[str] = None
    country         : Optional[str] = None
    original_title  : Optional[str] = None
    
    seen            : bool = False
    user_rating     : Optional[float] = Field(None, ge=0, le=10)
    loaned_to       : Optional[str] = None
    backed_up       : BackupStatus = BackupStatus.NOT_BACKED_UP
    quality         : Optional[Quality] = None
    notes           : Optional[str] = None
    date_added      : Optional[datetime] = None
    date_watched    : Optional[datetime] = None
    location        : Optional[str] = None
    file_size       : Optional[str] = None
    seasons         : Optional[int] = None
    episodes        : Optional[int] = None
    episode_details : Optional[List[Dict[str, Any]]] = []
    
    @field_validator('overview', 'director', 'cast_names', 'country', 'original_title', 
                     'loaned_to', 'notes', 'location', 'file_size', 'poster_path', 
                     'backdrop_path', 'release_date', 'date_watched', mode='before')
    @classmethod
    def empty_str_to_none(cls, v):
        if v == '' or v is None:
            return None
        return v
    
    @field_validator('quality', mode='before')
    @classmethod
    def quality_empty_to_none(cls, v):
        if v == '' or v is None:
            return None
        # Try to convert string to Quality enum if it's a valid value
        if isinstance(v, str):
            try:
                return Quality(v)
            except ValueError:
                return None
        return v
    
    @field_validator('backed_up', mode='before')
    @classmethod
    def backed_up_default(cls, v):
        if v == '' or v is None:
            return BackupStatus.NOT_BACKED_UP
        # Try to convert string to BackupStatus enum if it's a valid value
        if isinstance(v, str):
            try:
                return BackupStatus(v)
            except ValueError:
                return BackupStatus.NOT_BACKED_UP
        return v
    
    @field_validator('runtime', 'seasons', 'episodes', 'tmdb_id', 'tmdb_vote_count', mode='before')
    @classmethod
    def int_empty_to_none(cls, v):
        if v == '' or v is None:
            return None
        if isinstance(v, str):
            try:
                return int(v) if v.strip() else None
            except (ValueError, AttributeError):
                return None
        return v
    
    @field_validator('tmdb_rating', 'user_rating', mode='before')
    @classmethod
    def float_empty_to_none(cls, v):
        if v == '' or v is None:
            return None
        if isinstance(v, str):
            try:
                return float(v) if v.strip() else None
            except (ValueError, AttributeError):
                return None
        return v
    
    @field_validator('media_type', mode='before')
    @classmethod
    def media_type_from_str(cls, v):
        if isinstance(v, str):
            try:
                return MediaType(v)
            except ValueError:
                return MediaType.MOVIE  # Default to movie
        return v

#============================================================
class MediaCreate(MediaBase):
    pass

#============================================================
class MediaUpdate(MediaBase):
    title               : Optional[str] = None
    media_type          : Optional[MediaType] = None

#============================================================
class Media(MediaBase):
    id                  : int
    date_added          : datetime
    last_updated        : datetime
    
    model_config        = ConfigDict(from_attributes=True)

#============================================================
class DashboardStats(BaseModel):
    total_movies            : int
    total_tv_series         : int
    total_media             : int
    seen_count              : int
    unseen_count            : int
    backed_up_count         : int
    not_backed_up_count     : int
    quality_distribution    : Dict[str, int]
    genre_distribution      : Dict[str, int]
    pending_by_quality      : Dict[str, int]
    average_user_rating     : Optional[float]
    average_tmdb_rating     : Optional[float]
    total_runtime_minutes   : int
    loaned_out_count        : int
    recently_added          : List[Dict[str, Any]]
    top_rated_by_user       : List[Dict[str, Any]]

#============================================================
class LoginRequest(BaseModel):
    username                : str
    password                : str
    remember_me             : bool = False

#============================================================
class LoginResponse(BaseModel):
    success                 : bool
    token                   : Optional[str] = None
    username                : Optional[str] = None
    message                 : Optional[str] = None

#============================================================
def init_database():
    conn    = psycopg2.connect(DATABASE_URL)
    cursor  = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS media (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            media_type TEXT NOT NULL,
            tmdb_id INTEGER,
            overview TEXT,
            release_date TEXT,
            runtime INTEGER,
            genres JSONB,
            cast_list JSONB,
            crew JSONB,
            poster_path TEXT,
            backdrop_path TEXT,
            tmdb_rating NUMERIC(3, 1),
            tmdb_vote_count INTEGER,
            director TEXT,
            cast_names TEXT,
            country TEXT,
            original_title TEXT,
            seen BOOLEAN DEFAULT FALSE,
            user_rating NUMERIC(3, 1),
            loaned_to TEXT,
            backed_up TEXT DEFAULT 'not_backed_up',
            quality TEXT,
            notes TEXT,
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            date_watched TIMESTAMP,
            location TEXT,
            file_size TEXT,
            seasons INTEGER,
            episodes INTEGER,
            episode_details JSONB,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT unique_tmdb_media UNIQUE(tmdb_id, media_type)
        )
    """)
    
    # Create indexes for better search performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_media_seen ON media(seen)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_media_backed_up ON media(backed_up)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_media_quality ON media(quality)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_media_type ON media(media_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_media_tmdb_id ON media(tmdb_id)")
    
    # Create users table for authentication
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create default admin account if no users exist
    cursor.execute("SELECT COUNT(*) as count FROM users")
    result = cursor.fetchone()
    
    # Handle both dict (RealDictCursor) and tuple results
    if isinstance(result, dict):
        user_count = result.get('count', 0)
    else:
        user_count = result[0] if result else 0
    
    if user_count == 0:
        # Default account: administrator / cinema@25
        default_password_hash = hash_password("cinema@25")
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
            ("administrator", default_password_hash)
        )
        print("Created default admin account: administrator / cinema@25")
    
    conn.commit()
    conn.close()

#============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_database()
    yield
    # Shutdown
    pass

#============================================================
# Initialize FastAPI app
app = FastAPI(
    title       = "Cinematheque Backend API",
    description = "API for managing personal media collection with TMDB integration",
    version     = "1.0.0",
    lifespan    = lifespan
)

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "https://your-netlify-site.netlify.app",  # replace with real Netlify URL once you have it
]

if os.getenv("ALLOW_ALL_ORIGINS", "true").lower() == "true":
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#============================================================
def get_db() -> Generator:
    """Get a thread-safe database connection"""
    # Create a new connection for each request
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()

#============================================================
async def fetch_from_tmdb(endpoint: str, params: dict = None):
    """Fetch data from TMDB API with better error handling"""
    if not TMDB_API_KEY or TMDB_API_KEY == "your_tmdb_api_key_here":
        print("TMDB API key not configured")
        return None
    
    async with httpx.AsyncClient(timeout=10.0) as client:  # Add timeout
        try:
            params = params or {}
            params["api_key"] = TMDB_API_KEY
            
            url = f"{TMDB_BASE_URL}{endpoint}"
            print(f"Fetching from TMDB: {url}")
            print(f"Parameters: {params}")
            
            response = await client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            print(f"TMDB response received, status: {response.status_code}")
            return data
            
        except httpx.ConnectError as e:
            print(f"TMDB connection error: {e}")
            return None
        except httpx.TimeoutException as e:
            print(f"TMDB timeout error: {e}")
            return None
        except httpx.HTTPStatusError as e:
            print(f"TMDB HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            print(f"TMDB API unexpected error: {type(e).__name__}: {e}")
            return None

async def search_tmdb(query: str, media_type: str):
    """Search for media on TMDB"""
    endpoint = "/search/movie" if media_type == "movie" else "/search/tv"
    return await fetch_from_tmdb(endpoint, {"query": query})

async def get_tmdb_details(tmdb_id: int, media_type: str):
    """Get detailed information from TMDB"""
    endpoint = f"/movie/{tmdb_id}" if media_type == "movie" else f"/tv/{tmdb_id}"
    details  = await fetch_from_tmdb(endpoint, {"append_to_response": "credits"})
    return details

#============================================================
def verify_session(authorization: Optional[str] = Header(None)) -> str:
    """Verify session token from Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = authorization.replace("Bearer ", "")
    
    if token not in active_sessions:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    return active_sessions[token]

#============================================================
@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest, conn = Depends(get_db)):
    """Authenticate user and create session"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get user from database
    cursor.execute(
        "SELECT username, password_hash FROM users WHERE username = %s",
        (request.username,)
    )
    user        = cursor.fetchone()
    
    if not user:
        return LoginResponse(success=False, message="Invalid username or password")
    
    # user is a dict with RealDictCursor
    username    = user['username']
    stored_hash = user['password_hash']
    
    # Verify password
    if not verify_password(request.password, stored_hash):
        return LoginResponse(success=False, message="Invalid username or password")
    
    # Create session token
    token = create_session_token()
    active_sessions[token] = username
    
    return LoginResponse(
        success  = True,
        token    = token,
        username = username,
        message  = "Login successful"
    )

@app.post("/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    """Logout user and invalidate session"""
    if authorization:
        token = authorization.replace("Bearer ", "")
        if token in active_sessions:
            del active_sessions[token]
    
    return {"message": "Logged out successfully"}

@app.get("/auth/verify")
async def verify_auth(username: str = Depends(verify_session)):
    """Verify if session is still valid"""
    return {"authenticated": True, "username": username}

@app.get("/")
async def root():
    return {
        "message"   : "Cinematheque Backend API",
        "version"   : "1.0.0",
        "endpoints" : {
            "login"       : "/auth/login",
            "media"       : "/media",
            "search_tmdb" : "/tmdb/search",
            "stats"       : "/stats"
        }
    }

@app.get("/media", response_model=List[Media])
async def get_media(
    conn       = Depends(get_db),
    media_type : Optional[MediaType] = None,
    seen       : Optional[bool] = None,
    backed_up  : Optional[BackupStatus] = None,
    quality    : Optional[Quality] = None,
    loaned     : Optional[bool] = None,
    search     : Optional[str] = None,
    limit      : int = Query(100, le=500),
    offset     : int = 0
):
    """Get all media with optional filters"""
    query  = "SELECT * FROM media WHERE 1=1"
    params = []
    
    if media_type:
        query += " AND media_type = %s"
        params.append(media_type)
    
    if seen is not None:
        query += " AND seen = %s"
        params.append(seen)  # PostgreSQL handles boolean directly
    
    if backed_up:
        query += " AND backed_up = %s"
        params.append(backed_up)
    
    if quality:
        query += " AND quality = %s"
        params.append(quality)
    
    if loaned is not None:
        if loaned:
            query += " AND loaned_to IS NOT NULL AND loaned_to != ''"
        else:
            query += " AND (loaned_to IS NULL OR loaned_to = '')"
    
    if search:
        query += " AND (title ILIKE %s OR overview ILIKE %s OR notes ILIKE %s OR location ILIKE %s)"
        search_param = f"%{search}%"
        params.extend([search_param, search_param, search_param, search_param])
    
    query += " ORDER BY date_added DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(query, params)
    rows   = cursor.fetchall()
    
    media_list = []
    for row in rows:
        media_dict = dict(row)
        # JSONB fields are automatically parsed by PostgreSQL
        for field in ['genres', 'cast_list', 'crew', 'episode_details']:
            if media_dict.get(field) is None:
                media_dict[field] = []
        
        media_list.append(media_dict)
    
    return media_list

@app.get("/media/{media_id}", response_model=Media)
async def get_media_by_id(media_id: int, conn = Depends(get_db)):
    """Get a specific media item by ID"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM media WHERE id = %s", (media_id,))
    row = cursor.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="Media not found")
    
    media_dict = dict(row)
    # JSONB fields are already parsed by PostgreSQL, just ensure they're not None
    for field in ['genres', 'cast_list', 'crew', 'episode_details']:
        if media_dict.get(field) is None:
            media_dict[field] = []
    
    return media_dict

@app.post("/media", response_model=Media)
async def create_media(media: MediaCreate, conn = Depends(get_db)):
    """Create a new media entry"""
    cursor               = conn.cursor(cursor_factory=RealDictCursor)
    
    # Convert lists to JSON strings for storage
    # Prepare JSONB fields - PostgreSQL handles these natively
    genres_json          = Json(media.genres) if media.genres else Json([])
    cast_list_json       = Json(media.cast_list) if media.cast_list else Json([])
    crew_json            = Json(media.crew) if media.crew else Json([])
    episode_details_json = Json(media.episode_details) if media.episode_details else Json([])
    
    try:
        cursor.execute("""
            INSERT INTO media (
                title, media_type, tmdb_id, overview, release_date, runtime,
                genres, cast_list, crew, poster_path, backdrop_path, tmdb_rating,
                tmdb_vote_count, director, cast_names, country, original_title,
                seen, user_rating, loaned_to, backed_up, quality, notes, 
                date_watched, location, file_size, seasons, episodes,
                episode_details
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
        """, (
            media.title, 
            media.media_type.value if hasattr(media.media_type, 'value') else media.media_type, 
            media.tmdb_id, 
            media.overview,
            media.release_date, 
            media.runtime,
            genres_json, 
            cast_list_json, 
            crew_json,
            media.poster_path, 
            media.backdrop_path, 
            media.tmdb_rating,
            media.tmdb_vote_count, 
            media.director, 
            media.cast_names, 
            media.country,
            media.original_title, 
            media.seen, 
            media.user_rating,
            media.loaned_to, 
            media.backed_up.value if hasattr(media.backed_up, 'value') else media.backed_up, 
            media.quality.value if media.quality and hasattr(media.quality, 'value') else media.quality, 
            media.notes, 
            media.date_watched, 
            media.location, 
            media.file_size,
            media.seasons, 
            media.episodes, 
            episode_details_json
        ))
        
        conn.commit()
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=500, detail="Failed to retrieve created media")
            
        media_dict = dict(row)
        
        # JSONB fields are automatically returned as dict/list, no parsing needed
        # Ensure they're not None
        for field in ['genres', 'cast_list', 'crew', 'episode_details']:
            if media_dict.get(field) is None:
                media_dict[field] = []
        
        return media_dict
        
    except psycopg2.IntegrityError as e:
        conn.rollback()
        if "unique_tmdb_media" in str(e):
            raise HTTPException(status_code=400, detail="This media already exists in your collection")
        raise HTTPException(status_code=400, detail=f"Invalid data: {str(e)}")
    except Exception as e:
        conn.rollback()
        print(f"Error creating media: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create media: {str(e)}")

@app.put("/media/{media_id}", response_model=Media)
async def update_media(
    media_id: int,
    media: MediaUpdate,
    conn = Depends(get_db)
):
    """Update an existing media entry"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Check if media exists
    cursor.execute("SELECT * FROM media WHERE id = %s", (media_id,))
    if not cursor.fetchone():
        raise HTTPException(status_code=404, detail="Media not found")
    
    # Build update query dynamically
    update_fields = []
    params = []
    
    for field, value in media.dict(exclude_unset=True).items():
        if field in ['genres', 'cast_list', 'crew', 'episode_details']:
            update_fields.append(f"{field} = %s")
            # Ensure genres is always a list
            if field == 'genres' and isinstance(value, str):
                value = [g.strip() for g in value.split(',') if g.strip()]
            params.append(Json(value) if value else Json([]))
        elif field == 'seen':
            update_fields.append(f"{field} = %s")
            params.append(value)  # PostgreSQL uses TRUE/FALSE
        elif field in ['media_type', 'backed_up', 'quality'] and value is not None:
            update_fields.append(f"{field} = %s")
            params.append(value.value if hasattr(value, 'value') else value)
        elif value is not None:
            update_fields.append(f"{field} = %s")
            params.append(value)
    
    if update_fields:
        update_fields.append("last_updated = CURRENT_TIMESTAMP")
        query = f"UPDATE media SET {', '.join(update_fields)} WHERE id = %s"
        params.append(media_id)
        
        try:
            cursor.execute(query, params)
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to update media: {str(e)}")
    
    # Fetch and return updated media
    cursor.execute("SELECT * FROM media WHERE id = %s", (media_id,))
    row        = cursor.fetchone()
    media_dict = dict(row)
    
    # JSONB fields are already parsed by PostgreSQL, just ensure they're not None
    for field in ['genres', 'cast_list', 'crew', 'episode_details']:
        if media_dict.get(field) is None:
            media_dict[field] = []
    
    return media_dict

@app.delete("/media/{media_id}")
async def delete_media(media_id: int, conn = Depends(get_db)):
    """Delete a media entry"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("SELECT * FROM media WHERE id = %s", (media_id,))
    if not cursor.fetchone():
        raise HTTPException(status_code=404, detail="Media not found")
    
    cursor.execute("DELETE FROM media WHERE id = %s", (media_id,))
    conn.commit()
    
    return {"message": "Media deleted successfully"}

@app.get("/tmdb/search")
async def search_tmdb_endpoint(
    query: str,
    media_type: MediaType = MediaType.MOVIE
):
    """Search TMDB for movies or TV shows"""
    if not query:
        return {"results": [], "error": "No search query provided"}
    
    print(f"TMDB Search: query='{query}', type={media_type}")
    
    results = await search_tmdb(query, media_type)
    if results is None:
        return {
            "results": [], 
            "error": "Could not connect to TMDB. Check your internet connection and API key."
        }
    
    if not results:
        return {"results": [], "message": "No results found"}
    
    # Format results for frontend
    formatted_results = []
    for item in results.get("results", []):
        formatted_results.append({
            "tmdb_id"       : item.get("id"),
            "title"         : item.get("title" if media_type == "movie" else "name"),
            "overview"      : item.get("overview"),
            "release_date"  : item.get("release_date" if media_type == "movie" else "first_air_date"),
            "poster_path"   : item.get("poster_path"),
            "backdrop_path" : item.get("backdrop_path"),
            "vote_average"  : item.get("vote_average"),
            "vote_count"    : item.get("vote_count")
        })
    
    print(f"TMDB Search found {len(formatted_results)} results")
    return {"results": formatted_results}

@app.get("/tmdb/details/{tmdb_id}")
async def get_tmdb_details_endpoint(
    tmdb_id: int,
    media_type: MediaType = MediaType.MOVIE
):
    """Get detailed information from TMDB"""
    print(f"TMDB Details: id={tmdb_id}, type={media_type}")
    
    details = await get_tmdb_details(tmdb_id, media_type)
    if not details:
        print(f"Failed to fetch TMDB details for {tmdb_id}")
        raise HTTPException(
            status_code = 503, 
            detail      = "Could not fetch details from TMDB. You can still add the media manually."
        )
    
    # Extract country information
    country_name = None
    production_countries = details.get("production_countries") or []
    if production_countries:
        country_name = production_countries[0].get("name")
    if not country_name:
        origin_countries = details.get("origin_country") or []
        if origin_countries:
            # origin_country is typically a list of ISO codes
            country_name = origin_countries[0]

    # Extract relevant information
    formatted_details = {
        "tmdb_id"           : details.get("id"),
        "title"             : details.get("title" if media_type == MediaType.MOVIE else "name"),
        "original_title"    : details.get("original_title" if media_type == MediaType.MOVIE else "original_name"),
        "overview"          : details.get("overview"),
        "release_date"      : details.get("release_date" if media_type == MediaType.MOVIE else "first_air_date"),
        "runtime"           : details.get("runtime") if media_type == MediaType.MOVIE else None,
        "genres"            : [g["name"] for g in details.get("genres", [])],
        "poster_path"       : details.get("poster_path"),
        "backdrop_path"     : details.get("backdrop_path"),
        "tmdb_rating"       : details.get("vote_average"),
        "tmdb_vote_count"   : details.get("vote_count"),
        "country"           : country_name,
        "cast"              : [],
        "crew"              : []
    }
    
    # Add cast and crew information
    if "credits" in details:
        for person in details["credits"].get("cast", [])[:10]:
            formatted_details["cast"].append({
                "name"         : person.get("name"),
                "character"    : person.get("character"),
                "profile_path" : person.get("profile_path")
            })
        
        for person in details["credits"].get("crew", []):
            if person.get("job") in ["Director", "Producer", "Screenplay", "Writer"]:
                formatted_details["crew"].append({
                    "name"         : person.get("name"),
                    "job"          : person.get("job"),
                    "profile_path" : person.get("profile_path")
                })
    
    # TV series specific information
    if media_type == MediaType.TV_SERIES:
        formatted_details["seasons"]  = details.get("number_of_seasons")
        formatted_details["episodes"] = details.get("number_of_episodes")

        # Fetch all episode-level details from TMDB to pre-fill episode_details
        # Only keep season number, episode number and title; do not store plot/overview.
        episode_details: list[dict[str, Any]] = []

        seasons_info = details.get("seasons") or []
        print(f"TV Series: Fetching {len(seasons_info)} seasons for TMDB ID {tmdb_id}")
        
        for idx, season in enumerate(seasons_info):
            season_number = season.get("season_number")
            # Skip specials or invalid season numbers
            if not season_number or season_number <= 0:
                print(f"Skipping season {season_number} (specials or invalid)")
                continue

            # Add small delay between requests to avoid rate limiting (except first request)
            if idx > 0:
                import asyncio
                await asyncio.sleep(0.3)  # 300ms delay between seasons
            
            print(f"Fetching season {season_number}...")
            try:
                season_data = await fetch_from_tmdb(
                    f"/tv/{tmdb_id}/season/{season_number}",
                    params={}
                )
                if not season_data:
                    print(f"Warning: No data returned for season {season_number}")
                    continue

                episodes = season_data.get("episodes") or []
                print(f"Season {season_number}: Found {len(episodes)} episodes")
                
                for ep in episodes:
                    episode_details.append({
                        "season_number"  : season_number,
                        "episode_number" : ep.get("episode_number"),
                        "title"          : ep.get("name")
                    })
            except Exception as e:
                print(f"Error fetching season {season_number}: {e}")
                continue
        
        print(f"Total episodes fetched: {len(episode_details)}")
        formatted_details["episode_details"] = episode_details
    else:
        # For movies, keep the field present but empty for consistency
        formatted_details["episode_details"] = []

    return formatted_details

@app.get("/stats", response_model=DashboardStats)
async def get_stats(conn = Depends(get_db)):
    """Get dashboard statistics"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Basic counts
    cursor.execute("SELECT COUNT(*) as count FROM media WHERE media_type = 'movie'")
    total_movies = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM media WHERE media_type = 'tv_series'")
    total_tv_series = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM media")
    total_media = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM media WHERE seen = TRUE")
    seen_count = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM media WHERE seen = FALSE")
    unseen_count = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM media WHERE backed_up = 'backed_up'")
    backed_up_count = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM media WHERE backed_up != 'backed_up'")
    not_backed_up_count = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM media WHERE loaned_to IS NOT NULL AND loaned_to != ''")
    loaned_out_count = cursor.fetchone()['count']
    
    # Quality distribution
    cursor.execute("SELECT quality as quality, COUNT(*) as count FROM media WHERE quality IS NOT NULL GROUP BY quality")
    quality_rows         = cursor.fetchall()
    quality_distribution = {row['quality']: row['count'] for row in quality_rows}
    
    # Pending backup by quality
    cursor.execute("SELECT quality as quality, COUNT(*) as count FROM media WHERE backed_up = 'pending' AND quality IS NOT NULL GROUP BY quality")
    pending_rows         = cursor.fetchall()
    pending_by_quality   = {row['quality']: row['count'] for row in pending_rows}
    
    # Genre distribution - JSONB is automatically parsed
    cursor.execute("SELECT genres FROM media WHERE genres IS NOT NULL AND genres != '[]'::jsonb")
    genre_rows   = cursor.fetchall()
    genre_counts = {}
    for row in genre_rows:
        genres = row['genres'] if row.get('genres') else []  # RealDictCursor returns dict
        if isinstance(genres, list):
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    # Sort genres by count and take top 10
    genre_distribution = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Average ratings
    cursor.execute("SELECT AVG(user_rating) as avg FROM media WHERE user_rating IS NOT NULL")
    avg_user = cursor.fetchone()['avg']
    
    cursor.execute("SELECT AVG(tmdb_rating) as avg FROM media WHERE tmdb_rating IS NOT NULL")
    avg_tmdb = cursor.fetchone()['avg']
    
    # Total runtime
    cursor.execute("SELECT SUM(runtime) as sum FROM media WHERE runtime IS NOT NULL")
    total_runtime = cursor.fetchone()['sum'] or 0
    
    # Recently added (last 5)
    cursor.execute("""
        SELECT id, title, 
        case when media_type = 'tv_series' then 'tv series' else media_type end as media_type, poster_path, date_added 
        FROM media 
        ORDER BY date_added DESC 
        LIMIT 4
    """)
    recent_rows = cursor.fetchall()
    # RealDictCursor returns dicts
    recently_added = [
        {
            'id'          : row['id'],
            'title'       : row['title'],
            'media_type'  : row['media_type'],
            'poster_path' : row['poster_path'],
            'date_added'  : row['date_added']
        }
        for row in recent_rows
    ]
    
    # Top rated by user (top 5)
    cursor.execute("""
        SELECT id, title, media_type, user_rating, poster_path 
        FROM media 
        WHERE user_rating IS NOT NULL 
        ORDER BY user_rating DESC 
        LIMIT 4
    """)
    top_rated_rows = cursor.fetchall()
    # RealDictCursor returns dicts, just extract the fields we need
    top_rated = [
        {
            'id'          : row['id'],
            'title'       : row['title'],
            'media_type'  : row['media_type'],
            'user_rating' : row['user_rating'],
            'poster_path' : row['poster_path']
        }
        for row in top_rated_rows
    ]
    
    return DashboardStats(
        total_movies          = total_movies,
        total_tv_series       = total_tv_series,
        total_media           = total_media,
        seen_count            = seen_count,
        unseen_count          = unseen_count,
        backed_up_count       = backed_up_count,
        not_backed_up_count   = not_backed_up_count,
        quality_distribution  = quality_distribution,
        genre_distribution    = genre_distribution,
        pending_by_quality    = pending_by_quality,
        average_user_rating   = round(avg_user, 2) if avg_user else None,
        average_tmdb_rating   = round(avg_tmdb, 2) if avg_tmdb else None,
        total_runtime_minutes = total_runtime,
        loaned_out_count      = loaned_out_count,
        recently_added        = recently_added,
        top_rated_by_user     = top_rated
    )

if __name__ == "__main__":
    import uvicorn
    # Get host from environment or default to 0.0.0.0 for network access
    import os
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    print(f"Starting server on {host}:{port}")
    print(f"API will be accessible at:")
    print(f"  - http://localhost:{port}")
    print(f"  - http://YOUR_LOCAL_IP:{port}")
    print(f"API documentation at: http://localhost:{port}/docs")
    
    uvicorn.run("backend:app", host=host, port=port, reload=True)

# For Vercel deployment
app_handler = app