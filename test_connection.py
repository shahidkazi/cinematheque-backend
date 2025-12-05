#!/usr/bin/env python3
"""
Supabase Connection Diagnostic Tool
Run this to test your database connection
"""

import sys

print("🔍 Supabase Connection Diagnostic\n")
print("=" * 50)

# Test 1: Check environment file
print("\n1️⃣ Checking .env file...")
try:
    with open('.env', 'r') as f:
        lines = f.readlines()
    
    db_url = None
    api_key = None
    
    for line in lines:
        if line.startswith('DATABASE_URL='):
            db_url = line.split('=', 1)[1].strip()
        elif line.startswith('TMDB_API_KEY='):
            api_key = line.split('=', 1)[1].strip()
    
    if db_url:
        print("   ✅ DATABASE_URL found")
        # Hide password for display
        safe_url = db_url.split('@')[0].split(':')[0:2]
        print(f"   URL format: {safe_url[0]}://...")
    else:
        print("   ❌ DATABASE_URL not found in .env")
        sys.exit(1)
    
    if api_key:
        print("   ✅ TMDB_API_KEY found")
    else:
        print("   ⚠️  TMDB_API_KEY not found")
        
except FileNotFoundError:
    print("   ❌ .env file not found!")
    print("   Create .env file in this directory")
    sys.exit(1)

# Test 2: Check psycopg2 installation
print("\n2️⃣ Checking psycopg2 installation...")
try:
    import psycopg2
    print("   ✅ psycopg2 installed")
    print(f"   Version: {psycopg2.__version__}")
except ImportError:
    print("   ❌ psycopg2 not installed")
    print("   Run: pip3 install psycopg2-binary")
    sys.exit(1)

# Test 3: Check dotenv
print("\n3️⃣ Checking python-dotenv...")
try:
    from dotenv import load_dotenv
    print("   ✅ python-dotenv installed")
    load_dotenv()
except ImportError:
    print("   ❌ python-dotenv not installed")
    print("   Run: pip3 install python-dotenv")
    sys.exit(1)

# Test 4: Parse connection string
print("\n4️⃣ Parsing connection string...")
try:
    from urllib.parse import urlparse
    parsed = urlparse(db_url)
    
    print(f"   Hostname: {parsed.hostname}")
    print(f"   Port: {parsed.port}")
    print(f"   Database: {parsed.path[1:]}")
    print(f"   Username: {parsed.username}")
    
    # Check hostname format
    if '.pooler.supabase.com' in parsed.hostname:
        print("   ✅ Using pooler connection (recommended)")
    elif '.supabase.co' in parsed.hostname:
        print("   ⚠️  Using direct connection (might have issues)")
        print("   💡 Try Session/Transaction mode in Supabase")
    else:
        print("   ❌ Hostname doesn't look like Supabase")
        
except Exception as e:
    print(f"   ❌ Error parsing URL: {e}")
    sys.exit(1)

# Test 5: DNS Resolution
print("\n5️⃣ Testing DNS resolution...")
import socket
try:
    ip = socket.gethostbyname(parsed.hostname)
    print(f"   ✅ DNS resolved: {ip}")
except socket.gaierror:
    print(f"   ❌ Cannot resolve hostname: {parsed.hostname}")
    print("   💡 Possible fixes:")
    print("      - Check internet connection")
    print("      - Try different network")
    print("      - Get fresh connection string from Supabase")
    print("      - Use Session/Transaction mode (not Direct)")
    sys.exit(1)

# Test 6: Network connectivity
print("\n6️⃣ Testing network connectivity...")
import socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result = sock.connect_ex((parsed.hostname, parsed.port or 5432))
    sock.close()
    
    if result == 0:
        print(f"   ✅ Port {parsed.port or 5432} is reachable")
    else:
        print(f"   ❌ Cannot connect to port {parsed.port or 5432}")
        print("   💡 Firewall or network issue")
except Exception as e:
    print(f"   ❌ Connection test failed: {e}")

# Test 7: Database connection
print("\n7️⃣ Testing database connection...")
try:
    conn = psycopg2.connect(db_url)
    print("   ✅ Successfully connected to database!")
    
    # Test query
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()[0]
    print(f"   PostgreSQL version: {version[:50]}...")
    
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Your connection works!")
    print("=" * 50)
    print("\nYou can now run: python3 backend.py")
    
except psycopg2.OperationalError as e:
    print(f"   ❌ Connection failed: {str(e)[:100]}")
    print("\n" + "=" * 50)
    print("💡 Troubleshooting Steps:")
    print("=" * 50)
    print("\n1. Get fresh connection string from Supabase:")
    print("   - Dashboard → Settings → Database")
    print("   - Use 'Session mode' or 'Transaction mode'")
    print("   - Copy the URI (should have .pooler.supabase.com)")
    print("\n2. Update .env with new connection string")
    print("   - Remember to encode @ as %40 in password")
    print("\n3. Check Supabase project status:")
    print("   - Make sure project is 'Active' (not Paused)")
    print("\n4. Try different network:")
    print("   - Disconnect VPN if using one")
    print("   - Try phone hotspot")
    
except Exception as e:
    print(f"   ❌ Unexpected error: {e}")

print()
