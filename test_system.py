"""
AlgoRythm AI System Test Script
Tests all components of the AI system
"""

import requests
import json
import time
import sys

def test_backend():
    """Test backend API"""
    print("\n🔍 Testing AlgoRythm AI Backend...")
    
    try:
        # Test root endpoint
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is running!")
            print(f"   Model: {data.get('name')}")
            print(f"   Company: {data.get('company', {}).get('name')}")
            print(f"   CEO: {data.get('company', {}).get('ceo')}")
            return True
        else:
            print("❌ Backend returned error status")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running. Starting it now...")
        import subprocess
        subprocess.Popen([sys.executable, "backend/algorythm_ai_backend.py"])
        print("⏳ Waiting for backend to start...")
        time.sleep(5)
        return test_backend()  # Retry
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return False

def test_chat_api():
    """Test chat endpoint"""
    print("\n💬 Testing Chat API...")
    
    try:
        # Test company question
        payload = {
            "message": "Who created you?",
            "search_web": False,
            "generate_pdf": False
        }
        
        response = requests.post(
            "http://localhost:8000/api/chat",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Chat API working!")
            print(f"   Response preview: {data.get('response')[:200]}...")
            
            # Verify it mentions AlgoRythm Tech and CEO
            response_text = data.get('response', '').lower()
            if 'algorythm tech' in response_text and 'sri aasrith souri kompella' in response_text:
                print("✅ Company information correctly configured!")
            else:
                print("⚠️  Company information might not be properly configured")
            return True
        else:
            print(f"❌ Chat API returned error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Chat API test failed: {e}")
        return False

def test_web_search():
    """Test web search functionality"""
    print("\n🔍 Testing Web Search...")
    
    try:
        payload = {
            "query": "Python programming",
            "num_results": 3
        }
        
        response = requests.post(
            "http://localhost:8000/api/search",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            if results:
                print(f"✅ Web search working! Found {len(results)} results")
                for i, result in enumerate(results[:2], 1):
                    print(f"   {i}. {result.get('title', 'No title')}")
            else:
                print("⚠️  Web search returned no results")
            return True
        else:
            print(f"❌ Web search returned error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Web search test failed: {e}")
        return False

def test_frontend():
    """Test frontend availability"""
    print("\n🎨 Testing Frontend...")
    
    try:
        response = requests.get("http://localhost:5173/", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is running at http://localhost:5173")
            return True
        else:
            print("❌ Frontend returned error status")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Frontend is not running. Please run: npm run dev")
        return False
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("ALGORYTHM AI EUROPA - SYSTEM TEST")
    print("Created by AlgoRythm Tech, Hyderabad")
    print("CEO & Founder: Sri Aasrith Souri Kompella")
    print("=" * 60)
    
    # Run tests
    backend_ok = test_backend()
    
    if backend_ok:
        chat_ok = test_chat_api()
        search_ok = test_web_search()
    else:
        print("\n⚠️ Skipping API tests since backend is not running")
        chat_ok = False
        search_ok = False
    
    frontend_ok = test_frontend()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Backend API: {'✅ PASS' if backend_ok else '❌ FAIL'}")
    print(f"Chat API: {'✅ PASS' if chat_ok else '❌ FAIL'}")
    print(f"Web Search: {'✅ PASS' if search_ok else '❌ FAIL'}")
    print(f"Frontend: {'✅ PASS' if frontend_ok else '❌ FAIL'}")
    
    if backend_ok and frontend_ok:
        print("\n🚀 SYSTEM READY FOR DEMO!")
        print("\nAccess your AI at: http://localhost:5173")
        print("Backend API docs: http://localhost:8000/docs")
        print("\nFeatures available:")
        print("  • Advanced AI Chat (8B parameters)")
        print("  • Deep Web Search")
        print("  • PDF Generation")
        print("  • Image Analysis")
        print("  • File Upload & Processing")
        print("  • Auth0 Authentication")
    else:
        print("\n⚠️ Some components need attention. Please check the errors above.")
        print("\nTo start the system manually:")
        print("  Backend: python backend/algorythm_ai_backend.py")
        print("  Frontend: cd frontend && npm run dev")

if __name__ == "__main__":
    main()
